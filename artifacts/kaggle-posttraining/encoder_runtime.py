"""Self-contained Colab retrieval training; embedded by build_notebooks.py.

Single-device FP32 VGG16-GeM, clean mining, train-only augmentation, Drive
epoch-boundary checkpoints. No Kaggle credentials or repository install needed.
"""
import copy
import gc
import hashlib
import io
import json
import math
import os
import pickle
import random
import shutil
import stat
import sys
import time
import traceback
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import Dataset, DataLoader

VERSION = 'colab_b0_robust_v1'
WEIGHTS_HASH = 'd369c8e593250e421eda192512263dccd35edb5c4b57e97312651bac10a51b7d'
WEIGHTS_URL = 'https://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth'
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_CACHE = {}
IMAGE_CACHE_INFO = {}
IMAGE_CACHE_KEY = None


def system_memory():
    """Effective physical/cgroup memory; cgroup headroom is conservative."""
    import psutil
    memory = psutil.virtual_memory()
    total, available = memory.total, memory.available
    for limit_path, usage_path in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                                    ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            limit = int(Path(limit_path).read_text().strip())
            usage = int(Path(usage_path).read_text().strip())
            if limit < total:
                total, available = limit, min(available,max(0,limit-usage))
                break
        except (OSError,ValueError):
            continue
    return dict(total_bytes=total,available_bytes=available,used_bytes=total-available)


def loader_options(cfg):
    workers = cfg['workers']
    options = dict(num_workers=workers)
    if workers:
        options['prefetch_factor'] = cfg.get('prefetch_factor',2)
        # Cache is read-only parent memory, shared by fork (not pickled per worker).
        if cfg.get('cache_mode','off') != 'off':
            options['multiprocessing_context'] = 'fork'
    return options


def prepare_image_cache(paths,cfg):
    """Bounded read-only compressed/RGB cache. No augmented image/tensor is cached."""
    global IMAGE_CACHE, IMAGE_CACHE_INFO, IMAGE_CACHE_KEY
    mode = cfg.get('cache_mode','off')
    if mode not in {'off','bytes','decoded'}:
        raise ValueError('cache_mode must be off, bytes or decoded')
    unique = list(dict.fromkeys(map(str,paths)))
    key = (mode,cfg['image_size'],cfg.get('cache_gib',0),hash(tuple(unique)))
    if IMAGE_CACHE_KEY == key:
        return dict(IMAGE_CACHE_INFO,reused=True)
    IMAGE_CACHE = {}
    IMAGE_CACHE_INFO = {}
    IMAGE_CACHE_KEY = None
    gc.collect()
    started = time.monotonic()
    memory = system_memory()
    reserve = max(4*1024**3,int(memory['total_bytes']*.2))
    budget = max(0,min(int(cfg.get('cache_gib',0)*1024**3),memory['available_bytes']-reserve))
    if mode != 'off' and budget <= 0:
        raise MemoryError('Insufficient host RAM for configured cache and 20%/4 GiB reserve')
    stored = 0

    def read(path):
        if mode == 'bytes':
            return path,Path(path).read_bytes()
        with Image.open(path) as source:
            image = source.convert('RGB')
            image.thumbnail((cfg['image_size'],cfg['image_size']),Image.Resampling.LANCZOS)
            value = np.array(image,dtype=np.uint8,copy=True)
            value.flags.writeable = False
        return path,value

    if mode != 'off':
        print(f'Building {mode} cache, cap {budget/1024**3:.1f} GiB; JPEG/RGB before augmentation.',flush=True)
        with ThreadPoolExecutor(max_workers=min(8,os.cpu_count() or 1)) as executor:
            full = False
            for start in range(0,len(unique),16):
                if system_memory()['available_bytes'] < reserve:
                    break
                for path,value in executor.map(read,unique[start:start+16]):
                    size = len(value) if mode == 'bytes' else value.nbytes
                    if stored+size > budget:
                        full = True
                        continue
                    IMAGE_CACHE[path] = value
                    stored += size
                if start and start % 1024 == 0:
                    print(f'Cache: {len(IMAGE_CACHE)}/{len(unique)} images, {stored/1024**3:.1f} GiB',flush=True)
                if full:
                    break
    IMAGE_CACHE_KEY = key
    IMAGE_CACHE_INFO = dict(mode=mode,image_size=cfg['image_size'],stored_bytes=stored,
        cached_images=len(IMAGE_CACHE),total_images=len(unique),
        coverage=len(IMAGE_CACHE)/max(1,len(unique)),budget_bytes=budget,
        build_seconds=time.monotonic()-started,reused=False)
    print('Host image cache:',IMAGE_CACHE_INFO,flush=True)
    return dict(IMAGE_CACHE_INFO)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class Tee:
    def __init__(self, original, logfile):
        self.original, self.logfile = original, logfile

    def write(self, text):
        self.original.write(text)
        self.logfile.write(text)
        if '\n' in text:
            self.logfile.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


def start_logging(run_dir):
    """Append directly to Drive, including notebook tracebacks and stderr."""
    stop = getattr(sys, '_iir_stop_logging', None)
    if stop:
        stop()
    log_dir = Path(run_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f'session_{time.time_ns()}.log'
    stream = path.open('a', buffering=1)
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(stdout, stream), Tee(stderr, stream)

    def close():
        sys.stdout, sys.stderr = stdout, stderr
        stream.flush()
        stream.close()
        sys._iir_stop_logging = None

    sys._iir_stop_logging = close
    print('Drive session log:', path, flush=True)
    return path


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def extract_zip(source, cache_root):
    """Copy ZIP to local disk, validate paths/CRC, publish only complete extraction."""
    source, cache_root = Path(source), Path(cache_root)
    if not source.is_file() or source.suffix.lower() != '.zip':
        raise FileNotFoundError(f'Expected ZIP on Drive: {source}')
    info = source.stat()
    identity = dict(path=str(source.resolve()), bytes=info.st_size, mtime_ns=info.st_mtime_ns)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    destination = cache_root / key
    marker = destination / '.extraction_complete.json'
    if marker.exists() and json.loads(marker.read_text())['source'] == identity:
        print('Reuse completed local extraction:', destination)
        return destination
    cache_root.mkdir(parents=True, exist_ok=True)
    local_zip = cache_root / f'{key}.zip'
    partial = cache_root / f'{key}.partial'
    if partial.exists():
        shutil.rmtree(partial)
    if destination.exists():
        raise RuntimeError(f'Incomplete extraction at {destination}; remove that local folder before retrying.')
    try:
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            needed = sum(e.file_size for e in entries) + info.st_size + 2 * 1024**3
            if shutil.disk_usage(cache_root).free < needed:
                raise OSError(f'ZIP needs approximately {needed / 1024**3:.1f} GiB free local disk.')
        print('Copying ZIP Drive -> local:', source, flush=True)
        shutil.copyfile(source, local_zip)
        digest = sha256(local_zip)
        partial.mkdir()
        with zipfile.ZipFile(local_zip) as archive:
            seen = set()
            for entry in archive.infolist():
                target = (partial / entry.filename).resolve()
                if (not target.is_relative_to(partial.resolve()) or '\\' in entry.filename
                        or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError(f'Unsafe ZIP entry: {entry.filename}')
                if entry.filename in seen:
                    raise ValueError(f'Duplicate ZIP entry: {entry.filename}')
                seen.add(entry.filename)
                archive.extract(entry, partial)  # zipfile validates CRC while reading.
        atomic_json(partial / marker.name, dict(source=identity, zip_sha256=digest))
        partial.rename(destination)
        print('Extraction complete:', destination, flush=True)
        return destination
    finally:
        local_zip.unlink(missing_ok=True)


def only_match(root, pattern, override=None):
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    matches = list(Path(root).rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one {pattern}, found {len(matches)}. Set an exact override. {matches[:8]}')
    return matches[0]


def cid_path(cid, root):
    return Path(root) / cid[-2:] / cid[-4:-2] / cid[-6:-4] / cid


def load_sfm(root, metadata_override=None, images_override=None):
    metadata = only_match(root, 'retrieval-SfM-120k.pkl', metadata_override)
    with metadata.open('rb') as f:
        database = pickle.load(f)  # Only load your trusted dataset archive.
    expected = dict(train=(15953, 100, 34337), val=(739, 20, 181))
    for split, counts in expected.items():
        part = database[split]
        actual = (len(part['cids']), len(set(part['cluster'])), len(part['qidxs']))
        if actual != counts:
            raise ValueError(f'{split} expected prepared subset counts {counts}, got {actual}. '
                             'Use the prepared seed42 subset, not full/whitening metadata.')
        assert len(part['cluster']) == len(part['cids'])
        assert len(part['qidxs']) == len(part['pidxs'])
        assert len(set(part['cids'])) == len(part['cids'])
        for cid in part['cids']:
            assert isinstance(cid, str) and Path(cid).name == cid and '\\' not in cid
        for q, p in zip(part['qidxs'], part['pidxs']):
            assert 0 <= q < len(part['cids']) and 0 <= p < len(part['cids'])
            assert q != p and part['cluster'][q] == part['cluster'][p]
    assert not set(database['train']['cids']) & set(database['val']['cids'])
    assert not set(database['train']['cluster']) & set(database['val']['cluster'])
    images = Path(images_override) if images_override else metadata.parent / 'ims'
    # Accept canonical CID folders or a flat ims directory; reject mixed layouts.
    all_cids = database['train']['cids'] + database['val']['cids']
    canonical = [cid_path(cid, images) for cid in all_cids]
    flat = [images / cid for cid in all_cids]
    if all(p.is_file() for p in canonical):
        paths = canonical
    elif all(p.is_file() for p in flat):
        paths = flat
    else:
        missing = [str(p) for p in canonical if not p.is_file()]
        raise FileNotFoundError(f'SfM ZIP incomplete or wrong ims path; missing {len(missing)}. {missing[:5]}')
    manifest = metadata.parent / 'prepared_manifest.json'
    if manifest.exists():
        prepared = json.loads(manifest.read_text())
        assert prepared['format'] == 'b0_prepared_subset_v1'
        for name, expected_hash in prepared['metadata_files'].items():
            assert Path(name).name == name
            assert sha256(metadata.parent / name) == expected_hash, name
        for relative in random.Random(42).sample(sorted(prepared['sfm_images']), 64):
            actual = (images / relative).resolve()
            assert actual.is_relative_to(images.resolve())
            assert sha256(actual) == prepared['sfm_images'][relative]['sha256'], relative
    # All paths checked, sample decode. Unsampled corruption remains a possible loader error.
    for path in random.Random(42).sample(paths, min(128, len(paths))):
        with Image.open(path) as im:
            im.convert('RGB').load()
    offset = len(database['train']['cids'])
    partitions = {}
    for split, split_paths in [('train', paths[:offset]), ('val', paths[offset:])]:
        partitions[split] = dict(database[split], paths=[str(p) for p in split_paths])
    print('SfM verified: train=15953 images/34337 pairs, val=739 images/181 pairs.')
    return partitions, dict(metadata_sha256=sha256(metadata),
                            manifest_sha256=sha256(manifest) if manifest.exists() else None)


def resolve_weights(path, cache_root):
    if path:
        weights = Path(path)
    else:
        weights = Path(cache_root) / 'imagenet-caffe-vgg16-features-d369c8e.pth'
        weights.parent.mkdir(parents=True, exist_ok=True)
        if not weights.exists():
            print('Downloading original Caffe VGG16 features:', WEIGHTS_URL)
            temp = weights.with_suffix('.download')
            torch.hub.download_url_to_file(WEIGHTS_URL, str(temp))
            if sha256(temp) != WEIGHTS_HASH:
                temp.unlink()
                raise ValueError('Downloaded weights hash mismatch')
            temp.replace(weights)
    if not weights.is_file() or sha256(weights) != WEIGHTS_HASH:
        raise ValueError(f'Expected original Caffe VGG16 feature weights SHA256 {WEIGHTS_HASH}: {weights}')
    return weights


class SpatialAttention(nn.Module):
    """CBAM-style spatial-only gate, identity at initialization (2*sigmoid)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        summary = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * (2 * self.conv(summary).sigmoid())


class RetrievalModel(nn.Module):
    def __init__(self, attention=False, features=None, output_dim=512):
        super().__init__()
        if features is None:
            layers, channels = [], 3
            # Exactly torchvision VGG16 features[:-1], without allocating the classifier.
            for item in [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                         512, 512, 512, 'M', 512, 512, 512]:
                if item == 'M':
                    layers.append(nn.MaxPool2d(2, 2))
                else:
                    layers.extend([nn.Conv2d(channels, item, 3, padding=1), nn.ReLU(inplace=True)])
                    channels = item
            features = nn.Sequential(*layers)
        self.features = features
        self.attention = SpatialAttention() if attention else nn.Identity()
        self.p = nn.Parameter(torch.tensor([3.0]))
        self.output_dim = output_dim
        self.precision = 'fp32'

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                            enabled=self.precision == 'bf16'):
            x = self.attention(self.features(x))
        # GeM powers, normalization and contrastive loss stay FP32 under BF16.
        x = x.float()
        x = x.clamp(min=1e-6).pow(self.p).mean((-2, -1)).pow(1 / self.p)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-6)
        return x.T


def make_model(cfg, weights, device):
    configure_acceleration(cfg)
    model = RetrievalModel(attention=cfg['attention'])
    model.features.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.precision = cfg.get('precision', 'FP32').lower()
    if model.precision == 'tf32':
        model.precision = 'fp32'
    return model.to(device)


def configure_acceleration(cfg):
    precision = cfg.get('precision', 'FP32').upper()
    if precision not in {'FP32', 'TF32', 'BF16'}:
        raise ValueError('precision must be FP32, TF32 or BF16')
    if precision == 'BF16' and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This CUDA device does not support BF16.')
    torch.backends.cuda.matmul.allow_tf32 = precision == 'TF32'
    torch.backends.cudnn.allow_tf32 = precision == 'TF32'


class ImageTransform:
    def __init__(self, augment=False):
        self.augment = augment

    def __call__(self, image):
        if self.augment:
            if random.random() < 0.8:
                operations = [(ImageEnhance.Brightness, 0.8, 1.2),
                              (ImageEnhance.Contrast, 0.8, 1.2),
                              (ImageEnhance.Color, 0.85, 1.15)]
                random.shuffle(operations)
                for enhancer, low, high in operations:
                    image = enhancer(image).enhance(random.uniform(low, high))
            if random.random() < 0.15:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.0)))
            if random.random() < 0.2:
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=random.randint(55, 95))
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    image = decoded.convert('RGB').copy()
        array = np.array(image, dtype=np.float32, copy=True) / 255.
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensor = (tensor - tensor.new_tensor(MEAN)[:, None, None]) / tensor.new_tensor(STD)[:, None, None]
        if self.augment and random.random() < 0.15:
            h, w = tensor.shape[-2:]
            area = random.uniform(0.01, 0.05) * h * w
            ratio = random.uniform(0.5, 2.0)
            rh, rw = int(math.sqrt(area / ratio)), int(math.sqrt(area * ratio))
            if 0 < rh <= h and 0 < rw <= w:
                y, x = random.randint(0, h-rh), random.randint(0, w-rw)
                tensor[:, y:y+rh, x:x+rw] = 0  # Normalized channel mean.
        return tensor.contiguous()


class ImageList(Dataset):
    def __init__(self, paths, size, augment=False, boxes=None):
        self.paths, self.size, self.boxes = paths, size, boxes
        self.transform = ImageTransform(augment)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = str(self.paths[index])
        cached = IMAGE_CACHE.get(path)
        if (isinstance(cached,np.ndarray) and self.boxes is None
                and IMAGE_CACHE_INFO.get('image_size') == self.size):
            image = Image.fromarray(cached).copy()
            return self.transform(image).unsqueeze(0)
        source_path = io.BytesIO(cached) if isinstance(cached,bytes) else path
        with Image.open(source_path) as source:
            image = source.convert('RGB')
        original_long = max(image.size)
        target = self.size
        if self.boxes is not None:
            image = image.crop(self.boxes[index])
            # Preserve the upstream query-crop resize protocol.
            target = max(1, int(self.size * max(image.size) / original_long))
        image.thumbnail((target, target), Image.Resampling.LANCZOS)
        return self.transform(image).unsqueeze(0)


def identity_collate(items):
    return items


def forward_images(model, images, microbatch, pixel_limit):
    groups, vectors = defaultdict(list), [None] * len(images)
    device = next(model.parameters()).device
    for i, image in enumerate(images):
        groups[tuple(image.shape[1:])].append(i)
    for shape, indices in groups.items():
        limit = max(1, min(microbatch, pixel_limit // (shape[-2] * shape[-1])))
        for start in range(0, len(indices), limit):
            selection = indices[start:start+limit]
            batch = torch.cat([images[i].to(device, non_blocking=True) for i in selection])
            output = model(batch)
            for j, index in enumerate(selection):
                vectors[index] = output[:, j:j+1]
    return torch.cat(vectors, dim=1)


@torch.no_grad()
def extract(model, paths, cfg, boxes=None):
    model.eval()
    data = ImageList(paths, cfg['image_size'], boxes=boxes)
    loader = DataLoader(data, batch_size=cfg['extraction_window'], shuffle=False,
                        **loader_options(cfg), pin_memory=next(model.parameters()).is_cuda,
                        collate_fn=identity_collate, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(0))
    device = next(model.parameters()).device
    result = torch.empty((model.output_dim, len(paths)), device=device)
    offset = 0
    for images in loader:
        vectors = forward_images(model, images, cfg['microbatch'], cfg['max_batch_pixels'])
        result[:, offset:offset+len(images)] = vectors
        offset += len(images)
    return result


def tuple_loss(vectors, margin=0.7):
    tuples = vectors.reshape(vectors.shape[0], -1, 7)
    distances = (tuples[:, :, :1] - tuples[:, :, 1:] + 1e-6).square().sum(0).sqrt()
    return 0.5 * (distances[:, 0].square().sum() + (margin-distances[:, 1:]).clamp(min=0).square().sum())


@torch.no_grad()
def mine(model, part, cfg, seed, validation=False, smoke=False):
    started = time.monotonic()
    generator = torch.Generator().manual_seed(seed)
    qsize = len(part['qidxs']) if validation else min(cfg['query_size'], len(part['qidxs']))
    if smoke:
        qsize = cfg['batch_size']
    sampled = torch.randperm(len(part['qidxs']), generator=generator)[:qsize].tolist()
    qs = [int(part['qidxs'][i]) for i in sampled]
    ps = [int(part['pidxs'][i]) for i in sampled]
    poolsize = len(part['paths']) if validation else min(cfg['pool_size'], len(part['paths']))
    if smoke:
        poolsize = min(256, len(part['paths']))
    pool = torch.randperm(len(part['paths']), generator=generator)[:poolsize].tolist()
    unique = list(dict.fromkeys(pool + qs + ps))
    positions = {index: pos for pos, index in enumerate(unique)}
    descriptors = extract(model, [part['paths'][i] for i in unique], cfg)
    qvectors = descriptors[:, [positions[i] for i in qs]]
    pvectors = descriptors[:, [positions[i] for i in pool]]
    ranks = torch.argsort(pvectors.T @ qvectors, dim=0, descending=True).cpu().numpy()
    tuples = []
    for column, (q, p) in enumerate(zip(qs, ps)):
        seen, negatives = {part['cluster'][q]}, []
        for rank in ranks[:, column]:
            candidate = pool[int(rank)]
            cluster = part['cluster'][candidate]
            if cluster not in seen:
                negatives.append(candidate)
                seen.add(cluster)
                if len(negatives) == 5:
                    break
        if len(negatives) != 5:
            raise RuntimeError('Mining pool has fewer than five distinct negative clusters.')
        tuples.append([q, p] + negatives)
    seconds = time.monotonic() - started
    print(f'Mining {"val" if validation else "train"}: {len(unique)} unique images, '
          f'{len(tuples)} tuples, {seconds/60:.1f} min', flush=True)
    if validation:
        return tuples, descriptors, positions, seconds
    return tuples, None, None, seconds


class TupleImages(Dataset):
    def __init__(self, part, tuples, cfg):
        self.images = ImageList(part['paths'], cfg['image_size'], augment=True)
        self.tuples = tuples

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, index):
        return [self.images[i] for i in self.tuples[index]]


def train_epoch(model, optimizer, part, cfg, epoch, smoke=False):
    tuples, _, _, mining_seconds = mine(model, part, cfg, cfg['seed']+epoch, smoke=smoke)
    data = TupleImages(part, tuples, cfg)
    loader = DataLoader(data, batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                        **loader_options(cfg), pin_memory=next(model.parameters()).is_cuda,
                        collate_fn=identity_collate, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(cfg['seed']+epoch))
    model.train()
    total, count, pending = 0., 0, []
    for step, batch in enumerate(loader, 1):
        pending.extend(optimizer_batch(model, optimizer, batch, cfg))
        count += len(batch)
        if step == 1 or step % 20 == 0 or step == len(loader):
            total += torch.stack(pending).double().sum().item()
            pending.clear()
            print(f'Epoch {epoch+1} train {step}/{len(loader)}: loss/tuple={total/count:.6f}', flush=True)
    if not count:
        raise RuntimeError('No full training batch')
    return total/count, mining_seconds, len(loader)


def optimizer_batch(model, optimizer, batch, cfg):
    """Shared measured training step: same code in training and GPU profiler."""
    optimizer.zero_grad(set_to_none=True)
    losses,_ = backward_tuples(model,batch,cfg)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not torch.stack([g.isfinite().all() for g in grads]).all().item():
        raise FloatingPointError('Non-finite or missing gradients; aborting optimizer update.')
    optimizer.step()
    if not torch.isfinite(model.p).all().item() or model.p.item() <= 0:
        raise FloatingPointError('Invalid learned GeM p')
    return losses


def backward_tuples(model,batch,cfg,collect_descriptors=False):
    fusion = cfg.get('tuple_fusion',1)
    if fusion not in {1,2,5}:
        raise ValueError('tuple_fusion must be 1, 2 or 5; effective batch remains five tuples')
    if fusion > 1 and any(isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,
                                       nn.Dropout,nn.Dropout2d,nn.Dropout3d)) for m in model.modules()):
        raise ValueError('Tuple fusion requires this VGG/attention model without batchnorm/dropout')
    losses,descriptors = [],[]
    for start in range(0,len(batch),fusion):
        images = [image for item in batch[start:start+fusion] for image in item]
        vectors = forward_images(model, images, cfg['microbatch'], cfg['max_batch_pixels'])
        loss = tuple_loss(vectors, cfg['margin'])
        loss.backward()  # Summed tuple objective; optimizer still steps once after five tuples.
        losses.append(loss.detach())
        if collect_descriptors:
            descriptors.append(vectors.detach().float().cpu())
    return losses,descriptors


@torch.no_grad()
def validate(model, part, cfg, smoke=False):
    tuples, vectors, positions, mining_seconds = mine(
        model, part, cfg, cfg['seed']+100000, validation=True, smoke=smoke)
    # Fixed pair sampling seed; negatives still change with the current model.
    stop = len(tuples) // cfg['batch_size'] * cfg['batch_size']
    if not stop:
        raise RuntimeError('No full validation batch')
    total = 0.
    for start in range(0, stop, cfg['batch_size']):
        indices = [positions[i] for t in tuples[start:start+cfg['batch_size']] for i in t]
        total += tuple_loss(vectors[:, indices], cfg['margin']).item()
    return total/stop, mining_seconds


def make_optimizer(model, cfg):
    groups = [dict(params=model.features.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']),
              dict(params=[model.p], lr=cfg['lr']*10, weight_decay=0)]
    if cfg['attention']:
        groups.append(dict(params=model.attention.parameters(), lr=cfg['attention_lr'], weight_decay=0))
    return torch.optim.Adam(groups)


def set_epoch_lr(optimizer, cfg, epoch):
    # Explicit historical pre-epoch exponential schedule, without scheduler.step warning.
    base = [cfg['lr'], cfg['lr']*10] + ([cfg['attention_lr']] if cfg['attention'] else [])
    for group, lr in zip(optimizer.param_groups, base):
        group['lr'] = lr * math.exp(-0.01 * (epoch+1))


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def save_checkpoint(path, state, scratch):
    """Serialize locally, copy+hash+rename on Drive. Keep previous latest for recovery."""
    path, scratch = Path(path), Path(scratch)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    local = scratch / f'{path.stem}_{time.time_ns()}.pt'
    pending = path.with_name(path.name + '.pending')
    try:
        torch.save(cpu_tree(state), local)
        digest = sha256(local)
        shutil.copyfile(local, pending)
        if sha256(pending) != digest:
            raise IOError(f'Drive checkpoint copy failed hash verification: {pending}')
        if path.name == 'latest.pt' and path.exists():
            path.replace(path.with_name('previous_latest.pt'))
            old_digest = path.with_name('latest.pt.sha256')
            if old_digest.exists():
                old_digest.replace(path.with_name('previous_latest.pt.sha256'))
        pending.replace(path)
        path.with_name(path.name + '.sha256').write_text(digest + '\n')
    finally:
        local.unlink(missing_ok=True)


def publish_checkpoint_alias(source, destination):
    source, destination = Path(source), Path(destination)
    pending = destination.with_name(destination.name + '.pending')
    digest = sha256(source)
    shutil.copyfile(source, pending)
    if sha256(pending) != digest:
        raise IOError('Drive alias copy failed verification')
    pending.replace(destination)
    destination.with_name(destination.name + '.sha256').write_text(digest + '\n')


def load_checkpoint(path, identity):
    path = Path(path)
    digest_file = path.with_name(path.name + '.sha256')
    if not digest_file.exists() or sha256(path) != digest_file.read_text().strip():
        raise IOError(f'Missing/mismatched checkpoint SHA256: {path}. Try previous_latest.pt explicitly.')
    state = torch.load(path, map_location='cpu', weights_only=False)  # Trusted own checkpoint only.
    if state.get('format') != VERSION or state.get('identity') != identity:
        raise ValueError('Checkpoint run/source/data/config mismatch; never resume the other run.')
    return state


def discover_benchmarks(roots):
    result = {}
    for name in ['oxford5k', 'paris6k']:
        matches = [p for root in roots for p in Path(root).rglob(f'gnd_{name}.pkl')]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f'Ambiguous {name} metadata: {matches}')
        path = matches[0]
        with path.open('rb') as f:
            gt = pickle.load(f)
        images = path.parent / 'jpg'
        names = set(gt['imlist']) | set(gt['qimlist'])
        for item in names:
            assert isinstance(item, str) and Path(item).name == item
            if not (images / (item+'.jpg')).is_file():
                raise FileNotFoundError(images / (item+'.jpg'))
        result[name] = dict(gt=gt, root=images, metadata_sha256=sha256(path))
    return result


def benchmark(model, benchmarks, cfg, output, compute_map, export=False):
    result = {}
    for name, info in benchmarks.items():
        print('Benchmark:', name, flush=True)
        gt, root = info['gt'], info['root']
        gallery = extract(model, [str(root/(n+'.jpg')) for n in gt['imlist']], cfg).cpu()
        query = extract(model, [str(root/(n+'.jpg')) for n in gt['qimlist']], cfg,
                        boxes=[tuple(g['bbx']) for g in gt['gnd']]).cpu()
        ranks = np.argsort(-(gallery.numpy().T @ query.numpy()), axis=0)
        score, aps, _, _ = compute_map(ranks, gt['gnd'])
        result[name] = dict(map=float(score), aps=np.asarray(aps).tolist(),
                            protocol='legacy_ok_junk', scale=[1], whitening=False, query_expansion=False,
                            metadata_sha256=info['metadata_sha256'])
        if export:
            # Small descriptor files are saved on Drive for later CPU search experiments.
            np.savez_compressed(Path(output).parent / f'{name}_best_descriptors.npz',
                                gallery=gallery.numpy(), query=query.numpy(),
                                gallery_ids=np.asarray(gt['imlist']), query_ids=np.asarray(gt['qimlist']))
        print(name, f'mAP={score*100:.3f}%', flush=True)
    atomic_json(output, result)
    return result


def _run_smoke_impl(cfg, partitions, weights, device, run_dir, identity, scratch):
    seed_all(cfg['seed'])
    model = make_model(cfg, weights, device)
    optimizer = make_optimizer(model, cfg)
    before = model.features[0].weight.detach().clone()
    started = time.monotonic()
    train_loss, _, _ = train_epoch(model, optimizer, partitions['train'], cfg, 0, smoke=True)
    val_loss, _ = validate(model, partitions['val'], cfg, smoke=True)
    assert math.isfinite(train_loss) and math.isfinite(val_loss)
    assert not torch.equal(before, model.features[0].weight)
    if cfg['attention']:
        assert torch.count_nonzero(model.attention.conv.weight).item() > 0
    path = Path(run_dir) / 'smoke' / 'roundtrip.pt'
    state = dict(format=VERSION, identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict())
    save_checkpoint(path, state, scratch)
    restored = load_checkpoint(path, identity)
    assert all(torch.equal(t.cpu(), restored['model'][key]) for key, t in model.state_dict().items())
    optimizer.load_state_dict(restored['optimizer'])
    result = dict(status='PASS', train_loss=train_loss, val_loss=val_loss,
                  seconds=time.monotonic()-started, image_size=cfg['image_size'])
    atomic_json(Path(run_dir)/'smoke'/'result.json', result)
    del model, optimizer, state, restored, before
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('SMOKE PASS:', result, flush=True)
    return result


def run_smoke(cfg, partitions, weights, device, run_dir, identity, scratch):
    try:
        return _run_smoke_impl(cfg, partitions, weights, device, run_dir, identity, scratch)
    except Exception as error:
        traceback.print_exc()
        failure = dict(status='FAILED', error_type=type(error).__name__, error=str(error),
                       image_size=cfg['image_size'], note='Full training was not started.')
        atomic_json(Path(run_dir)/'smoke'/'result.json', failure)
        # Clear exception frames holding CUDA activations before a notebook retains the error.
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        raise RuntimeError('Smoke failed; inspect Drive logs and smoke/result.json. No full training started.') from None


def train_run(cfg, partitions, weights, run_dir, identity, operational, benchmarks, compute_map, *, device=None):
    # CPU override is used only by the local integration verifier, not notebook cells.
    device = torch.device(device) if device is not None else torch.device('cuda:0')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Choose a GPU runtime in Colab before training.')
    run_dir = Path(run_dir)
    scratch = Path(operational['scratch'])
    started = time.monotonic()
    cache_info = prepare_image_cache(partitions['train']['paths']+partitions['val']['paths'],cfg)
    atomic_json(run_dir/'image_cache.json',cache_info)
    latest = run_dir/'latest.pt'
    resume = operational['resume']
    if resume == 'auto':
        resume = latest if latest.exists() else None
        if resume is None and any((run_dir/name).exists() for name in ['previous_latest.pt', 'history.json']):
            raise RuntimeError('Existing run has no latest.pt. Select previous_latest.pt explicitly or a new RUN_ID.')
    elif resume:
        resume = Path(resume)
        if resume.resolve().parent != run_dir.resolve():
            raise ValueError('Resume from this RUN_ID folder; latest/best must remain together.')
    if resume is None and any((run_dir/name).exists() for name in ['latest.pt', 'previous_latest.pt', 'history.json']):
        raise RuntimeError('Existing latest.pt; use auto resume or a new RUN_ID.')
    state = load_checkpoint(resume, identity) if resume else None
    # Versioned best files allow an interrupted latest/best publication to recover.
    best_file = state['best_file'] if state else None
    best_path = run_dir/best_file if best_file else run_dir/'best.pt'
    if state and not best_path.exists():
        raise FileNotFoundError('Keep the versioned best checkpoint beside latest.pt on Drive.')
    start_epoch = state['epoch'] if state else 0
    if cfg['epochs'] < start_epoch:
        raise ValueError('Epoch target is below the checkpoint epoch.')
    previous_seconds = state.get('total_session_seconds', 0) if state else 0
    usage_file = run_dir/'usage_total.json'
    if usage_file.exists():
        previous_seconds = max(previous_seconds, json.loads(usage_file.read_text())['total_session_seconds'])
    rate = operational['cu_per_hour']
    cu_cap = operational['run_cu_budget']
    wall_cap = operational['session_hours'] * 3600

    def remaining():
        elapsed = time.monotonic()-started
        seconds = wall_cap-elapsed
        if rate is not None:
            seconds = min(seconds, cu_cap/rate*3600 - previous_seconds-elapsed)
        return seconds

    if remaining() < 600:
        raise RuntimeError('Less than ten minutes of configured session/CU budget remains.')
    # Complete runs can be reopened for evaluation without paying for another smoke.
    if start_epoch < cfg['epochs']:
        try:
            run_smoke(cfg, partitions, weights, device, run_dir, identity, scratch)
        finally:
            atomic_json(usage_file, dict(total_session_seconds=previous_seconds+time.monotonic()-started))
    seed_all(cfg['seed'])
    model = make_model(cfg, weights, device)
    optimizer = make_optimizer(model, cfg)
    history, best_loss = [], float('inf')
    if state:
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        history, best_loss = state['history'], state['best_loss']
        restore_rng(state['rng'])
        best_state = load_checkpoint(best_path, identity)
        assert best_state['best_loss'] == best_loss
        del best_state, state
        publish_checkpoint_alias(best_path, run_dir/'best.pt')
    atomic_json(run_dir/'run_record.json', dict(identity=identity, config=cfg, operational=operational,
                environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                 gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU_VERIFIER', precision=cfg.get('precision','FP32')),
                resume=str(resume), selection='clean_validation_contrastive_loss',
                validation_pairs_processed=180, gradient_reduction='sum per tuple; five tuples per update',
                schedule='lr_group * exp(-0.01 * (epoch_index+1)); historical pre-epoch convention'))
    status = 'COMPLETED'
    try:
        for epoch in range(start_epoch, cfg['epochs']):
            estimate = history[-1].get('total_seconds', history[-1]['train_val_seconds']*1.25) if history else 0
            if estimate and remaining() < 1.2*estimate + 300:
                status = 'PAUSED_BEFORE_EPOCH_FOR_BUDGET'
                print(status, 'Resume this same notebook/RUN_ID in a new session.', flush=True)
                break
            epoch_started = time.monotonic()
            seed_all(cfg['seed']+epoch)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            set_epoch_lr(optimizer, cfg, epoch)
            print(f'\n=== Epoch {epoch+1}/{cfg["epochs"]}; LR {[g["lr"] for g in optimizer.param_groups]} ===', flush=True)
            train_start = time.monotonic()
            train_loss, train_mining, steps = train_epoch(model, optimizer, partitions['train'], cfg, epoch)
            train_seconds = time.monotonic()-train_start
            val_start = time.monotonic()
            val_loss, val_mining = validate(model, partitions['val'], cfg)
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                raise FloatingPointError('Non-finite epoch loss')
            improved = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
            if improved:
                best_file = f'best_epoch_{epoch+1:03d}.pt'
                best_path = run_dir/best_file
            row = dict(epoch=epoch+1, train_loss=train_loss, val_loss=val_loss, gem_p=model.p.item(),
                       optimizer_steps=steps, train_seconds=train_seconds, train_mining_seconds=train_mining,
                       val_seconds=time.monotonic()-val_start, val_mining_seconds=val_mining,
                       peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3 if device.type == 'cuda' else None,
                       train_val_seconds=time.monotonic()-epoch_started)
            history.append(row)
            checkpoint = dict(format=VERSION, identity=identity, epoch=epoch+1,
                              model=model.state_dict(), optimizer=optimizer.state_dict(),
                              schedule=dict(gamma=math.exp(-0.01), timing='pre_epoch', completed_epochs=epoch+1),
                              rng=rng_state(), best_loss=best_loss, best_file=best_file, history=history,
                              total_session_seconds=previous_seconds+time.monotonic()-started)
            # Publish best before latest, so a failed best upload cannot silently continue.
            if improved:
                save_checkpoint(best_path, checkpoint, scratch)
            save_checkpoint(latest, checkpoint, scratch)
            if improved:
                publish_checkpoint_alias(best_path, run_dir/'best.pt')
            del checkpoint
            atomic_json(run_dir/'history.json', history)
            if epoch+1 == 7:
                shutil.copyfile(latest, run_dir/'epoch_007.pt')
                shutil.copyfile(run_dir/'latest.pt.sha256', run_dir/'epoch_007.pt.sha256')
            benchmark_file = run_dir/f'evaluation_epoch_{epoch+1:03d}.json'
            if benchmarks and (epoch+1) % operational['eval_every'] == 0 and remaining() > 1800:
                benchmark(model, benchmarks, cfg, benchmark_file, compute_map)
            row['total_seconds'] = time.monotonic()-epoch_started
            atomic_json(run_dir/'history.json', history)
            atomic_json(run_dir/'timing_latest.json', dict(last_epoch=row,
                        total_session_seconds=previous_seconds+time.monotonic()-started,
                        estimated_cu_used=(previous_seconds+time.monotonic()-started)*rate/3600 if rate else None))
            print('Epoch complete:', json.dumps(row), flush=True)
        if operational['final_evaluation'] and benchmarks and best_path.exists():
            if remaining() > 1800:
                best = load_checkpoint(best_path, identity)
                model.load_state_dict(best['model'], strict=True)
                result = benchmark(model, benchmarks, cfg, run_dir/'evaluation_best.json', compute_map, export=True)
                atomic_json(run_dir/'evaluation_best_identity.json', dict(epoch=best['epoch'],
                            checkpoint_sha256=sha256(best_path), metrics=result, identity=identity))
                del best
            else:
                print('Best-checkpoint evaluation deferred: low time/CU budget.', flush=True)
        elif not benchmarks:
            print('Oxford/Paris inputs absent: training/validation complete; no mAP claimed.', flush=True)
        result = dict(status=status, completed_epochs=len(history), target_epochs=cfg['epochs'],
                      best_val_loss=best_loss, latest=str(latest), best=str(run_dir/'best.pt'),
                      total_session_seconds=previous_seconds+time.monotonic()-started)
        atomic_json(run_dir/'training_result.json', result)
        print(result, flush=True)
        return result
    except BaseException as error:
        traceback.print_exc()
        atomic_json(run_dir/'interruption.json', dict(error_type=type(error).__name__, error=str(error),
                    recovery='Resume latest.pt (completed epoch). Partial current epoch is discarded.',
                    session_seconds=time.monotonic()-started))
        raise
    finally:
        atomic_json(usage_file, dict(total_session_seconds=previous_seconds+time.monotonic()-started,
                    note='Measured notebook execution wall time; CU is an estimate, not Colab billing.'))
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def make_identity(cfg, data_identity, runtime_hash):
    # Epoch target can grow; all model/data/augmentation/learning choices must stay fixed.
    fixed = {key: value for key, value in cfg.items() if key != 'epochs'}
    return dict(version=VERSION, config=fixed, data=data_identity, runtime_sha256=runtime_hash,
                weights_sha256=WEIGHTS_HASH)
