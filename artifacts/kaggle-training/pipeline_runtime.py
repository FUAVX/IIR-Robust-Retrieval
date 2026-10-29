"""Embedded into the pipeline notebook. Uses its pinned cirtorch and torch globals.
No source-file mutation; no descriptor survives a training epoch or optimizer update.
"""
from collections import defaultdict
import torch

PIPELINE_SETTINGS = dict(version='pipeline_v2', inference_microbatch=2,
                         training_microbatch=2, max_batch_pixels=2 * 1024 * 1024,
                         extraction_window=8)


class PipelineContrastiveLoss(torch.nn.Module):
    """Original summed contrastive formula for fixed q+p+5 tuples, no GPU label scan."""
    def __init__(self, margin=0.7, eps=1e-6):
        super().__init__()
        self.margin, self.eps = margin, eps
    def forward(self, x, label):
        if x.shape[1] % 7 or label.numel() != x.shape[1]:
            raise ValueError('Expected fixed seven-image retrieval tuples.')
        tuples = x.reshape(x.shape[0], -1, 7)
        difference = tuples[:, :, :1] - tuples[:, :, 1:]
        distance = (difference + self.eps).square().sum(dim=0).sqrt()
        labels = label.reshape(-1, 7)[:, 1:]
        losses = (0.5 * labels * distance.square()
                  + 0.5 * (1-labels) * torch.clamp(self.margin-distance, min=0).square())
        return losses.sum()


def _shape_groups(images, limit):
    buckets = defaultdict(list)
    for i, image in enumerate(images):
        assert image.ndim == 4 and image.shape[0] == 1
        buckets[tuple(image.shape[1:])].append(i)
    for shape, indices in buckets.items():
        pixels = shape[-2] * shape[-1]
        size = max(1, min(limit, PIPELINE_SETTINGS['max_batch_pixels'] // pixels))
        for start in range(0, len(indices), size):
            yield indices[start:start + size]


def pipeline_tuple_vectors(model, images, limit=None):
    """Group only equal shapes, preserve output order and autograd; never pad/resize."""
    if limit is None:
        limit = PIPELINE_SETTINGS['training_microbatch']
    vectors = [None] * len(images)
    device = next(model.parameters()).device
    for indices in _shape_groups(images, limit):
        # Transfer pinned loader tensors first; a CPU torch.cat would lose pinning.
        tensors = [images[i].to(device, non_blocking=True) for i in indices]
        batch = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        descriptors = model(batch)
        assert descriptors.shape == (model.meta['outputdim'], len(indices))
        for j, index in enumerate(indices):
            vectors[index] = descriptors[:, j:j+1]
    return torch.cat(vectors, dim=1)


def pipeline_extract_gpu(model, images, image_size, transform, bbxs=None):
    """Bounded CPU window, batch equal shapes, retain descriptors on GPU until done."""
    dataset = traindataset.ImagesFromList(root='', images=images, imsize=image_size,
                                         bbxs=bbxs, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
        num_workers=effective_workers(8), pin_memory=True, persistent_workers=False,
        generator=torch.Generator().manual_seed(0))
    device = next(model.parameters()).device
    result = torch.empty(model.meta['outputdim'], len(images), device=device)
    window, offset = [], 0
    with torch.no_grad():
        for image in loader:
            window.append(image)
            if len(window) == PIPELINE_SETTINGS['extraction_window']:
                vectors = pipeline_tuple_vectors(model, window, PIPELINE_SETTINGS['inference_microbatch'])
                result[:, offset:offset+len(window)] = vectors
                offset += len(window)
                window.clear()
        if window:
            vectors = pipeline_tuple_vectors(model, window, PIPELINE_SETTINGS['inference_microbatch'])
            result[:, offset:] = vectors
    return result


def pipeline_extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    if list(ms) != [1]:
        raise ValueError('Pipeline supports the notebook single-scale protocol only.')
    net.eval()
    torch.empty((), dtype=torch.int64).random_()  # Original extraction loader base seed.
    return pipeline_extract_gpu(net, images, image_size, transform, bbxs).cpu()


def _select_negative_indices(ranks_cpu, pool_indices, query_indices, clusters, nnum):
    """CPU-only cluster selection after one bulk rank transfer, exact ranked scan."""
    selected = []
    for q, query in enumerate(query_indices):
        seen = {clusters[query]}
        negatives = []
        for rank in ranks_cpu[:, q]:
            candidate = int(pool_indices[int(rank)])
            cluster = clusters[candidate]
            if cluster not in seen:
                negatives.append(candidate)
                seen.add(cluster)
                if len(negatives) == nnum:
                    break
        if len(negatives) != nnum:
            raise RuntimeError('Insufficient distinct negative clusters in sampled pool.')
        selected.append(negatives)
    return selected


def pipeline_mine(dataset, net):
    # Keep the upstream CPU RNG draw order exactly: pairs first, then pool.
    pair_indices = torch.randperm(len(dataset.qpool))[:dataset.qsize].tolist()
    dataset.qidxs = [dataset.qpool[i] for i in pair_indices]
    dataset.pidxs = [dataset.ppool[i] for i in pair_indices]
    pool = torch.randperm(len(dataset.images))[:dataset.poolsize].tolist()
    unique = list(dict.fromkeys(pool + dataset.qidxs))
    positions = {index: pos for pos, index in enumerate(unique)}
    net.eval()
    # Original query and pool loader iterators each consume one CPU base seed.
    for _ in range(2):
        torch.empty((), dtype=torch.int64).random_()
    with torch.no_grad():
        descriptors = pipeline_extract_gpu(net, [dataset.images[i] for i in unique],
                                           dataset.imsize, dataset.transform)
        qvecs = descriptors[:, [positions[i] for i in dataset.qidxs]]
        poolvecs = descriptors[:, [positions[i] for i in pool]]
        _, ranks = torch.sort(poolvecs.t() @ qvecs, dim=0, descending=True)
        dataset.nidxs = _select_negative_indices(ranks.cpu().numpy(), pool,
                                                 dataset.qidxs, dataset.clusters, dataset.nnum)
        negative_positions = [[positions[i] for i in ns] for ns in dataset.nidxs]
        # Original epsilon placement retained; only diagnostic reduction is vectorized.
        negatives = descriptors[:, negative_positions]
        mean_distance = (qvecs.unsqueeze(-1) - negatives + 1e-6).square().sum(dim=0).sqrt().mean().item()
        if dataset.mode == 'val':
            # Validation weights do not change between mining and validation loss.
            dataset._pipeline_validation_vectors = descriptors
            dataset._pipeline_validation_positions = positions
        dataset.last_extraction_count = len(unique)
        print('Mining extracted', len(unique), 'unique images; original separate passes:', len(pool)+len(dataset.qidxs))
    return mean_distance


def pipeline_train(loader, model, criterion, optimizer, epoch):
    loader.dataset.create_epoch_tuples(model)
    model.train()
    model.apply(original_train.set_batchnorm_eval)
    optimizer.zero_grad()
    total_loss, count = 0.0, 0
    pending_losses = []
    started = time.monotonic()
    # Worker replicas must be recreated after each epoch's tuple remine.
    for i, (images, targets) in enumerate(loader):
        for image_tuple, target in zip(images, targets):
            vectors = pipeline_tuple_vectors(model, image_tuple)
            loss = criterion(vectors, target.to(vectors.device, non_blocking=True))
            loss.backward()  # Original per-tuple backward and summed gradient accumulation.
            pending_losses.append(loss.detach())
            count += 1
        if (i+1) % original_train.args.update_every == 0:
            optimizer.step()
            optimizer.zero_grad()
        if i == 0 or (i+1) % original_train.args.print_freq == 0 or i+1 == len(loader):
            # Defer scalar host synchronization until a progress boundary.
            total_loss += torch.stack(pending_losses).double().sum().item()
            pending_losses.clear()
            print(f'>> Train: [{epoch+1}][{i+1}/{len(loader)}] loss={total_loss/count:.6f} elapsed={time.monotonic()-started:.1f}s')
    return total_loss / count


def pipeline_validate(loader, model, criterion, epoch):
    loader.dataset.create_epoch_tuples(model)
    model.eval()
    dataset = loader.dataset
    try:
        vectors = dataset._pipeline_validation_vectors
        positions = dataset._pipeline_validation_positions
        # Positives may be absent with a partial pool. Extract these once as needed.
        needed = list(dict.fromkeys(dataset.qidxs + dataset.pidxs + [i for ns in dataset.nidxs for i in ns]))
        missing = [i for i in needed if i not in positions]
        if missing:
            extras = pipeline_extract_gpu(model, [dataset.images[i] for i in missing], dataset.imsize, dataset.transform)
            offset = vectors.shape[1]
            vectors = torch.cat([vectors, extras], dim=1)
            positions.update({i: offset+j for j, i in enumerate(missing)})
        # Original validation loader iterator consumes a CPU base seed.
        torch.empty((), dtype=torch.int64).random_()
        total_loss, count = 0.0, 0
        batch_size = loader.batch_size
        # Same non-shuffled tuples and drop_last as upstream; no image decode/forward again.
        stop = len(dataset) - len(dataset) % batch_size if loader.drop_last else len(dataset)
        for start in range(0, stop, batch_size):
            tuples = range(start, min(start+batch_size, stop))
            indices, labels = [], []
            for q in tuples:
                indices.extend(positions[i] for i in [dataset.qidxs[q], dataset.pidxs[q]] + dataset.nidxs[q])
                labels.extend([-1, 1] + [0]*dataset.nnum)
            target = torch.tensor(labels, dtype=torch.float32, device=vectors.device)
            total_loss += criterion(vectors[:, indices], target).item()
            count += len(tuples)
        print(f'Validation: reused mining descriptors for {count} tuples; no second image pass.')
        return total_loss / count
    finally:
        del dataset._pipeline_validation_vectors
        del dataset._pipeline_validation_positions
