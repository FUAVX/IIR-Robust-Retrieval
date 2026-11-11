"""CPU verification of notebook syntax, VGG topology, training and Drive-style resume.

Run with Python containing torch, torchvision, NumPy, Pillow.
Does not mount Drive, download data/weights, or claim Colab GPU validation.
"""
import copy
import importlib.util
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

import colab_runtime as rt

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent


def test_notebooks():
    for path in sorted(ROOT.glob('*.ipynb')):
        nb = json.loads(path.read_text())
        assert nb['nbformat'] == 4
        for index, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                body = source.split('\n', 1)[1] if source.startswith('%%writefile ') else source
                compile(body, f'{path.name}:cell{index}', 'exec')
                assert not cell['outputs'] and cell['execution_count'] is None
        namespace = {'RUNTIME_SOURCE': '', 'EVALUATE_SOURCE': '', 'PROFILE_SOURCE': ''}
        # Reconstruct readable %%writefile cells; never mount Drive or train.
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            if source.startswith('%%writefile '):
                command, body = source.split('\n', 1)
                name = ('RUNTIME_SOURCE' if command.endswith('colab_runtime.py') else
                        'PROFILE_SOURCE' if command.endswith('profile_runtime.py') else 'EVALUATE_SOURCE')
                if '-a ' not in command:
                    namespace[name] = ''
                namespace[name] += body
        assert namespace['RUNTIME_SOURCE'] == (ROOT/'colab_runtime.py').read_text()
        if namespace['PROFILE_SOURCE']:
            assert namespace['PROFILE_SOURCE'] == (ROOT/'profile_runtime.py').read_text()
        assert 'np.in1d(' not in namespace['EVALUATE_SOURCE']
        # Exercise the generated hash gate against Colab-style terminal newlines.
        import hashlib
        header = next(''.join(c['source']) for c in nb['cells']
                      if c['cell_type'] == 'code' and ''.join(c['source']).startswith('RUNTIME_SOURCE ='))
        header = header.split('if str(LOCAL_CODE)')[0]
        with tempfile.TemporaryDirectory() as directory:
            code = Path(directory)
            (code/'colab_runtime.py').write_text(namespace['RUNTIME_SOURCE'])
            for ending in ['', chr(10), chr(10)*2, chr(13)+chr(10)]:
                (code/'legacy_evaluate.py').write_text(namespace['EVALUATE_SOURCE'].rstrip(chr(10))+ending)
                exec(header, dict(LOCAL_CODE=code, hashlib=hashlib))
            stripped = chr(10).join(line.rstrip() for line in namespace['EVALUATE_SOURCE'].splitlines())
            (code/'legacy_evaluate.py').write_text(stripped+chr(10))
            checked = dict(LOCAL_CODE=code, hashlib=hashlib)
            exec(header, checked)
            assert checked['EVALUATE_HASH'] == 'ced1d110dd19f3f887e75f18ac9120aaff8900f5d6b52dc8c45c66c823141fd0'
            (code/'legacy_evaluate.py').write_text(namespace['EVALUATE_SOURCE'].replace('def compute_ap(', 'def edited_compute_ap('))
            try:
                exec(header, dict(LOCAL_CODE=code, hashlib=hashlib))
                raise RuntimeError('Actual evaluator edit was accepted')
            except AssertionError:
                pass
        evaluator = {}
        exec(namespace['EVALUATE_SOURCE'], evaluator)
        score, *_ = evaluator['compute_map'](np.array([[0], [1]]), [{'ok': [0], 'junk': []}])
        assert score == 1.0
    print('PASS notebooks: compile/source/evaluator; hash gate accepts terminal newlines and rejects actual edits')


def test_architecture():
    from torchvision.models.vgg import make_layers, cfgs
    original = nn.Sequential(*list(make_layers(cfgs['D']).children())[:-1])
    model = rt.RetrievalModel()
    assert list(original.state_dict()) == list(model.features.state_dict())
    assert all(original.state_dict()[k].shape == v.shape for k,v in model.features.state_dict().items())
    model.features.load_state_dict(original.state_dict())
    attended = rt.RetrievalModel(attention=True)
    attended.features.load_state_dict(original.state_dict())
    assert sum(p.numel() for p in attended.attention.parameters()) == 99
    x = torch.rand(1, 3, 32, 48)
    with torch.no_grad():
        result = model(x)
        reference = original(x).clamp(min=1e-6).pow(3).mean((-2,-1)).pow(1/3)
        reference = (reference / (reference.norm(dim=1, keepdim=True)+1e-6)).T
        torch.testing.assert_close(result, reference)
        torch.testing.assert_close(attended(x), result, rtol=0, atol=0)
    print('PASS VGG topology/state keys/output parity; 99-parameter gate starts as identity')


def test_loss_and_batching():
    x = torch.randn(16, 35, requires_grad=True)
    actual = rt.tuple_loss(x)
    reference = 0
    for start in range(0, 35, 7):
        distances = (x[:, start:start+1] - x[:, start+1:start+7] + 1e-6).square().sum(0).sqrt()
        reference += 0.5*distances[0].square() + 0.5*(0.7-distances[1:]).clamp(min=0).square().sum()
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, x, retain_graph=True)[0],
                               torch.autograd.grad(reference, x)[0])
    model = rt.RetrievalModel(features=nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()), output_dim=8)
    images = [torch.rand(1,3,h,w) for h,w in [(16,20),(20,16),(16,20),(18,18)]]
    together = rt.forward_images(model, images, 2, 2048)
    separate = torch.cat([model(image) for image in images], dim=1)
    torch.testing.assert_close(together, separate)
    print('PASS tuple loss/gradients; mixed-shape microbatch descriptor order')


def make_fixture(root):
    part = dict(cids=[], cluster=[], qidxs=[], pidxs=[], paths=[])
    generator = np.random.default_rng(10)
    for cluster in range(8):
        for member in range(2):
            i = len(part['paths'])
            path = root/f'{i}.jpg'
            Image.fromarray(generator.integers(0,256,(24,32,3),dtype=np.uint8)).save(path)
            part['paths'].append(str(path))
            part['cids'].append(path.name)
            part['cluster'].append(cluster)
        part['qidxs'].append(cluster*2)
        part['pidxs'].append(cluster*2+1)
    return dict(train=part, val=copy.deepcopy(part))


def test_zip(root):
    archive = root/'input.zip'
    with zipfile.ZipFile(archive, 'w') as f:
        f.writestr('nested/meta.txt', 'example')
    extracted = rt.extract_zip(archive, root/'cache')
    assert (extracted/'nested/meta.txt').read_text() == 'example'
    assert rt.extract_zip(archive, root/'cache') == extracted
    malicious = root/'malicious.zip'
    with zipfile.ZipFile(malicious, 'w') as f:
        f.writestr('../escape.txt', 'bad')
    try:
        rt.extract_zip(malicious, root/'cache')
        raise AssertionError('Traversal accepted')
    except ValueError:
        pass
    assert not (root/'escape.txt').exists()
    print('PASS ZIP extraction completion/reuse/path traversal rejection')


def test_sfm_contract(root):
    folder = root/'prepared'
    images = folder/'ims'
    images.mkdir(parents=True)
    import io, pickle
    buffer = io.BytesIO()
    Image.new('RGB', (16, 16), color=(120, 80, 60)).save(buffer, format='JPEG')
    payload = buffer.getvalue()
    database = {}
    offset = 0
    for split, n, clusters, pairs in [('train',15953,100,34337), ('val',739,20,181)]:
        cids = [f'{offset+i:016x}' for i in range(n)]
        for cid in cids:
            (images/cid).write_bytes(payload)
        qs = [i % n for i in range(pairs)]
        ps = [q+clusters if q+clusters < n else q-clusters for q in qs]
        database[split] = dict(cids=cids, cluster=[offset+i % clusters for i in range(n)], qidxs=qs, pidxs=ps)
        offset += n
    metadata = folder/'retrieval-SfM-120k.pkl'
    metadata.write_bytes(pickle.dumps(database))
    partitions, identity = rt.load_sfm(folder)
    assert len(partitions['train']['paths']) == 15953 and len(partitions['val']['paths']) == 739
    assert identity['metadata_sha256'] == rt.sha256(metadata)
    database['train']['pidxs'][0] = database['train']['qidxs'][0]
    metadata.write_bytes(pickle.dumps(database))
    try:
        rt.load_sfm(folder)
        raise AssertionError('Self pair accepted')
    except AssertionError as error:
        assert str(error) != 'Self pair accepted'
    print('PASS real-count SfM contract/flat ZIP layout/split paths; invalid self pair rejected')


def test_train_resume(root):
    fixture_root = root/'images'
    fixture_root.mkdir()
    parts = make_fixture(fixture_root)
    cfg = dict(attention=False, attention_lr=1e-4, epochs=2, seed=42,
               image_size=32, query_size=5, pool_size=16, batch_size=5,
               lr=1e-4, weight_decay=1e-6, margin=.7, workers=0,
               microbatch=2, max_batch_pixels=2048, extraction_window=8,
               augmentation_version='test', precision='FP32')
    operational = dict(resume='auto', scratch=str(root/'scratch'), cu_per_hour=None,
                       run_cu_budget=110., session_hours=1., eval_every=3, final_evaluation=False)
    factory = rt.make_model

    def tiny(config, weights, device):
        rt.seed_all(7)
        net = rt.RetrievalModel(config['attention'],
                nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()), output_dim=8).to(device)
        return net

    rt.make_model = tiny
    try:
        image = Image.open(parts['train']['paths'][0]).convert('RGB')
        assert torch.equal(rt.ImageTransform(False)(image), rt.ImageTransform(False)(image))
        # Different seeded augmentation outcomes, never called by extraction/mining.
        rt.seed_all(1)
        a = rt.ImageTransform(True)(image)
        rt.seed_all(2)
        b = rt.ImageTransform(True)(image)
        assert not torch.equal(a, b)
        for attention in [False, True]:
            cfg['attention'] = attention
            identity = rt.make_identity(cfg, {'metadata_sha256':'fixture'}, 'test_runtime')
            full, resumed = root/f'full_{attention}', root/f'resumed_{attention}'
            for folder in [full,resumed]:
                folder.mkdir()
            rt.train_run(cfg, parts, None, full, identity, operational, {}, None, device='cpu')
            one = dict(cfg, epochs=1)
            rt.train_run(one, parts, None, resumed, identity, operational, {}, None, device='cpu')
            rt.train_run(cfg, parts, None, resumed, identity, operational, {}, None, device='cpu')
            first, second = rt.load_checkpoint(full/'latest.pt', identity), rt.load_checkpoint(resumed/'latest.pt', identity)
            assert first['epoch'] == second['epoch'] == 2
            for key in first['model']:
                torch.testing.assert_close(first['model'][key], second['model'][key], rtol=0, atol=0)
            assert first['history'][-1]['train_loss'] == second['history'][-1]['train_loss']
            assert (resumed/'previous_latest.pt').is_file() and (resumed/'best.pt').is_file()
            # A truncated best alias is repaired from the versioned best referenced by latest.
            (resumed/'best.pt').write_bytes(b'interrupted alias publication')
            rt.train_run(cfg, parts, None, resumed, identity, operational, {}, None, device='cpu')
            rt.load_checkpoint(resumed/'best.pt', identity)
            # Mismatched run/model config and damaged files must not be resumed.
            wrong = copy.deepcopy(identity)
            wrong['config']['attention'] = not attention
            try:
                rt.load_checkpoint(resumed/'latest.pt', wrong)
                raise AssertionError('Wrong run accepted')
            except ValueError:
                pass
            (resumed/'latest.pt').write_bytes(b'corrupt')
            try:
                rt.load_checkpoint(resumed/'latest.pt', identity)
                raise AssertionError('Corrupt checkpoint accepted')
            except IOError:
                pass
        print('PASS both variants: mining/train/val/smoke/checkpoint; resumed weights exactly equal uninterrupted run')
    finally:
        rt.make_model = factory


if __name__ == '__main__':
    test_notebooks()
    test_architecture()
    test_loss_and_batching()
    with tempfile.TemporaryDirectory(prefix='iir_colab_verify_') as temporary:
        root = Path(temporary)
        test_zip(root)
        test_sfm_contract(root)
        test_train_resume(root)
    print('ALL CPU CHECKS PASSED. Real Colab/VGG16 GPU throughput and VRAM remain untested.')
