"""Export the trusted B0 validation-selected weights for one short GPU job."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'artifacts/kaggle-posttraining'))
from encoder_runtime import RetrievalModel
from cirtorch.networks.imageretrievalnet import ImageRetrievalNet
from cirtorch.layers.pooling import GeM

EXPECTED = '8a56bad2e1d585c68de42ff841f061e5469f76792750d8b4993e500294f19f27'


def prepare():
    torch.set_num_threads(2)
    folder = ROOT/'artifacts/revision-validation/publish/b0-inputs'
    folder.mkdir(parents=True, exist_ok=True)
    archive = ROOT/'B0_VGG16_GeM_baseline_reproduction_evaluation_artifacts.zip'
    with zipfile.ZipFile(archive) as z:
        name = 'B0_VGG16_GeM_Evaluation/run_artifacts/best.pt'
        payload = z.read(name)
    assert hashlib.sha256(payload).hexdigest() == EXPECTED
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    assert state['epoch'] == 12 and state['config']['selection'] == 'validation_loss'
    weights = dict(state['state_dict'])
    assert 'pool.p' in weights and 'p' not in weights
    weights['p'] = weights.pop('pool.p')
    model = RetrievalModel(attention=False).eval()
    model.load_state_dict(weights, strict=True)
    original = ImageRetrievalNet(list(RetrievalModel().features.children()), None, GeM(), None, state['meta']).eval()
    original.load_state_dict(state['state_dict'], strict=True)
    max_error = 0.
    torch.manual_seed(42)
    with torch.inference_mode():
        for shape in [(1, 3, 64, 80), (1, 3, 96, 64)]:
            x = torch.rand(shape)
            a, b = original(x), model(x)
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
            max_error = max(max_error, float((a-b).abs().max()))
    torch.save(weights, folder/'b0_epoch12.pth')
    sources = ROOT/'artifacts/kaggle-posttraining'
    for name in ['runtime.py', 'protocol.py', 'encoder_runtime.py', 'legacy_evaluate.py', 'artifacts_io.py', 'LICENSE']:
        shutil.copy2(sources/name, folder/name)
    for name in ['diffusion_checks.py', 'b0_clean_eval.py']:
        shutil.copy2(Path(__file__).parent/name, folder/name)
    manifest = dict(model='b0_epoch12', epoch=12, selection='validation_loss',
                    source_checkpoint_sha256=EXPECTED, source_commit=state['source_commit'],
                    state_key_mapping={'pool.p': 'p'}, smoke_max_abs_error=max_error,
                    precision='fp32', tf32=False, image_size=1024, search='fixed_clean_v1',
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in folder.iterdir() if p.is_file() and p.name not in ['manifest.json', 'dataset-metadata.json']})
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (folder/'dataset-metadata.json').write_text(json.dumps(dict(id='ndhoang2310/iir-revision-b0-inputs-v1',
        title='IIR Revision B0 Epoch12 Inputs v1', licenses=[dict(name='other')]), indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    prepare()
