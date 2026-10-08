# B0 configuration placeholder

No final or runnable training configuration has been agreed or created.
The current specification and all unresolved values are recorded in
[B0_GeM.md](../../report/B0/B0_GeM.md). Values marked **TBD** must be resolved
explicitly before producing a training configuration; source defaults do not
automatically become TEAM B0 decisions.

Future configuration files should contain only lightweight, reviewable settings
and references to local datasets/checkpoints, never the artifacts themselves.

## Modern environment: Phase 2.2A / C0

Use Kaggle's preinstalled, compatible PyTorch/torchvision pair for B0. Record the
actual Python, PyTorch, torchvision, NumPy, Pillow, and CUDA versions in the run
record; the verified Kaggle version inventory is **TBD**. This is environment
guidance, not a final training configuration or a claim that Kaggle has been tested.

The root `requirements.txt` (`torch==1.0.0`, `torchvision==0.2.1`) and `setup.py`
(`torch>=1.0.0,<1.4.0`) retain the imported historical dependency declarations.
**Do not run `pip install -r requirements.txt` or `pip install .` for modern B0.**
Run from the repository root so Python can import the local package without an
installation. If a notebook imports from another working directory, put this
checkout on `PYTHONPATH` before starting Python.

Before any run, execute this lightweight preflight from the repository root:

```bash
python -B - <<'PY'
import sys
import numpy as np
import PIL
from PIL import Image
import torch
import torchvision
import cirtorch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("NumPy:", np.__version__)
print("Pillow:", PIL.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("cirtorch source:", cirtorch.__file__)
assert hasattr(Image.Resampling, "LANCZOS")
assert callable(np.isin)
PY
```

A missing dependency or torch/torchvision import error requires a compatible
modern environment, not installation of the historical pins. Record and review
any required environment adjustment before training.

C0 replaces only the removed Pillow resampling constant and NumPy membership API.
VGG16 construction is unchanged: it creates an untrained torchvision model and
loads the original custom Caffe-imported feature weights
(`imagenet-caffe-vgg16-features-d369c8e.pth`). Do not substitute torchvision
ImageNet weights or their classification transforms. Compatibility with the
actual Kaggle environment still needs verification; later audit findings remain
outside this patch.
