# Installing the repository

The historical `torch==1.0.0` and `torchvision==0.2.1` pins have been removed
from the active package metadata. They remain in Git history for provenance.
The imported `cirtorch` code needs PyTorch, torchvision, NumPy, Pillow, and
SciPy. The B0 evaluation notebook additionally uses pandas, Matplotlib, and
IPython.

1. Create a Python environment supported by the PyTorch build you choose. For
   example, on Linux or macOS:

   ```sh
   python3 -m venv .venv
   . .venv/bin/activate
   ```

   On Windows, activate the virtual environment with the command appropriate
   for your shell.
2. Select your operating system, pip, Python, and CPU/CUDA platform in the
   [official PyTorch installer](https://pytorch.org/get-started/locally/).
   Run its command to install a compatible `torch` and `torchvision` pair.
   On Kaggle, use the compatible pair already supplied by the image and run
   the notebook's own preflight; do not replace it merely to match a local
   environment.
3. From the repository root, install the project and analysis tools:

   ```sh
   python -m pip install -r requirements.txt
   ```

4. Check imports and the CPU evaluator tests:

   ```sh
   python -c "import torch, torchvision, cirtorch; print(torch.__version__, torchvision.__version__)"
   python -m unittest discover -s tests -p 'test_*.py'
   ```

For GPU work, also check `torch.cuda.is_available()` and the GPU count required
by the chosen notebook. This installation guide does not claim that the
historical training CLI or the new Kaggle pipeline has been run end to end in
every supported PyTorch version. Use the [notebook entrypoint guide](../notebooks/README.md)
to select the correct workflow and the [external artifact index](../artifacts/README.md)
for data, weights, and archived checkpoints.
