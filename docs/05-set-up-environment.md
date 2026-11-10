# 5. Set up your environment

## What you need

- Python. The commands in this guide were run with Python 3.14.6; any Python version supported by your PyTorch and Ultralytics releases should work. If an install fails on a very new Python, try an older one such as 3.12.
- About 2 GB of disk for PyTorch and Ultralytics, plus space for your data and results.
- An NVIDIA GPU is optional for the smoke test and strongly recommended for real training.

## Create an isolated environment

```bash
git clone <this-repo-url>
cd yolo-custom-dataset-guide
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
```

## Install PyTorch, then the rest

Install PyTorch **first**, choosing the build that matches your machine. The selector on <https://pytorch.org/get-started/locally/> gives the exact command.

```bash
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU: use the command from pytorch.org for your CUDA version, for example
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then the other packages:

```bash
pip install -r requirements.txt
```

If you skip the first step, `pip` may install a CPU-only PyTorch even though you have a GPU, and training will be very slow.

## Check that it works

```bash
python -c "import torch, ultralytics; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| ultralytics', ultralytics.__version__)"
```

- `CUDA available: True` means the GPU will be used with `device=0`.
- `False` means CPU. That is fine for the smoke test; on a machine with an NVIDIA GPU it usually means the wrong PyTorch build or a driver problem (see [troubleshooting](10-troubleshooting.md)).

Run this repo's tests to confirm the tools work:

```bash
python -m unittest discover -s tests
```

## Windows notes

- If data loading hangs or crashes, add `workers=0` to the training command. It is slower but removes multiprocessing problems.
- Use forward slashes or quote paths that contain spaces.

## Licence of the tools you are about to use

Ultralytics YOLO is released under the **AGPL-3.0** licence, with a paid enterprise licence as the alternative. For learning and personal projects that is usually fine. If you plan to ship a product or run a network service built on it, read the current terms at <https://www.ultralytics.com/legal> first. Weights you train with it are generally treated as derived from it. This is not legal advice.

Next: [6. Train a model](06-train.md)
