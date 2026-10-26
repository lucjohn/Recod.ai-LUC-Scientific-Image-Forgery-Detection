# Recod.ai/LUC - Scientific Image Forgery Detection

Work for the Kaggle competition *Recod.ai/LUC - Scientific Image Forgery Detection*.

## Layout

```
data/       competition data (git-ignored)
models/     checkpoints / weights (git-ignored)
src/        source code (the `forgery` package)
tests/      pytest tests
scripts/    one-off utilities
notebooks/  exploration notebooks
```

## Environment

Python 3.12, NVIDIA RTX 4060. The venv lives outside OneDrive at `~/.venvs/forgery`, so the multi-GB PyTorch install doesn't sync.

```bash
py -3.12 -m venv ~/.venvs/forgery
~/.venvs/forgery/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
~/.venvs/forgery/Scripts/python -m pip install -r requirements.txt
~/.venvs/forgery/Scripts/python -m pip install -e .   # installs the `forgery` package from src/
~/.venvs/forgery/Scripts/python scripts/check_env.py   # should print CUDA available: True
~/.venvs/forgery/Scripts/python -m pytest              # run the tests
```

In Jupyter / VS Code notebooks, select the **Python (forgery)** kernel.

## Kaggle data

```bash
kaggle auth login    # one-time
kaggle competitions download -c recodai-luc-scientific-image-forgery-detection -p data
```
