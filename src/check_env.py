"""Sanity check that the environment is set up and PyTorch sees the GPU."""
import cv2
import faiss
import timm
import torch
import ultralytics
from lightglue import LightGlue  # noqa: F401

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CPU-only PyTorch. Reinstall from the CUDA index (see requirements.txt).")
print("ultralytics", ultralytics.__version__, "| timm", timm.__version__)
print("opencv", cv2.__version__, "| SIFT:", cv2.SIFT_create() is not None, "| faiss", faiss.__version__)
print("nextvit_small in timm:", "nextvit_small" in timm.list_models("nextvit*"))
