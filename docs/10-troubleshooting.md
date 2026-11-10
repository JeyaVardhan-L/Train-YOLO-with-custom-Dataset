# 10. Troubleshooting

Entries marked **seen** were hit while building this guide and the aerial detector. The others are common causes of the same symptoms; check them, but they were not reproduced here.

| Symptom | Cause | Fix |
| :-- | :-- | :-- |
| `FileNotFoundError: Dataset '...yaml' images not found, missing path '...'` (**seen**) | A relative `path:` in `data.yaml` is resolved from the folder you run the command in, not from the YAML's folder. Or the `val:` folder name is wrong. | Use an absolute `path:` (`python -m yolotools yaml ...`), run `python -m yolotools check --data data.yaml` |
| Training starts but reports 0 images or "no labels found" | Labels folder not found. The labels folder is the images path with `images` replaced by `labels`; label names must match image names | Fix the folder names; run the checker |
| Warnings about "corrupt" images or labels, files being ignored | Label lines with the wrong number of values, coordinates outside 0 to 1, or unreadable images | The checker lists the exact files and lines |
| mAP stays near 0 | Class ids do not match `names`, boxes in pixels instead of fractions, or labels shifted | Open a few `train_batch*.jpg` to see whether boxes sit on objects; check the label format (chapter 2) |
| `CUDA out of memory` | Batch or image size too large for your GPU | Lower `batch` (for example 8 or 4), or `imgsz`, or use a smaller model |
| `torch.cuda.is_available()` is `False` on a machine with an NVIDIA GPU | CPU-only PyTorch installed, or driver problem | Reinstall PyTorch with the CUDA build from pytorch.org; update the driver |
| Data loading hangs or crashes on Windows | Multiprocessing data workers | Add `workers=0` |
| Very slow training | Running on CPU | Check `device=0` and the CUDA check above |
| `ModuleNotFoundError` when running `python -m yolotools` | Wrong folder or environment | Run from the repo root with the virtual environment active |
| Great validation score, poor real-world results | Data differs from real use, or validation leakage | Chapters 1 and 4 |
| Validation loss rises while training loss falls | Overfitting | Fewer epochs, more or more varied data |
| Weights file not found when predicting | Wrong run folder (`train`, `train2`, ...) | Look inside `runs/detect/` for the newest folder |
| Result numbers differ from an earlier run | Different Ultralytics version, hardware, batch size or random seed | Record versions and hardware with every result (chapter 7) |

## Asking for help

Include: your OS, Python, PyTorch and Ultralytics versions, the exact command, the **full** error text, and the output of `python -m yolotools check --data data.yaml`. Most questions are answered by those four things.

Next: [11. The aerial detector](11-the-aerial-detector.md)
