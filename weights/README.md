# Weights

`weights/best.pt` is the trained model behind the results in this repository: YOLO11s fine-tuned on the aerial dataset, 3 classes (`Aircrafts`, `Bird`, `Drone`).

| | |
| :-- | :-- |
| File | `best.pt`, 19,178,451 bytes |
| SHA-256 | `22eea7251a0c8db2e0d0aa2436e45095e1943083bfbf022afd60feee556b20bf` |
| Training | 57 epochs, 640 px, batch 16, see `results/logs/run2_args.yaml` |
| Evaluation | `results/README.md` |

The file is **not stored in git** (it is larger than the 5 MB limit this repository keeps for committed files, and model files are better distributed as release assets). Put your copy at `weights/best.pt`. Check it with:

```bash
# Linux / macOS
sha256sum weights/best.pt
# Windows PowerShell
Get-FileHash weights/best.pt -Algorithm SHA256
```

If the hash differs, you do not have this model, and the numbers in this repository do not apply to it.

No weights file? Train your own with `scripts/train.py` or `yolo detect train` (chapter 6), then evaluate it yourself (chapter 7).

The weights are derived from Ultralytics YOLO, which is licensed AGPL-3.0 (or commercially); see the README for the licence note.
