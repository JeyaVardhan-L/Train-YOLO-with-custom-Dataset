# 6. Train a model

You start from a model that has already learned general visual features on a large dataset (COCO) and **fine-tune** it on your images. Fine-tuning needs far less data and time than training from scratch.

## The command

```bash
yolo detect train data=data.yaml model=yolo11s.pt epochs=60 imgsz=640 batch=16 device=0
```

- `model=yolo11s.pt` is the starting point. Ultralytics downloads it on first use.
- `device=0` uses the first GPU. Use `device=cpu` on a machine without one.
- Results go to `runs/detect/train` (then `train2`, `train3`, ... for later runs).

The same thing from Python:

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(data="data.yaml", epochs=60, imgsz=640, batch=16, device=0)
```

## The settings you will actually change

| Setting | What it does | Starting point |
| :-- | :-- | :-- |
| `model` | Size of the network: `yolo11n` (smallest), `s`, `m`, `l`, `x` (largest) | `yolo11s.pt` for real projects, `yolo11n.pt` for quick tests |
| `epochs` | Passes over the training set | 50 to 100; watch the curves (chapter 7) |
| `imgsz` | Images are resized to this size | 640. Larger helps small objects but costs time and memory |
| `batch` | Images per step | 16. Lower it if you run out of GPU memory |
| `patience` | Stop early if validation fitness has not improved for this many epochs | 10 to 50. This project used 5, which can stop early on a noisy curve |
| `device` | `0`, `0,1`, or `cpu` | Your GPU if you have one |
| `workers` | Data loading processes | Default; use `0` on Windows if loading hangs |
| `name` | Name of the output folder | Something that says what you changed |

Everything else can stay at its default until you have a reason (chapter 8). Ultralytics saves the full list of arguments of every run in `args.yaml`, so you can always see what a run used.

## What happens while it trains

Each epoch prints training losses, then validation precision, recall and mAP. You want the validation mAP to rise over the first epochs, then flatten.

```text
 Epoch   box_loss   cls_loss   dfl_loss   Instances   Size
 ...
   Class     Images  Instances   Box(P      R      mAP50  mAP50-95)
     all         17         31   0.539   0.334   0.320   0.201
```

## What you get

```text
runs/detect/train/
├── weights/best.pt        best checkpoint on the validation set
├── weights/last.pt        final epoch (used to resume)
├── results.csv            numbers for every epoch
├── results.png            the curves
├── confusion_matrix*.png
├── *PR_curve.png, *F1_curve.png, ...
├── train_batch*.jpg       examples of what the model saw, with boxes
├── val_batch*_labels.jpg  validation images with the true boxes
├── val_batch*_pred.jpg    the same images with the predicted boxes
└── args.yaml              the settings of this run
```

Open `train_batch0.jpg` before anything else. If the boxes do not sit on the objects, your labels or class ids are wrong; stop and fix them.

## How long does it take?

| Run | Hardware | Time |
| :-- | :-- | :-- |
| Sample data (33 train + 17 val images, 320 px), `yolo11n`, 20 epochs | Laptop CPU (13th Gen Intel Core i5-13420H) | 0.050 hours (about 3 minutes) as reported by Ultralytics |
| Aerial detector (7,110 train images, 640 px), `yolo11s`, 57 epochs | GPU, model not recorded | About 3.2 hours |

The full dataset on a CPU was not timed; it is far larger than the sample and a GPU is the sensible choice. For a quick check that your setup works, the sample data with `epochs=3` is enough.

## Continue an interrupted run

```bash
yolo detect train resume model=runs/detect/train/weights/last.pt
```

## Train again from the best weights

You can use a previous `best.pt` as the starting model (`model=runs/detect/train/weights/best.pt`) to continue improving on new data. Keep track of what each run started from; otherwise you cannot explain your results later.

Next: [7. Evaluate the model](07-evaluate.md)
