# 3. Dataset layout and data.yaml

Ultralytics finds your labels by a naming rule, not by configuration. If the folders are not named as it expects, training fails or silently finds no labels.

## The layout

```text
my-dataset/
├── train/
│   ├── images/   photo001.jpg, photo002.jpg, ...
│   └── labels/   photo001.txt, photo002.txt, ...
└── val/
    ├── images/
    └── labels/
```

- The labels folder is found by replacing the **last `images` folder** in the image path with `labels`. So `my-dataset/train/images` uses `my-dataset/train/labels`.
- Each image needs a label file with the **same name** and the extension `.txt`.
- Image and label files can be any of the usual image types; keep names unique.

Many exports use `images/train` and `images/val` instead (with `labels/train`, `labels/val`). That works too, as long as the word `images` is replaced by `labels` in the path. This repo's tools assume the layout above; adapt the `train:` and `val:` lines of the YAML if yours differs.

If you do not have a train/validation split yet, read [chapter 4](04-split-without-leaking.md) first. Do **not** split at random without thinking about it.

## The data.yaml file

```yaml
path: /absolute/path/to/my-dataset
train: train/images
val: val/images
names:
  0: Aircrafts
  1: Bird
  2: Drone
```

- `train` and `val` are folders of images, relative to `path`.
- `names` maps the class numbers in your label files to readable names. The order must match your class ids.

### The path trap

Ultralytics resolves a **relative** `path:` against the folder you run the command from, not against the folder that contains the YAML. If you write `path: ../data` and run from somewhere else, you get an error like this (seen while building the [aerial detector project](11-the-aerial-detector.md)):

```text
FileNotFoundError: Dataset 'configs/data.yaml' images not found, missing path '.../data/validation/images'
```

The reliable fix is to write an **absolute** `path:`. This repo can generate the file for you:

```bash
python -m yolotools yaml --root my-dataset --names Aircrafts,Bird,Drone --out data.yaml
```

Add `--relative` only if you want a relative path and will always run from the same folder.

## Check it

```bash
python -m yolotools check --data data.yaml
```

Example output on the sample data from [chapter 0](00-start-here.md):

```text
Classes (3): {0: 'Aircrafts', 1: 'Bird', 2: 'Drone'}

[train] .../sample-data/train/images
  images: 33   boxes: 46   background images (empty label): 0
    class 0 (Aircrafts): 21
    class 1 (Bird): 9
    class 2 (Drone): 16
  note: 1 box covers under 0.1% of the image (hard to detect)

[val] .../sample-data/val/images
  images: 17   boxes: 31   background images (empty label): 0
    class 0 (Aircrafts): 7
    class 1 (Bird): 9
    class 2 (Drone): 15
  note: 1 box covers under 0.1% of the image (hard to detect)
Result: no problems found
```

Fix every `ERROR` line before you train. The command exits with a non-zero status when it finds problems, so you can use it in scripts.

Next: [4. Split without leaking](04-split-without-leaking.md)
