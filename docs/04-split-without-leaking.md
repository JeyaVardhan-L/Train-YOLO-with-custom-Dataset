# 4. Split without leaking

You train on one set of images and measure on another (the **validation** set). The measurement only means something if the validation images are genuinely new to the model. If near-copies of validation images are in the training set, the score is inflated and you will not find out until the model meets real data.

## Where leakage comes from

- **Augmented copies.** Dataset exporters such as Roboflow can save several flipped, rotated or brightened versions of each photo as separate files. A random split by file puts some copies in training and some in validation.
- **Video frames.** Frames a fraction of a second apart look almost the same.
- **Duplicates.** The same photo downloaded twice under different names.
- **Burst shots** of the same scene.

The fix is to split by **group**: every file that comes from the same source photo, video or scene goes to the same side.

## A real example

In the [aerial detector project](11-the-aerial-detector.md) the dataset came from Roboflow and the files were named like this:

```text
00000002_jpg.rf.433b7ff22d8cdaa0b249cf0369166aae.jpg
```

Everything before `.rf.` is the source photo; the part after it is a hash that differs between augmented copies. The 80/20 split had been done at random per file. Counting afterwards showed:

- 8,888 images came from 8,298 distinct source photos.
- 154 of those photos had copies on **both** sides of the split.
- **173 of the 1,778 validation images (9.7%)** shared a source photo with a training image.

Those validation scores are therefore probably optimistic. The model was never re-trained on a clean split, so how much they would drop is unknown. The point is that this is easy to do by accident and easy to check for.

## Check an existing split

```bash
python -m yolotools leakage --train data/train/images --val data/val/images --group roboflow
```

`--group roboflow` uses the `<source>_jpg.rf.<hash>` naming. For your own naming scheme, give a regular expression whose first group is the name of the source:

```bash
# frames named  clipA_000123.jpg, clipA_000124.jpg, clipB_000001.jpg ...  (group = clip)
python -m yolotools leakage --train data/train/images --val data/val/images --group-regex "^(.+?)_\d+\."
```

A result of `0` means no validation image shares a group with a training image. If you cannot describe your groups with a filename pattern, rename your files so that you can (for example, put the video name at the start).

## Make a clean split

Put all your images and labels in one folder (either flat, or with `images/` and `labels/` inside), then:

```bash
# Group-aware: keeps every copy of a photo together
python -m yolotools split --source all-data --out my-dataset --val-pct 0.2 --group roboflow

# Or with your own pattern
python -m yolotools split --source all-data --out my-dataset --group-regex "^(.+?)_\d+\."

# Random per image (only correct when every image really is independent)
python -m yolotools split --source all-data --out my-dataset
```

What it does:

- Groups the files, shuffles the **groups** with a fixed seed (`--seed`, default 42, so the result is repeatable), and fills the validation set with whole groups until it has about `--val-pct` of the images.
- Copies files, so your originals stay untouched.
- Writes `train/images`, `train/labels`, `val/images`, `val/labels` under `--out`.

Run on the merged 8,888 images of this project with `--group roboflow`, this produced 7,109 training and 1,779 validation images and **0** validation images sharing a source photo with training (the original split had 173). The result is slightly off 80/20 because whole groups move together. No model was re-trained on this split.

Because whole groups move together, the validation share is approximate: with groups of several copies you may get slightly more than 20%.

The repo's tests build a small collection of 20 photos with 3 augmented copies each. A random split leaks copies into validation; the group split gives 0 leaking images (`tests/test_tools.py`).

## Other things worth checking

- **Class balance per split.** `python -m yolotools check` prints the box count per class for train and validation. A rare class may vanish from validation after a random split; with very small classes, check by hand.
- **Same distribution.** The validation set should look like your real use, not like the easiest part of your data.
- **A separate test set.** If you pick the best checkpoint or tune settings on the validation set, its score becomes optimistic as well. For a number you will quote, keep a third, untouched **test** set and evaluate on it once. This project has none, which is one of its limits.

Next: [5. Set up your environment](05-set-up-environment.md)
