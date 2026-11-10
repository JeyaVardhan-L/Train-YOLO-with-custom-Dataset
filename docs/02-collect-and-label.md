# 2. Collect and label images

The model can only learn what your labels show it. This chapter is about getting images and boxes that are worth training on.

## Collect images

- **Take them where the model will work.** Same camera type, height, distance, lighting and weather as the real use.
- **Vary everything you can.** Time of day, backgrounds, object size, partly hidden objects, motion blur.
- **If you extract frames from video, do not keep every frame.** Neighbouring frames are almost identical. Take one every second or so. Remember which video each frame came from: you will need that in [chapter 4](04-split-without-leaking.md).
- **Add images with none of your objects** (empty sky, empty road, a cluttered desk). Their label files are empty. They teach the model what *not* to detect and reduce false alarms. A common rule of thumb is that some percent of the set is such images; the right share depends on your use.
- **Check licences and privacy.** Do not train on images you may not use, and avoid faces, number plates and personal data unless you have a reason and permission.

## Label the images

You draw a tight rectangle around every object of your classes and pick its class. Tools that can do this and export YOLO format include [Roboflow](https://roboflow.com), [CVAT](https://www.cvat.ai) and [Label Studio](https://labelstud.io). Check each tool's export options for "YOLO" (detection) when you pick one.

Rules that make labels good:

1. **Label every visible instance.** An unlabelled object teaches the model that it is background.
2. **Boxes snug to the object**, not loose, and not cutting off parts of it.
3. **Partly hidden objects**: label the visible part, and apply one rule for the whole dataset.
4. **Be consistent.** Two people labelling "the same" objects differently teach the model noise. Write your class definitions from chapter 1 down and use them.
5. **Do not label things that are too small to recognise yourself**, or decide a minimum size and stick to it.
6. **Review a sample of finished labels** by eye. Most dataset bugs are visible in ten minutes of looking.

## The YOLO label format

Each image `photo.jpg` has a text file `photo.txt` with the same name. One line per object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id` starts at **0** and follows the order of `names:` in your `data.yaml`.
- The four numbers are **fractions of the image size**, between 0 and 1: the box centre and the box size.

### Worked example

An image is 1280 px wide and 720 px tall. An object occupies the pixel box from (320, 180) to (640, 450).

| Quantity | Pixels | Normalised |
| :-- | :-: | :-: |
| Box width | 640 - 320 = 320 | 320 / 1280 = **0.25** |
| Box height | 450 - 180 = 270 | 270 / 720 = **0.375** |
| Centre x | (320 + 640) / 2 = 480 | 480 / 1280 = **0.375** |
| Centre y | (180 + 450) / 2 = 315 | 315 / 720 = **0.4375** |

If this object is class 2, the line is:

```text
2 0.375 0.4375 0.25 0.375
```

An image with several objects has several lines. An image with no objects has an **empty** label file (or no file; an empty file is clearer and the checker in this repo accepts it).

Labelling tools do this conversion for you when you export. You only need the formula to understand an error or to write your own converter.

## Check the labels before training

Wrong labels are the most common reason a model "does not learn". Run the checker from this repo once your files are in the layout of [chapter 3](03-dataset-layout-and-yaml.md):

```bash
python -m yolotools check --data data.yaml
```

It reports images without labels, labels without images, lines that are not five numbers, class ids that do not exist, coordinates outside 0 to 1, and how many boxes each class has per split.

Next: [3. Dataset layout and data.yaml](03-dataset-layout-and-yaml.md)
