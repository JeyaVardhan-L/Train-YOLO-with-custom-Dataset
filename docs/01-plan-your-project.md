# 1. Plan your project

Most failed YOLO projects fail before any training starts: the classes are vague, the data does not look like the real use case, or nobody decided what "good enough" means. Spend an hour on this chapter.

## Decide what you want the model to output

YOLO **object detection** answers one question: *which objects of which classes are in this image, and where?* Each answer is a box, a class and a confidence score.

Check that this is the task you need:

| If you need... | Use |
| :-- | :-- |
| Boxes around objects (count them, locate them) | Detection (this guide) |
| One label per whole image ("cat" or "dog") | Classification |
| Pixel-accurate outlines | Segmentation |
| Body or hand keypoints | Pose |

This guide covers detection only. The same ideas carry over to the other tasks, but the label format differs.

## Choose your classes

Write one sentence for each class that a stranger could use to label an image the same way you would.

- **Make classes visually distinct.** "Truck" and "lorry" are one class. "Small drone" and "large drone" are two classes only if the model can see the difference at your camera distance.
- **Merge what you cannot tell apart.** In the [aerial detector project](11-the-aerial-detector.md), helicopters, airliners and light aircraft share one `Aircrafts` class because the source labels did not separate them.
- **Keep the list short at first.** You can add classes later; fixing a messy class list later means relabelling.
- **Decide what is "nothing".** Images that contain none of your objects are useful. Plan to include some (see chapter 2).

## Decide how you will judge success

Pick the number you will check at the end, and a target, before you look at results. For detection the usual choice is **mAP@50-95** (strict) or **mAP@50** (lenient), plus precision and recall for the class you care about most. Chapter 7 explains all of them.

Also write down the cost of each mistake. A missed object (false negative) and a false alarm (false positive) rarely cost the same. That decides which threshold you will use later.

## How much data do you need?

There is no exact number. Ultralytics' own training tips suggest roughly 1,500 or more images per class and many instances per class as a starting point (check their current documentation for the exact figures). This project used about 3,000 images per class, and its weakest class was the one with the hardest objects, not the one with the fewest examples.

What matters more than the raw count:

- **Variety.** Different backgrounds, lighting, distances, camera angles and object sizes. A thousand near-identical frames are worth far less than a few hundred varied ones.
- **Matching the real use.** Train on images that look like what the model will see. A model trained on clean stock photos struggles on a blurry phone video.
- **Balanced-enough classes.** One class with 10 times more examples than another tends to dominate.
- **Hard cases.** Include small, partly hidden and blurry objects, and look-alikes of your classes that should *not* be detected.

## Check your hardware

| Setup | What to expect |
| :-- | :-- |
| CPU only | Fine for the smoke test and tiny datasets. Real training on thousands of images takes many hours or days. |
| NVIDIA GPU | The normal choice. Training on about 7,000 images for around 57 epochs took about 3.2 hours in this project (the GPU model was not recorded). |
| Cloud notebook with a GPU | A workable option when you have no GPU. It is not tested in this guide. |

You can do chapters 1 to 4 and the smoke test on any laptop.

## Checklist before you move on

- [ ] I can say in one sentence what the model must detect.
- [ ] Each class has a written definition.
- [ ] I know the metric and the target I will judge by.
- [ ] I have a plan for collecting varied images that look like the real use.

Next: [2. Collect and label images](02-collect-and-label.md)
