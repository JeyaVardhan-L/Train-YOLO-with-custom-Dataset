# 8. Improve the model

Improving a detector is a loop: look at the mistakes, form a guess about the cause, change **one** thing, measure again. Tuning settings blindly is the slowest way.

## Step 1: look at the mistakes

1. Run `val` and open the per-class table (chapter 7).
2. Open the `val_batch*_pred.jpg` images and the confusion matrix.
3. Sort what you see into categories:

| What you see | Likely cause | First thing to try |
| :-- | :-- | :-- |
| Real objects missed, especially small or far ones | Objects too small at your image size; too few small examples | Larger `imgsz`, more labelled small objects |
| Boxes on empty background | No background images; background that looks like an object | Add images without objects; add the specific false-alarm scenes |
| One class often predicted as another | Classes look alike, labels inconsistent, or too few examples | Fix label consistency; add examples of both; merge the classes if they cannot be told apart |
| Boxes in roughly the right place but loose | Loose labels; low image resolution | Re-label more tightly; larger `imgsz` |
| Good on validation, bad in real use | Training data differs from real use | Collect data from the real setting |

## Step 2: fix the data first

In most projects, better data beats better settings.

- **Fix wrong labels.** Open the training images where the model is confidently "wrong"; often the label is the mistake.
- **Add the failure cases.** If it misses backlit objects, collect and label more backlit images.
- **Add background images** (empty label files) for scenes that cause false alarms.
- **Balance classes** by collecting more of the weak ones rather than only duplicating existing images.
- **Remove near-duplicates** and keep the split clean (chapter 4).

## Step 3: then try settings, one at a time

| Change | When it may help | Cost |
| :-- | :-- | :-- |
| Larger `imgsz` (for example 960 instead of 640) | Small objects | Slower training and inference, more memory |
| Larger model (`yolo11s` to `yolo11m`) | Underfitting: training metrics are also poor | Slower, more memory, more data needed |
| More epochs | Curves still rising at the end | More time; stop if validation loss rises |
| Smaller model | Needs to run on weak hardware | Usually lower accuracy |
| Less or more augmentation | Overfitting (more) or unrealistic images (less) | Needs experiments |

Keep a small table of every run: what you changed, the settings, and the per-class results. This project kept only two runs, which is one reason its conclusions are limited.

## Worked example: what this project's results show

The aerial model reached mAP@50-95 of 0.588 for aircraft, 0.654 for drones and only **0.428 for birds**. The confusion matrix shows birds and drones are rarely confused with each other, but about half of the false alarms (detections matching no label) are birds. Reasonable next steps, **none of which has been tried**:

1. Look at the Bird false alarms in the validation images. If they are clouds, tree branches or distant specks, add that background as empty-label images.
2. Check Bird labelling consistency: whole flock in one box or one box per bird?
3. Try `imgsz=960`, since birds are small.
4. First re-split the data by source photo (chapter 4), so the baseline you improve on is honest.

## When to stop

Stop when the metric you chose in chapter 1 meets its target **on data that looks like the real use**, or when the remaining errors are label noise you cannot fix. A model that is good enough and honestly measured beats one that is slightly higher on a leaky validation set.

Next: [9. Use the trained model](09-use-the-model.md)
