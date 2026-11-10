# Sample data

50 real images with YOLO labels from the project's dataset, so you can run every step of the guide without downloading anything.

| Split | Images | Boxes (Aircrafts / Bird / Drone) |
| :-- | :-: | :-: |
| `train/` | 33 | 21 / 9 / 16 |
| `val/` | 17 | 7 / 9 / 15 |

- No validation image shares a source photo with a training image (checked with `python -m yolotools leakage`).
- `data.yaml` uses `path: sample-data`, a relative path, so run commands from the repository root.
- This is far too little data to train a useful model. Use it to check that your setup works.

## Where the images come from

They are a hand-picked part of Roboflow Universe `subhranil-dey-wttsv/drone-detection-axkso`, version 3, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by its publisher. All 8,888 images of that dataset are 227 x 227 px.

The dataset collects images from many third-party sources. While choosing this sample, images were **left out** if they showed a visible stock-photo watermark or credit, an identifiable person, or looked like a flight-simulator screenshot. The rights of the remaining images could not be verified individually; they are included under the dataset's stated licence. If you own one of them and want it removed, open an issue.
