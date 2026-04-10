# Share Screw Detection With Teammates

This note explains how to let another teammate run screw detection on their own computer.

## What To Share

Share these files:

- `detect_screw.py`
- `dobot_rac_workshop-master/scripts/detect_screw_realsense.py`
- `requirements.txt`
- Your trained YOLO model, usually named `best.pt`

If your teammate only needs detection, sharing `best.pt` is enough. They do not need to download the Roboflow dataset again.

If your teammate also needs to retrain the model, they can use the Roboflow code in `yolov8.py` with the team API key.

## Recommended Folder Layout

Put the trained model here:

```text
dobot_rac_workshop-master/
  models/
    screw_best.pt
  scripts/
    detect_screw_realsense.py
```

Create the folder if it does not exist:

```bash
mkdir -p dobot_rac_workshop-master/models
```

Then copy `best.pt` into:

```text
dobot_rac_workshop-master/models/screw_best.pt
```

## Install Dependencies

Use the same Python environment that already runs `click_and_go.py`.

Install YOLOv8 if it is not installed:

```bash
pip install ultralytics
```

If they are setting up from scratch:

```bash
pip install -r requirements.txt
pip install ultralytics
```

Note: RealSense still needs the same `pyrealsense2` setup used by Click-and-Go.

## Option A: Detect From An Image

This does not need RealSense. It only needs OpenCV, Ultralytics, and the trained model.

```bash
python3 detect_screw.py --model dobot_rac_workshop-master/models/screw_best.pt --image "traing data/IMG_01.jpg" --save
```

The output JSON includes:

```json
{
  "best": {
    "center_xy": [320.0, 240.0],
    "bbox_xyxy": [280.0, 210.0, 360.0, 270.0],
    "confidence": 0.91
  }
}
```

The important value for the next robotic-arm step is `center_xy`.

## Option B: Detect From A Normal Webcam

```bash
python3 detect_screw.py --model dobot_rac_workshop-master/models/screw_best.pt --camera 0 --live
```

Press `q` to quit.

## Option C: Detect From RealSense

Run this from inside `dobot_rac_workshop-master`:

```bash
python3 scripts/detect_screw_realsense.py --model models/screw_best.pt
```

For one frame only:

```bash
python3 scripts/detect_screw_realsense.py --model models/screw_best.pt --once --save results/screw_detection.png
```

The RealSense version outputs both:

- `center_xy`: screw center in image pixel coordinates
- `camera_xyz_mm`: screw center in RealSense camera coordinates, using depth

It does not move the robot. This is intentional, so detection can be tested safely before connecting it to pick-and-place.

## Next Step

Once detection is stable, connect this output to Click-and-Go:

```text
YOLO center_xy -> RealSense depth -> camera_xyz_mm -> base_T_camera -> robot base XYZ -> pick command
```

The existing Click-and-Go pipeline already contains the later pieces: depth lookup, `base_T_camera`, workspace checks, and Dobot movement.
