#!/usr/bin/env python3
"""Detect screw center pixels with a trained YOLOv8 model.

Examples:
    python detect_screw.py --model runs/detect/train/weights/best.pt --image "traing data/IMG_01.jpg"
    python detect_screw.py --model best.pt --camera 0 --save results/screw_detection.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("runs") / "detect" / "train" / "weights" / "best.pt"
_CV2 = None


def require_cv2() -> Any:
    global _CV2
    if _CV2 is not None:
        return _CV2
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: opencv-python. Install project requirements first.") from exc
    _CV2 = cv2
    return _CV2


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    center_xy: list[float]
    width: float
    height: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a YOLOv8 screw detector and print the detected center pixel."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"Path to trained YOLO weights. Default: {DEFAULT_MODEL}",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to an image file.")
    source.add_argument(
        "--camera",
        type=int,
        help="Camera index for cv2.VideoCapture, usually 0 for the default webcam.",
    )

    parser.add_argument(
        "--class-name",
        default="screw",
        help="Class name to keep. Matching is case-insensitive and accepts singular/plural. Use empty string to keep all classes.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument(
        "--save",
        nargs="?",
        const="results/screw_detection.png",
        help="Optional path to save an annotated detection image.",
    )
    parser.add_argument("--show", action="store_true", help="Show the annotated frame in a window.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="With --camera, keep detecting until q is pressed. Without --live, only one frame is captured.",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height.")
    return parser


def normalize_class_name(name: str) -> str:
    return name.strip().lower().rstrip("s")


def class_matches(class_name: str, wanted: str) -> bool:
    if not wanted.strip():
        return True
    return normalize_class_name(class_name) == normalize_class_name(wanted)


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Train the YOLOv8 model first and copy best.pt to this path, "
            "or pass the correct path with --model."
        )
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: ultralytics. Run: pip install ultralytics") from exc
    return YOLO(str(path))


def predict_frame(model: Any, frame: Any, conf: float, imgsz: int) -> Any:
    return model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]


def detections_from_result(result: Any, class_name: str) -> list[Detection]:
    names = result.names
    detections: list[Detection] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        xyxy = [float(value) for value in box.xyxy[0].tolist()]
        class_id = int(box.cls[0])
        detected_class_name = str(names.get(class_id, class_id))
        if not class_matches(detected_class_name, class_name):
            continue

        x1, y1, x2, y2 = xyxy
        width = x2 - x1
        height = y2 - y1
        detections.append(
            Detection(
                class_id=class_id,
                class_name=detected_class_name,
                confidence=float(box.conf[0]),
                bbox_xyxy=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                center_xy=[round((x1 + x2) / 2.0, 2), round((y1 + y2) / 2.0, 2)],
                width=round(width, 2),
                height=round(height, 2),
            )
        )

    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections


def detect_frame(model: Any, frame: Any, class_name: str, conf: float, imgsz: int) -> list[Detection]:
    result = predict_frame(model, frame, conf, imgsz)
    return detections_from_result(result, class_name)


def annotate_result(result: Any) -> Any:
    return result.plot()


def print_result(source: str, detections: list[Detection]) -> None:
    payload = {
        "source": source,
        "count": len(detections),
        "best": asdict(detections[0]) if detections else None,
        "detections": [asdict(detection) for detection in detections],
    }
    print(json.dumps(payload, indent=2))


def save_or_show_frame(frame: Any, save_path: str | None, show: bool) -> None:
    cv2 = require_cv2()
    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), frame)
        print(f"Annotated image saved to: {output}")

    if show:
        cv2.imshow("Screw Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def read_image(path: Path) -> Any:
    cv2 = require_cv2()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not read image: {path}")
    return frame


def detect_image(args: argparse.Namespace, model: Any) -> int:
    image_path = Path(args.image).expanduser()
    frame = read_image(image_path)
    result = predict_frame(model, frame, args.conf, args.imgsz)
    detections = detections_from_result(result, args.class_name)
    print_result(str(image_path), detections)

    if args.save or args.show:
        annotated = annotate_result(result)
        save_or_show_frame(annotated, args.save, args.show)

    return 0 if detections else 2


def open_camera(index: int, width: int, height: int) -> Any:
    cv2 = require_cv2()
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {index}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def detect_camera_once(args: argparse.Namespace, model: Any) -> int:
    capture = open_camera(args.camera, args.width, args.height)
    try:
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        raise RuntimeError("Could not capture a frame from the camera.")

    result = predict_frame(model, frame, args.conf, args.imgsz)
    detections = detections_from_result(result, args.class_name)
    print_result(f"camera:{args.camera}", detections)

    if args.save or args.show:
        annotated = annotate_result(result)
        save_or_show_frame(annotated, args.save, args.show)

    return 0 if detections else 2


def detect_camera_live(args: argparse.Namespace, model: Any) -> int:
    cv2 = require_cv2()
    capture = open_camera(args.camera, args.width, args.height)
    last_print_time = 0.0
    last_detections: list[Detection] = []

    print("Live detection started. Press q to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Could not capture a frame from the camera.")

            result = predict_frame(model, frame, args.conf, args.imgsz)
            detections = detections_from_result(result, args.class_name)
            annotated = annotate_result(result)

            now = time.monotonic()
            if now - last_print_time >= 1.0:
                print_result(f"camera:{args.camera}", detections)
                last_print_time = now
                last_detections = detections

            cv2.imshow("Screw Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0 if last_detections else 2


def main() -> int:
    args = build_parser().parse_args()
    model = load_model(Path(args.model).expanduser())

    if args.image:
        return detect_image(args, model)

    if args.live:
        return detect_camera_live(args, model)

    return detect_camera_once(args, model)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
