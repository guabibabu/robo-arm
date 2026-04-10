#!/usr/bin/env python3
"""Detect screws with YOLOv8 using the same RealSense setup as Click-and-Go.

This script only detects and reports coordinates. It does not move the robot.

Examples:
    python scripts/detect_screw_realsense.py --model models/screw_best.pt
    python scripts/detect_screw_realsense.py --model models/screw_best.pt --once --save results/screw.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from click_and_go_shared import load_app_config, load_device_config


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT_DIR / "models" / "screw_best.pt"
cv2 = None
np = None
rs = None
YOLO = None


def ensure_runtime_dependencies() -> None:
    global cv2, np, rs, YOLO
    missing = []

    if cv2 is None:
        try:
            import cv2 as cv2_module
            cv2 = cv2_module
        except ModuleNotFoundError:
            missing.append("opencv-python")

    if np is None:
        try:
            import numpy as np_module
            np = np_module
        except ModuleNotFoundError:
            missing.append("numpy")

    if rs is None:
        try:
            import pyrealsense2 as rs_module
            rs = rs_module
        except ModuleNotFoundError:
            missing.append("pyrealsense2")

    if YOLO is None:
        try:
            from ultralytics import YOLO as yolo_class
            YOLO = yolo_class
        except ModuleNotFoundError:
            missing.append("ultralytics")

    if missing:
        raise RuntimeError(
            "Missing Python packages: "
            + ", ".join(sorted(set(missing)))
            + ". Use the same environment that runs Click-and-Go, then install ultralytics if needed."
        )


@dataclass
class ScrewDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    center_xy: list[float]
    camera_xyz_mm: list[float] | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect screw center pixels and RealSense camera-frame XYZ coordinates."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"Path to trained screw YOLO weights. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--class-name",
        default="screw",
        help="Class name to keep. Matching is case-insensitive and accepts singular/plural. Use empty string to keep all classes.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture one aligned RealSense frame, print detections, then exit.",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=str(ROOT_DIR / "results" / "screw_detection.png"),
        help="Optional path to save an annotated image.",
    )
    parser.add_argument(
        "--depth-window",
        type=int,
        default=None,
        help="Median depth window in pixels. Default comes from config/click_and_go.yaml.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=1.0,
        help="Seconds between JSON prints in live mode.",
    )
    return parser


def normalize_class_name(name: str) -> str:
    return name.strip().lower().rstrip("s")


def class_matches(class_name: str, wanted: str) -> bool:
    if not wanted.strip():
        return True
    return normalize_class_name(class_name) == normalize_class_name(wanted)


def initialize_pipeline(camera_cfg: dict[str, Any], serial_number: str | None):
    pipeline = rs.pipeline()
    config = rs.config()

    if serial_number:
        config.enable_device(serial_number)

    config.enable_stream(
        rs.stream.color,
        int(camera_cfg["width"]),
        int(camera_cfg["height"]),
        rs.format.bgr8,
        int(camera_cfg["fps"]),
    )
    config.enable_stream(
        rs.stream.depth,
        int(camera_cfg["width"]),
        int(camera_cfg["height"]),
        rs.format.z16,
        int(camera_cfg["fps"]),
    )

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not start the RealSense pipeline. Check camera USB/power and camera serial."
        ) from exc

    return pipeline, profile, rs.align(rs.stream.color)


def get_color_intrinsics(profile: Any) -> Any:
    return profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()


def get_aligned_frames(pipeline: Any, align: Any) -> tuple[Any | None, Any | None]:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None
    return color_frame, depth_frame


def median_depth_at_pixel(depth_frame: Any, pixel_x: int, pixel_y: int, window_size: int) -> float | None:
    half = max(0, window_size // 2)
    valid_depths = []

    for y in range(pixel_y - half, pixel_y + half + 1):
        for x in range(pixel_x - half, pixel_x + half + 1):
            if x < 0 or y < 0 or x >= depth_frame.get_width() or y >= depth_frame.get_height():
                continue
            depth_m = depth_frame.get_distance(x, y)
            if depth_m > 0:
                valid_depths.append(depth_m)

    if not valid_depths:
        return None

    return float(np.median(valid_depths))


def pixel_to_camera_point_mm(
    intrinsics: Any,
    pixel_x: int,
    pixel_y: int,
    depth_frame: Any,
    window_size: int,
) -> list[float] | None:
    depth_m = median_depth_at_pixel(depth_frame, pixel_x, pixel_y, window_size)
    if depth_m is None:
        return None

    point_m = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth_m)
    return [round(float(value) * 1000.0, 2) for value in point_m]


def load_model(path: Path) -> YOLO:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Copy your trained best.pt to dobot_rac_workshop-master/models/screw_best.pt, "
            "or pass the correct path with --model."
        )
    return YOLO(str(path))


def predict_screws(
    model: YOLO,
    color_image: Any,
    depth_frame: Any,
    intrinsics: Any,
    class_name: str,
    conf: float,
    imgsz: int,
    depth_window_px: int,
) -> list[ScrewDetection]:
    result = model.predict(color_image, conf=conf, imgsz=imgsz, verbose=False)[0]
    detections: list[ScrewDetection] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        detected_class_name = str(result.names.get(class_id, class_id))
        if not class_matches(detected_class_name, class_name):
            continue

        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        center_x = int(round((x1 + x2) / 2.0))
        center_y = int(round((y1 + y2) / 2.0))
        camera_xyz_mm = pixel_to_camera_point_mm(
            intrinsics,
            center_x,
            center_y,
            depth_frame,
            depth_window_px,
        )
        detections.append(
            ScrewDetection(
                class_id=class_id,
                class_name=detected_class_name,
                confidence=round(float(box.conf[0]), 4),
                bbox_xyxy=[round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                center_xy=[float(center_x), float(center_y)],
                camera_xyz_mm=camera_xyz_mm,
            )
        )

    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections


def print_detections(detections: list[ScrewDetection]) -> None:
    payload = {
        "count": len(detections),
        "best": asdict(detections[0]) if detections else None,
        "detections": [asdict(detection) for detection in detections],
    }
    print(json.dumps(payload, indent=2))


def draw_detections(image: Any, detections: list[ScrewDetection]) -> Any:
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox_xyxy]
        center_x, center_y = [int(round(value)) for value in detection.center_xy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.circle(annotated, (center_x, center_y), 5, (0, 255, 0), -1)

        label = f"{detection.class_name} {detection.confidence:.2f} px=({center_x},{center_y})"
        if detection.camera_xyz_mm is not None:
            x_mm, y_mm, z_mm = detection.camera_xyz_mm
            label += f" cam=({x_mm:.0f},{y_mm:.0f},{z_mm:.0f})mm"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def save_image(path: str, image: Any) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    print(f"Annotated image saved to: {output}")


def main() -> int:
    args = build_parser().parse_args()
    ensure_runtime_dependencies()
    app_config = load_app_config()
    device_config = load_device_config()
    camera_cfg = app_config["camera"]
    depth_window_px = int(args.depth_window or camera_cfg["depth_window_px"])

    serial_number = device_config.get("camera_serial")
    if serial_number:
        print(f"Using RealSense serial: {serial_number}")

    model = load_model(Path(args.model).expanduser())
    pipeline, profile, align = initialize_pipeline(camera_cfg, serial_number)
    intrinsics = get_color_intrinsics(profile)

    window_name = "Screw Detection - RealSense"
    last_print_time = 0.0
    last_detections: list[ScrewDetection] = []

    try:
        while True:
            color_frame, depth_frame = get_aligned_frames(pipeline, align)
            if color_frame is None or depth_frame is None:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            detections = predict_screws(
                model,
                color_image,
                depth_frame,
                intrinsics,
                args.class_name,
                args.conf,
                args.imgsz,
                depth_window_px,
            )
            annotated = draw_detections(color_image, detections)
            last_detections = detections

            now = time.monotonic()
            if args.once or now - last_print_time >= float(args.print_every):
                print_detections(detections)
                last_print_time = now

            if args.save and (args.once or detections):
                save_image(args.save, annotated)
                args.save = None

            if args.once:
                return 0 if detections else 2

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0 if last_detections else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
