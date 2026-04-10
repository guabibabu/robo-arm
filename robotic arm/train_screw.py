#!/usr/bin/env python3
"""Train a YOLOv8 screw detector from the local Roboflow dataset.

Example:
    python3 train_screw.py
    python3 train_screw.py --epochs 50 --imgsz 640
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT_DIR / "SCREW DETECTION WORKSHOP" / "data.yaml"
DEFAULT_EXPORT = ROOT_DIR / "dobot_rac_workshop-master" / "models" / "screw_best.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the local screw dataset.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Path to YOLO data.yaml.")
    parser.add_argument("--base-model", default="yolov8n.pt", help="YOLO base model.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument(
        "--batch",
        default="auto",
        help="Batch size. Use an integer or 'auto'.",
    )
    parser.add_argument(
        "--project",
        default=str(ROOT_DIR / "runs" / "detect"),
        help="YOLO output project directory.",
    )
    parser.add_argument("--name", default="screw_train", help="YOLO run name.")
    parser.add_argument(
        "--export",
        default=str(DEFAULT_EXPORT),
        help="Where to copy the trained best.pt for sharing/detection.",
    )
    return parser


def load_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: ultralytics. Run: pip install ultralytics") from exc
    return YOLO


def normalize_batch(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--batch must be an integer or 'auto'.") from exc


def main() -> int:
    args = build_parser().parse_args()
    data_path = Path(args.data).expanduser()
    export_path = Path(args.export).expanduser()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    YOLO = load_yolo()
    model = YOLO(args.base_model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=normalize_batch(args.batch),
        project=str(Path(args.project).expanduser()),
        name=args.name,
        plots=True,
    )

    save_dir = Path(results.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training finished, but best.pt was not found at: {best_path}")

    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, export_path)

    print("=" * 70)
    print(f"Training run: {save_dir}")
    print(f"Best model: {best_path}")
    print(f"Copied shared model to: {export_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
