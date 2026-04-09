import json
from pathlib import Path

from dobot_backend import create_dobot, sleep_for_device


ROOT_DIR = Path(__file__).resolve().parents[2]
POINTS_CONFIG_PATH = ROOT_DIR / "offline_arm_points.json"

DEFAULT_DEMO_CONFIG = {
    "pause_after_move_s": 2.0,
    "pause_after_report_s": 1.0,
    "points": [
        {"name": "P1", "x": 200.0, "y": 0.0, "z": 50.0, "r": 0.0},
        {"name": "P2", "x": 250.0, "y": 50.0, "z": 70.0, "r": 270.0},
        {"name": "P3", "x": 200.0, "y": -50.0, "z": 50.0, "r": 0.0},
        {"name": "P4", "x": 300.0, "y": 0.0, "z": 50.0, "r": -90.0},
    ],
}


def _write_default_config(config_path: Path) -> None:
    config_path.write_text(
        json.dumps(DEFAULT_DEMO_CONFIG, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_demo_config(config_path: Path = POINTS_CONFIG_PATH):
    if not config_path.exists():
        _write_default_config(config_path)

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Point config is not valid JSON: {config_path}") from exc

    if not isinstance(raw_config, dict):
        raise RuntimeError(f"Point config must be a JSON object: {config_path}")

    points = raw_config.get("points")
    if not isinstance(points, list) or not points:
        raise RuntimeError(f"`points` must be a non-empty array in {config_path}")

    normalized_points = []
    for index, point in enumerate(points, start=1):
        if not isinstance(point, dict):
            raise RuntimeError(f"Point #{index} must be an object in {config_path}")

        try:
            normalized_points.append(
                {
                    "name": str(point.get("name") or f"P{index}"),
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "z": float(point["z"]),
                    "r": float(point["r"]),
                }
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Point #{index} is missing `{exc.args[0]}` in {config_path}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Point #{index} must use numeric x/y/z/r values in {config_path}"
            ) from exc

    pause_after_move = float(raw_config.get("pause_after_move_s", 2.0))
    pause_after_report = float(raw_config.get("pause_after_report_s", 1.0))

    return normalized_points, pause_after_move, pause_after_report


if __name__ == "__main__":
    points, pause_after_move_s, pause_after_report_s = load_demo_config()
    device = create_dobot()
    print(f"Loaded demo points from: {POINTS_CONFIG_PATH}")
    print("Homing the robotic arm...")
    device.home()
    print(f"Arm homed. Starting demo moves to {len(points)} points.")

    for point in points:
        coordinates = (point["x"], point["y"], point["z"], point["r"])
        print(f"Moving to {point['name']}: {coordinates}")
        device.move_to(*coordinates, wait=True)
        sleep_for_device(device, pause_after_move_s)

        print("Current pose:", device.get_pose())
        sleep_for_device(device, pause_after_report_s)

    device.close()
