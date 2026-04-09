import json
from pathlib import Path

from dobot_backend import create_dobot, sleep_for_device


ROOT_DIR = Path(__file__).resolve().parents[2]
PICK_PLACE_CONFIG_PATH = ROOT_DIR / "offline_pick_place.json"

DEFAULT_TASK_CONFIG = {
    "jump_height_mm": 50.0,
    "move_delay_s": 0.5,
    "grip_delay_s": 1.0,
    "release_delay_s": 1.0,
    "pick": {"name": "Pick", "x": 200.0, "y": 100.0, "z": 0.0, "r": 0.0},
    "place": {"name": "Place", "x": 200.0, "y": -100.0, "z": 0.0, "r": 0.0},
    "safe_pose": {"name": "Safe", "x": 200.0, "y": 0.0, "z": 50.0, "r": 0.0},
}


def _write_default_config(config_path: Path) -> None:
    config_path.write_text(
        json.dumps(DEFAULT_TASK_CONFIG, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _normalize_pose(raw_pose, field_name: str, config_path: Path) -> dict:
    if not isinstance(raw_pose, dict):
        raise RuntimeError(f"`{field_name}` must be an object in {config_path}")

    try:
        return {
            "name": str(raw_pose.get("name") or field_name.title()),
            "x": float(raw_pose["x"]),
            "y": float(raw_pose["y"]),
            "z": float(raw_pose["z"]),
            "r": float(raw_pose["r"]),
        }
    except KeyError as exc:
        raise RuntimeError(f"`{field_name}` is missing `{exc.args[0]}` in {config_path}") from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"`{field_name}` must use numeric x/y/z/r values in {config_path}") from exc


def load_task_config(config_path: Path = PICK_PLACE_CONFIG_PATH) -> dict:
    if not config_path.exists():
        _write_default_config(config_path)

    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Pick/place config is not valid JSON: {config_path}") from exc

    if not isinstance(raw_config, dict):
        raise RuntimeError(f"Pick/place config must be a JSON object: {config_path}")

    try:
        config = {
            "jump_height_mm": float(raw_config.get("jump_height_mm", DEFAULT_TASK_CONFIG["jump_height_mm"])),
            "move_delay_s": float(raw_config.get("move_delay_s", DEFAULT_TASK_CONFIG["move_delay_s"])),
            "grip_delay_s": float(raw_config.get("grip_delay_s", DEFAULT_TASK_CONFIG["grip_delay_s"])),
            "release_delay_s": float(raw_config.get("release_delay_s", DEFAULT_TASK_CONFIG["release_delay_s"])),
            "pick": _normalize_pose(raw_config.get("pick", DEFAULT_TASK_CONFIG["pick"]), "pick", config_path),
            "place": _normalize_pose(raw_config.get("place", DEFAULT_TASK_CONFIG["place"]), "place", config_path),
            "safe_pose": _normalize_pose(
                raw_config.get("safe_pose", DEFAULT_TASK_CONFIG["safe_pose"]),
                "safe_pose",
                config_path,
            ),
        }
    except ValueError as exc:
        raise RuntimeError(f"Jump height and delay values must be numbers in {config_path}") from exc

    return config


def pose_to_tuple(pose: dict) -> tuple[float, float, float, float]:
    return (pose["x"], pose["y"], pose["z"], pose["r"])


def jump_to(device, target_pose: dict, jump_height_mm: float, move_delay_s: float):
    current_pose = device.get_pose()
    current_z = current_pose.position.z
    x, y, z, r = pose_to_tuple(target_pose)

    lift_height = max(current_z, z) + jump_height_mm
    print(f"  Lifting to Z={lift_height:.1f}mm")
    device.move_to(current_pose.position.x, current_pose.position.y, lift_height, r, wait=True)
    sleep_for_device(device, move_delay_s)

    print(f"  Moving to X={x:.1f}, Y={y:.1f} at Z={lift_height:.1f}mm")
    device.move_to(x, y, lift_height, r, wait=True)
    sleep_for_device(device, move_delay_s)

    print(f"  Descending to Z={z:.1f}mm")
    device.move_to(x, y, z, r, wait=True)
    sleep_for_device(device, move_delay_s)


def pick_and_place(device, config: dict):
    pick_pose = config["pick"]
    place_pose = config["place"]

    print("\n=== Starting Pick and Place Operation ===")

    print("Opening gripper...")
    device.grip(False)
    sleep_for_device(device, config["grip_delay_s"])

    print(
        f"\nMoving to {pick_pose['name']}: "
        f"({pick_pose['x']}, {pick_pose['y']}, {pick_pose['z']}, {pick_pose['r']})"
    )
    jump_to(device, pick_pose, config["jump_height_mm"], config["move_delay_s"])

    print("Gripping object...")
    device.grip(True)
    sleep_for_device(device, config["grip_delay_s"])

    print(
        f"\nMoving to {place_pose['name']}: "
        f"({place_pose['x']}, {place_pose['y']}, {place_pose['z']}, {place_pose['r']})"
    )
    jump_to(device, place_pose, config["jump_height_mm"], config["move_delay_s"])

    print("Releasing object...")
    device.grip(False)
    sleep_for_device(device, config["release_delay_s"])

    print("\n=== Pick and Place Complete ===\n")


if __name__ == "__main__":
    config = load_task_config()
    device = create_dobot()
    print(f"Loaded pick/place config from: {PICK_PLACE_CONFIG_PATH}")

    try:
        pick_and_place(device, config)
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
    finally:
        safe_pose = pose_to_tuple(config["safe_pose"])
        device.move_to(*safe_pose, wait=True)
        print("Closing connection...")
        device.grip(False)
        device.close()
        print("Done!")
