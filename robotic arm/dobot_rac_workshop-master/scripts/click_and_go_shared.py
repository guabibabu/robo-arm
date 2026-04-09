#!/usr/bin/env python3
"""Shared configuration and transform helpers for Click-and-Go."""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
DEVICE_CONFIG_PATH = ROOT_DIR / "config" / "device_port.yaml"
APP_CONFIG_PATH = ROOT_DIR / "config" / "click_and_go.yaml"

DEFAULT_APP_CONFIG = {
    "camera": {
        "width": 640,
        "height": 480,
        "fps": 30,
        "tag_size_m": 0.0792,
        "depth_window_px": 5,
    },
    "motion": {
        "safe_pose": {"x": 200.0, "y": 0.0, "z": 80.0, "r": 0.0},
        "hover_z_mm": 90.0,
        "approach_offset_z_mm": 30.0,
        "target_offset_z_mm": 15.0,
        "target_r_deg": 0.0,
    },
    "workspace": {
        "x_min_mm": 120.0,
        "x_max_mm": 320.0,
        "y_min_mm": -180.0,
        "y_max_mm": 180.0,
        "z_min_mm": -10.0,
        "z_max_mm": 120.0,
        "max_radius_mm": 320.0,
    },
    "calibration": {
        "gripper_T_tag": [
            [-1.0, 0.0, 0.0, 30.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 153.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "base_T_camera": None,
    },
}


def _load_yaml_module():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required for Click-and-Go configuration.") from exc
    return yaml


def deep_update(base, updates):
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path):
    yaml = _load_yaml_module()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def save_yaml(path: Path, data):
    yaml = _load_yaml_module()
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def load_device_config():
    return load_yaml(DEVICE_CONFIG_PATH)


def load_app_config():
    config = deep_update(DEFAULT_APP_CONFIG, load_yaml(APP_CONFIG_PATH))
    if not APP_CONFIG_PATH.exists():
        save_yaml(APP_CONFIG_PATH, config)
    return config


def invert_transform(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def point_to_homogeneous(point_xyz):
    return np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=float)


def transform_point(transform, point_xyz):
    return (transform @ point_to_homogeneous(point_xyz))[:3]


def get_tag_to_camera_matrix(tag):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(tag.pose_R, dtype=float)
    transform[:3, 3] = np.asarray(tag.pose_t, dtype=float).reshape(3) * 1000.0
    return transform


def get_robot_arm_matrix(pose):
    x = float(pose.position.x)
    y = float(pose.position.y)
    z = float(pose.position.z)
    r_deg = float(pose.position.r)
    yaw = np.deg2rad(r_deg)
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0, x],
            [np.sin(yaw), np.cos(yaw), 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
