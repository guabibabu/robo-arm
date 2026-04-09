#!/usr/bin/env python3
"""Shared Dobot backend selection for real hardware or offline simulation."""

from __future__ import annotations

import os
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEVICE_CONFIG_PATH = ROOT_DIR / "config" / "device_port.yaml"


def get_dobot_backend() -> str:
    return os.environ.get("DOBOT_BACKEND", "real").strip().lower() or "real"


def _load_device_port() -> str:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "PyYAML is required for the real Dobot backend. Install requirements or use DOBOT_BACKEND=sim."
        ) from exc

    if not DEVICE_CONFIG_PATH.exists():
        raise RuntimeError(f"Device config not found: {DEVICE_CONFIG_PATH}")

    with DEVICE_CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    port = config.get("device_port")
    if not port:
        raise RuntimeError(f"`device_port` is missing in {DEVICE_CONFIG_PATH}")
    return str(port)


def create_dobot(port: str | None = None):
    backend = get_dobot_backend()
    if backend == "sim":
        from simulated_dobot import SimulatedDobot

        return SimulatedDobot()

    if backend != "real":
        raise RuntimeError(f"Unsupported DOBOT_BACKEND={backend!r}. Use `real` or `sim`.")

    if port is None:
        port = _load_device_port()

    try:
        from pydobotplus import Dobot
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError(
            "pydobotplus is not installed, so the real Dobot backend cannot start. "
            "Install requirements or run with DOBOT_BACKEND=sim."
        ) from exc

    return Dobot(port=port)


def sleep_for_device(device, seconds: float) -> None:
    if hasattr(device, "sleep"):
        device.sleep(seconds)
        return

    time.sleep(seconds)
