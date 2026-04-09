#!/usr/bin/env python3
"""Small control panel for editing pick/place poses and launching the simulator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

from manual_customized_task import DEFAULT_TASK_CONFIG, PICK_PLACE_CONFIG_PATH, load_task_config


ROOT_DIR = Path(__file__).resolve().parents[2]
TASK_SCRIPT = Path(__file__).resolve().with_name("manual_customized_task.py")


def save_task_config(config_path: Path, config: dict) -> None:
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class PickPlaceSimControlPanel:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Dobot Pick and Place Editor")
        self.root.geometry("760x360")
        self.root.minsize(760, 340)

        self.pose_rows: dict[str, dict[str, tk.StringVar]] = {}
        self.jump_height_var = tk.StringVar()
        self.move_delay_var = tk.StringVar()
        self.grip_delay_var = tk.StringVar()
        self.release_delay_var = tk.StringVar()
        self.status_var = tk.StringVar(value="改数字后按 Run Simulation。")

        self._build_ui()
        self._load_current_values()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="改这一个窗口就好，按 Run 就会保存并启动 Pick and Place。",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text="你只要改 Pick / Place / Safe 三组座标，下一次运行就会立刻生效。",
        ).pack(anchor="w", pady=(4, 12))

        table = ttk.Frame(outer)
        table.pack(fill="x")

        headers = ("Pose", "X", "Y", "Z", "R")
        for column, header in enumerate(headers):
            ttk.Label(table, text=header, font=("Arial", 10, "bold")).grid(
                row=0, column=column, padx=6, pady=(0, 8), sticky="w"
            )

        for row_index, key in enumerate(("pick", "place", "safe_pose"), start=1):
            row_vars = {
                "name": tk.StringVar(),
                "x": tk.StringVar(),
                "y": tk.StringVar(),
                "z": tk.StringVar(),
                "r": tk.StringVar(),
            }
            self.pose_rows[key] = row_vars

            ttk.Entry(table, textvariable=row_vars["name"], width=12).grid(
                row=row_index, column=0, padx=6, pady=4, sticky="ew"
            )
            for offset, field in enumerate(("x", "y", "z", "r"), start=1):
                ttk.Entry(table, textvariable=row_vars[field], width=12).grid(
                    row=row_index, column=offset, padx=6, pady=4, sticky="ew"
                )

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(16, 10))

        ttk.Label(controls, text="Jump height (mm)").grid(row=0, column=0, padx=(0, 8), sticky="w")
        ttk.Entry(controls, textvariable=self.jump_height_var, width=8).grid(row=0, column=1, padx=(0, 18))
        ttk.Label(controls, text="Move delay (s)").grid(row=0, column=2, padx=(0, 8), sticky="w")
        ttk.Entry(controls, textvariable=self.move_delay_var, width=8).grid(row=0, column=3, padx=(0, 18))
        ttk.Label(controls, text="Grip delay (s)").grid(row=1, column=0, padx=(0, 8), pady=(10, 0), sticky="w")
        ttk.Entry(controls, textvariable=self.grip_delay_var, width=8).grid(
            row=1, column=1, padx=(0, 18), pady=(10, 0)
        )
        ttk.Label(controls, text="Release delay (s)").grid(
            row=1, column=2, padx=(0, 8), pady=(10, 0), sticky="w"
        )
        ttk.Entry(controls, textvariable=self.release_delay_var, width=8).grid(
            row=1, column=3, padx=(0, 18), pady=(10, 0)
        )

        button_row = ttk.Frame(outer)
        button_row.pack(fill="x", pady=(8, 0))
        ttk.Button(button_row, text="Run Simulation", command=self.run_simulation).pack(side="left")
        ttk.Button(button_row, text="Reset Defaults", command=self.reset_defaults).pack(side="left", padx=8)

        ttk.Label(outer, textvariable=self.status_var, foreground="#333333").pack(anchor="w", pady=(16, 0))

    def _load_current_values(self) -> None:
        config = load_task_config(PICK_PLACE_CONFIG_PATH)
        for key in ("pick", "place", "safe_pose"):
            pose = config[key]
            row_vars = self.pose_rows[key]
            row_vars["name"].set(pose["name"])
            row_vars["x"].set(str(pose["x"]))
            row_vars["y"].set(str(pose["y"]))
            row_vars["z"].set(str(pose["z"]))
            row_vars["r"].set(str(pose["r"]))

        self.jump_height_var.set(str(config["jump_height_mm"]))
        self.move_delay_var.set(str(config["move_delay_s"]))
        self.grip_delay_var.set(str(config["grip_delay_s"]))
        self.release_delay_var.set(str(config["release_delay_s"]))

    def _collect_values(self) -> dict:
        config = {}
        for key in ("pick", "place", "safe_pose"):
            row_vars = self.pose_rows[key]
            name = row_vars["name"].get().strip() or DEFAULT_TASK_CONFIG[key]["name"]
            try:
                config[key] = {
                    "name": name,
                    "x": float(row_vars["x"].get().strip()),
                    "y": float(row_vars["y"].get().strip()),
                    "z": float(row_vars["z"].get().strip()),
                    "r": float(row_vars["r"].get().strip()),
                }
            except ValueError as exc:
                raise ValueError(f"{name} 的 X/Y/Z/R 需要是数字。") from exc

        try:
            config["jump_height_mm"] = float(self.jump_height_var.get().strip())
            config["move_delay_s"] = float(self.move_delay_var.get().strip())
            config["grip_delay_s"] = float(self.grip_delay_var.get().strip())
            config["release_delay_s"] = float(self.release_delay_var.get().strip())
        except ValueError as exc:
            raise ValueError("Jump height 和 delay 都必须是数字。") from exc

        return config

    def reset_defaults(self) -> None:
        for key in ("pick", "place", "safe_pose"):
            pose = DEFAULT_TASK_CONFIG[key]
            row_vars = self.pose_rows[key]
            row_vars["name"].set(pose["name"])
            row_vars["x"].set(str(pose["x"]))
            row_vars["y"].set(str(pose["y"]))
            row_vars["z"].set(str(pose["z"]))
            row_vars["r"].set(str(pose["r"]))

        self.jump_height_var.set(str(DEFAULT_TASK_CONFIG["jump_height_mm"]))
        self.move_delay_var.set(str(DEFAULT_TASK_CONFIG["move_delay_s"]))
        self.grip_delay_var.set(str(DEFAULT_TASK_CONFIG["grip_delay_s"]))
        self.release_delay_var.set(str(DEFAULT_TASK_CONFIG["release_delay_s"]))
        self.status_var.set("已恢复默认值，按 Run 生效。")

    def run_simulation(self) -> None:
        try:
            config = self._collect_values()
            save_task_config(PICK_PLACE_CONFIG_PATH, config)
        except Exception as exc:
            messagebox.showerror("无法启动", str(exc))
            return

        env = os.environ.copy()
        env["DOBOT_BACKEND"] = "sim"
        env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            subprocess.Popen([sys.executable, str(TASK_SCRIPT)], cwd=str(ROOT_DIR), env=env)
        except Exception as exc:
            messagebox.showerror("无法启动模拟", str(exc))
            return

        self.status_var.set("已保存并启动 Pick and Place。改完数字后可以再按一次 Run。")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = PickPlaceSimControlPanel()
    app.run()


if __name__ == "__main__":
    main()
