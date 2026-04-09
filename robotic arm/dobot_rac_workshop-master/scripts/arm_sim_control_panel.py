#!/usr/bin/env python3
"""Small control panel for editing demo points and launching the offline simulator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

from arm_move import DEFAULT_DEMO_CONFIG, POINTS_CONFIG_PATH, load_demo_config


ROOT_DIR = Path(__file__).resolve().parents[2]
ARM_MOVE_SCRIPT = Path(__file__).resolve().with_name("arm_move.py")


def save_demo_config(config_path: Path, pause_after_move_s: float, pause_after_report_s: float, points: list[dict]) -> None:
    payload = {
        "pause_after_move_s": pause_after_move_s,
        "pause_after_report_s": pause_after_report_s,
        "points": points,
    }
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class ArmSimControlPanel:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Dobot Arm Quick Editor")
        self.root.geometry("720x340")
        self.root.minsize(720, 320)

        self.point_rows: list[dict[str, tk.StringVar]] = []
        self.pause_after_move_var = tk.StringVar()
        self.pause_after_report_var = tk.StringVar()
        self.status_var = tk.StringVar(value="改数字后按 Run Simulation。")

        self._build_ui()
        self._load_current_values()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(
            outer,
            text="改这一个窗口就好，按 Run 就会保存并启动离线机械臂。",
            font=("Arial", 13, "bold"),
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            outer,
            text="改完数字后按 Run，下面这四个点会立刻用于下一次模拟。",
        )
        subtitle.pack(anchor="w", pady=(4, 12))

        table = ttk.Frame(outer)
        table.pack(fill="x")

        headers = ("Point", "X", "Y", "Z", "R")
        for column, header in enumerate(headers):
            ttk.Label(table, text=header, font=("Arial", 10, "bold")).grid(
                row=0,
                column=column,
                padx=6,
                pady=(0, 8),
                sticky="w",
            )

        for row_index in range(4):
            row_vars = {
                "name": tk.StringVar(),
                "x": tk.StringVar(),
                "y": tk.StringVar(),
                "z": tk.StringVar(),
                "r": tk.StringVar(),
            }
            self.point_rows.append(row_vars)

            ttk.Entry(table, textvariable=row_vars["name"], width=10).grid(
                row=row_index + 1, column=0, padx=6, pady=4, sticky="ew"
            )
            for offset, key in enumerate(("x", "y", "z", "r"), start=1):
                ttk.Entry(table, textvariable=row_vars[key], width=12).grid(
                    row=row_index + 1, column=offset, padx=6, pady=4, sticky="ew"
                )

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(16, 10))

        ttk.Label(controls, text="Pause after move (s)").grid(row=0, column=0, padx=(0, 8), sticky="w")
        ttk.Entry(controls, textvariable=self.pause_after_move_var, width=8).grid(
            row=0, column=1, padx=(0, 18), sticky="w"
        )
        ttk.Label(controls, text="Pause after report (s)").grid(row=0, column=2, padx=(0, 8), sticky="w")
        ttk.Entry(controls, textvariable=self.pause_after_report_var, width=8).grid(
            row=0, column=3, padx=(0, 18), sticky="w"
        )

        button_row = ttk.Frame(outer)
        button_row.pack(fill="x", pady=(8, 0))

        ttk.Button(button_row, text="Run Simulation", command=self.run_simulation).pack(side="left")
        ttk.Button(button_row, text="Reset Defaults", command=self.reset_defaults).pack(side="left", padx=8)

        status = ttk.Label(outer, textvariable=self.status_var, foreground="#333333")
        status.pack(anchor="w", pady=(16, 0))

    def _load_current_values(self) -> None:
        points, pause_after_move_s, pause_after_report_s = load_demo_config(POINTS_CONFIG_PATH)

        points = points[:4]
        while len(points) < 4:
            default_index = len(points)
            points.append(DEFAULT_DEMO_CONFIG["points"][default_index])

        for row_vars, point in zip(self.point_rows, points):
            row_vars["name"].set(point["name"])
            row_vars["x"].set(str(point["x"]))
            row_vars["y"].set(str(point["y"]))
            row_vars["z"].set(str(point["z"]))
            row_vars["r"].set(str(point["r"]))

        self.pause_after_move_var.set(str(pause_after_move_s))
        self.pause_after_report_var.set(str(pause_after_report_s))

    def _collect_values(self) -> tuple[list[dict], float, float]:
        points = []
        for index, row_vars in enumerate(self.point_rows, start=1):
            name = row_vars["name"].get().strip() or f"P{index}"
            try:
                point = {
                    "name": name,
                    "x": float(row_vars["x"].get().strip()),
                    "y": float(row_vars["y"].get().strip()),
                    "z": float(row_vars["z"].get().strip()),
                    "r": float(row_vars["r"].get().strip()),
                }
            except ValueError as exc:
                raise ValueError(f"{name or f'P{index}'} 的 X/Y/Z/R 需要是数字。") from exc
            points.append(point)

        try:
            pause_after_move_s = float(self.pause_after_move_var.get().strip())
            pause_after_report_s = float(self.pause_after_report_var.get().strip())
        except ValueError as exc:
            raise ValueError("暂停秒数必须是数字。") from exc

        return points, pause_after_move_s, pause_after_report_s

    def reset_defaults(self) -> None:
        for row_vars, default_point in zip(self.point_rows, DEFAULT_DEMO_CONFIG["points"]):
            row_vars["name"].set(default_point["name"])
            row_vars["x"].set(str(default_point["x"]))
            row_vars["y"].set(str(default_point["y"]))
            row_vars["z"].set(str(default_point["z"]))
            row_vars["r"].set(str(default_point["r"]))

        self.pause_after_move_var.set(str(DEFAULT_DEMO_CONFIG["pause_after_move_s"]))
        self.pause_after_report_var.set(str(DEFAULT_DEMO_CONFIG["pause_after_report_s"]))
        self.status_var.set("已恢复默认值，按 Save 或 Run 生效。")

    def run_simulation(self) -> None:
        try:
            points, pause_after_move_s, pause_after_report_s = self._collect_values()
            save_demo_config(POINTS_CONFIG_PATH, pause_after_move_s, pause_after_report_s, points)
        except Exception as exc:
            messagebox.showerror("无法启动", str(exc))
            return

        env = os.environ.copy()
        env["DOBOT_BACKEND"] = "sim"
        env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            subprocess.Popen(
                [sys.executable, str(ARM_MOVE_SCRIPT)],
                cwd=str(ROOT_DIR),
                env=env,
            )
        except Exception as exc:
            messagebox.showerror("无法启动模拟", str(exc))
            return

        self.status_var.set("已保存并启动模拟。改完数字后可以再按一次 Run。")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = ArmSimControlPanel()
    app.run()


if __name__ == "__main__":
    main()
