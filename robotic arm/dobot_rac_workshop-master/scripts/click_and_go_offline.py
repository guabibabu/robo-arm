#!/usr/bin/env python3
"""Offline Click-and-Go simulator with a synthetic camera view."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from click_and_go_shared import get_robot_arm_matrix, get_tag_to_camera_matrix, invert_transform, load_app_config, transform_point
from dobot_backend import create_dobot
from simulated_dobot import compute_link_geometry


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class SyntheticTag:
    tag_id: int
    pose_R: np.ndarray
    pose_t: np.ndarray
    corners: np.ndarray
    center: np.ndarray


@dataclass(frozen=True)
class BoxTarget:
    name: str
    center: np.ndarray
    size_mm: np.ndarray
    color: str


BOX_TARGETS = [
    BoxTarget(name="Box A", center=np.array([180.0, -70.0, 0.0]), size_mm=np.array([34.0, 34.0, 28.0]), color="#ff7b72"),
    BoxTarget(name="Box B", center=np.array([220.0, 35.0, 0.0]), size_mm=np.array([34.0, 34.0, 36.0]), color="#4caf50"),
    BoxTarget(name="Box C", center=np.array([285.0, -20.0, 0.0]), size_mm=np.array([34.0, 34.0, 24.0]), color="#5aa9ff"),
]

OFFLINE_CAMERA_POSITION_MM = np.array([220.0, -80.0, 620.0], dtype=float)
OFFLINE_CAMERA_TARGET_MM = np.array([220.0, 0.0, 10.0], dtype=float)
OFFLINE_CLICK_AND_GO_TIME_SCALE = 3.0


def _normalize(vector):
    array = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(array)
    if norm <= 1e-9:
        raise ValueError("Cannot normalize a zero-length vector.")
    return array / norm


def make_look_at_transform(camera_position_mm, target_point_mm, up_guess=(0.0, 0.0, -1.0)):
    camera_position = np.asarray(camera_position_mm, dtype=float)
    target_point = np.asarray(target_point_mm, dtype=float)

    z_axis = _normalize(target_point - camera_position)
    up_axis = np.asarray(up_guess, dtype=float)
    x_axis = _normalize(np.cross(z_axis, up_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))

    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    transform[:3, 3] = camera_position
    return transform


def project_base_point_to_pixel(point_base_mm, base_T_camera, intrinsics):
    camera_T_base = invert_transform(base_T_camera)
    camera_point = transform_point(camera_T_base, point_base_mm)
    if camera_point[2] <= 1e-6:
        return None

    pixel_x = intrinsics.fx * camera_point[0] / camera_point[2] + intrinsics.cx
    pixel_y = intrinsics.fy * camera_point[1] / camera_point[2] + intrinsics.cy
    return np.array([pixel_x, pixel_y], dtype=float)


def intersect_pixel_with_plane(pixel_x, pixel_y, base_T_camera, intrinsics, plane_z_mm):
    ray_camera = np.array(
        [
            (float(pixel_x) - intrinsics.cx) / intrinsics.fx,
            (float(pixel_y) - intrinsics.cy) / intrinsics.fy,
            1.0,
        ],
        dtype=float,
    )
    ray_base = base_T_camera[:3, :3] @ _normalize(ray_camera)
    origin_base = base_T_camera[:3, 3]

    if abs(ray_base[2]) <= 1e-9:
        return None

    travel = (float(plane_z_mm) - origin_base[2]) / ray_base[2]
    if travel <= 0.0:
        return None

    return origin_base + ray_base * travel


def build_synthetic_tag(base_T_camera, base_T_gripper, gripper_T_tag, tag_size_mm, intrinsics):
    base_T_tag = base_T_gripper @ gripper_T_tag
    camera_T_tag = invert_transform(base_T_camera) @ base_T_tag

    half_size = float(tag_size_mm) / 2.0
    tag_corners = [
        (-half_size, -half_size, 0.0),
        (half_size, -half_size, 0.0),
        (half_size, half_size, 0.0),
        (-half_size, half_size, 0.0),
    ]

    pixel_corners = []
    for corner in tag_corners:
        base_corner = transform_point(base_T_tag, corner)
        pixel = project_base_point_to_pixel(base_corner, base_T_camera, intrinsics)
        if pixel is None:
            return None
        pixel_corners.append(pixel)

    pixel_corners = np.asarray(pixel_corners, dtype=float)
    if (
        np.all(pixel_corners[:, 0] < -20.0)
        or np.all(pixel_corners[:, 0] > intrinsics.width + 20.0)
        or np.all(pixel_corners[:, 1] < -20.0)
        or np.all(pixel_corners[:, 1] > intrinsics.height + 20.0)
    ):
        return None

    return SyntheticTag(
        tag_id=0,
        pose_R=camera_T_tag[:3, :3],
        pose_t=(camera_T_tag[:3, 3] / 1000.0).reshape(3, 1),
        corners=pixel_corners,
        center=np.mean(pixel_corners, axis=0),
    )


def make_box_faces(box):
    half_x = float(box.size_mm[0]) / 2.0
    half_y = float(box.size_mm[1]) / 2.0
    height = float(box.size_mm[2])
    cx, cy, cz = box.center

    bottom = np.array(
        [
            [cx - half_x, cy - half_y, cz],
            [cx + half_x, cy - half_y, cz],
            [cx + half_x, cy + half_y, cz],
            [cx - half_x, cy + half_y, cz],
        ],
        dtype=float,
    )
    top = bottom.copy()
    top[:, 2] = cz + height

    return {
        "top": top,
        "front": np.array([bottom[0], bottom[1], top[1], top[0]], dtype=float),
        "right": np.array([bottom[1], bottom[2], top[2], top[1]], dtype=float),
        "left": np.array([bottom[0], bottom[3], top[3], top[0]], dtype=float),
        "bottom": bottom,
        "center_top": np.array([cx, cy, cz + height], dtype=float),
    }


class OfflineClickAndGoDemo:
    def __init__(self):
        self.app_config = load_app_config()
        self.camera_cfg = self.app_config["camera"]
        self.motion_cfg = self.app_config["motion"]
        self.workspace_cfg = self.app_config["workspace"]
        self.calibration_cfg = self.app_config["calibration"]

        os.environ.setdefault("DOBOT_SIM_DISABLE_VIEWER", "1")
        os.environ.setdefault("DOBOT_SIM_TIME_SCALE", str(OFFLINE_CLICK_AND_GO_TIME_SCALE))
        try:
            self.sim_time_scale = max(0.0, float(os.environ.get("DOBOT_SIM_TIME_SCALE", str(OFFLINE_CLICK_AND_GO_TIME_SCALE))))
        except ValueError:
            self.sim_time_scale = OFFLINE_CLICK_AND_GO_TIME_SCALE
        self.device = create_dobot()
        self.gripper_T_tag = np.asarray(self.calibration_cfg["gripper_T_tag"], dtype=float)
        self.base_T_camera = None
        self.synthetic_base_T_camera = make_look_at_transform(
            camera_position_mm=OFFLINE_CAMERA_POSITION_MM,
            target_point_mm=OFFLINE_CAMERA_TARGET_MM,
        )
        self.intrinsics = CameraIntrinsics(
            width=int(self.camera_cfg["width"]),
            height=int(self.camera_cfg["height"]),
            fx=540.0,
            fy=540.0,
            cx=float(self.camera_cfg["width"]) / 2.0,
            cy=float(self.camera_cfg["height"]) / 2.0,
        )
        self.target_surface_z_mm = 0.0

        self.device_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.pending_click = None
        self.latest_click = None
        self.latest_camera_point = None
        self.latest_base_point = None
        self.latest_command_point = None
        self.latest_target_name = None
        self.latest_tag = None
        self.motion_thread = None
        self.busy = False
        self.status_message = (
            f"Offline mode ready. Motion is slowed to {self.sim_time_scale:.1f}x for visibility. "
            "Press 'k' to simulate calibration."
        )
        self.closed = False

        self.fig = plt.figure(figsize=(16.2, 6.8))
        grid = self.fig.add_gridspec(1, 3, width_ratios=[4.0, 3.8, 1.9])
        self.ax_camera = self.fig.add_subplot(grid[0, 0])
        self.ax_robot = self.fig.add_subplot(grid[0, 1], projection="3d")
        self.ax_info = self.fig.add_subplot(grid[0, 2])
        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        try:
            self.fig.canvas.manager.set_window_title("Offline Click-and-Go Camera View")
        except Exception:
            pass
        self.fig.tight_layout()

    def on_close(self, _event):
        self.closed = True

    def set_status(self, message):
        with self.state_lock:
            self.status_message = message

    def set_busy(self, busy):
        with self.state_lock:
            self.busy = busy

    def is_busy(self):
        with self.state_lock:
            return self.busy

    def on_mouse_click(self, event):
        if event.inaxes != self.ax_camera or event.xdata is None or event.ydata is None:
            return

        with self.state_lock:
            self.pending_click = (float(event.xdata), float(event.ydata))
            self.latest_click = (float(event.xdata), float(event.ydata))

    def on_key_press(self, event):
        if event.key == "q":
            self.closed = True
            plt.close(self.fig)
        elif event.key == "c":
            self.latest_click = None
            self.latest_camera_point = None
            self.latest_base_point = None
            self.latest_command_point = None
            self.latest_target_name = None
            self.set_status("Cleared target marker.")
        elif event.key == "h":
            self.move_to_safe_pose()
        elif event.key == "k":
            if self.latest_tag is None:
                self.set_status("No synthetic AprilTag visible right now. Try again after the arm settles.")
            elif self.is_busy():
                self.set_status("Wait for the robot to stop before recalibrating.")
            else:
                self.calibrate_from_tag(self.latest_tag)

    def point_in_workspace(self, point_mm):
        x, y, z = point_mm
        cfg = self.workspace_cfg
        radius = float(np.hypot(x, y))

        if not (cfg["x_min_mm"] <= x <= cfg["x_max_mm"]):
            return False, f"X out of range: {x:.1f} mm"
        if not (cfg["y_min_mm"] <= y <= cfg["y_max_mm"]):
            return False, f"Y out of range: {y:.1f} mm"
        if not (cfg["z_min_mm"] <= z <= cfg["z_max_mm"]):
            return False, f"Z out of range: {z:.1f} mm"
        if radius > cfg["max_radius_mm"]:
            return False, f"Radius out of range: {radius:.1f} mm"
        return True, ""

    def calibrate_from_tag(self, tag):
        with self.device_lock:
            pose = self.device.get_pose()

        base_T_gripper = get_robot_arm_matrix(pose)
        base_T_tag = base_T_gripper @ self.gripper_T_tag
        camera_T_tag = get_tag_to_camera_matrix(tag)
        tag_T_camera = invert_transform(camera_T_tag)
        self.base_T_camera = base_T_tag @ tag_T_camera

        error_mm = float(np.linalg.norm(self.base_T_camera[:3, 3] - self.synthetic_base_T_camera[:3, 3]))
        self.set_status(f"Calibration complete. Camera pose error ≈ {error_mm:.1f} mm.")
        print("=" * 70)
        print("Offline calibration complete: base_T_camera")
        print(np.array2string(self.base_T_camera, precision=3, suppress_small=True))
        print("=" * 70)

    def queue_move(self, target_point_mm, reason):
        if self.is_busy():
            self.set_status("Robot is already moving. Wait for the current move to finish.")
            return

        target_r = float(self.motion_cfg["target_r_deg"])
        self.set_busy(True)
        self.motion_thread = threading.Thread(
            target=self.execute_move_thread,
            args=(np.asarray(target_point_mm, dtype=float), target_r, reason),
            daemon=True,
        )
        self.motion_thread.start()

    def execute_move_thread(self, target_point_mm, target_r_deg, reason):
        self.set_status(reason)

        try:
            with self.device_lock:
                current_pose = self.device.get_pose()

            current_x = float(current_pose.position.x)
            current_y = float(current_pose.position.y)
            current_z = float(current_pose.position.z)

            hover_z = max(
                float(self.motion_cfg["hover_z_mm"]),
                current_z,
                float(target_point_mm[2]) + float(self.motion_cfg["approach_offset_z_mm"]),
            )

            waypoints = [
                (current_x, current_y, hover_z, target_r_deg),
                (float(target_point_mm[0]), float(target_point_mm[1]), hover_z, target_r_deg),
                (float(target_point_mm[0]), float(target_point_mm[1]), float(target_point_mm[2]), target_r_deg),
            ]

            print("=" * 70)
            print(f"Offline click-and-go move: {reason}")
            for index, waypoint in enumerate(waypoints, start=1):
                print(
                    f"Waypoint {index}: X={waypoint[0]:.1f}, Y={waypoint[1]:.1f}, "
                    f"Z={waypoint[2]:.1f}, R={waypoint[3]:.1f}"
                )
                with self.device_lock:
                    self.device.move_to(*waypoint, wait=True)
            print("=" * 70)
            self.set_status("Move complete. Click another point or press 'k' to recalibrate.")
        except Exception as exc:
            self.set_status(f"Move failed: {exc}")
            print(f"Move failed: {exc}")
        finally:
            self.set_busy(False)

    def handle_click(self):
        with self.state_lock:
            if self.pending_click is None:
                return
            click = self.pending_click
            self.pending_click = None

        if self.base_T_camera is None:
            self.set_status("No calibration yet. Press 'k' to simulate calibration first.")
            return

        if self.is_busy():
            self.set_status("Robot is busy. Click again after the move finishes.")
            return

        pixel_x, pixel_y = click
        raw_base_point_gt, hit_name = self._resolve_click_target(pixel_x, pixel_y)
        if raw_base_point_gt is None:
            self.set_status("This click does not hit the synthetic workspace plane.")
            return

        camera_T_base_truth = invert_transform(self.synthetic_base_T_camera)
        camera_point_mm = transform_point(camera_T_base_truth, raw_base_point_gt)
        raw_base_point_mm = transform_point(self.base_T_camera, camera_point_mm)
        command_point_mm = np.asarray(raw_base_point_mm, dtype=float).copy()
        command_point_mm[2] += float(self.motion_cfg["target_offset_z_mm"])

        valid, reason = self.point_in_workspace(command_point_mm)
        self.latest_camera_point = camera_point_mm
        self.latest_base_point = raw_base_point_mm
        self.latest_command_point = command_point_mm
        self.latest_target_name = hit_name

        if not valid:
            self.set_status(f"Rejected target: {reason}")
            return

        target_label = f"target {hit_name}" if hit_name else "clicked target"
        reason = (
            f"Moving to {target_label}: pixel=({int(pixel_x)}, {int(pixel_y)}) "
            f"base=({command_point_mm[0]:.1f}, {command_point_mm[1]:.1f}, {command_point_mm[2]:.1f})"
        )
        self.queue_move(command_point_mm, reason)

    def move_to_safe_pose(self):
        safe = self.motion_cfg["safe_pose"]
        target = np.array([safe["x"], safe["y"], safe["z"]], dtype=float)
        self.latest_command_point = target
        if self.is_busy():
            self.set_status("Robot is already moving. Wait for the current move to finish.")
            return

        self.set_busy(True)
        self.motion_thread = threading.Thread(
            target=self.execute_move_thread,
            args=(target, float(safe["r"]), "Moving to safe pose."),
            daemon=True,
        )
        self.motion_thread.start()

    def _project_pose(self, pose_xyz):
        return project_base_point_to_pixel(pose_xyz, self.synthetic_base_T_camera, self.intrinsics)

    def _project_polygon(self, points_xyz):
        projected = []
        for point in points_xyz:
            pixel = self._project_pose(point)
            if pixel is None:
                return None
            projected.append(pixel)
        return np.asarray(projected, dtype=float)

    def _resolve_click_target(self, pixel_x, pixel_y):
        click_point = (float(pixel_x), float(pixel_y))
        for box in sorted(BOX_TARGETS, key=lambda item: item.center[1], reverse=True):
            faces = make_box_faces(box)
            polygon = self._project_polygon(faces["top"])
            if polygon is None:
                continue
            if MplPath(polygon).contains_point(click_point):
                return faces["center_top"], box.name

        plane_hit = intersect_pixel_with_plane(
            pixel_x,
            pixel_y,
            self.synthetic_base_T_camera,
            self.intrinsics,
            self.target_surface_z_mm,
        )
        return plane_hit, None

    def _draw_workspace(self):
        corners = [
            (self.workspace_cfg["x_min_mm"], self.workspace_cfg["y_min_mm"], self.target_surface_z_mm),
            (self.workspace_cfg["x_min_mm"], self.workspace_cfg["y_max_mm"], self.target_surface_z_mm),
            (self.workspace_cfg["x_max_mm"], self.workspace_cfg["y_max_mm"], self.target_surface_z_mm),
            (self.workspace_cfg["x_max_mm"], self.workspace_cfg["y_min_mm"], self.target_surface_z_mm),
        ]
        projected = [self._project_pose(corner) for corner in corners]
        if any(point is None for point in projected):
            return

        polygon = np.asarray(projected, dtype=float)
        self.ax_camera.fill(polygon[:, 0], polygon[:, 1], color="#dcecf9", alpha=0.9)
        self.ax_camera.plot(
            np.append(polygon[:, 0], polygon[0, 0]),
            np.append(polygon[:, 1], polygon[0, 1]),
            color="#2a5c88",
            linewidth=2.0,
        )

        for x in np.arange(self.workspace_cfg["x_min_mm"], self.workspace_cfg["x_max_mm"] + 1, 40.0):
            endpoints = [
                self._project_pose((x, self.workspace_cfg["y_min_mm"], self.target_surface_z_mm)),
                self._project_pose((x, self.workspace_cfg["y_max_mm"], self.target_surface_z_mm)),
            ]
            if all(point is not None for point in endpoints):
                self.ax_camera.plot(
                    [endpoints[0][0], endpoints[1][0]],
                    [endpoints[0][1], endpoints[1][1]],
                    color="#9bb8d1",
                    linewidth=0.8,
                )

    def _draw_box_targets_camera(self):
        camera_T_base = invert_transform(self.synthetic_base_T_camera)
        draw_order = []
        for box in BOX_TARGETS:
            faces = make_box_faces(box)
            center_camera = transform_point(camera_T_base, faces["center_top"])
            draw_order.append((center_camera[2], box, faces))

        for _, box, faces in sorted(draw_order, key=lambda item: item[0], reverse=True):
            top = self._project_polygon(faces["top"])
            front = self._project_polygon(faces["front"])
            right = self._project_polygon(faces["right"])
            if top is None:
                continue

            if front is not None:
                self.ax_camera.fill(front[:, 0], front[:, 1], color=box.color, alpha=0.35, linewidth=0)
            if right is not None:
                self.ax_camera.fill(right[:, 0], right[:, 1], color=box.color, alpha=0.22, linewidth=0)

            self.ax_camera.fill(top[:, 0], top[:, 1], color=box.color, alpha=0.9, linewidth=0)
            self.ax_camera.plot(
                np.append(top[:, 0], top[0, 0]),
                np.append(top[:, 1], top[0, 1]),
                color="#303030",
                linewidth=1.2,
            )
            center_pixel = np.mean(top, axis=0)
            self.ax_camera.text(
                center_pixel[0],
                center_pixel[1],
                box.name,
                color="white",
                fontsize=9,
                weight="bold",
                ha="center",
                va="center",
            )

        for y in np.arange(self.workspace_cfg["y_min_mm"], self.workspace_cfg["y_max_mm"] + 1, 40.0):
            endpoints = [
                self._project_pose((self.workspace_cfg["x_min_mm"], y, self.target_surface_z_mm)),
                self._project_pose((self.workspace_cfg["x_max_mm"], y, self.target_surface_z_mm)),
            ]
            if all(point is not None for point in endpoints):
                self.ax_camera.plot(
                    [endpoints[0][0], endpoints[1][0]],
                    [endpoints[0][1], endpoints[1][1]],
                    color="#9bb8d1",
                    linewidth=0.8,
                )

    def _draw_tag(self):
        with self.device_lock:
            pose = self.device.get_pose()
        base_T_gripper = get_robot_arm_matrix(pose)
        self.latest_tag = build_synthetic_tag(
            self.synthetic_base_T_camera,
            base_T_gripper,
            self.gripper_T_tag,
            float(self.camera_cfg["tag_size_m"]) * 1000.0,
            self.intrinsics,
        )
        if self.latest_tag is None:
            return

        corners = np.vstack([self.latest_tag.corners, self.latest_tag.corners[0]])
        self.ax_camera.fill(corners[:-1, 0], corners[:-1, 1], color="#85d684", alpha=0.75)
        self.ax_camera.plot(corners[:, 0], corners[:, 1], color="#207520", linewidth=2.2)
        self.ax_camera.scatter(self.latest_tag.center[0], self.latest_tag.center[1], c="#ffdd55", s=40, zorder=5)
        self.ax_camera.text(
            self.latest_tag.center[0] + 10.0,
            self.latest_tag.center[1] - 10.0,
            "Tag",
            color="#1d5f1d",
            fontsize=10,
            weight="bold",
        )

    def _draw_markers(self):
        with self.device_lock:
            ee_pose = self.device.get_pose()
        ee_pixel = self._project_pose((ee_pose.position.x, ee_pose.position.y, ee_pose.position.z))
        if ee_pixel is not None:
            self.ax_camera.scatter(ee_pixel[0], ee_pixel[1], c="#f28500", s=60, zorder=6)
            self.ax_camera.text(ee_pixel[0] + 8.0, ee_pixel[1], "Gripper", color="#b85f00", fontsize=9)

        if self.latest_click is not None:
            self.ax_camera.scatter(self.latest_click[0], self.latest_click[1], c="#00d9ff", s=90, marker="x")

        if self.latest_base_point is not None:
            base_pixel = self._project_pose(self.latest_base_point)
            if base_pixel is not None:
                self.ax_camera.scatter(base_pixel[0], base_pixel[1], c="#ff2e63", s=40, zorder=6)
                self.ax_camera.text(base_pixel[0] + 8.0, base_pixel[1] + 10.0, "Raw", color="#b5103a", fontsize=9)

        if self.latest_command_point is not None:
            command_pixel = self._project_pose(self.latest_command_point)
            if command_pixel is not None:
                self.ax_camera.scatter(command_pixel[0], command_pixel[1], c="#8739ff", s=40, zorder=6)
                self.ax_camera.text(
                    command_pixel[0] + 8.0,
                    command_pixel[1] - 10.0,
                    self.latest_target_name or "Cmd",
                    color="#5e21b0",
                    fontsize=9,
                )

    def _draw_info(self):
        self.ax_info.clear()
        self.ax_info.axis("off")
        with self.device_lock:
            current_pose = self.device.get_pose()
        lines = [
            "Offline Click-and-Go",
            "",
            "Controls:",
            "click: move to target",
            "k: simulate calibration",
            "h: move to safe pose",
            "c: clear markers",
            "q: quit",
            "",
            f"Calibration: {'READY' if self.base_T_camera is not None else 'NOT READY'}",
            f"Busy: {self.is_busy()}",
            f"Motion scale: {self.sim_time_scale:.1f}x",
            "",
            (
                "Gripper pose (mm):\n"
                f"{current_pose.position.x:.1f}, {current_pose.position.y:.1f}, "
                f"{current_pose.position.z:.1f}, R={current_pose.position.r:.1f}"
            ),
            "",
            f"Hit target: {self.latest_target_name or 'table'}",
            "",
            f"Status: {self.status_message}",
        ]

        if self.latest_camera_point is not None:
            lines += [
                "",
                "Camera point (mm):",
                f"{self.latest_camera_point[0]:.1f}, {self.latest_camera_point[1]:.1f}, {self.latest_camera_point[2]:.1f}",
            ]
        if self.latest_base_point is not None:
            lines += [
                "Raw base point (mm):",
                f"{self.latest_base_point[0]:.1f}, {self.latest_base_point[1]:.1f}, {self.latest_base_point[2]:.1f}",
            ]
        if self.latest_command_point is not None:
            lines += [
                "Command point (mm):",
                f"{self.latest_command_point[0]:.1f}, {self.latest_command_point[1]:.1f}, {self.latest_command_point[2]:.1f}",
            ]

        self.ax_info.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)

    def _draw_robot_gripper(self, ee_point, yaw_deg):
        yaw = np.deg2rad(yaw_deg)
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=float)
        up = np.array([0.0, 0.0, 1.0], dtype=float)

        def point(origin, forward_mm=0.0, lateral_mm=0.0, up_mm=0.0):
            return (
                np.asarray(origin, dtype=float)
                + forward * forward_mm
                + lateral * lateral_mm
                + up * up_mm
            )

        wrist_mount = point(ee_point, forward_mm=-10.0, up_mm=2.0)
        palm_center = point(ee_point, forward_mm=4.0, up_mm=5.0)
        left_palm = point(palm_center, lateral_mm=12.0)
        right_palm = point(palm_center, lateral_mm=-12.0)

        self.ax_robot.plot(
            [wrist_mount[0], palm_center[0]],
            [wrist_mount[1], palm_center[1]],
            [wrist_mount[2], palm_center[2]],
            color="slategray",
            linewidth=4.5,
        )
        self.ax_robot.plot(
            [left_palm[0], right_palm[0]],
            [left_palm[1], right_palm[1]],
            [left_palm[2], right_palm[2]],
            color="dimgray",
            linewidth=5.0,
        )

        for direction in (-1.0, 1.0):
            finger_root = point(palm_center, lateral_mm=12.0 * direction)
            finger_mid = point(finger_root, forward_mm=16.0, lateral_mm=2.0 * direction, up_mm=2.0)
            finger_tip = point(finger_mid, forward_mm=12.0, lateral_mm=-8.0 * direction, up_mm=-3.0)

            self.ax_robot.plot(
                [finger_root[0], finger_mid[0]],
                [finger_root[1], finger_mid[1]],
                [finger_root[2], finger_mid[2]],
                color="gray",
                linewidth=3.6,
            )
            self.ax_robot.plot(
                [finger_mid[0], finger_tip[0]],
                [finger_mid[1], finger_tip[1]],
                [finger_mid[2], finger_tip[2]],
                color="firebrick",
                linewidth=3.2,
            )

    def _draw_robot_view(self):
        self.ax_robot.clear()
        with self.device_lock:
            pose = self.device.get_pose()

        joints = SimpleNamespace(
            j1=float(pose.joint.j1),
            j2=float(pose.joint.j2),
            j3=float(pose.joint.j3),
            j5=float(pose.joint.j5),
        )
        geometry = compute_link_geometry(joints)
        ee_point = np.array(geometry.ee, dtype=float)

        self.ax_robot.plot(
            [geometry.base[0], geometry.shoulder[0]],
            [geometry.base[1], geometry.shoulder[1]],
            [geometry.base[2], geometry.shoulder[2]],
            color="black",
            linewidth=4.0,
        )
        self.ax_robot.plot(
            [geometry.shoulder[0], geometry.elbow[0]],
            [geometry.shoulder[1], geometry.elbow[1]],
            [geometry.shoulder[2], geometry.elbow[2]],
            color="steelblue",
            linewidth=6.0,
        )
        self.ax_robot.plot(
            [geometry.elbow[0], geometry.wrist[0]],
            [geometry.elbow[1], geometry.wrist[1]],
            [geometry.elbow[2], geometry.wrist[2]],
            color="royalblue",
            linewidth=6.0,
        )
        self.ax_robot.plot(
            [geometry.wrist[0], geometry.ee[0]],
            [geometry.wrist[1], geometry.ee[1]],
            [geometry.wrist[2], geometry.ee[2]],
            color="darkorange",
            linewidth=5.0,
        )

        for point, size, color, label in (
            (geometry.base, 60, "black", "Base"),
            (geometry.shoulder, 45, "gray", "Shoulder"),
            (geometry.elbow, 45, "royalblue", "Elbow"),
            (geometry.wrist, 45, "deepskyblue", "Wrist"),
            (geometry.ee, 70, "orange", "Gripper"),
        ):
            self.ax_robot.scatter(point[0], point[1], point[2], s=size, c=color)
            self.ax_robot.text(point[0], point[1], point[2], f" {label}", fontsize=8)

        for box in BOX_TARGETS:
            faces = make_box_faces(box)
            poly = Poly3DCollection(
                [faces["top"], faces["front"], faces["right"], faces["left"]],
                facecolors=[box.color, box.color, box.color, box.color],
                alpha=0.55,
                edgecolors="#333333",
                linewidths=0.8,
            )
            self.ax_robot.add_collection3d(poly)
            center_top = faces["center_top"]
            self.ax_robot.text(center_top[0], center_top[1], center_top[2] + 6.0, box.name, fontsize=8, ha="center")

        if self.latest_command_point is not None:
            target = np.asarray(self.latest_command_point, dtype=float)
            self.ax_robot.scatter(target[0], target[1], target[2], c="purple", s=90, marker="*", zorder=6)
            self.ax_robot.plot(
                [ee_point[0], target[0]],
                [ee_point[1], target[1]],
                [ee_point[2], target[2]],
                color="purple",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )

        yaw_rad = np.deg2rad(float(pose.position.r))
        self.ax_robot.quiver(
            ee_point[0],
            ee_point[1],
            ee_point[2],
            np.cos(yaw_rad) * 45.0,
            np.sin(yaw_rad) * 45.0,
            0.0,
            color="darkred",
            linewidth=2.0,
            arrow_length_ratio=0.15,
        )
        self._draw_robot_gripper(ee_point, float(pose.position.r))

        self.ax_robot.set_title("Third-Person Robot View", fontsize=12, weight="bold")
        self.ax_robot.set_xlabel("X (mm)")
        self.ax_robot.set_ylabel("Y (mm)")
        self.ax_robot.set_zlabel("Z (mm)")
        self.ax_robot.set_xlim([-320, 320])
        self.ax_robot.set_ylim([-320, 320])
        self.ax_robot.set_zlim([-40, 320])
        self.ax_robot.set_box_aspect([1, 1, 0.9])
        self.ax_robot.view_init(elev=24.0, azim=-58.0)
        self.ax_robot.grid(True, alpha=0.25)

    def render(self):
        self.ax_camera.clear()
        self.ax_camera.set_title("Synthetic RealSense View", fontsize=12, weight="bold")
        self.ax_camera.set_xlim(0, self.intrinsics.width)
        self.ax_camera.set_ylim(self.intrinsics.height, 0)
        self.ax_camera.set_aspect("equal")
        self.ax_camera.set_facecolor("#f5f6f8")
        self.ax_camera.set_xlabel("Pixel X")
        self.ax_camera.set_ylabel("Pixel Y")

        self._draw_workspace()
        self._draw_box_targets_camera()
        self._draw_tag()
        self._draw_markers()
        self._draw_robot_view()
        self._draw_info()
        self.fig.canvas.draw_idle()

    def run(self):
        print("=" * 70)
        print("Offline Click-and-Go controls:")
        print("  - Press 'k' to simulate AprilTag calibration")
        print("  - Click inside the synthetic camera view to move the arm")
        print("  - Press 'h' to move to the configured safe pose")
        print("  - Press 'q' to quit")
        print("=" * 70)

        while not self.closed:
            self.handle_click()
            self.render()
            plt.pause(0.03 if self.is_busy() else 0.05)

        if self.motion_thread is not None and self.motion_thread.is_alive():
            self.motion_thread.join(timeout=5.0)
        with self.device_lock:
            self.device.close()


def main():
    demo = OfflineClickAndGoDemo()
    demo.run()


if __name__ == "__main__":
    main()
