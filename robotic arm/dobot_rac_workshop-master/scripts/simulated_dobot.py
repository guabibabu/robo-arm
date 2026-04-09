#!/usr/bin/env python3
"""Offline Dobot simulator with a lightweight 3D visualizer."""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


L1 = 85.0
L2 = 135.0
L3 = 147.0
L4 = 59.0

HOME_POSE = (200.0, 0.0, 50.0, 0.0)
SIM_UPDATE_INTERVAL_S = 0.03
SIM_LINEAR_SPEED_MM_S = 180.0
SIM_ROTATION_SPEED_DEG_S = 150.0


@dataclass(frozen=True)
class CartesianPose:
    x: float
    y: float
    z: float
    r: float


@dataclass(frozen=True)
class JointState:
    j1: float
    j2: float
    j3: float
    j5: float


@dataclass(frozen=True)
class LinkGeometry:
    base: tuple[float, float, float]
    shoulder: tuple[float, float, float]
    elbow: tuple[float, float, float]
    wrist: tuple[float, float, float]
    ee: tuple[float, float, float]


class PositionView:
    def __init__(self, pose: CartesianPose):
        self.x = pose.x
        self.y = pose.y
        self.z = pose.z
        self.r = pose.r

    def __repr__(self) -> str:
        return (
            f"Position(x={self.x:.2f}, y={self.y:.2f}, "
            f"z={self.z:.2f}, r={self.r:.2f})"
        )


class JointView:
    def __init__(self, joints: JointState):
        self.j1 = joints.j1
        self.j2 = joints.j2
        self.j3 = joints.j3
        self.j5 = joints.j5

    def __repr__(self) -> str:
        return (
            f"Joint(j1={self.j1:.2f}, j2={self.j2:.2f}, "
            f"j3={self.j3:.2f}, j5={self.j5:.2f})"
        )


class ThetaView:
    def __init__(self, joints: JointState):
        self.theta1 = joints.j1
        self.theta2 = joints.j2
        self.theta3 = joints.j3
        self.theta5 = joints.j5

    def __repr__(self) -> str:
        return (
            f"Theta(theta1={self.theta1:.2f}, theta2={self.theta2:.2f}, "
            f"theta3={self.theta3:.2f}, theta5={self.theta5:.2f})"
        )


class PoseView:
    """Match the attribute shape used by the workshop scripts."""

    def __init__(self, pose: CartesianPose, joints: JointState):
        self.position = PositionView(pose)
        joint_view = JointView(joints)
        self.joint = joint_view
        self.joints = joint_view
        self.angles = joint_view
        self.joint_angle = joint_view
        self.theta = ThetaView(joints)

    def __repr__(self) -> str:
        return f"Pose(position={self.position!r}, joint={self.joint!r})"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _shortest_angle_delta_deg(start: float, end: float) -> float:
    return ((end - start + 180.0) % 360.0) - 180.0


def _interpolate_angle_deg(start: float, end: float, alpha: float) -> float:
    return start + _shortest_angle_delta_deg(start, end) * alpha


def _smoothstep(alpha: float) -> float:
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _distance_mm(a: CartesianPose, b: CartesianPose) -> float:
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))


def _as_tuple(point: tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(point[0]), float(point[1]), float(point[2]))


def compute_link_geometry(joints: JointState) -> LinkGeometry:
    theta1 = math.radians(joints.j1)
    theta2 = math.radians(joints.j2)
    theta3 = math.radians(joints.j3)

    base = (0.0, 0.0, 0.0)
    shoulder = (0.0, 0.0, L1)
    elbow = (
        math.cos(theta1) * L2 * math.cos(theta2),
        math.sin(theta1) * L2 * math.cos(theta2),
        L1 + L2 * math.sin(theta2),
    )
    wrist = (
        elbow[0] + math.cos(theta1) * L3 * math.cos(theta2 + theta3),
        elbow[1] + math.sin(theta1) * L3 * math.cos(theta2 + theta3),
        elbow[2] + L3 * math.sin(theta2 + theta3),
    )
    ee = (
        wrist[0] + math.cos(theta1) * L4,
        wrist[1] + math.sin(theta1) * L4,
        wrist[2],
    )

    return LinkGeometry(
        base=_as_tuple(base),
        shoulder=_as_tuple(shoulder),
        elbow=_as_tuple(elbow),
        wrist=_as_tuple(wrist),
        ee=_as_tuple(ee),
    )


def forward_kinematics(joints: JointState) -> CartesianPose:
    theta1 = math.radians(joints.j1)
    theta2 = math.radians(joints.j2)
    theta3 = math.radians(joints.j3)
    theta5 = math.radians(joints.j5)

    radial = L4 + L3 * math.cos(theta2 + theta3) + L2 * math.cos(theta2)
    x = math.cos(theta1) * radial
    y = math.sin(theta1) * radial
    z = L1 + L3 * math.sin(theta2 + theta3) + L2 * math.sin(theta2)
    r = math.degrees(theta1 + theta5)

    return CartesianPose(x=float(x), y=float(y), z=float(z), r=float(r))


def inverse_kinematics(
    x: float,
    y: float,
    z: float,
    r: float,
    current_joints: Optional[JointState] = None,
) -> JointState:
    radial = math.hypot(x, y)
    wrist_radius = radial - L4
    dz = z - L1

    reach_sq = wrist_radius * wrist_radius + dz * dz
    cos_theta3 = (reach_sq - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
    if cos_theta3 < -1.0 - 1e-6 or cos_theta3 > 1.0 + 1e-6:
        raise ValueError(
            f"Target ({x:.1f}, {y:.1f}, {z:.1f}, {r:.1f}) is outside the simulated workspace."
        )

    cos_theta3 = _clamp(cos_theta3, -1.0, 1.0)
    base_deg = math.degrees(math.atan2(y, x))
    target_yaw_deg = r

    candidates: list[tuple[int, float, JointState]] = []
    for elbow_sign in (1.0, -1.0):
        sin_theta3 = elbow_sign * math.sqrt(max(0.0, 1.0 - cos_theta3 * cos_theta3))
        theta3 = math.atan2(sin_theta3, cos_theta3)
        theta2 = math.atan2(dz, wrist_radius) - math.atan2(
            L3 * math.sin(theta3), L2 + L3 * math.cos(theta3)
        )

        joints = JointState(
            j1=base_deg,
            j2=math.degrees(theta2),
            j3=math.degrees(theta3),
            j5=_shortest_angle_delta_deg(base_deg, target_yaw_deg),
        )

        geometry = compute_link_geometry(joints)
        below_ground = int(min(geometry.elbow[2], geometry.wrist[2], geometry.ee[2]) < -5.0)
        if current_joints is None:
            delta_score = abs(joints.j2) + abs(joints.j3) * 0.5 + abs(joints.j5) * 0.1
        else:
            delta_score = sum(
                abs(_shortest_angle_delta_deg(current, target))
                for current, target in (
                    (current_joints.j1, joints.j1),
                    (current_joints.j2, joints.j2),
                    (current_joints.j3, joints.j3),
                    (current_joints.j5, joints.j5),
                )
            )

        candidates.append((below_ground, delta_score, joints))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


@dataclass(frozen=True)
class SimulationSnapshot:
    pose: CartesianPose
    joints: JointState
    path: tuple[tuple[float, float, float], ...]
    target_pose: Optional[CartesianPose]
    error_target: Optional[CartesianPose]
    gripper_closed: bool
    busy: bool
    status_message: str


class SimulationVisualizer:
    def __init__(self) -> None:
        try:
            import matplotlib.pyplot as pyplot
        except Exception as exc:  # pragma: no cover - exercised only when matplotlib is absent
            raise RuntimeError(
                "matplotlib is required for DOBOT_BACKEND=sim. Install requirements first."
            ) from exc

        self._plt = pyplot
        self._plt.ion()
        self.fig = self._plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.info_text = self.fig.text(
            0.02,
            0.98,
            "",
            va="top",
            family="monospace",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88),
        )
        self._closed = False
        self._last_draw_at = 0.0

        self.fig.canvas.mpl_connect("close_event", self._on_close)
        try:
            self.fig.canvas.manager.set_window_title("Dobot Offline Simulator")
        except Exception:
            pass

        self.fig.tight_layout()
        self._plt.show(block=False)

    def _on_close(self, _event) -> None:
        self._closed = True

    def _draw_gripper(self, ee: tuple[float, float, float], yaw_deg: float, closed: bool) -> None:
        yaw = math.radians(yaw_deg)
        forward = (math.cos(yaw), math.sin(yaw), 0.0)
        lateral = (-math.sin(yaw), math.cos(yaw), 0.0)
        up = (0.0, 0.0, 1.0)

        def point(origin, forward_mm=0.0, lateral_mm=0.0, up_mm=0.0):
            return (
                origin[0] + forward[0] * forward_mm + lateral[0] * lateral_mm + up[0] * up_mm,
                origin[1] + forward[1] * forward_mm + lateral[1] * lateral_mm + up[1] * up_mm,
                origin[2] + forward[2] * forward_mm + lateral[2] * lateral_mm + up[2] * up_mm,
            )

        wrist_mount = point(ee, forward_mm=-10.0, up_mm=2.0)
        palm_center = point(ee, forward_mm=4.0, up_mm=5.0)

        palm_half_width = 12.0
        jaw_gap = 8.0 if closed else 24.0
        outer_reach = 16.0
        inner_reach = 10.0
        tip_drop = -3.0

        left_palm = point(palm_center, lateral_mm=palm_half_width)
        right_palm = point(palm_center, lateral_mm=-palm_half_width)

        self.ax.plot(
            [wrist_mount[0], palm_center[0]],
            [wrist_mount[1], palm_center[1]],
            [wrist_mount[2], palm_center[2]],
            color="slategray",
            linewidth=5.0,
        )
        self.ax.plot(
            [left_palm[0], right_palm[0]],
            [left_palm[1], right_palm[1]],
            [left_palm[2], right_palm[2]],
            color="dimgray",
            linewidth=6.0,
        )

        for direction in (-1.0, 1.0):
            finger_root = point(palm_center, lateral_mm=palm_half_width * direction)
            finger_mid = point(
                finger_root,
                forward_mm=outer_reach,
                lateral_mm=2.0 * direction,
                up_mm=2.0,
            )
            finger_tip = point(
                finger_mid,
                forward_mm=inner_reach,
                lateral_mm=-(palm_half_width - jaw_gap / 2.0 + 2.0) * direction,
                up_mm=tip_drop,
            )
            pad_tip = point(finger_tip, forward_mm=4.0)

            self.ax.plot(
                [finger_root[0], finger_mid[0]],
                [finger_root[1], finger_mid[1]],
                [finger_root[2], finger_mid[2]],
                color="gray",
                linewidth=4.5,
            )
            self.ax.plot(
                [finger_mid[0], finger_tip[0]],
                [finger_mid[1], finger_tip[1]],
                [finger_mid[2], finger_tip[2]],
                color="gray",
                linewidth=4.0,
            )
            self.ax.plot(
                [finger_tip[0], pad_tip[0]],
                [finger_tip[1], pad_tip[1]],
                [finger_tip[2], pad_tip[2]],
                color="firebrick",
                linewidth=3.5,
            )
            self.ax.scatter(finger_mid[0], finger_mid[1], finger_mid[2], s=26, c="black")
            self.ax.scatter(pad_tip[0], pad_tip[1], pad_tip[2], s=22, c="firebrick")

    def render(self, snapshot: SimulationSnapshot, force: bool = False) -> None:
        if self._closed:
            return

        now = time.time()
        if not force and now - self._last_draw_at < (1.0 / 30.0):
            self._pump_events()
            return

        self._last_draw_at = now
        geometry = compute_link_geometry(snapshot.joints)
        path = snapshot.path

        self.ax.clear()

        self.ax.plot(
            [geometry.base[0], geometry.shoulder[0]],
            [geometry.base[1], geometry.shoulder[1]],
            [geometry.base[2], geometry.shoulder[2]],
            color="black",
            linewidth=4.0,
        )
        self.ax.plot(
            [geometry.shoulder[0], geometry.elbow[0]],
            [geometry.shoulder[1], geometry.elbow[1]],
            [geometry.shoulder[2], geometry.elbow[2]],
            color="steelblue",
            linewidth=6.0,
        )
        self.ax.plot(
            [geometry.elbow[0], geometry.wrist[0]],
            [geometry.elbow[1], geometry.wrist[1]],
            [geometry.elbow[2], geometry.wrist[2]],
            color="royalblue",
            linewidth=6.0,
        )
        self.ax.plot(
            [geometry.wrist[0], geometry.ee[0]],
            [geometry.wrist[1], geometry.ee[1]],
            [geometry.wrist[2], geometry.ee[2]],
            color="darkorange",
            linewidth=5.0,
        )

        for point, size, color, label in (
            (geometry.base, 90, "black", "Base"),
            (geometry.shoulder, 70, "gray", "Shoulder"),
            (geometry.elbow, 70, "royalblue", "Elbow"),
            (geometry.wrist, 70, "deepskyblue", "Wrist"),
            (geometry.ee, 110, "orange", "EE"),
        ):
            self.ax.scatter(point[0], point[1], point[2], s=size, c=color)
            self.ax.text(point[0], point[1], point[2], f" {label}", fontsize=9)

        if len(path) > 1:
            xs = [point[0] for point in path]
            ys = [point[1] for point in path]
            zs = [point[2] for point in path]
            self.ax.plot(xs, ys, zs, color="magenta", linewidth=1.8, alpha=0.8)

        if snapshot.target_pose is not None:
            self.ax.scatter(
                snapshot.target_pose.x,
                snapshot.target_pose.y,
                snapshot.target_pose.z,
                c="cyan",
                s=130,
                marker="*",
                edgecolors="black",
                linewidths=1.0,
            )

        if snapshot.error_target is not None:
            self.ax.scatter(
                snapshot.error_target.x,
                snapshot.error_target.y,
                snapshot.error_target.z,
                c="red",
                s=140,
                marker="x",
                linewidths=2.2,
            )

        yaw_rad = math.radians(snapshot.pose.r)
        self.ax.quiver(
            geometry.ee[0],
            geometry.ee[1],
            geometry.ee[2],
            math.cos(yaw_rad) * 45.0,
            math.sin(yaw_rad) * 45.0,
            0.0,
            color="darkred",
            linewidth=2.5,
            arrow_length_ratio=0.16,
        )

        self._draw_gripper(geometry.ee, snapshot.pose.r, snapshot.gripper_closed)

        self.ax.set_title("Dobot Offline 3D Simulator", fontsize=13, weight="bold")
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.set_zlabel("Z (mm)")
        self.ax.set_xlim([-320, 320])
        self.ax.set_ylim([-320, 320])
        self.ax.set_zlim([-40, 320])
        self.ax.set_box_aspect([1, 1, 0.9])
        self.ax.view_init(elev=24.0, azim=-58.0)
        self.ax.grid(True, alpha=0.3)

        info = "\n".join(
            [
                f"status : {snapshot.status_message}",
                f"busy   : {snapshot.busy}",
                f"grip   : {'closed' if snapshot.gripper_closed else 'open'}",
                "",
                (
                    f"pose   : X={snapshot.pose.x:7.1f}  Y={snapshot.pose.y:7.1f}  "
                    f"Z={snapshot.pose.z:7.1f}  R={snapshot.pose.r:7.1f}"
                ),
                (
                    f"joints : J1={snapshot.joints.j1:7.1f}  J2={snapshot.joints.j2:7.1f}  "
                    f"J3={snapshot.joints.j3:7.1f}  J5={snapshot.joints.j5:7.1f}"
                ),
                "",
                "legend : magenta path, cyan target, red x rejected target",
            ]
        )
        self.info_text.set_text(info)

        self.fig.canvas.draw_idle()
        self._pump_events()

    def _pump_events(self) -> None:
        if self._closed:
            return
        try:
            self._plt.pause(0.001)
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._plt.close(self.fig)
        except Exception:
            pass


class SimulatedDobot:
    """Drop-in offline replacement for the subset of pydobotplus used here."""

    def __init__(self, enable_viewer: bool = True, time_scale: Optional[float] = None):
        self._time_scale = self._resolve_time_scale(time_scale)
        self._lock = threading.RLock()
        self._queue: queue.Queue[Optional[tuple[int, CartesianPose]]] = queue.Queue()
        self._stop_event = threading.Event()
        self._known_command_ids: set[int] = set()
        self._completed_command_ids: set[int] = set()
        self._path: deque[tuple[float, float, float]] = deque(maxlen=400)
        self._next_command_id = 1
        self._busy = False
        self._gripper_closed = False
        self._status_message = "Simulation ready."
        self._current_target: Optional[CartesianPose] = None
        self._error_target: Optional[CartesianPose] = None

        starting_joints = inverse_kinematics(*HOME_POSE)
        self._joints = starting_joints
        self._pose = forward_kinematics(starting_joints)
        self._path.append((self._pose.x, self._pose.y, self._pose.z))

        viewer_disabled = os.environ.get("DOBOT_SIM_DISABLE_VIEWER", "").strip() == "1"
        self._viewer = SimulationVisualizer() if enable_viewer and not viewer_disabled else None
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._render(force=True)

    @staticmethod
    def _resolve_time_scale(time_scale: Optional[float]) -> float:
        if time_scale is not None:
            return max(0.0, float(time_scale))

        raw_value = os.environ.get("DOBOT_SIM_TIME_SCALE", "1.0").strip()
        try:
            return max(0.0, float(raw_value))
        except ValueError:
            return 1.0

    def _snapshot(self) -> SimulationSnapshot:
        with self._lock:
            return SimulationSnapshot(
                pose=CartesianPose(self._pose.x, self._pose.y, self._pose.z, self._pose.r),
                joints=JointState(self._joints.j1, self._joints.j2, self._joints.j3, self._joints.j5),
                path=tuple(self._path),
                target_pose=self._current_target,
                error_target=self._error_target,
                gripper_closed=self._gripper_closed,
                busy=self._busy,
                status_message=self._status_message,
            )

    def _render(self, force: bool = False) -> None:
        if self._viewer is None:
            return
        self._viewer.render(self._snapshot(), force=force)

    def _estimate_duration(self, start_pose: CartesianPose, end_pose: CartesianPose) -> float:
        linear_component = _distance_mm(start_pose, end_pose) / SIM_LINEAR_SPEED_MM_S
        rotation_component = abs(_shortest_angle_delta_deg(start_pose.r, end_pose.r)) / SIM_ROTATION_SPEED_DEG_S
        base_duration = max(0.35, linear_component, rotation_component)
        return base_duration * self._time_scale

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is None:
                break

            command_id, target_pose = item
            with self._lock:
                start_joints = self._joints
                start_pose = self._pose
                self._busy = True
                self._current_target = target_pose
                self._error_target = None
                self._status_message = f"Executing simulated move #{command_id}"

            try:
                target_joints = inverse_kinematics(
                    target_pose.x,
                    target_pose.y,
                    target_pose.z,
                    target_pose.r,
                    current_joints=start_joints,
                )
                target_pose = forward_kinematics(target_joints)
                duration_s = self._estimate_duration(start_pose, target_pose)
                steps = max(1, int(duration_s / SIM_UPDATE_INTERVAL_S))
                step_duration = 0.0 if steps <= 1 else duration_s / steps

                for step_index in range(1, steps + 1):
                    if self._stop_event.is_set():
                        return

                    alpha = _smoothstep(step_index / steps)
                    joints = JointState(
                        j1=_interpolate_angle_deg(start_joints.j1, target_joints.j1, alpha),
                        j2=_interpolate_angle_deg(start_joints.j2, target_joints.j2, alpha),
                        j3=_interpolate_angle_deg(start_joints.j3, target_joints.j3, alpha),
                        j5=_interpolate_angle_deg(start_joints.j5, target_joints.j5, alpha),
                    )
                    pose = forward_kinematics(joints)

                    with self._lock:
                        self._joints = joints
                        self._pose = pose
                        self._path.append((pose.x, pose.y, pose.z))

                    if step_duration > 0.0:
                        time.sleep(step_duration)

                with self._lock:
                    self._joints = target_joints
                    self._pose = target_pose
                    self._path.append((target_pose.x, target_pose.y, target_pose.z))
                    self._busy = False
                    self._current_target = None
                    self._completed_command_ids.add(command_id)
                    self._status_message = f"Reached simulated target #{command_id}"

            except Exception as exc:
                with self._lock:
                    self._busy = False
                    self._current_target = None
                    self._completed_command_ids.add(command_id)
                    self._error_target = target_pose
                    self._status_message = f"Simulation error: {exc}"

    def get_pose(self) -> PoseView:
        with self._lock:
            return PoseView(self._pose, self._joints)

    def move_to(self, x: float, y: float, z: float, r: float, wait: bool = True, mode=None) -> int:
        del mode  # Accepted for compatibility with the real Dobot API.

        with self._lock:
            current_joints = self._joints

        target_pose = CartesianPose(float(x), float(y), float(z), float(r))
        try:
            inverse_kinematics(x, y, z, r, current_joints=current_joints)
        except ValueError as exc:
            with self._lock:
                self._error_target = target_pose
                self._status_message = str(exc)
            self._render(force=True)
            raise

        with self._lock:
            command_id = self._next_command_id
            self._next_command_id += 1
            self._known_command_ids.add(command_id)
            self._status_message = f"Queued simulated move #{command_id}"

        self._queue.put((command_id, target_pose))
        if wait:
            self.wait_for_cmd(command_id)
        return command_id

    def wait_for_cmd(self, command_id: int) -> None:
        if command_id not in self._known_command_ids:
            raise ValueError(f"Unknown simulated command id: {command_id}")

        while True:
            with self._lock:
                if command_id in self._completed_command_ids:
                    break
            self._render()
            time.sleep(0.02)

        self._render(force=True)

    def home(self) -> None:
        with self._lock:
            self._status_message = "Homing to the default simulated pose."
        self.move_to(*HOME_POSE, wait=True)

    def grip(self, closed: bool) -> None:
        with self._lock:
            self._gripper_closed = bool(closed)
            self._status_message = (
                "Gripper closed in simulation." if self._gripper_closed else "Gripper opened in simulation."
            )
        self._render(force=True)

    def clear_alarms(self) -> None:
        with self._lock:
            self._status_message = "Simulated alarms cleared."
        self._render(force=True)

    def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            self._render(force=True)
            return

        duration = max(0.0, float(seconds)) * self._time_scale
        end_time = time.time() + duration
        while time.time() < end_time:
            self._render()
            time.sleep(0.02)
        self._render(force=True)

    def close(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._render(force=True)
        if self._viewer is not None:
            self._viewer.close()
