#!/usr/bin/env python3
"""
Click-and-Go demo for Dobot + RealSense.

Workflow:
1. Press `k` while the AprilTag mounted on the gripper is visible.
2. The script computes base_T_camera from the live tag detection.
3. Click on the camera view.
4. The clicked pixel is converted to a 3D point in the camera frame.
5. The point is transformed into the robot base frame and the arm moves there.
"""

from __future__ import annotations

import threading
from pathlib import Path
import sys

import numpy as np

_MISSING_DEPENDENCIES = []

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
    _MISSING_DEPENDENCIES.append("opencv-python")

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    rs = None
    _MISSING_DEPENDENCIES.append("pyrealsense2")

try:
    from pupil_apriltags import Detector
except ModuleNotFoundError:
    Detector = None
    _MISSING_DEPENDENCIES.append("pupil-apriltags")

try:
    from pydobotplus import Dobot
except ModuleNotFoundError:
    Dobot = None
    _MISSING_DEPENDENCIES.append("pydobotplus")

CURRENT_DIR = Path(__file__).resolve().parent
for candidate in (CURRENT_DIR, CURRENT_DIR.parent):
    if (candidate / "scripts").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

from scripts.click_and_go_shared import (
    APP_CONFIG_PATH,
    get_robot_arm_matrix,
    get_tag_to_camera_matrix,
    is_dobot_target_reachable,
    invert_transform,
    load_app_config,
    load_device_config,
    save_yaml,
    transform_point,
)


def ensure_runtime_dependencies():
    if not _MISSING_DEPENDENCIES:
        return

    message_lines = [
        "Real Click-and-Go is missing required Python packages:",
        "  - " + "\n  - ".join(sorted(_MISSING_DEPENDENCIES)),
        "",
        "Install them in your active venv before running this script.",
    ]
    if "pyrealsense2" in _MISSING_DEPENDENCIES:
        message_lines += [
            "",
            "macOS note: `pip install pyrealsense2` often has no wheel on Mac.",
            "If you are on macOS, either install `pyrealsense2-macosx` or build the Intel RealSense Python wrapper from source.",
        ]
    raise RuntimeError("\n".join(message_lines))


def validate_command_point(point_mm, workspace_cfg, target_r_deg):
    x, y, z = (float(point_mm[0]), float(point_mm[1]), float(point_mm[2]))
    radius = float(np.hypot(x, y))

    if z < max(0.0, float(workspace_cfg["z_min_mm"])):
        return False, f"Z below the tabletop safety floor: {z:.1f} mm"
    if x < float(workspace_cfg["x_min_mm"]) or x > float(workspace_cfg["x_max_mm"]):
        return False, f"X out of range: {x:.1f} mm"
    if y < float(workspace_cfg["y_min_mm"]) or y > float(workspace_cfg["y_max_mm"]):
        return False, f"Y out of range: {y:.1f} mm"
    if z > float(workspace_cfg["z_max_mm"]):
        return False, f"Z out of range: {z:.1f} mm"
    if radius > float(workspace_cfg["max_radius_mm"]):
        return False, f"Radius out of range: {radius:.1f} mm"

    reachable, reason = is_dobot_target_reachable(x, y, z, float(target_r_deg))
    if not reachable:
        return False, f"Unreachable target: {reason}"
    return True, ""


def initialize_pipeline(camera_cfg, serial_number=None):
    pipeline = rs.pipeline()
    config = rs.config()

    if serial_number:
        config.enable_device(serial_number)

    config.enable_stream(
        rs.stream.color,
        camera_cfg["width"],
        camera_cfg["height"],
        rs.format.bgr8,
        camera_cfg["fps"],
    )
    config.enable_stream(
        rs.stream.depth,
        camera_cfg["width"],
        camera_cfg["height"],
        rs.format.z16,
        camera_cfg["fps"],
    )

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not start the RealSense pipeline. Make sure the camera is connected, powered, "
            "and visible to the system before running real Click-and-Go."
        ) from exc
    align = rs.align(rs.stream.color)
    return pipeline, profile, align


def get_color_intrinsics(profile):
    return profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()


def get_aligned_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None

    return color_frame, depth_frame


def median_depth_at_pixel(depth_frame, pixel_x, pixel_y, window_size):
    half = max(0, window_size // 2)
    valid_depths = []

    for y in range(pixel_y - half, pixel_y + half + 1):
        for x in range(pixel_x - half, pixel_x + half + 1):
            if x < 0 or y < 0 or x >= depth_frame.get_width() or y >= depth_frame.get_height():
                continue
            depth = depth_frame.get_distance(x, y)
            if depth > 0:
                valid_depths.append(depth)

    if not valid_depths:
        return None

    return float(np.median(valid_depths))


def pixel_to_camera_point_mm(intrinsics, pixel_x, pixel_y, depth_frame, window_size):
    depth_m = median_depth_at_pixel(depth_frame, pixel_x, pixel_y, window_size)
    if depth_m is None:
        return None

    point_m = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth_m)
    return np.asarray(point_m, dtype=float) * 1000.0


class ClickAndGoDemo:
    def __init__(self):
        ensure_runtime_dependencies()
        self.device_config = load_device_config()
        self.app_config = load_app_config()
        self.camera_cfg = self.app_config["camera"]
        self.motion_cfg = self.app_config["motion"]
        self.workspace_cfg = self.app_config["workspace"]
        self.calibration_cfg = self.app_config["calibration"]

        serial_number = self.device_config.get("camera_serial")
        if serial_number:
            print(f"Using RealSense serial: {serial_number}")

        self.pipeline, self.profile, self.align = initialize_pipeline(self.camera_cfg, serial_number)
        self.color_intrinsics = get_color_intrinsics(self.profile)
        self.detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        port = self.device_config["device_port"]
        print(f"Connecting to Dobot on port: {port}")
        try:
            self.device = Dobot(port=port)
        except Exception as exc:
            raise RuntimeError(
                f"Could not connect to Dobot on port {port}. Check the USB cable, power, and "
                f"{APP_CONFIG_PATH.parent / 'device_port.yaml'}."
            ) from exc
        print("Connected to Dobot.")

        self.gripper_T_tag = np.asarray(self.calibration_cfg["gripper_T_tag"], dtype=float)
        self.base_T_camera = self._load_saved_base_to_camera()

        self.device_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.pending_click = None
        self.latest_click = None
        self.latest_camera_point = None
        self.latest_base_point = None
        self.latest_command_point = None
        self.latest_tag = None

        self.motion_thread = None
        self.busy = False
        self.status_message = (
            "Loaded saved calibration. Press 'k' to recalibrate."
            if self.base_T_camera is not None
            else "Press 'k' with the gripper AprilTag visible to calibrate."
        )

    def _load_saved_base_to_camera(self):
        matrix = self.calibration_cfg.get("base_T_camera")
        if matrix is None:
            return None

        array = np.asarray(matrix, dtype=float)
        if array.shape != (4, 4):
            return None
        return array

    def _save_base_to_camera(self, transform):
        self.app_config["calibration"]["base_T_camera"] = np.asarray(transform, dtype=float).tolist()
        save_yaml(APP_CONFIG_PATH, self.app_config)

    def set_status(self, message):
        with self.state_lock:
            self.status_message = message

    def is_busy(self):
        with self.state_lock:
            return self.busy

    def set_busy(self, busy):
        with self.state_lock:
            self.busy = busy

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.state_lock:
                self.pending_click = (x, y)
                self.latest_click = (x, y)

    def calibrate_from_tag(self, tag):
        with self.device_lock:
            pose = self.device.get_pose()

        base_T_gripper = get_robot_arm_matrix(pose)
        base_T_tag = base_T_gripper @ self.gripper_T_tag
        camera_T_tag = get_tag_to_camera_matrix(tag)
        tag_T_camera = invert_transform(camera_T_tag)
        base_T_camera = base_T_tag @ tag_T_camera

        self.base_T_camera = base_T_camera
        self._save_base_to_camera(base_T_camera)
        self.set_status("Calibration updated and saved. Click the image to move.")

        print("=" * 70)
        print("Calibration complete: base_T_camera")
        print(np.array2string(base_T_camera, precision=3, suppress_small=True))
        print("=" * 70)

    def point_in_workspace(self, point_mm):
        return validate_command_point(point_mm, self.workspace_cfg, self.motion_cfg["target_r_deg"])

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
                if hasattr(self.device, "clear_alarms"):
                    self.device.clear_alarms()
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
            print(f"Executing move: {reason}")
            for index, (x, y, z, r) in enumerate(waypoints, start=1):
                print(f"Waypoint {index}: X={x:.1f}, Y={y:.1f}, Z={z:.1f}, R={r:.1f}")
                with self.device_lock:
                    self.device.move_to(x, y, z, r, wait=True)
            print("=" * 70)
            self.set_status("Move complete. Click another point or press 'k' to recalibrate.")
        except Exception as exc:
            self.set_status(f"Move failed: {exc}")
            print(f"Move failed: {exc}")
        finally:
            self.set_busy(False)

    def handle_click(self, depth_frame):
        with self.state_lock:
            if self.pending_click is None:
                return
            click = self.pending_click
            self.pending_click = None

        if self.base_T_camera is None:
            self.set_status("No calibration yet. Press 'k' while the gripper AprilTag is visible.")
            return

        if self.is_busy():
            self.set_status("Robot is busy. Click again after the move finishes.")
            return

        x, y = click
        camera_point_mm = pixel_to_camera_point_mm(
            self.color_intrinsics,
            x,
            y,
            depth_frame,
            int(self.camera_cfg["depth_window_px"]),
        )

        if camera_point_mm is None:
            self.set_status("Invalid depth at the clicked pixel. Try another point.")
            return

        raw_base_point_mm = transform_point(self.base_T_camera, camera_point_mm)
        command_point_mm = np.asarray(raw_base_point_mm, dtype=float).copy()
        command_point_mm[2] += float(self.motion_cfg["target_offset_z_mm"])

        valid, reason = self.point_in_workspace(command_point_mm)
        if not valid:
            self.latest_camera_point = camera_point_mm
            self.latest_base_point = raw_base_point_mm
            self.latest_command_point = command_point_mm
            self.set_status(f"Rejected target: {reason}")
            return

        self.latest_camera_point = camera_point_mm
        self.latest_base_point = raw_base_point_mm
        self.latest_command_point = command_point_mm

        reason = (
            f"Moving to clicked target: pixel=({x}, {y}) "
            f"base=({command_point_mm[0]:.1f}, {command_point_mm[1]:.1f}, {command_point_mm[2]:.1f})"
        )
        self.queue_move(command_point_mm, reason)

    def detect_tag(self, color_image):
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(
            gray_image,
            estimate_tag_pose=True,
            camera_params=[
                self.color_intrinsics.fx,
                self.color_intrinsics.fy,
                self.color_intrinsics.ppx,
                self.color_intrinsics.ppy,
            ],
            tag_size=float(self.camera_cfg["tag_size_m"]),
        )
        return tags

    def draw_tag_overlay(self, image, tag):
        for idx in range(len(tag.corners)):
            cv2.line(
                image,
                tuple(tag.corners[idx - 1, :].astype(int)),
                tuple(tag.corners[idx, :].astype(int)),
                (0, 255, 0),
                2,
            )
        center = tuple(tag.center.astype(int))
        cv2.circle(image, center, 5, (0, 255, 255), -1)
        cv2.putText(
            image,
            f"Tag ID: {tag.tag_id}",
            (center[0] + 12, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    def draw_click_overlay(self, image):
        if self.latest_click is None:
            return

        x, y = self.latest_click
        cv2.drawMarker(
            image,
            (x, y),
            (255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    def draw_info_panel(self, image):
        def shorten(text, limit):
            return text if len(text) <= limit else text[: limit - 3] + "..."

        height, width = image.shape[:2]
        panel_width = min(310, max(250, width // 2 - 20))
        panel_x0 = width - panel_width - 14
        panel_y0 = 12

        panel_lines = [
            "Click-and-Go",
            f"Calib: {'READY' if self.base_T_camera is not None else 'NOT READY'} | Busy: {'YES' if self.is_busy() else 'NO'}",
            shorten(f"Status: {self.status_message}", 42),
        ]

        if self.latest_command_point is not None:
            panel_lines.append(
                "Cmd: "
                f"{self.latest_command_point[0]:.1f}, {self.latest_command_point[1]:.1f}, {self.latest_command_point[2]:.1f}"
            )
        elif self.latest_base_point is not None:
            panel_lines.append(
                "Raw: "
                f"{self.latest_base_point[0]:.1f}, {self.latest_base_point[1]:.1f}, {self.latest_base_point[2]:.1f}"
            )

        panel_height = 14 + 24 * len(panel_lines)
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (panel_x0, panel_y0),
            (panel_x0 + panel_width, panel_y0 + panel_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.42, image, 0.58, 0, image)
        cv2.rectangle(
            image,
            (panel_x0, panel_y0),
            (panel_x0 + panel_width, panel_y0 + panel_height),
            (60, 220, 60),
            1,
        )

        for index, line in enumerate(panel_lines):
            cv2.putText(
                image,
                line,
                (panel_x0 + 12, panel_y0 + 24 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
            )

        control_text = "click move | k calibrate | h safe | c clear | q quit"
        footer_y0 = height - 34
        overlay = image.copy()
        cv2.rectangle(overlay, (10, footer_y0), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
        cv2.putText(
            image,
            control_text,
            (18, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )

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

    def run(self):
        print("=" * 70)
        print("Click-and-Go controls:")
        print("  - Press 'k' to calibrate from the gripper-mounted AprilTag")
        print("  - Left click on the camera view to move the arm")
        print("  - Press 'h' to move to the configured safe pose")
        print("  - Press 'q' to quit")
        print("=" * 70)

        window_name = "Click-and-Go Camera View"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        try:
            while True:
                color_frame, depth_frame = get_aligned_frames(self.pipeline, self.align)
                if color_frame is None or depth_frame is None:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                tags = self.detect_tag(color_image)
                self.latest_tag = tags[0] if tags else None

                if self.latest_tag is not None:
                    self.draw_tag_overlay(color_image, self.latest_tag)

                self.handle_click(depth_frame)
                self.draw_click_overlay(color_image)
                self.draw_info_panel(color_image)

                cv2.imshow(window_name, color_image) 
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("c"):
                    self.latest_click = None
                    self.latest_camera_point = None
                    self.latest_base_point = None
                    self.latest_command_point = None
                    self.set_status("Cleared target marker.")
                if key == ord("k"):
                    if self.latest_tag is None:
                        self.set_status("No AprilTag visible. Put the gripper tag in view and press 'k' again.")
                    elif self.is_busy():
                        self.set_status("Wait for the robot to stop before recalibrating.")
                    else:
                        self.calibrate_from_tag(self.latest_tag)
                if key == ord("h"):
                    self.move_to_safe_pose()

        finally:
            if self.motion_thread is not None and self.motion_thread.is_alive():
                self.motion_thread.join(timeout=5.0)
            self.pipeline.stop()
            cv2.destroyAllWindows()
            with self.device_lock:
                self.device.close()


def main():
    demo = ClickAndGoDemo()
    demo.run()


if __name__ == "__main__":
    main()
