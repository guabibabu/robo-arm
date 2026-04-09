#!/usr/bin/env python3
"""Coverage for the offline Dobot simulator."""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest

import numpy as np


SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from arm_move import load_demo_config
from click_and_go_offline import (
    BOX_TARGETS,
    CameraIntrinsics,
    OFFLINE_CAMERA_POSITION_MM,
    OFFLINE_CAMERA_TARGET_MM,
    make_box_faces,
    build_synthetic_tag,
    intersect_pixel_with_plane,
    make_look_at_transform,
    project_base_point_to_pixel,
)
from click_and_go_shared import get_robot_arm_matrix, get_tag_to_camera_matrix, invert_transform
from dobot_backend import create_dobot
from manual_customized_task import load_task_config
from simulated_dobot import HOME_POSE, SimulatedDobot, forward_kinematics, inverse_kinematics


class SimulatedDobotTests(unittest.TestCase):
    def test_inverse_and_forward_kinematics_match_known_targets(self) -> None:
        points = [
            HOME_POSE,
            (250.0, 50.0, 50.0, 90.0),
            (200.0, -50.0, 50.0, 0.0),
            (250.0, 0.0, 50.0, -90.0),
            (200.0, 100.0, 0.0, 0.0),
            (200.0, -100.0, 0.0, 0.0),
        ]

        current_joints = None
        for point in points:
            joints = inverse_kinematics(*point, current_joints=current_joints)
            pose = forward_kinematics(joints)
            self.assertAlmostEqual(pose.x, point[0], places=4)
            self.assertAlmostEqual(pose.y, point[1], places=4)
            self.assertAlmostEqual(pose.z, point[2], places=4)
            self.assertAlmostEqual(pose.r, point[3], places=4)
            current_joints = joints

    def test_move_to_wait_false_then_wait_for_cmd_updates_pose(self) -> None:
        device = SimulatedDobot(enable_viewer=False, time_scale=0.0)
        try:
            command_id = device.move_to(250.0, 50.0, 50.0, 90.0, wait=False)
            device.wait_for_cmd(command_id)
            pose = device.get_pose()
            self.assertAlmostEqual(pose.position.x, 250.0, places=4)
            self.assertAlmostEqual(pose.position.y, 50.0, places=4)
            self.assertAlmostEqual(pose.position.z, 50.0, places=4)
            self.assertAlmostEqual(pose.position.r, 90.0, places=4)
        finally:
            device.close()

    def test_unreachable_target_raises_clear_error(self) -> None:
        device = SimulatedDobot(enable_viewer=False, time_scale=0.0)
        try:
            with self.assertRaisesRegex(ValueError, "outside the simulated workspace"):
                device.move_to(500.0, 0.0, 300.0, 0.0, wait=True)
        finally:
            device.close()

    def test_real_backend_missing_dependencies_raises_runtime_error(self) -> None:
        previous_backend = os.environ.get("DOBOT_BACKEND")
        os.environ["DOBOT_BACKEND"] = "real"
        try:
            with self.assertRaisesRegex(RuntimeError, "Install requirements|DOBOT_BACKEND=sim"):
                create_dobot()
        finally:
            if previous_backend is None:
                os.environ.pop("DOBOT_BACKEND", None)
            else:
                os.environ["DOBOT_BACKEND"] = previous_backend

    def test_load_demo_config_reads_editable_point_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "points.json"
            config_path.write_text(
                """
                {
                  "pause_after_move_s": 0.5,
                  "pause_after_report_s": 0.25,
                  "points": [
                    {"name": "A", "x": 210, "y": 10, "z": 55, "r": 15},
                    {"x": 220, "y": -20, "z": 45, "r": -30}
                  ]
                }
                """.strip(),
                encoding="utf-8",
            )

            points, pause_after_move_s, pause_after_report_s = load_demo_config(config_path)

            self.assertEqual(points[0]["name"], "A")
            self.assertEqual(points[1]["name"], "P2")
            self.assertEqual(points[1]["x"], 220.0)
            self.assertEqual(pause_after_move_s, 0.5)
            self.assertEqual(pause_after_report_s, 0.25)

    def test_load_task_config_reads_editable_pick_place_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = pathlib.Path(temp_dir) / "pick_place.json"
            config_path.write_text(
                """
                {
                  "jump_height_mm": 42,
                  "move_delay_s": 0.2,
                  "grip_delay_s": 0.8,
                  "release_delay_s": 0.4,
                  "pick": {"x": 210, "y": 30, "z": 5, "r": 10},
                  "place": {"name": "Drop", "x": 190, "y": -20, "z": 8, "r": -10},
                  "safe_pose": {"x": 200, "y": 0, "z": 50, "r": 0}
                }
                """.strip(),
                encoding="utf-8",
            )

            config = load_task_config(config_path)

            self.assertEqual(config["jump_height_mm"], 42.0)
            self.assertEqual(config["pick"]["name"], "Pick")
            self.assertEqual(config["place"]["name"], "Drop")
            self.assertEqual(config["place"]["y"], -20.0)
            self.assertEqual(config["release_delay_s"], 0.4)

    def test_click_and_go_projection_round_trip_hits_workspace_plane(self) -> None:
        intrinsics = CameraIntrinsics(width=640, height=480, fx=540.0, fy=540.0, cx=320.0, cy=240.0)
        base_T_camera = make_look_at_transform(OFFLINE_CAMERA_POSITION_MM, OFFLINE_CAMERA_TARGET_MM)
        base_point = np.array([220.0, 20.0, 0.0], dtype=float)

        pixel = project_base_point_to_pixel(base_point, base_T_camera, intrinsics)
        reconstructed = intersect_pixel_with_plane(pixel[0], pixel[1], base_T_camera, intrinsics, plane_z_mm=0.0)

        np.testing.assert_allclose(reconstructed, base_point, atol=1e-3)

    def test_click_and_go_synthetic_tag_reconstructs_camera_pose(self) -> None:
        intrinsics = CameraIntrinsics(width=640, height=480, fx=540.0, fy=540.0, cx=320.0, cy=240.0)
        ground_truth_base_T_camera = make_look_at_transform(OFFLINE_CAMERA_POSITION_MM, OFFLINE_CAMERA_TARGET_MM)
        gripper_T_tag = np.array(
            [
                [-1.0, 0.0, 0.0, 30.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 153.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        device = SimulatedDobot(enable_viewer=False, time_scale=0.0)
        try:
            pose = device.get_pose()
            base_T_gripper = get_robot_arm_matrix(pose)
            tag = build_synthetic_tag(
                ground_truth_base_T_camera,
                base_T_gripper,
                gripper_T_tag,
                tag_size_mm=79.2,
                intrinsics=intrinsics,
            )
            self.assertIsNotNone(tag)

            base_T_tag = base_T_gripper @ gripper_T_tag
            camera_T_tag = get_tag_to_camera_matrix(tag)
            estimated_base_T_camera = base_T_tag @ invert_transform(camera_T_tag)

            np.testing.assert_allclose(estimated_base_T_camera, ground_truth_base_T_camera, atol=1e-6)
        finally:
            device.close()

    def test_click_and_go_box_top_projects_inside_camera_frame(self) -> None:
        intrinsics = CameraIntrinsics(width=640, height=480, fx=540.0, fy=540.0, cx=320.0, cy=240.0)
        base_T_camera = make_look_at_transform(OFFLINE_CAMERA_POSITION_MM, OFFLINE_CAMERA_TARGET_MM)
        box = BOX_TARGETS[0]
        box_top_center = make_box_faces(box)["center_top"]

        pixel = project_base_point_to_pixel(box_top_center, base_T_camera, intrinsics)

        self.assertIsNotNone(pixel)
        self.assertGreaterEqual(pixel[0], 0.0)
        self.assertLessEqual(pixel[0], intrinsics.width)
        self.assertGreaterEqual(pixel[1], 0.0)
        self.assertLessEqual(pixel[1], intrinsics.height)


if __name__ == "__main__":
    unittest.main()
