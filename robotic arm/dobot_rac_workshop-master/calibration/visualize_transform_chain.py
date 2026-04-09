#!/usr/bin/env python3
"""
Interactive transform-chain visualizer.

This script does not need a real robot or camera. It builds a simple example
chain:

    Base -> Gripper -> Tag -> Camera

and visualizes how a point expressed in the Tag frame is transformed into the
Base frame step by step.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R


def make_transform(translation_mm, euler_deg_xyz):
    """Create a 4x4 homogeneous transform from translation and xyz Euler angles."""
    transform = np.eye(4)
    transform[:3, :3] = R.from_euler("xyz", euler_deg_xyz, degrees=True).as_matrix()
    transform[:3, 3] = np.asarray(translation_mm, dtype=float)
    return transform


def invert_transform(transform):
    """Invert a rigid 4x4 homogeneous transform."""
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_point(transform, point_xyz):
    """Apply a 4x4 transform to a 3D point."""
    point_h = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0])
    return (transform @ point_h)[:3]


def draw_frame(ax, transform, label, scale=40.0, colors=None):
    """Draw a coordinate frame defined by a 4x4 transform."""
    if colors is None:
        colors = ["red", "green", "blue"]

    origin = transform[:3, 3]
    local_axes = np.array(
        [
            [scale, 0.0, 0.0, 1.0],
            [0.0, scale, 0.0, 1.0],
            [0.0, 0.0, scale, 1.0],
        ]
    )
    world_axes = (transform @ local_axes.T).T

    for idx, axis_name in enumerate(("X", "Y", "Z")):
        endpoint = world_axes[idx, :3]
        vector = endpoint - origin
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vector[0],
            vector[1],
            vector[2],
            color=colors[idx],
            linewidth=2.4,
            arrow_length_ratio=0.14,
        )
        ax.text(endpoint[0], endpoint[1], endpoint[2], f"{label}_{axis_name}", fontsize=9)

    ax.scatter(origin[0], origin[1], origin[2], s=55, c="black")
    ax.text(origin[0], origin[1], origin[2], f" {label}", fontsize=10, weight="bold")


class TransformChainVisualizer:
    """Animate a transform chain and a point moving through it."""

    def __init__(self):
        self.fig = plt.figure(figsize=(15, 8))
        self.ax_3d = self.fig.add_subplot(121, projection="3d")
        self.ax_text = self.fig.add_subplot(122)
        self.ax_text.axis("off")
        self.animation = None

        self.tag_point = np.array([35.0, 15.0, 20.0])
        self.gripper_T_tag = np.array(
            [
                [-1.0, 0.0, 0.0, 30.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 153.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.base_T_camera = make_transform(
            translation_mm=[220.0, -180.0, 260.0],
            euler_deg_xyz=[-20.0, 15.0, 125.0],
        )

    def compute_chain(self, frame_idx):
        """Build a slightly changing transform chain."""
        gripper_yaw = 8.0 + 0.9 * frame_idx
        gripper_z = 120.0 + 15.0 * np.sin(np.deg2rad(frame_idx * 6.0))

        base_T_gripper = make_transform(
            translation_mm=[180.0, 90.0, gripper_z],
            euler_deg_xyz=[0.0, 0.0, gripper_yaw],
        )
        gripper_T_tag = self.gripper_T_tag
        base_T_tag = base_T_gripper @ gripper_T_tag
        camera_T_tag = invert_transform(self.base_T_camera) @ base_T_tag

        return {
            "base_T_gripper": base_T_gripper,
            "gripper_T_tag": gripper_T_tag,
            "base_T_tag": base_T_tag,
            "camera_T_tag": camera_T_tag,
            "tag_T_camera": invert_transform(camera_T_tag),
        }

    def format_matrix(self, name, matrix):
        rows = []
        for row in matrix:
            rows.append("[" + " ".join(f"{value:8.3f}" for value in row) + "]")
        return f"{name} =\n" + "\n".join(rows)

    def update(self, frame_idx):
        chain = self.compute_chain(frame_idx)

        base_T_gripper = chain["base_T_gripper"]
        gripper_T_tag = chain["gripper_T_tag"]
        base_T_tag = chain["base_T_tag"]
        camera_T_tag = chain["camera_T_tag"]

        point_in_tag = self.tag_point
        point_in_gripper = transform_point(gripper_T_tag, point_in_tag)
        point_in_base = transform_point(base_T_tag, point_in_tag)
        point_in_camera = transform_point(camera_T_tag, point_in_tag)

        self.ax_3d.clear()
        self.ax_text.clear()
        self.ax_text.axis("off")

        draw_frame(self.ax_3d, np.eye(4), "Base", scale=55.0)
        draw_frame(self.ax_3d, self.base_T_camera, "Camera", scale=45.0, colors=["salmon", "lime", "skyblue"])
        draw_frame(self.ax_3d, base_T_gripper, "Gripper", scale=42.0, colors=["darkred", "darkgreen", "darkblue"])
        draw_frame(self.ax_3d, base_T_tag, "Tag", scale=38.0, colors=["orange", "gold", "purple"])

        base_origin = np.zeros(3)
        gripper_origin = base_T_gripper[:3, 3]
        tag_origin = base_T_tag[:3, 3]
        camera_origin = self.base_T_camera[:3, 3]

        self.ax_3d.plot(
            [base_origin[0], gripper_origin[0]],
            [base_origin[1], gripper_origin[1]],
            [base_origin[2], gripper_origin[2]],
            linestyle="--",
            color="gray",
            linewidth=1.5,
        )
        self.ax_3d.plot(
            [gripper_origin[0], tag_origin[0]],
            [gripper_origin[1], tag_origin[1]],
            [gripper_origin[2], tag_origin[2]],
            linestyle=":",
            color="purple",
            linewidth=1.8,
        )
        self.ax_3d.plot(
            [camera_origin[0], tag_origin[0]],
            [camera_origin[1], tag_origin[1]],
            [camera_origin[2], tag_origin[2]],
            linestyle="-.",
            color="teal",
            linewidth=1.6,
        )

        self.ax_3d.scatter(point_in_base[0], point_in_base[1], point_in_base[2], c="magenta", s=100)
        self.ax_3d.text(
            point_in_base[0], point_in_base[1], point_in_base[2], " Tag point in Base", fontsize=10, color="magenta"
        )

        self.ax_3d.set_title("Transform Chain: Base -> Gripper -> Tag -> Camera", fontsize=12, weight="bold")
        self.ax_3d.set_xlabel("X (mm)")
        self.ax_3d.set_ylabel("Y (mm)")
        self.ax_3d.set_zlabel("Z (mm)")
        self.ax_3d.set_xlim([-150, 350])
        self.ax_3d.set_ylim([-300, 250])
        self.ax_3d.set_zlim([-120, 380])
        self.ax_3d.set_box_aspect([1, 1, 1])
        self.ax_3d.view_init(elev=24.0, azim=-58.0)

        explanation = "\n".join(
            [
                "Point picked in Tag frame:",
                f"p_tag = [{point_in_tag[0]:.1f}, {point_in_tag[1]:.1f}, {point_in_tag[2]:.1f}] mm",
                "",
                "Step-by-step coordinate conversion:",
                f"1. p_gripper = gripper_T_tag * p_tag",
                f"   -> [{point_in_gripper[0]:.1f}, {point_in_gripper[1]:.1f}, {point_in_gripper[2]:.1f}] mm",
                f"2. p_base = base_T_gripper * p_gripper",
                f"   -> [{point_in_base[0]:.1f}, {point_in_base[1]:.1f}, {point_in_base[2]:.1f}] mm",
                "",
                "Equivalent one-shot form:",
                "p_base = (base_T_gripper * gripper_T_tag) * p_tag",
                "",
                "What the camera sees:",
                f"p_camera = camera_T_tag * p_tag",
                f"   -> [{point_in_camera[0]:.1f}, {point_in_camera[1]:.1f}, {point_in_camera[2]:.1f}] mm",
                "",
                self.format_matrix("base_T_gripper", base_T_gripper),
                "",
                self.format_matrix("gripper_T_tag", gripper_T_tag),
                "",
                self.format_matrix("camera_T_tag", camera_T_tag),
            ]
        )
        self.ax_text.text(0.0, 1.0, explanation, va="top", ha="left", family="monospace", fontsize=9)

        return (self.ax_3d, self.ax_text)

    def run(self, animate=True):
        if animate:
            self.animation = FuncAnimation(
                self.fig, self.update, frames=180, interval=80, blit=False, cache_frame_data=False
            )
        else:
            self.update(frame_idx=0)
        plt.tight_layout()
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize how coordinates move through a transform chain.")
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Draw a single snapshot instead of an animation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    visualizer = TransformChainVisualizer()
    visualizer.run(animate=not args.no_animate)


if __name__ == "__main__":
    main()
