#!/usr/bin/env python3
"""
End-to-End Multi-Robot Camera-Only Test for CamSKiD-SLAM.
Renders textured 3D landmarks into RGB + Depth images for two robots (robot0, robot1)
with overlapping trajectories, writes groundtruth.txt, and streams to ROS 2.
"""

import argparse
import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time


def make_ros_time(t_sec):
    msg = Time()
    msg.sec = int(t_sec)
    msg.nanosec = int((t_sec - int(t_sec)) * 1e9)
    return msg


def mat_to_image_msg(img, encoding, frame_id, stamp):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = img.strides[0]
    msg.data = img.tobytes()
    return msg


def build_3d_scene(seed=7):
    rng = np.random.default_rng(seed)
    pts = []
    colors = []
    # Rich textured landmarks on walls and objects around a 6m x 6m room
    for _ in range(450):
        side = rng.integers(0, 4)
        if side == 0:
            p = [rng.uniform(-2.5, 4.5), rng.uniform(-1.5, 1.5), rng.uniform(2.5, 3.8)]
        elif side == 1:
            p = [rng.uniform(-2.5, 4.5), rng.uniform(-1.5, 1.5), rng.uniform(-2.5, -1.5)]
        elif side == 2:
            p = [rng.uniform(2.5, 4.5), rng.uniform(-1.5, 1.5), rng.uniform(-2.0, 3.5)]
        else:
            p = [rng.uniform(-2.0, 3.5), rng.uniform(0.6, 1.4), rng.uniform(-1.5, 3.0)]
        pts.append(p)
        colors.append(rng.integers(40, 245, size=3).tolist())
    return np.array(pts, dtype=np.float64), np.array(colors, dtype=np.uint8)


def render_frame(pts_w, colors, R_wc, t_wc, fx=517.306408, fy=516.469215, cx=318.643040, cy=255.313989):
    w, h = 640, 480
    bgr = np.full((h, w, 3), 28, dtype=np.uint8)
    depth_m = np.zeros((h, w), dtype=np.float32)

    # Transform world -> camera: p_c = R_wc^T * (p_w - t_wc)
    pts_c = (R_wc.T @ (pts_w - t_wc).T).T
    z = pts_c[:, 2]
    valid = (z > 0.35) & (z < 5.0)
    pts_c = pts_c[valid]
    cols = colors[valid]

    order = np.argsort(-pts_c[:, 2])  # painter's algo: far to near
    pts_c = pts_c[order]
    cols = cols[order]

    us = np.round(fx * pts_c[:, 0] / pts_c[:, 2] + cx).astype(int)
    vs = np.round(fy * pts_c[:, 1] / pts_c[:, 2] + cy).astype(int)

    for u, v, d, c in zip(us, vs, pts_c[:, 2], cols):
        if 8 <= u < w - 8 and 8 <= v < h - 8:
            rad = max(4, min(14, int(18.0 / d)))
            cv2.circle(bgr, (u, v), rad, (int(c[0]), int(c[1]), int(c[2])), -1)
            cv2.circle(bgr, (u, v), max(2, rad // 2), (255 - int(c[0]), int(c[1]), 255), 1)
            cv2.circle(depth_m, (u, v), rad, float(d), -1)

    depth_u16 = np.clip(depth_m * 5000.0, 0, 65535).astype(np.uint16)
    return bgr, depth_u16


def rot_y(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


class SyntheticPlayer(Node):
    def __init__(self, gt_path):
        super().__init__("synthetic_multirobot_player")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pubs = {
            "robot0": (
                self.create_publisher(Image, "/robot0/camera/rgb/image_raw", qos),
                self.create_publisher(Image, "/robot0/camera/depth/image_raw", qos),
            ),
            "robot1": (
                self.create_publisher(Image, "/robot1/camera/rgb/image_raw", qos),
                self.create_publisher(Image, "/robot1/camera/depth/image_raw", qos),
            ),
        }
        self.gt_path = gt_path

    def run(self):
        # Wait briefly for DDS discovery
        for _ in range(30):
            if self.pubs["robot0"][0].get_subscription_count() > 0 and self.pubs["robot1"][0].get_subscription_count() > 0:
                break
            time.sleep(0.1)
        pts_w, colors = build_3d_scene()
        n_frames = 55
        t0 = 1700000000.0

        gt_lines = []
        for i in range(n_frames):
            if not rclpy.ok():
                break
            t_sec = t0 + i * 0.1
            stamp = make_ros_time(t_sec)

            # robot0 moves forward + slight yaw
            x0 = 0.04 * i
            z0 = 0.01 * np.sin(0.1 * i)
            yaw0 = 0.008 * i
            R0 = rot_y(yaw0)
            t_wc0 = np.array([x0, 0.0, z0])

            # robot1 starts slightly offset and overlaps robot0's path
            x1 = 0.3 + 0.035 * i
            z1 = -0.1 + 0.02 * np.cos(0.1 * i)
            yaw1 = 0.006 * i
            R1 = rot_y(yaw1)
            t_wc1 = np.array([x1, 0.0, z1])

            gt_lines.append(f"{t_sec:.6f} {t_wc0[0]:.6f} {t_wc0[1]:.6f} {t_wc0[2]:.6f} 0 0 0 1\n")

            for rid, R, twc in [("robot0", R0, t_wc0), ("robot1", R1, t_wc1)]:
                bgr, depth = render_frame(pts_w, colors, R, twc)
                self.pubs[rid][0].publish(mat_to_image_msg(bgr, "bgr8", f"{rid}/camera_link", stamp))
                self.pubs[rid][1].publish(mat_to_image_msg(depth, "16UC1", f"{rid}/camera_link", stamp))

            time.sleep(0.12)

        with open(self.gt_path, "w") as f:
            f.writelines(gt_lines)
        self.get_logger().info(f"Finished streaming {n_frames} frames per robot.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-out", default="/media/amir/New Volume/Project/cam_skid_results/groundtruth.txt")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.gt_out), exist_ok=True)

    rclpy.init()
    node = SyntheticPlayer(args.gt_out)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
