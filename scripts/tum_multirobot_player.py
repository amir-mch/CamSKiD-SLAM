#!/usr/bin/env python3
"""
Multi-Robot TUM RGB-D Player for CamSKiD-SLAM.
Splits a single-robot TUM RGB-D sequence into 2 overlapping robot trajectories
(robot0 and robot1) so inter-robot loop closure and map fusion can be benchmarked
against real ground-truth trajectories.
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


def read_file_list(filename):
    entries = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            entries.append((float(parts[0]), parts[1]))
    return entries


def associate_rgb_depth(rgb_list, depth_list, max_diff=0.04):
    depth_times = np.array([d[0] for d in depth_list])
    pairs = []
    for t_rgb, rgb_rel in rgb_list:
        idx = int(np.argmin(np.abs(depth_times - t_rgb)))
        if abs(depth_times[idx] - t_rgb) < max_diff:
            pairs.append((t_rgb, rgb_rel, depth_list[idx][1]))
    return pairs


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


class MultiRobotPlayer(Node):
    def __init__(self, dataset_dir, fps=10.0, stride=2):
        super().__init__("tum_multirobot_player")
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

        rgb_list = read_file_list(os.path.join(dataset_dir, "rgb.txt"))
        depth_list = read_file_list(os.path.join(dataset_dir, "depth.txt"))
        pairs = associate_rgb_depth(rgb_list, depth_list)[::stride]

        n = len(pairs)
        split0_end = int(0.60 * n)
        split1_start = int(0.40 * n)

        self.seq0 = pairs[:split0_end]
        self.seq1 = pairs[split1_start:]
        self.dataset_dir = dataset_dir
        self.fps = fps

        self.get_logger().info(
            f"Loaded {n} RGB-D pairs (stride={stride}). "
            f"robot0={len(self.seq0)} frames, robot1={len(self.seq1)} frames (20% overlap)."
        )

    def run(self):
        for _ in range(30):
            if self.pubs["robot0"][0].get_subscription_count() > 0 and self.pubs["robot1"][0].get_subscription_count() > 0:
                break
            time.sleep(0.1)
        period = 1.0 / self.fps
        max_len = max(len(self.seq0), len(self.seq1))
        for i in range(max_len):
            if not rclpy.ok():
                break
            t_start = time.time()

            for rid, seq in [("robot0", self.seq0), ("robot1", self.seq1)]:
                if i < len(seq):
                    t_sec, rgb_rel, depth_rel = seq[i]
                    bgr = cv2.imread(os.path.join(self.dataset_dir, rgb_rel), cv2.IMREAD_COLOR)
                    depth = cv2.imread(os.path.join(self.dataset_dir, depth_rel), cv2.IMREAD_UNCHANGED)
                    if bgr is not None and depth is not None:
                        stamp = make_ros_time(t_sec)
                        self.pubs[rid][0].publish(
                            mat_to_image_msg(bgr, "bgr8", f"{rid}/camera_link", stamp)
                        )
                        self.pubs[rid][1].publish(
                            mat_to_image_msg(depth, "16UC1", f"{rid}/camera_link", stamp)
                        )

            if i % 25 == 0:
                self.get_logger().info(f"Published frame {i}/{max_len}")

            dt = time.time() - t_start
            if dt < period:
                time.sleep(period - dt)

        self.get_logger().info("Finished playing multi-robot TUM sequence.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to TUM RGB-D sequence folder")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    rclpy.init()
    node = MultiRobotPlayer(args.dataset, fps=args.fps, stride=args.stride)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
