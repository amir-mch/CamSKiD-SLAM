import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("cam_skid_slam")
    rviz_cfg = os.path.join(pkg_share, "config", "cam_skid.rviz")

    save_dir = LaunchConfiguration("save_dir")
    use_rviz = LaunchConfiguration("rviz")

    return LaunchDescription([
        DeclareLaunchArgument("save_dir", default_value="/media/amir/New Volume/Project/cam_skid_results"),
        DeclareLaunchArgument("rviz", default_value="true"),

        Node(
            package="cam_skid_slam",
            executable="cam_skid_vo",
            namespace="robot0",
            name="cam_skid_vo",
            output="screen",
            parameters=[{
                "robot_id": "robot0",
                "keyframe_dist_thresh": 0.10,
                "keyframe_angle_thresh": 0.12,
                "save_dir": save_dir,
            }],
        ),
        Node(
            package="cam_skid_slam",
            executable="cam_skid_vo",
            namespace="robot1",
            name="cam_skid_vo",
            output="screen",
            parameters=[{
                "robot_id": "robot1",
                "keyframe_dist_thresh": 0.10,
                "keyframe_angle_thresh": 0.12,
                "save_dir": save_dir,
            }],
        ),
        Node(
            package="cam_skid_slam",
            executable="cam_skid_fusion",
            name="cam_skid_fusion",
            output="screen",
            parameters=[{
                "anchor_robot": "robot0",
                "global_desc_thresh": 0.65,
                "min_ransac_inliers": 10,
                "icp_fitness_thresh": 0.12,
                "save_dir": save_dir,
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="cam_skid_rviz",
            arguments=["-d", rviz_cfg],
            condition=IfCondition(use_rviz),
            output="screen",
        ),
    ])
