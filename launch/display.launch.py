from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("robot"))
    urdf_path = package_share / "urdf" / "机器人1.urdf"
    rviz_config_path = package_share / "rviz" / "display.rviz"

    robot_description = urdf_path.read_text(encoding="utf-8")

    use_rviz = LaunchConfiguration("rviz")
    use_gui = LaunchConfiguration("gui")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Start RViz2.",
            ),
            DeclareLaunchArgument(
                "gui",
                default_value="true",
                description="Start the joint slider GUI.",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="robot",
                executable="joint_state_slider_gui.py",
                name="joint_state_slider_gui",
                output="screen",
                condition=IfCondition(use_gui),
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="robot",
                executable="zero_joint_state_publisher.py",
                name="zero_joint_state_publisher",
                output="screen",
                condition=UnlessCondition(use_gui),
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", str(rviz_config_path)],
                condition=IfCondition(use_rviz),
                output="screen",
            ),
        ]
    )
