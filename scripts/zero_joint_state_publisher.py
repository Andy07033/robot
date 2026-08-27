#!/usr/bin/python3

import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class ZeroJointStatePublisher(Node):
    def __init__(self):
        super().__init__("zero_joint_state_publisher")
        self.declare_parameter("robot_description", "")

        robot_description = self.get_parameter("robot_description").value
        self.joint_names = self._get_movable_joint_names(robot_description)

        self.publisher = self.create_publisher(JointState, "joint_states", 10)
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        self.get_logger().info(
            f"publishing zero joint states for: {', '.join(self.joint_names) or 'none'}"
        )

    def _get_movable_joint_names(self, robot_description):
        if not robot_description:
            return []

        root = ET.fromstring(robot_description)
        names = []
        for joint in root.findall("joint"):
            joint_type = joint.attrib.get("type", "")
            joint_name = joint.attrib.get("name", "")
            if joint_name and joint_type not in ("fixed", ""):
                names.append(joint_name)
        return names

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [0.0] * len(self.joint_names)
        self.publisher.publish(msg)


def main():
    rclpy.init()
    node = ZeroJointStatePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
