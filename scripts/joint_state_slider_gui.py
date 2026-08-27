#!/usr/bin/python3

import math
import tkinter as tk
from tkinter import ttk
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateSliderGui(Node):
    def __init__(self):
        super().__init__("joint_state_slider_gui")
        self.declare_parameter("robot_description", "")

        robot_description = self.get_parameter("robot_description").value
        self.joints = self._parse_joints(robot_description)
        self.values = {joint["name"]: joint["initial"] for joint in self.joints}

        self.publisher = self.create_publisher(JointState, "joint_states", 10)
        self.root = tk.Tk()
        self.root.title("Robot Joint Control")
        self.root.geometry("760x760")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._running = True

        self._build_ui()
        self.root.after(50, self._publish_loop)

        active = [
            joint["name"]
            for joint in self.joints
            if not joint["locked"] and not joint["mimic_joint"]
        ]
        self.get_logger().info(
            f"joint slider GUI ready: {len(active)} controllable joints, "
            f"{len(self.joints) - len(active)} locked joints"
        )

    def _parse_joints(self, robot_description):
        if not robot_description:
            return []

        root = ET.fromstring(robot_description)
        joints = []
        for joint in root.findall("joint"):
            joint_name = joint.attrib.get("name", "")
            joint_type = joint.attrib.get("type", "")
            if not joint_name or joint_type in ("fixed", ""):
                continue

            lower = -math.pi
            upper = math.pi
            if joint_type == "prismatic":
                lower = -0.2
                upper = 0.2

            limit = joint.find("limit")
            if limit is not None:
                lower = float(limit.attrib.get("lower", lower))
                upper = float(limit.attrib.get("upper", upper))

            mimic = joint.find("mimic")
            mimic_joint = None
            mimic_multiplier = 1.0
            mimic_offset = 0.0
            if mimic is not None:
                mimic_joint = mimic.attrib.get("joint")
                mimic_multiplier = float(mimic.attrib.get("multiplier", 1.0))
                mimic_offset = float(mimic.attrib.get("offset", 0.0))

            locked = abs(upper - lower) < 1e-9
            initial = lower if locked else max(min(0.0, upper), lower)
            joints.append(
                {
                    "name": joint_name,
                    "type": joint_type,
                    "lower": lower,
                    "upper": upper,
                    "initial": initial,
                    "locked": locked,
                    "mimic_joint": mimic_joint,
                    "mimic_multiplier": mimic_multiplier,
                    "mimic_offset": mimic_offset,
                }
            )
        return joints

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="Joint sliders publish /joint_states").pack(side="left")
        ttk.Button(header, text="Reset", command=self._reset).pack(side="right")

        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self.joint_frame = ttk.Frame(canvas)
        self.joint_frame.bind(
            "<Configure>",
            lambda event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.joint_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.scales = {}
        row = 0
        for joint in self.joints:
            if joint["mimic_joint"]:
                continue

            frame = ttk.Frame(self.joint_frame, padding=(0, 4))
            frame.grid(row=row, column=0, sticky="ew")
            frame.columnconfigure(1, weight=1)

            ttk.Label(frame, text=joint["name"], width=34).grid(row=0, column=0, sticky="w")
            value_label = ttk.Label(frame, text=f"{joint['initial']:.3f}", width=9)
            value_label.grid(row=0, column=2, sticky="e")

            if joint["locked"]:
                ttk.Label(frame, text="locked").grid(row=0, column=1, sticky="w")
                row += 1
                continue

            variable = tk.DoubleVar(value=joint["initial"])
            scale = ttk.Scale(
                frame,
                from_=joint["lower"],
                to=joint["upper"],
                orient="horizontal",
                variable=variable,
                command=lambda value, name=joint["name"], label=value_label: self._set_joint(
                    name, value, label
                ),
            )
            scale.grid(row=0, column=1, sticky="ew", padx=8)
            self.scales[joint["name"]] = (variable, value_label)
            row += 1

    def _set_joint(self, name, value, label):
        numeric = float(value)
        self.values[name] = numeric
        label.configure(text=f"{numeric:.3f}")

    def _reset(self):
        for joint in self.joints:
            self.values[joint["name"]] = joint["initial"]
            if joint["name"] in self.scales:
                variable, label = self.scales[joint["name"]]
                variable.set(joint["initial"])
                label.configure(text=f"{joint['initial']:.3f}")

    def _publish_loop(self):
        if not self._running:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [joint["name"] for joint in self.joints]
        msg.position = []
        for joint in self.joints:
            if joint["mimic_joint"]:
                parent_value = self.values.get(joint["mimic_joint"], 0.0)
                value = parent_value * joint["mimic_multiplier"] + joint["mimic_offset"]
                self.values[joint["name"]] = value
            msg.position.append(self.values[joint["name"]])
        self.publisher.publish(msg)
        self.root.after(50, self._publish_loop)

    def _on_close(self):
        self._running = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    node = JointStateSliderGui()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
