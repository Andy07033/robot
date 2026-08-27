#!/usr/bin/python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time

import casadi as ca
import mujoco
import mujoco.viewer
import numpy as np


BASE_JOINTS = ["base_x", "base_y", "base_yaw"]
BASE_ACTUATORS = ["base_vx_ctrl", "base_vy_ctrl", "base_yaw_rate_ctrl"]

UR3_JOINT_SUFFIXES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

GRIPPER_ACTUATOR_SUFFIXES = [
    "finger_joint_pos",
    "left_inner_finger_joint_pos",
    "left_inner_knuckle_joint_pos",
    "right_outer_knuckle_joint_pos",
    "right_inner_finger_joint_pos",
    "right_inner_knuckle_joint_pos",
]

STRICT_INSERT_MAX_TIP_ERROR = 0.006
STRICT_INSERT_MAX_AXIAL_ERROR = 0.004
STRICT_INSERT_MAX_LATERAL_ERROR = 0.0035
STRICT_INSERT_MAX_VERTICAL_ERROR = 0.003
STRICT_INSERT_MIN_X_DOT = 0.995
STRICT_INSERT_MIN_YZ_DOT = 0.990


@dataclass(frozen=True)
class ArmConfig:
    prefix: str
    base_pos: tuple[float, float, float]
    base_pitch: float
    home_q: tuple[float, float, float, float, float, float]
    side_sign: float

    @property
    def joints(self) -> list[str]:
        return [f"{self.prefix}_ur3_{suffix}" for suffix in UR3_JOINT_SUFFIXES]

    @property
    def actuators(self) -> list[str]:
        return [name + "_pos" for name in self.joints]

    @property
    def gripper_actuators(self) -> list[str]:
        return [f"{self.prefix}_gripper_{suffix}" for suffix in GRIPPER_ACTUATOR_SUFFIXES]

    @property
    def tcp_site(self) -> str:
        return f"{self.prefix}_gripper_tcp"


@dataclass
class VisionEstimate:
    port_target: np.ndarray
    port_entry: np.ndarray
    port_rot: np.ndarray
    port_hold_point: np.ndarray
    plug_tip: np.ndarray
    plug_grasp: np.ndarray
    plug_rot: np.ndarray
    port_error: np.ndarray
    plug_tip_error: np.ndarray
    port_detections: int
    plug_detections: int


ARMS = {
    "left": ArmConfig(
        prefix="left",
        base_pos=(0.215, 0.0075416, 0.43),
        base_pitch=np.pi / 2.0,
        home_q=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        side_sign=1.0,
    ),
    # This is the visually left arm in the current mirrored MJCF.
    "right": ArmConfig(
        prefix="right",
        base_pos=(-0.215, 0.0075416, 0.43),
        base_pitch=-np.pi / 2.0,
        home_q=(0.0, np.pi, 0.0, 0.0, 0.0, 0.0),
        side_sign=-1.0,
    ),
}


def casadi_rot_x(a):
    c = ca.cos(a)
    s = ca.sin(a)
    return ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, c, -s),
        ca.horzcat(0, s, c),
    )


def casadi_rot_y(a):
    c = ca.cos(a)
    s = ca.sin(a)
    return ca.vertcat(
        ca.horzcat(c, 0, s),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-s, 0, c),
    )


def casadi_rot_z(a):
    c = ca.cos(a)
    s = ca.sin(a)
    return ca.vertcat(
        ca.horzcat(c, -s, 0),
        ca.horzcat(s, c, 0),
        ca.horzcat(0, 0, 1),
    )


def casadi_rpy(roll: float, pitch: float, yaw: float):
    return casadi_rot_x(roll) @ casadi_rot_y(pitch) @ casadi_rot_z(yaw)


def fk_expr(q, base_pose, arm: ArmConfig):
    p = ca.vertcat(base_pose[0], base_pose[1], 0.995)
    rot = casadi_rot_z(base_pose[2])

    p = p + rot @ ca.DM(arm.base_pos)
    rot = rot @ casadi_rpy(0.0, arm.base_pitch, 0.0)
    points = {}

    def translate(local_pos):
        nonlocal p
        p = p + rot @ ca.DM(local_pos)

    translate((0.0, 0.0, 0.1519))
    rot = rot @ casadi_rot_z(q[0])
    points["shoulder"] = p

    translate((0.0, 0.1198, 0.0))
    rot = rot @ casadi_rpy(0.0, np.pi / 2.0, 0.0) @ casadi_rot_y(q[1])
    points["upper"] = p

    translate((0.0, -0.0925, 0.24365))
    rot = rot @ casadi_rot_y(q[2])
    points["forearm"] = p

    translate((0.0, 0.0, 0.21325))
    rot = rot @ casadi_rpy(0.0, np.pi / 2.0, 0.0) @ casadi_rot_y(q[3])
    points["wrist1"] = p

    translate((0.0, 0.08505, 0.0))
    rot = rot @ casadi_rot_z(q[4])
    points["wrist2"] = p

    translate((0.0, 0.0, 0.08535))
    rot = rot @ casadi_rot_y(q[5])
    points["wrist3"] = p

    translate((0.0, 0.0819, 0.0))
    rot = rot @ casadi_rpy(-np.pi / 2.0, 0.0, 0.0)
    points["tool0"] = p
    points["rot"] = rot
    points["tcp"] = p + rot @ ca.DM((0.0, 0.0, 0.12))
    return points


def smoothstep(alpha: float) -> float:
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _as_array_log(rows: list[np.ndarray], width: int = 3) -> np.ndarray:
    if not rows:
        return np.empty((0, width), dtype=float)
    return np.vstack(rows)


def rotation_from_axes(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    x = np.asarray(x_axis, dtype=float)
    z = np.asarray(z_axis, dtype=float)
    y_hint = np.asarray(y_axis, dtype=float)
    x /= np.linalg.norm(x)
    if np.linalg.norm(z) < 1e-9:
        z = np.cross(x, y_hint)
    z -= x * float(x @ z)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, z])


def small_angle_rotation(vec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vec))
    if angle < 1e-12:
        return np.eye(3)
    axis = np.asarray(vec, dtype=float) / angle
    kx = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)


def average_rotations(rotations: list[np.ndarray], weights: list[float]) -> np.ndarray:
    acc = np.zeros((3, 3), dtype=float)
    for rot, weight in zip(rotations, weights):
        acc += float(weight) * rot
    u, _, vt = np.linalg.svd(acc)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot


def configure_viewer(viewer):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0.15, 0.55, 0.55]
    viewer.cam.distance = 4.1
    viewer.cam.azimuth = -140
    viewer.cam.elevation = -38


def show_viewer_before_planning(demo, viewer, seconds: float = 1.5):
    print("[viewer] MuJoCo visualization is open; showing the initial scene before planning starts.")
    steps = max(1, int(seconds / demo.model.opt.timestep))
    for _ in range(steps):
        if not viewer.is_running():
            raise RuntimeError("MuJoCo viewer was closed before the demo started.")
        demo.lock_base_pose()
        viewer.sync()
        if demo.camera_panel is not None:
            demo.camera_panel.update(demo.data)
        time.sleep(demo.model.opt.timestep / demo.playback_speed)


class ViewerRecorder:
    def __init__(self, model: mujoco.MjModel, out_dir: Path, fps: int = 30, width: int = 1280, height: int = 720):
        self.model = model
        self.out_dir = out_dir
        self.fps = fps
        self.width = width
        self.height = height
        try:
            self.renderer = mujoco.Renderer(model, height=height, width=width)
        except ValueError as exc:
            max_width = int(model.vis.global_.offwidth)
            max_height = int(model.vis.global_.offheight)
            self.width = min(width, max_width)
            self.height = min(height, max_height)
            print(
                f"[record] requested {width}x{height} exceeds MuJoCo offscreen buffer; "
                f"using {self.width}x{self.height}. Original error: {exc}"
            )
            self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.viewer = None
        self.writer = None
        self.path = None
        self.recording = False
        self.last_frame_time = 0.0
        self.frame_interval = 1.0 / fps
        self.frame_count = 0
        self.lock = threading.Lock()
        self.run_requested = False

    def set_viewer(self, viewer):
        self.viewer = viewer

    def handle_key(self, key: int):
        if key in (ord("R"), ord("r")):
            self.start()
            self.run_requested = True
        elif key in (ord("E"), ord("e")):
            self.stop()
        elif key == 32:
            self.run_requested = True

    def start(self):
        with self.lock:
            if self.recording:
                print("[record] already recording; press E to finish.")
                return
            try:
                import cv2
            except Exception as exc:
                print(f"[record] OpenCV is unavailable, cannot record video: {exc}")
                return
            self.out_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.path = self.out_dir / f"left_arm_insert_demo_{stamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.width, self.height))
            if not self.writer.isOpened():
                self.writer = None
                print(f"[record] failed to open video writer: {self.path}")
                return
            self.recording = True
            self.last_frame_time = 0.0
            self.frame_count = 0
            print(f"[record] recording started: {self.path}. Press E to save.")

    def stop(self):
        with self.lock:
            if not self.recording:
                print("[record] not recording; press R to start.")
                return
            self.recording = False
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            print(f"[record] saved {self.frame_count} frame(s): {self.path}")

    def capture(self, data: mujoco.MjData):
        if not self.recording or self.viewer is None:
            return
        now = time.time()
        if now - self.last_frame_time < self.frame_interval:
            return
        with self.lock:
            if not self.recording or self.writer is None:
                return
            try:
                import cv2

                self.renderer.update_scene(data, camera=self.viewer.cam)
                rgb = self.renderer.render()
                self.writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                self.frame_count += 1
                self.last_frame_time = now
            except Exception as exc:
                print(f"[record] capture failed, stopping recording: {exc}")
                self.recording = False
                self.writer.release()
                self.writer = None


class ArucoImageDetector:
    def __init__(self, marker_size: float = 0.08):
        import cv2

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV was built without the aruco module; install opencv-contrib-python.")
        self.cv2 = cv2
        self.marker_size = marker_size
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 35
        self.parameters.adaptiveThreshWinSizeStep = 4
        self.parameters.minMarkerPerimeterRate = 0.015
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    def camera_matrix(self, width: int, height: int, fovy_deg: float) -> np.ndarray:
        fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) * 0.5)
        fx = fy
        return np.array([[fx, 0.0, width * 0.5], [0.0, fy, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)

    def detect(self, rgb: np.ndarray, fovy_deg: float):
        gray = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        annotated = rgb.copy()
        camera_matrix = self.camera_matrix(rgb.shape[1], rgb.shape[0], fovy_deg)
        dist_coeffs = np.zeros(5, dtype=np.float64)
        detections = []
        if ids is not None and len(ids) > 0:
            self.cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            rvecs, tvecs, _ = self.cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_size,
                camera_matrix,
                dist_coeffs,
            )
            for idx, marker_id in enumerate(ids.reshape(-1)):
                pts = corners[idx].reshape(4, 2)
                center = pts.mean(axis=0)
                area = float(abs(self.cv2.contourArea(pts.astype(np.float32))))
                detections.append(
                    {
                        "id": int(marker_id),
                        "center_px": [float(center[0]), float(center[1])],
                        "corners_px": pts.astype(float).round(2).tolist(),
                        "area_px2": area,
                        "rvec": rvecs[idx].reshape(3).astype(float).tolist(),
                        "tvec_m": tvecs[idx].reshape(3).astype(float).tolist(),
                    }
                )
                try:
                    self.cv2.aruco.drawAxis(
                        annotated,
                        camera_matrix,
                        dist_coeffs,
                        rvecs[idx],
                        tvecs[idx],
                        self.marker_size * 0.45,
                    )
                except Exception:
                    pass
        return annotated, detections, camera_matrix, dist_coeffs


class RosArucoPublisher:
    def __init__(self, camera_keys: list[str]):
        import rclpy
        from cv_bridge import CvBridge
        from geometry_msgs.msg import Pose, PoseArray
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String

        if not rclpy.ok():
            rclpy.init(args=None)
        self.rclpy = rclpy
        self.Pose = Pose
        self.PoseArray = PoseArray
        self.CameraInfo = CameraInfo
        self.String = String
        self.bridge = CvBridge()
        self.node = rclpy.create_node("mujoco_aruco_camera_publisher")
        self.publishers = {}
        for key in camera_keys:
            base = f"/mujoco_cameras/{key}"
            self.publishers[key] = {
                "raw": self.node.create_publisher(Image, f"{base}/image_raw", 10),
                "annotated": self.node.create_publisher(Image, f"{base}/aruco/image", 10),
                "detections": self.node.create_publisher(String, f"{base}/aruco/detections", 10),
                "poses": self.node.create_publisher(PoseArray, f"{base}/aruco/poses", 10),
                "info": self.node.create_publisher(CameraInfo, f"{base}/camera_info", 10),
            }
        print("[ros2] publishing MuJoCo camera ArUco topics under /mujoco_cameras/*")

    def publish(
        self,
        key: str,
        rgb: np.ndarray,
        annotated_rgb: np.ndarray,
        detections: list[dict],
        camera_matrix: np.ndarray,
        width: int,
        height: int,
    ):
        pubs = self.publishers[key]
        stamp = self.node.get_clock().now().to_msg()
        frame_id = f"{key}_optical_frame"

        raw_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        raw_msg.header.stamp = stamp
        raw_msg.header.frame_id = frame_id
        pubs["raw"].publish(raw_msg)

        annotated_msg = self.bridge.cv2_to_imgmsg(annotated_rgb, encoding="rgb8")
        annotated_msg.header.stamp = stamp
        annotated_msg.header.frame_id = frame_id
        pubs["annotated"].publish(annotated_msg)

        info = self.CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = frame_id
        info.width = int(width)
        info.height = int(height)
        info.k = camera_matrix.reshape(-1).astype(float).tolist()
        info.p = [
            float(camera_matrix[0, 0]),
            0.0,
            float(camera_matrix[0, 2]),
            0.0,
            0.0,
            float(camera_matrix[1, 1]),
            float(camera_matrix[1, 2]),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        pubs["info"].publish(info)

        text = self.String()
        text.data = json.dumps({"camera": key, "frame_id": frame_id, "markers": detections}, ensure_ascii=False)
        pubs["detections"].publish(text)

        pose_array = self.PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = frame_id
        for det in detections:
            pose = self.Pose()
            pose.position.x, pose.position.y, pose.position.z = [float(v) for v in det["tvec_m"]]
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)
        pubs["poses"].publish(pose_array)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass


class CameraPanel:
    def __init__(
        self,
        model: mujoco.MjModel,
        cameras: list[tuple[str, str, str]],
        width: int = 640,
        height: int = 480,
        enable_ros: bool = True,
        show_window: bool = True,
        key_callback=None,
    ):
        self.cameras = cameras
        self.width = width
        self.height = height
        self.window_name = "MuJoCo ArUco camera views"
        self.enabled = True
        self.ros = None
        self.show_window = show_window
        self.key_callback = key_callback
        try:
            import cv2

            self.cv2 = cv2
            self.renderer = mujoco.Renderer(model, height=height, width=width)
            self.detector = ArucoImageDetector(marker_size=0.08)
            self.camera_fovy = {}
            for _, camera_name, _ in cameras:
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                self.camera_fovy[camera_name] = float(model.cam_fovy[cam_id]) if cam_id >= 0 else 45.0
        except Exception as exc:
            self.enabled = False
            self.cv2 = None
            self.renderer = None
            self.detector = None
            print(f"[camera] camera image panel disabled: {exc}")
        if self.enabled and enable_ros:
            try:
                self.ros = RosArucoPublisher([key for _, _, key in cameras])
            except Exception as exc:
                self.ros = None
                print(f"[ros2] ArUco camera topic publishing disabled: {exc}")

    def update(self, data: mujoco.MjData):
        if not self.enabled or self.cv2 is None or self.renderer is None or self.detector is None:
            return
        try:
            frames = []
            for label, camera_name, topic_key in self.cameras:
                self.renderer.update_scene(data, camera=camera_name)
                rgb = self.renderer.render()
                annotated_rgb, detections, camera_matrix, _ = self.detector.detect(rgb, self.camera_fovy[camera_name])
                if self.ros is not None:
                    self.ros.publish(topic_key, rgb, annotated_rgb, detections, camera_matrix, self.width, self.height)
                if self.show_window:
                    frame = self.cv2.cvtColor(annotated_rgb, self.cv2.COLOR_RGB2BGR)
                    self.cv2.rectangle(frame, (0, 0), (self.width, 30), (20, 20, 20), -1)
                    marker_text = f"{label}  ids={','.join(str(d['id']) for d in detections) if detections else 'none'}"
                    self.cv2.putText(
                        frame,
                        marker_text,
                        (10, 21),
                        self.cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        1,
                        self.cv2.LINE_AA,
                    )
                    frames.append(frame)
            if self.show_window and frames:
                tiled = np.hstack(frames)
                self.cv2.imshow(self.window_name, tiled)
                key = self.cv2.waitKey(1)
                if key != -1 and self.key_callback is not None:
                    self.key_callback(key & 0xFF)
        except Exception as exc:
            print(f"[camera] camera image panel disabled after render failure: {exc}")
            self.enabled = False

    def close(self):
        if self.ros is not None:
            self.ros.close()
            self.ros = None
        if not self.enabled or self.cv2 is None or not self.show_window:
            return
        try:
            self.cv2.destroyWindow(self.window_name)
        except Exception:
            pass


class CasadiInsertDemo:
    def __init__(self, xml_path: Path, arm_name: str, use_vision: bool = True):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.use_vision = use_vision
        self.vision_rng = np.random.default_rng(582)
        self.latest_vision: VisionEstimate | None = None
        self.vision_cache_step = -1

        self.home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self.home_key < 0:
            raise RuntimeError("MJCF must contain a keyframe named 'home'.")
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        mujoco.mj_forward(self.model, self.data)
        self.ctrl = self.model.key_ctrl[self.home_key].copy()

        self.plug_body = self._body_id("ethernet_plug")
        self.plug_free_jid = self._joint_id("ethernet_plug_free")
        self.plug_free_qadr = self.model.jnt_qposadr[self.plug_free_jid]
        self.plug_free_dofadr = self.model.jnt_dofadr[self.plug_free_jid]
        self.plug_grasp_site = self._site_id("rj45_grasp_site")
        self.plug_tip_site = self._site_id("rj45_tip_site")
        self.port_site = self._site_id("ethernet_port_target")
        self.port_entry_site = self._site_id("ethernet_port_entry")
        self.port_marker_site = self._site_id("aruco_port_marker_site")
        self.plug_marker_site = self._site_id("aruco_plug_marker_site")
        self.camera_sites = {
            "head_d405": self._site_id("head_d405_camera_site"),
            "plug_wrist": self._site_id("left_wrist_camera_site"),
            "port_wrist": self._site_id("right_wrist_camera_site"),
        }
        self.plug_latched = False
        self.latched_plug_qpos = None

        if arm_name == "nearest":
            plug_x = float(self.data.xpos[self.plug_body][0])
            arm_name = "left" if plug_x >= 0.0 else "right"
        self.arm = ARMS[arm_name]
        support_arm_name = "right" if arm_name == "left" else "left"
        self.support_arm = ARMS[support_arm_name]

        self.base_jids = [self._joint_id(name) for name in BASE_JOINTS]
        self.base_qadrs = np.array([self.model.jnt_qposadr[jid] for jid in self.base_jids])
        self.base_dofadrs = np.array([self.model.jnt_dofadr[jid] for jid in self.base_jids])
        self.base_actids = np.array([self._act_id(name) for name in BASE_ACTUATORS])
        self.base_hold_target = self.data.qpos[self.base_qadrs].copy()
        self.base_locked = True
        self.recorder: ViewerRecorder | None = None
        self.camera_panel: CameraPanel | None = None
        self.camera_view_stride = 8

        self.arm_jids = [self._joint_id(name) for name in self.arm.joints]
        self.arm_qadrs = np.array([self.model.jnt_qposadr[jid] for jid in self.arm_jids])
        self.arm_dofadrs = np.array([self.model.jnt_dofadr[jid] for jid in self.arm_jids])
        self.arm_actids = np.array([self._act_id(name) for name in self.arm.actuators])
        self.tcp_site = self._site_id(self.arm.tcp_site)
        self.gripper_body = self._body_id(f"{self.arm.prefix}_gripper_robotiq_arg2f_base_link")
        self.grasp_lock_site = self._site_id(f"{self.arm.prefix}_gripper_grasp_lock_site")
        self.grasp_weld_eq = self._eq_id(f"{self.arm.prefix}_plug_grasp_weld")
        self.grasp_weld_active = False
        self.gripper_actids = np.array([self._act_id(name) for name in self.arm.gripper_actuators])

        # In this MJCF the prefix "left" is mounted on the robot-body right side
        # (positive x while facing the table). Use it for the user's "right arm".
        self.right_arm = ARMS["left"]
        self.right_jids = [self._joint_id(name) for name in self.right_arm.joints]
        self.right_qadrs = np.array([self.model.jnt_qposadr[jid] for jid in self.right_jids])
        self.right_dofadrs = np.array([self.model.jnt_dofadr[jid] for jid in self.right_jids])
        self.right_actids = np.array([self._act_id(name) for name in self.right_arm.actuators])
        self.right_tcp_site = self._site_id(self.right_arm.tcp_site)
        self.right_gripper_body = self._body_id("left_gripper_robotiq_arg2f_base_link")
        self.right_grasp_lock_site = self._site_id("left_gripper_grasp_lock_site")
        self.right_grasp_weld_eq = self._eq_id("left_plug_grasp_weld")
        self.right_grasp_weld_active = False
        self.right_gripper_actids = np.array([self._act_id(name) for name in self.right_arm.gripper_actuators])

        self.support_jids = [self._joint_id(name) for name in self.support_arm.joints]
        self.support_qadrs = np.array([self.model.jnt_qposadr[jid] for jid in self.support_jids])
        self.support_dofadrs = np.array([self.model.jnt_dofadr[jid] for jid in self.support_jids])
        self.support_actids = np.array([self._act_id(name) for name in self.support_arm.actuators])
        self.support_tcp_site = self._site_id(self.support_arm.tcp_site)
        self.support_gripper_actids = np.array([self._act_id(name) for name in self.support_arm.gripper_actuators])
        self.left_tcp_site = self._site_id(ARMS["left"].tcp_site)
        self.trace_right_tcp_site = self._site_id(ARMS["right"].tcp_site)

        self.port_hold_site = self._site_id("ethernet_port_hold_target")
        self.step_count = 0
        self.log_stride = 5
        self.playback_speed = 40.0
        self.viewer_sync_stride = 4
        self.trace = {
            "time": [],
            "plug_tip": [],
            "port_target": [],
            "port_entry": [],
            "left_tcp": [],
            "right_tcp": [],
            "base_pose": [],
            "error_xyz": [],
            "error_axial_lateral": [],
            "right_tcp_force": [],
            "right_joint_actual": [],
            "right_joint_target": [],
            "vision_port_error": [],
            "vision_plug_tip_error": [],
            "vision_counts": [],
        }
        self.events = {
            "insert_success": None,
            "unplug_success": None,
        }

        self.settle_scene()
        self._init_vision_reference_transforms()

        print(f"[setup] plug arm: {self.arm.prefix}_ur3 ({self.arm.tcp_site})")
        print(f"[setup] port-support arm: {self.support_arm.prefix}_ur3 ({self.support_arm.tcp_site})")
        port_entry = self.data.site_xpos[self.port_entry_site].copy()
        port_bottom = self.data.site_xpos[self.port_site].copy()
        port_axis = self.data.site_xmat[self.port_site].reshape(3, 3)[:, 0].copy()
        print(f"[setup] RJ45 entry center = {np.round(port_entry, 4)}")
        print(f"[setup] RJ45 bottom target = {np.round(port_bottom, 4)}, insertion axis = {np.round(port_axis, 4)}")
        print(f"[setup] base lock target [x y yaw] = {np.round(self.base_hold_target, 4)}")
        print(f"[vision] ArUco visual servoing {'enabled' if self.use_vision else 'disabled'}; marker size = 0.080 m")
        self.record_sample(force=True)

    def settle_scene(self):
        for _ in range(int(0.35 / self.model.opt.timestep)):
            self.lock_base_pose()
            self.data.ctrl[:] = self.ctrl
            mujoco.mj_step(self.model, self.data)
            self.lock_base_pose()
        mujoco.mj_forward(self.model, self.data)

    def _site_pose(self, site_id: int) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site_xpos[site_id].copy(), self.data.site_xmat[site_id].reshape(3, 3).copy()

    def _body_pose(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[body_id].copy(), self.data.xmat[body_id].reshape(3, 3).copy()

    def _relative_pose(self, parent_pos: np.ndarray, parent_rot: np.ndarray, child_pos: np.ndarray, child_rot: np.ndarray):
        return parent_rot.T @ (child_pos - parent_pos), parent_rot.T @ child_rot

    def _init_vision_reference_transforms(self):
        mujoco.mj_forward(self.model, self.data)
        port_marker_pos, port_marker_rot = self._site_pose(self.port_marker_site)
        plug_marker_pos, plug_marker_rot = self._site_pose(self.plug_marker_site)
        port_target_pos, port_target_rot = self._site_pose(self.port_site)
        port_entry_pos, port_entry_rot = self._site_pose(self.port_entry_site)
        port_hold_pos, port_hold_rot = self._site_pose(self.port_hold_site)
        plug_tip_pos, plug_tip_rot = self._site_pose(self.plug_tip_site)
        plug_grasp_pos, plug_grasp_rot = self._site_pose(self.plug_grasp_site)
        plug_pos, plug_rot = self._body_pose(self.plug_body)

        self.port_marker_to_target_pos, self.port_marker_to_target_rot = self._relative_pose(
            port_marker_pos, port_marker_rot, port_target_pos, port_target_rot
        )
        self.port_marker_to_entry_pos, self.port_marker_to_entry_rot = self._relative_pose(
            port_marker_pos, port_marker_rot, port_entry_pos, port_entry_rot
        )
        self.port_marker_to_hold_pos, self.port_marker_to_hold_rot = self._relative_pose(
            port_marker_pos, port_marker_rot, port_hold_pos, port_hold_rot
        )
        self.plug_marker_to_tip_pos, self.plug_marker_to_tip_rot = self._relative_pose(
            plug_marker_pos, plug_marker_rot, plug_tip_pos, plug_tip_rot
        )
        self.plug_marker_to_grasp_pos, self.plug_marker_to_grasp_rot = self._relative_pose(
            plug_marker_pos, plug_marker_rot, plug_grasp_pos, plug_grasp_rot
        )
        self.plug_marker_to_body_pos, self.plug_marker_to_body_rot = self._relative_pose(
            plug_marker_pos, plug_marker_rot, plug_pos, plug_rot
        )

    def _marker_detection(self, camera_name: str, marker_name: str, marker_site: int):
        camera_site = self.camera_sites[camera_name]
        cam_pos, _ = self._site_pose(camera_site)
        marker_pos, marker_rot = self._site_pose(marker_site)
        distance = float(np.linalg.norm(marker_pos - cam_pos))
        config = {
            ("head_d405", "port"): (1.60, 0.0016, 0.006, 1.0),
            ("head_d405", "plug"): (1.60, 0.0018, 0.007, 1.0),
            ("plug_wrist", "plug"): (0.75, 0.0006, 0.003, 4.0),
            ("plug_wrist", "port"): (0.70, 0.0014, 0.006, 1.6),
            ("port_wrist", "port"): (0.75, 0.0006, 0.003, 4.0),
            ("port_wrist", "plug"): (0.70, 0.0014, 0.006, 1.6),
        }[(camera_name, marker_name)]
        max_range, pos_sigma, rot_sigma, role_weight = config
        if distance > max_range:
            return None
        range_weight = 1.0 / (1.0 + 1.5 * distance)
        noisy_pos = marker_pos + self.vision_rng.normal(0.0, pos_sigma, size=3)
        noisy_rot = marker_rot @ small_angle_rotation(self.vision_rng.normal(0.0, rot_sigma, size=3))
        return noisy_pos, noisy_rot, role_weight * range_weight

    def _fuse_marker_pose(self, marker_name: str, marker_site: int):
        detections = []
        for camera_name in ("head_d405", "plug_wrist", "port_wrist"):
            detection = self._marker_detection(camera_name, marker_name, marker_site)
            if detection is not None:
                detections.append(detection)
        if not detections:
            marker_pos, marker_rot = self._site_pose(marker_site)
            return marker_pos, marker_rot, 0
        weights = [det[2] for det in detections]
        total = float(sum(weights))
        pos = sum(det[0] * det[2] for det in detections) / total
        rot = average_rotations([det[1] for det in detections], weights)
        return pos, rot, len(detections)

    def estimate_visual_pose(self) -> VisionEstimate:
        if self.latest_vision is not None and self.vision_cache_step == self.step_count:
            return self.latest_vision
        if not self.use_vision:
            port_target, port_rot = self._site_pose(self.port_site)
            port_entry, _ = self._site_pose(self.port_entry_site)
            port_hold, _ = self._site_pose(self.port_hold_site)
            plug_tip, _ = self._site_pose(self.plug_tip_site)
            plug_grasp, _ = self._site_pose(self.plug_grasp_site)
            _, plug_rot = self._body_pose(self.plug_body)
            self.latest_vision = VisionEstimate(
                port_target,
                port_entry,
                port_rot,
                port_hold,
                plug_tip,
                plug_grasp,
                plug_rot,
                np.zeros(3),
                np.zeros(3),
                0,
                0,
            )
            self.vision_cache_step = self.step_count
            return self.latest_vision

        port_marker_pos, port_marker_rot, port_count = self._fuse_marker_pose("port", self.port_marker_site)
        plug_marker_pos, plug_marker_rot, plug_count = self._fuse_marker_pose("plug", self.plug_marker_site)
        port_target = port_marker_pos + port_marker_rot @ self.port_marker_to_target_pos
        port_entry = port_marker_pos + port_marker_rot @ self.port_marker_to_entry_pos
        port_hold = port_marker_pos + port_marker_rot @ self.port_marker_to_hold_pos
        port_rot = port_marker_rot @ self.port_marker_to_target_rot
        plug_tip = plug_marker_pos + plug_marker_rot @ self.plug_marker_to_tip_pos
        plug_grasp = plug_marker_pos + plug_marker_rot @ self.plug_marker_to_grasp_pos
        plug_rot = plug_marker_rot @ self.plug_marker_to_body_rot

        true_port_target, _ = self._site_pose(self.port_site)
        true_plug_tip, _ = self._site_pose(self.plug_tip_site)
        estimate = VisionEstimate(
            port_target=port_target,
            port_entry=port_entry,
            port_rot=port_rot,
            port_hold_point=port_hold,
            plug_tip=plug_tip,
            plug_grasp=plug_grasp,
            plug_rot=plug_rot,
            port_error=port_target - true_port_target,
            plug_tip_error=plug_tip - true_plug_tip,
            port_detections=port_count,
            plug_detections=plug_count,
        )
        self.latest_vision = estimate
        self.vision_cache_step = self.step_count
        return estimate

    def current_port_pose(self) -> tuple[np.ndarray, np.ndarray]:
        estimate = self.estimate_visual_pose()
        return estimate.port_target.copy(), estimate.port_rot.copy()

    def current_port_entry(self) -> np.ndarray:
        return self.estimate_visual_pose().port_entry.copy()

    def current_port_hold_point(self) -> np.ndarray:
        return self.estimate_visual_pose().port_hold_point.copy()

    def current_plug_grasp(self) -> np.ndarray:
        # The cable-side ArUco marker is mounted on a raised flag away from the
        # plug, so small marker rotation noise can move the inferred sleeve
        # grasp point by a few millimeters. Keep grasping tied to the physical
        # sleeve site; use vision for plug/port alignment during insertion.
        return self.data.site_xpos[self.plug_grasp_site].copy()

    def _joint_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if idx < 0:
            raise RuntimeError(f"missing joint: {name}")
        return idx

    def _act_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx < 0:
            raise RuntimeError(f"missing actuator: {name}")
        return idx

    def _body_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise RuntimeError(f"missing body: {name}")
        return idx

    def _site_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if idx < 0:
            raise RuntimeError(f"missing site: {name}")
        return idx

    def _eq_id(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if idx < 0:
            raise RuntimeError(f"missing equality: {name}")
        return idx

    def _current_arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadrs].copy()

    def _current_base_pose(self) -> np.ndarray:
        return self.data.qpos[self.base_qadrs].copy()

    def _sim_time(self) -> float:
        return float(self.step_count * self.model.opt.timestep)

    def _port_aligned_tcp_pose_for_tip_for_site(self, tcp_site: int, desired_tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        current_tip = self.data.site_xpos[self.plug_tip_site].copy()
        current_tcp = self.data.site_xpos[tcp_site].copy()
        current_tcp_rot = self.data.site_xmat[tcp_site].reshape(3, 3).copy()
        current_plug_rot = self.data.xmat[self.plug_body].reshape(3, 3).copy()
        _, port_rot = self.current_port_pose()

        tcp_offset_in_plug = current_plug_rot.T @ (current_tcp - current_tip)
        tcp_rot_in_plug = current_plug_rot.T @ current_tcp_rot
        desired_tcp = desired_tip + port_rot @ tcp_offset_in_plug
        desired_tcp_rot = port_rot @ tcp_rot_in_plug
        return desired_tcp, desired_tcp_rot

    def _solve_tcp_q_for(
        self,
        qadrs: np.ndarray,
        jids: list[int],
        dofadrs: np.ndarray,
        tcp_site: int,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        label: str = "tcp",
        rot_weight: float = 0.15,
        pos_tol: float = 0.006,
        rot_tol: float = 0.18,
    ) -> np.ndarray:
        work = mujoco.MjData(self.model)
        work.qpos[:] = self.data.qpos
        work.qvel[:] = 0.0
        mujoco.mj_forward(self.model, work)

        best = (np.inf, work.qpos[qadrs].copy())

        for _ in range(900):
            mujoco.mj_forward(self.model, work)
            tcp_pos = work.site_xpos[tcp_site]
            tcp_rot = work.site_xmat[tcp_site].reshape(3, 3)
            pos_err = target_pos - tcp_pos
            rot_err = self._orientation_error(tcp_rot, target_rot)
            score = np.linalg.norm(pos_err) + rot_weight * np.linalg.norm(rot_err)
            if score < best[0]:
                best = (score, work.qpos[qadrs].copy())
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
                return work.qpos[qadrs].copy()

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, work, jacp, jacr, tcp_site)
            jac = np.vstack([jacp[:, dofadrs], rot_weight * jacr[:, dofadrs]])
            err = np.concatenate([pos_err, rot_weight * rot_err])
            step = jac.T @ np.linalg.solve(jac @ jac.T + 0.04 * np.eye(6), err)
            step = np.clip(step, -0.04, 0.04)

            for adr, jid, delta in zip(qadrs, jids, step):
                lo, hi = self.model.jnt_range[jid]
                work.qpos[adr] = np.clip(work.qpos[adr] + delta, lo, hi)

        print(f"[{label}] IK best residual = {best[0]:.4f}")
        return best[1]

    def plan_grasp(self):
        plug_pos = self.current_plug_grasp()
        target = plug_pos + np.array([0.0, 0.0, 0.020])
        pregrasp = plug_pos + np.array([0.0, 0.0, 0.145])
        plug_yaw = np.pi
        plug_jaw_axis = np.array([np.cos(plug_yaw - np.pi / 2.0), np.sin(plug_yaw - np.pi / 2.0), 0.0])
        z_down = np.array([0.0, 0.0, -1.0])

        n_knots = 24
        mid = n_knots // 2
        q0 = self._current_arm_q()
        base0 = self._current_base_pose()

        q = ca.MX.sym("q", 6, n_knots)
        base = ca.MX.sym("base", 3)
        x = ca.vertcat(ca.reshape(q, -1, 1), base)

        cost = 0
        constraints = []
        lbg = []
        ubg = []

        for i in range(6):
            constraints.append(q[i, 0] - q0[i])
            lbg.append(0.0)
            ubg.append(0.0)

        for k in range(n_knots):
            points = fk_expr(q[:, k], base, self.arm)
            tcp = points["tcp"]
            rot = points["rot"]

            if k >= 2:
                constraints.append(tcp[1] - base[1])
                lbg.append(0.26)
                ubg.append(ca.inf)

            if k >= mid:
                constraints.append(self.arm.side_sign * (tcp[0] - base[0]))
                lbg.append(0.24)
                ubg.append(ca.inf)

            for name in ("wrist1", "wrist2", "wrist3", "tool0"):
                constraints.append(points[name][2])
                lbg.append(0.90)
                ubg.append(ca.inf)

            min_tcp_z = 0.95 if k < mid else 0.87
            constraints.append(tcp[2])
            lbg.append(min_tcp_z)
            ubg.append(ca.inf)

            if k >= mid:
                constraints.append(ca.dot(rot[:, 2], ca.DM(z_down)))
                lbg.append(0.992)
                ubg.append(1.0)
                constraints.append(ca.dot(rot[:, 1], ca.DM(plug_jaw_axis)))
                lbg.append(0.70)
                ubg.append(1.0)

            if k == mid:
                cost += 700.0 * ca.sumsqr(tcp - ca.DM(pregrasp))

            if k == n_knots - 1:
                constraints.append(tcp[0] - target[0])
                lbg.append(-0.008)
                ubg.append(0.008)
                constraints.append(tcp[1] - target[1])
                lbg.append(-0.008)
                ubg.append(0.008)
                constraints.append(tcp[2] - target[2])
                lbg.append(-0.015)
                ubg.append(0.015)
                cost += 12000.0 * ca.sumsqr(tcp - ca.DM(target))
                cost += 2200.0 * (1.0 - ca.dot(rot[:, 2], ca.DM(z_down))) ** 2
                cost += 650.0 * (1.0 - ca.dot(rot[:, 1], ca.DM(plug_jaw_axis))) ** 2

        for k in range(n_knots - 1):
            dq = q[:, k + 1] - q[:, k]
            cost += 18.0 * ca.dot(dq, dq)
        for k in range(n_knots - 2):
            ddq = q[:, k + 2] - 2.0 * q[:, k + 1] + q[:, k]
            cost += 8.0 * ca.dot(ddq, ddq)

        base_delta = base - ca.DM(base0)
        cost += 80.0 * base_delta[0] ** 2 + 80.0 * base_delta[1] ** 2 + 80.0 * base_delta[2] ** 2

        lbx = []
        ubx = []
        x0 = []
        for _ in range(n_knots):
            for jid, val in zip(self.arm_jids, q0):
                lo, hi = self.model.jnt_range[jid]
                lbx.append(float(lo))
                ubx.append(float(hi))
                x0.append(float(val))

        lbx.extend(base0.tolist())
        ubx.extend(base0.tolist())
        x0.extend(base0.tolist())

        solver = ca.nlpsol(
            "planner",
            "ipopt",
            {"x": x, "f": cost, "g": ca.vertcat(*constraints)},
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.max_iter": 900,
                "ipopt.tol": 1e-5,
                "ipopt.acceptable_tol": 1e-4,
            },
        )

        print("[plan] solving CasADi/IPOPT trajectory to the RJ45 plug")
        result = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        status = solver.stats()["return_status"]
        sol = np.array(result["x"]).reshape(-1)
        q_plan = sol[: 6 * n_knots].reshape((6, n_knots), order="F")
        base_target = sol[6 * n_knots : 6 * n_knots + 3]

        final_tcp, final_rot = self._fk_in_mujoco(base_target, q_plan[:, -1])
        pos_err = float(np.linalg.norm(final_tcp - target))
        z_alignment = float(final_rot[:, 2] @ z_down)
        jaw_alignment = float(final_rot[:, 1] @ plug_jaw_axis)
        strict_ok = pos_err < 0.025 and z_alignment > 0.990 and jaw_alignment > 0.70

        print(f"[plan] IPOPT status: {status}")
        print(f"[plan] locked base target [x y yaw] = {np.round(base_target, 4)}")
        print(f"[plan] desired plug grasp tcp = {np.round(target, 4)}")
        print(f"[plan] optimized final tcp   = {np.round(final_tcp, 4)}")
        print(f"[plan] final position error = {pos_err:.3f} m, vertical z-dot = {z_alignment:.4f}, yaw jaw-dot = {jaw_alignment:.4f}")
        if status not in {"Solve_Succeeded", "Solved_To_Acceptable_Level"} and strict_ok:
            print("[plan] using final-pose acceptance despite IPOPT's conservative status")
        if not strict_ok:
            print("[plan] strict vertical-down grasp is not reachable without violating scene safety.")
            print("[plan] The robot/table geometry needs a higher task surface, lower arm mount, or a different grasp posture.")

        return {
            "q_plan": q_plan,
            "base_target": base_target,
            "strict_ok": strict_ok,
            "target": target,
            "final_tcp": final_tcp,
        }

    def plan_port_hold(self, hover_height: float = 0.0):
        hold_point = self.current_port_hold_point()
        target = hold_point + np.array([0.0, 0.0, 0.020 + hover_height])
        if hover_height <= 1e-6:
            prehold = target + np.array([0.0, 0.0, 0.100])
        else:
            prehold = target + np.array([0.0, -0.08, 0.04])
        z_down = np.array([0.0, 0.0, -1.0])
        support_jaw_axis = np.array([0.0, -1.0, 0.0])

        n_knots = 20
        mid = n_knots // 2
        q0 = self.data.qpos[self.support_qadrs].copy()
        base0 = self._current_base_pose()

        q = ca.MX.sym("support_q", 6, n_knots)
        base = ca.MX.sym("support_base", 3)
        x = ca.vertcat(ca.reshape(q, -1, 1), base)
        constraints = []
        lbg = []
        ubg = []
        cost = 0

        for i in range(6):
            constraints.append(q[i, 0] - q0[i])
            lbg.append(0.0)
            ubg.append(0.0)

        for k in range(n_knots):
            points = fk_expr(q[:, k], base, self.support_arm)
            tcp = points["tcp"]
            rot = points["rot"]

            constraints.append(tcp[1] - base[1])
            lbg.append(0.275)
            ubg.append(ca.inf)

            for name in ("wrist1", "wrist2", "wrist3", "tool0"):
                constraints.append(points[name][2])
                lbg.append(0.82)
                ubg.append(ca.inf)

            constraints.append(tcp[2])
            lbg.append(0.91 if k < mid else 0.90)
            ubg.append(ca.inf)

            if k >= mid:
                constraints.append(ca.dot(rot[:, 2], ca.DM(z_down)))
                lbg.append(0.94)
                ubg.append(1.0)
                constraints.append(ca.dot(rot[:, 1], ca.DM(support_jaw_axis)))
                lbg.append(0.60)
                ubg.append(1.0)
                if hover_height <= 1e-6:
                    constraints.append(tcp[0] - target[0])
                    lbg.append(-0.035)
                    ubg.append(0.035)
                    constraints.append(tcp[1] - target[1])
                    lbg.append(-0.035)
                    ubg.append(0.035)

            if k == mid:
                cost += 900.0 * ca.sumsqr(tcp - ca.DM(prehold))

            if k == n_knots - 1:
                constraints.append(tcp[0] - target[0])
                lbg.append(-0.006)
                ubg.append(0.006)
                constraints.append(tcp[1] - target[1])
                lbg.append(-0.006)
                ubg.append(0.006)
                constraints.append(tcp[2] - target[2])
                lbg.append(-0.010)
                ubg.append(0.010)
                cost += 9000.0 * ca.sumsqr(tcp - ca.DM(target))
                cost += 1400.0 * (1.0 - ca.dot(rot[:, 2], ca.DM(z_down))) ** 2
                cost += 520.0 * (1.0 - ca.dot(rot[:, 1], ca.DM(support_jaw_axis))) ** 2

        for k in range(n_knots - 1):
            dq = q[:, k + 1] - q[:, k]
            cost += 20.0 * ca.dot(dq, dq)
        for k in range(n_knots - 2):
            ddq = q[:, k + 2] - 2.0 * q[:, k + 1] + q[:, k]
            cost += 8.0 * ca.dot(ddq, ddq)
        base_delta = base - ca.DM(base0)
        cost += 25.0 * base_delta[0] ** 2 + 70.0 * base_delta[1] ** 2 + 30.0 * base_delta[2] ** 2

        lbx = []
        ubx = []
        x0 = []
        for _ in range(n_knots):
            for jid, val in zip(self.support_jids, q0):
                lo, hi = self.model.jnt_range[jid]
                lbx.append(float(lo))
                ubx.append(float(hi))
                x0.append(float(val))
        lbx.extend(base0.tolist())
        ubx.extend(base0.tolist())
        x0.extend(base0.tolist())

        solver = ca.nlpsol(
            "support_planner",
            "ipopt",
            {"x": x, "f": cost, "g": ca.vertcat(*constraints)},
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.max_iter": 700,
                "ipopt.tol": 1e-5,
                "ipopt.acceptable_tol": 1e-4,
            },
        )

        print("[plan] solving CasADi/IPOPT trajectory to hold the port groove")
        result = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        status = solver.stats()["return_status"]
        sol = np.array(result["x"]).reshape(-1)
        q_plan = sol[: 6 * n_knots].reshape((6, n_knots), order="F")
        base_target = sol[6 * n_knots : 6 * n_knots + 3]

        final_tcp, final_rot = self._fk_arm_in_mujoco(
            self.support_qadrs,
            self.support_tcp_site,
            base_target,
            q_plan[:, -1],
        )
        pos_err = float(np.linalg.norm(final_tcp - target))
        z_alignment = float(final_rot[:, 2] @ z_down)
        jaw_alignment = float(final_rot[:, 1] @ support_jaw_axis)
        strict_ok = pos_err < 0.012 and z_alignment > 0.935 and jaw_alignment > 0.60

        print(f"[plan] support IPOPT status: {status}")
        print(f"[plan] support base target [x y yaw] = {np.round(base_target, 4)}")
        print(f"[plan] desired groove hold point = {np.round(hold_point, 4)}")
        print(f"[plan] desired groove {'hover' if hover_height > 0.0 else 'hold'} tcp = {np.round(target, 4)}")
        print(f"[plan] optimized support tcp  = {np.round(final_tcp, 4)}")
        print(f"[plan] support position error = {pos_err:.3f} m, vertical z-dot = {z_alignment:.4f}, yaw jaw-dot = {jaw_alignment:.4f}")
        if status not in {"Solve_Succeeded", "Solved_To_Acceptable_Level"} and strict_ok:
            print("[plan] using final-pose acceptance for the support arm")

        return {
            "q_plan": q_plan,
            "base_target": base_target,
            "strict_ok": strict_ok,
            "target": target,
            "final_tcp": final_tcp,
        }

    def _fk_in_mujoco(self, base_pose: np.ndarray, q: np.ndarray):
        return self._fk_arm_in_mujoco(self.arm_qadrs, self.tcp_site, base_pose, q)

    def _fk_arm_in_mujoco(self, qadrs: np.ndarray, tcp_site: int, base_pose: np.ndarray, q: np.ndarray):
        work = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, work, self.home_key)
        work.qpos[self.base_qadrs] = base_pose
        work.qpos[qadrs] = q
        mujoco.mj_forward(self.model, work)
        return (
            work.site_xpos[tcp_site].copy(),
            work.site_xmat[tcp_site].reshape(3, 3).copy(),
        )

    def _sync(self, viewer):
        if viewer is not None:
            if self.step_count % self.viewer_sync_stride == 0:
                viewer.sync()
                if self.recorder is not None:
                    self.recorder.capture(self.data)
                if self.camera_panel is not None and self.step_count % self.camera_view_stride == 0:
                    self.camera_panel.update(self.data)
            time.sleep(self.model.opt.timestep / self.playback_speed)

    def step(self, viewer=None):
        if self.plug_latched:
            self.enforce_plug_latch()
        if self.base_locked:
            self.lock_base_pose()
        self.data.ctrl[:] = self.ctrl
        mujoco.mj_step(self.model, self.data)
        if self.base_locked:
            self.lock_base_pose()
        if self.plug_latched:
            self.enforce_plug_latch()
        self.step_count += 1
        if self.step_count % self.log_stride == 0:
            self.record_sample()
        self._sync(viewer)

    def lock_base_pose(self):
        self.data.qpos[self.base_qadrs] = self.base_hold_target
        self.data.qvel[self.base_dofadrs] = 0.0
        self.ctrl[self.base_actids] = 0.0
        self.data.ctrl[self.base_actids] = 0.0

    def record_sample(self, force: bool = False):
        if not force and self.step_count % self.log_stride != 0:
            return
        mujoco.mj_forward(self.model, self.data)
        t = self.step_count * self.model.opt.timestep
        plug_tip = self.data.site_xpos[self.plug_tip_site].copy()
        port_target = self.data.site_xpos[self.port_site].copy()
        port_entry = self.data.site_xpos[self.port_entry_site].copy()
        port_rot = self.data.site_xmat[self.port_site].reshape(3, 3).copy()
        error = plug_tip - port_target
        axial = float(error @ port_rot[:, 0])
        lateral = float(np.linalg.norm(error - axial * port_rot[:, 0]))

        self.trace["time"].append(float(t))
        self.trace["plug_tip"].append(plug_tip)
        self.trace["port_target"].append(port_target)
        self.trace["port_entry"].append(port_entry)
        self.trace["left_tcp"].append(self.data.site_xpos[self.left_tcp_site].copy())
        self.trace["right_tcp"].append(self.data.site_xpos[self.trace_right_tcp_site].copy())
        self.trace["base_pose"].append(self.data.qpos[self.base_qadrs].copy())
        self.trace["error_xyz"].append(error)
        self.trace["error_axial_lateral"].append(np.array([axial, lateral], dtype=float))
        right_tcp_force, _ = self.contact_normal_force(
            [f"{self.right_arm.prefix}_gripper_", "finger_pad_collision"],
            ["rj45", "port_"],
        )
        self.trace["right_tcp_force"].append(np.array([right_tcp_force], dtype=float))
        self.trace["right_joint_actual"].append(self.data.qpos[self.right_qadrs].copy())
        self.trace["right_joint_target"].append(self.ctrl[self.right_actids].copy())
        vision = self.estimate_visual_pose()
        self.trace["vision_port_error"].append(vision.port_error.copy())
        self.trace["vision_plug_tip_error"].append(vision.plug_tip_error.copy())
        self.trace["vision_counts"].append(np.array([vision.port_detections, vision.plug_detections], dtype=float))

    def save_run_outputs(self):
        self.record_sample(force=True)
        times = np.asarray(self.trace["time"], dtype=float)
        if times.size < 2:
            print("[plot] skipped: not enough samples were recorded")
            return

        out_dir = self.xml_path.parent / "insert_demo_logs"
        out_dir.mkdir(parents=True, exist_ok=True)

        plug_tip = _as_array_log(self.trace["plug_tip"])
        port_target = _as_array_log(self.trace["port_target"])
        left_tcp = _as_array_log(self.trace["left_tcp"])
        right_tcp = _as_array_log(self.trace["right_tcp"])
        base_pose = _as_array_log(self.trace["base_pose"])
        error_xyz = _as_array_log(self.trace["error_xyz"])
        axial_lateral = _as_array_log(self.trace["error_axial_lateral"], width=2)
        right_tcp_force = _as_array_log(self.trace["right_tcp_force"], width=1)
        right_joint_actual = _as_array_log(self.trace["right_joint_actual"], width=6)
        right_joint_target = _as_array_log(self.trace["right_joint_target"], width=6)
        vision_port_error = _as_array_log(self.trace["vision_port_error"], width=3)
        vision_plug_tip_error = _as_array_log(self.trace["vision_plug_tip_error"], width=3)
        vision_counts = _as_array_log(self.trace["vision_counts"], width=2)
        error_norm = np.linalg.norm(error_xyz, axis=1)
        force_t_max = 47.0
        force_mask = times <= force_t_max

        csv_path = out_dir / "left_arm_insert_demo_trace.csv"
        table = np.column_stack(
            [
                times,
                plug_tip,
                port_target,
                error_xyz,
                axial_lateral,
                error_norm,
                left_tcp,
                right_tcp,
                base_pose,
                right_tcp_force,
                right_joint_actual,
                right_joint_target,
            ]
        )
        header = (
            "time,"
            "plug_tip_x,plug_tip_y,plug_tip_z,"
            "port_target_x,port_target_y,port_target_z,"
            "error_x,error_y,error_z,error_axial,error_lateral,error_norm,"
            "left_tcp_x,left_tcp_y,left_tcp_z,right_tcp_x,right_tcp_y,right_tcp_z,"
            "base_x,base_y,base_yaw,"
            "right_tcp_force,"
            "right_actual_j1,right_actual_j2,right_actual_j3,right_actual_j4,right_actual_j5,right_actual_j6,"
            "right_target_j1,right_target_j2,right_target_j3,right_target_j4,right_target_j5,right_target_j6"
        )
        np.savetxt(csv_path, table, delimiter=",", header=header, comments="")

        data_dir = self.xml_path.parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(data_dir / "left_arm_insert_demo_trace.csv", table, delimiter=",", header=header, comments="")
        np.savetxt(
            data_dir / "plug_socket_error_xyz.csv",
            np.column_stack([times, np.abs(error_xyz) * 1000.0]),
            delimiter=",",
            header="time,x_error_mm,y_error_mm,z_error_mm",
            comments="",
        )
        np.savetxt(
            data_dir / "right_arm_tcp_force_0_47s.csv",
            np.column_stack([times[force_mask], right_tcp_force[force_mask, 0]]),
            delimiter=",",
            header="time,right_tcp_force_N",
            comments="",
        )
        np.savetxt(
            data_dir / "right_arm_joint_tracking.csv",
            np.column_stack([times, right_joint_actual, right_joint_target]),
            delimiter=",",
            header=(
                "time,"
                "actual_j1,actual_j2,actual_j3,actual_j4,actual_j5,actual_j6,"
                "target_j1,target_j2,target_j3,target_j4,target_j5,target_j6"
            ),
            comments="",
        )
        np.savetxt(
            data_dir / "dual_arm_tcp_trajectory.csv",
            np.column_stack([times, left_tcp, right_tcp]),
            delimiter=",",
            header="time,left_tcp_x,left_tcp_y,left_tcp_z,right_tcp_x,right_tcp_y,right_tcp_z",
            comments="",
        )
        np.savetxt(
            data_dir / "visual_pose_error.csv",
            np.column_stack([times, vision_port_error * 1000.0, vision_plug_tip_error * 1000.0, vision_counts]),
            delimiter=",",
            header=(
                "time,"
                "port_target_error_x_mm,port_target_error_y_mm,port_target_error_z_mm,"
                "plug_tip_error_x_mm,plug_tip_error_y_mm,plug_tip_error_z_mm,"
                "port_marker_camera_count,plug_marker_camera_count"
            ),
            comments="",
        )
        event_rows = []
        for name, event_time in self.events.items():
            if event_time is not None:
                event_rows.append((name, float(event_time)))
        if event_rows:
            with (data_dir / "plot_events.csv").open("w", encoding="utf-8") as f:
                f.write("event,time\n")
                for name, event_time in event_rows:
                    f.write(f"{name},{event_time:.6f}\n")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.collections import LineCollection
            from matplotlib.colors import Normalize
            from matplotlib.lines import Line2D
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
        except Exception as exc:
            print(f"[plot] saved CSV only; matplotlib unavailable: {exc}")
            return

        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
                "font.weight": "normal",
                "font.size": 9.0,
                "axes.labelsize": 10.0,
                "axes.titlesize": 10.0,
                "legend.fontsize": 9.0,
                "xtick.labelsize": 9.0,
                "ytick.labelsize": 9.0,
                "axes.linewidth": 0.85,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        paper_colors = {
            "black": "#222222",
            "blue": "#0072B2",
            "vermillion": "#D55E00",
            "green": "#009E73",
            "purple": "#7E2F8E",
            "gray": "#6B7280",
        }

        def finish_axes(ax):
            ax.tick_params(direction="in", which="both", top=True, right=True, length=3.5, width=0.85)
            ax.tick_params(direction="in", which="minor", top=True, right=True, length=2.0, width=0.65)
            ax.minorticks_on()
            ax.grid(True, which="major", color="0.88", linewidth=0.65)
            for spine in ax.spines.values():
                spine.set_linewidth(0.85)

        error_fig, ax = plt.subplots(figsize=(6.4, 3.15), dpi=300)
        ax.plot(times, np.abs(error_xyz[:, 0]) * 1000.0, color=paper_colors["blue"], linewidth=3.0, label="X error")
        ax.plot(times, np.abs(error_xyz[:, 1]) * 1000.0, color=paper_colors["vermillion"], linewidth=3.0, label="Y error")
        ax.plot(times, np.abs(error_xyz[:, 2]) * 1000.0, color=paper_colors["green"], linewidth=3.0, label="Z error")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Plug-to-socket XYZ error (mm)")
        ax.legend(
            frameon=False,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            handlelength=2.6,
            columnspacing=1.4,
        )
        finish_axes(ax)
        error_fig.tight_layout()
        error_path = out_dir / "left_arm_insert_demo_error.png"
        error_fig.savefig(error_path)
        error_pdf_path = out_dir / "left_arm_insert_demo_error.pdf"
        error_fig.savefig(error_pdf_path, bbox_inches="tight")
        plt.close(error_fig)

        vision_fig, ax_vision = plt.subplots(figsize=(6.4, 3.0), dpi=300)
        port_norm_mm = np.linalg.norm(vision_port_error, axis=1) * 1000.0
        plug_norm_mm = np.linalg.norm(vision_plug_tip_error, axis=1) * 1000.0
        ax_vision.plot(times, port_norm_mm, color=paper_colors["blue"], linewidth=2.8, label="Socket marker fusion")
        ax_vision.plot(times, plug_norm_mm, color=paper_colors["vermillion"], linewidth=2.8, label="Plug marker fusion")
        ax_vision.set_xlabel("Time (s)")
        ax_vision.set_ylabel("Visual pose error (mm)")
        ax_vision.legend(frameon=False, loc="upper right")
        finish_axes(ax_vision)
        vision_fig.tight_layout()
        vision_path = out_dir / "visual_pose_error.png"
        vision_fig.savefig(vision_path)
        vision_pdf_path = out_dir / "visual_pose_error.pdf"
        vision_fig.savefig(vision_pdf_path, bbox_inches="tight")
        plt.close(vision_fig)

        force_fig, ax_force = plt.subplots(figsize=(6.4, 3.0), dpi=300)
        force_t_max = 47.0
        force_mask = times <= force_t_max
        force_times = times[force_mask]
        force_values = right_tcp_force[force_mask, 0]
        ax_force.plot(force_times, force_values, color=paper_colors["green"], linewidth=3.0)
        ax_force.fill_between(force_times, 0.0, force_values, color=paper_colors["green"], alpha=0.12, linewidth=0)
        ax_force.set_xlabel("Time (s)")
        ax_force.set_ylabel("Right TCP force magnitude (N)")
        y_top = max(1.0, float(np.nanmax(force_values)) * 1.16)
        ax_force.set_ylim(-0.04 * y_top, y_top)
        ax_force.set_xlim(float(force_times[0]), force_t_max)

        def force_peak_between(t_min: float, t_max: float) -> tuple[float, float] | None:
            mask = (force_times >= t_min) & (force_times <= min(t_max, force_t_max))
            if not np.any(mask):
                return None
            local_indices = np.flatnonzero(mask)
            peak_idx = local_indices[int(np.argmax(force_values[local_indices]))]
            return float(force_times[peak_idx]), float(force_values[peak_idx])

        insert_event = self.events.get("insert_success")
        unplug_event = self.events.get("unplug_success")
        force_peak_specs = []
        if insert_event is not None:
            peak = force_peak_between(max(float(times[0]), insert_event - 16.0), insert_event + 0.4)
            if peak is not None:
                force_peak_specs.append((peak, "Insertion success trigger", paper_colors["blue"], 0.93, 7, "left"))
        if unplug_event is not None:
            t0 = (insert_event + 1.0) if insert_event is not None else max(float(times[0]), unplug_event - 8.0)
            peak = force_peak_between(t0, unplug_event + 0.4)
            if peak is not None:
                force_peak_specs.append((peak, "Extraction success trigger", paper_colors["vermillion"], 0.78, -7, "right"))

        if force_peak_specs:
            with (data_dir / "right_arm_tcp_force_peak_triggers.csv").open("w", encoding="utf-8") as f:
                f.write("trigger,time,force_N\n")
                for (t_peak, f_peak), label, _, _, _, _ in force_peak_specs:
                    f.write(f"{label},{t_peak:.6f},{f_peak:.6f}\n")

        for (t_peak, f_peak), label, color, y_frac, x_offset, ha in force_peak_specs:
            ax_force.axvline(t_peak, color=color, linewidth=1.8, linestyle=(0, (5, 2.5)), zorder=3)
            ax_force.plot(
                t_peak,
                f_peak,
                marker="o",
                markersize=5.0,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=4,
            )
            ax_force.annotate(
                label,
                xy=(t_peak, f_peak),
                xytext=(t_peak + (0.9 if ha == "left" else -0.9), y_top * y_frac),
                textcoords="data",
                ha=ha,
                va="center",
                color=color,
                fontsize=9.0,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
            )
        finish_axes(ax_force)
        force_fig.tight_layout()
        force_path = out_dir / "right_arm_tcp_force.png"
        force_fig.savefig(force_path)
        force_pdf_path = out_dir / "right_arm_tcp_force.pdf"
        force_fig.savefig(force_pdf_path, bbox_inches="tight")
        plt.close(force_fig)

        joint_names = [f"J{i}" for i in range(1, 7)]
        joint_fig, axes = plt.subplots(2, 3, figsize=(8.8, 4.95), dpi=300, sharex=True)
        axes = axes.ravel()
        for idx, ax_joint in enumerate(axes):
            ax_joint.plot(
                times,
                right_joint_actual[:, idx],
                color=paper_colors["blue"],
                linewidth=2.25,
                label="Actual" if idx == 0 else None,
            )
            ax_joint.plot(
                times,
                right_joint_target[:, idx],
                color=paper_colors["vermillion"],
                linewidth=2.25,
                linestyle=(0, (4, 2)),
                label="Target" if idx == 0 else None,
            )
            ax_joint.set_ylabel(f"{joint_names[idx]} (rad)")
            finish_axes(ax_joint)
        for ax_joint in axes[3:]:
            ax_joint.set_xlabel("Time (s)")
        handles, labels = axes[0].get_legend_handles_labels()
        joint_fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
        joint_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        joint_path = out_dir / "right_arm_joint_tracking.png"
        joint_fig.savefig(joint_path)
        joint_pdf_path = out_dir / "right_arm_joint_tracking.pdf"
        joint_fig.savefig(joint_pdf_path, bbox_inches="tight")
        plt.close(joint_fig)

        def add_gradient_line_2d(ax, xy: np.ndarray, cmap_name: str, label: str):
            if len(xy) < 2:
                return
            segments = np.stack([xy[:-1], xy[1:]], axis=1)
            values = np.linspace(0.15, 1.0, len(segments))
            lc = LineCollection(segments, cmap=cm.get_cmap(cmap_name), norm=Normalize(0.0, 1.0))
            lc.set_array(values)
            lc.set_linewidth(2.8)
            lc.set_label(label)
            ax.add_collection(lc)

        traj_fig = plt.figure(figsize=(12.0, 5.6), dpi=160)
        ax3d = traj_fig.add_subplot(1, 2, 1, projection="3d")
        for points, cmap_name, label in [
            (left_tcp, "Blues", "left TCP"),
            (right_tcp, "Reds", "right TCP"),
        ]:
            if len(points) >= 2:
                segments = np.stack([points[:-1], points[1:]], axis=1)
                values = np.linspace(0.15, 1.0, len(segments))
                lc3d = Line3DCollection(segments, cmap=cm.get_cmap(cmap_name), norm=Normalize(0.0, 1.0))
                lc3d.set_array(values)
                lc3d.set_linewidth(2.6)
                ax3d.add_collection3d(lc3d)
                ax3d.scatter(*points[-1], s=28, color=cm.get_cmap(cmap_name)(1.0), label=label)
        all_tcp = np.vstack([left_tcp, right_tcp])
        span = np.maximum(np.ptp(all_tcp, axis=0), 0.02)
        center = np.mean(all_tcp, axis=0)
        radius = float(max(span) * 0.58)
        ax3d.set_xlim(center[0] - radius, center[0] + radius)
        ax3d.set_ylim(center[1] - radius, center[1] + radius)
        ax3d.set_zlim(center[2] - radius, center[2] + radius)
        ax3d.set_xlabel("x (m)")
        ax3d.set_ylabel("y (m)")
        ax3d.set_zlabel("z (m)")
        ax3d.set_title("Dual-arm TCP Trajectory")
        ax3d.tick_params(direction="in", which="both")
        ax3d.legend(frameon=False, loc="upper left")
        ax3d.view_init(elev=24, azim=-132)

        ax_xy = traj_fig.add_subplot(1, 2, 2)
        add_gradient_line_2d(ax_xy, left_tcp[:, :2], "Blues", "left TCP")
        add_gradient_line_2d(ax_xy, right_tcp[:, :2], "Reds", "right TCP")
        ax_xy.scatter(left_tcp[-1, 0], left_tcp[-1, 1], s=28, color="#1d4ed8")
        ax_xy.scatter(right_tcp[-1, 0], right_tcp[-1, 1], s=28, color="#dc2626")
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.autoscale()
        ax_xy.set_xlabel("x (m)")
        ax_xy.set_ylabel("y (m)")
        ax_xy.set_title("Top View")
        ax_xy.legend(
            handles=[
                Line2D([0], [0], color=cm.get_cmap("Blues")(0.85), lw=3, label="left TCP"),
                Line2D([0], [0], color=cm.get_cmap("Reds")(0.85), lw=3, label="right TCP"),
            ],
            frameon=False,
            loc="best",
        )
        finish_axes(ax_xy)

        traj_fig.tight_layout()
        traj_path = out_dir / "left_arm_insert_demo_tcp_trajectory.png"
        traj_fig.savefig(traj_path)
        plt.close(traj_fig)

        print(f"[plot] saved trace CSV: {csv_path}")
        print(f"[plot] saved error curve: {error_path} and {error_pdf_path}")
        print(f"[plot] saved visual pose error curve: {vision_path} and {vision_pdf_path}")
        print(f"[plot] saved right TCP force curve: {force_path} and {force_pdf_path}")
        print(f"[plot] saved right joint tracking curve: {joint_path} and {joint_pdf_path}")
        print(f"[plot] saved TCP trajectory: {traj_path}")

    def hold_base(self):
        if self.base_locked:
            self.lock_base_pose()
            return
        err = self.base_hold_target - self.data.qpos[self.base_qadrs]
        cmd = np.array([2.2 * err[0], 2.2 * err[1], 3.0 * err[2]])
        lo = self.model.actuator_ctrlrange[self.base_actids, 0]
        hi = self.model.actuator_ctrlrange[self.base_actids, 1]
        self.ctrl[self.base_actids] = np.clip(cmd, lo, hi)

    def move_base(self, target: np.ndarray, viewer=None):
        print(f"[base] kinematic move to table side -> {np.round(target, 4)}")
        start = self.data.qpos[self.base_qadrs].copy()
        self.base_locked = False
        self.ctrl[self.base_actids] = 0.0
        steps = max(1, int(1.15 / self.model.opt.timestep))
        for i in range(steps):
            alpha = smoothstep((i + 1) / steps)
            pose = (1.0 - alpha) * start + alpha * target
            self.data.qpos[self.base_qadrs] = pose
            self.data.qvel[self.base_dofadrs] = 0.0
            self.ctrl[self.base_actids] = 0.0
            self.step(viewer)
        self.ctrl[self.base_actids] = 0.0
        self.base_hold_target = target.copy()
        self.base_locked = True
        self.lock_base_pose()
        for _ in range(160):
            self.hold_base()
            self.step(viewer)
        print(f"[base] arrived and locked at {np.round(self.data.qpos[self.base_qadrs], 4)}")

    def move_arm_path(self, q_plan: np.ndarray, viewer=None):
        print("[arm] tracking optimized position-control trajectory")
        self.move_arm_path_for(q_plan, self.arm_actids, viewer)

    def move_support_arm_path(self, q_plan: np.ndarray, viewer=None):
        print("[support] tracking optimized groove-hold trajectory")
        self.move_arm_path_for(q_plan, self.support_actids, viewer)

    def solve_support_tcp_q(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        label: str = "support_tcp",
    ) -> np.ndarray:
        return self._solve_tcp_q_for(
            self.support_qadrs,
            self.support_jids,
            self.support_dofadrs,
            self.support_tcp_site,
            target_pos,
            target_rot,
            label=label,
            rot_weight=0.32,
            pos_tol=0.006,
            rot_tol=0.11,
        )

    def move_support_tcp_pose(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        duration: float,
        label: str,
        viewer=None,
    ):
        print(f"[support] {label} tcp -> {np.round(target_pos, 4)}")
        q_target = self.solve_support_tcp_q(target_pos, target_rot, label=label)
        self.move_arm_to_for(self.support_qadrs, self.support_actids, q_target, duration=duration, viewer=viewer)

    def move_support_to_port_hold_with_vision(self, viewer=None):
        hold_tcp_offset = np.array([0.0, 0.0, 0.020])
        z_down = np.array([0.0, 0.0, -1.0])
        support_jaw_axis = np.array([0.0, -1.0, 0.0])
        support_x_axis = np.cross(support_jaw_axis, z_down)
        support_rot = rotation_from_axes(support_x_axis, support_jaw_axis, z_down)

        hold_point = self.current_port_hold_point()
        hover = hold_point + hold_tcp_offset + np.array([0.0, 0.0, 0.100])
        self.move_support_tcp_pose(hover, support_rot, duration=0.75, label="Aruco hover 10cm above groove", viewer=viewer)

        print("[support] visual-servo vertical descent to the groove hold point")
        for idx, height in enumerate(np.linspace(0.080, 0.0, 9), start=1):
            estimate = self.estimate_visual_pose()
            hold_point = estimate.port_hold_point.copy()
            target = hold_point + hold_tcp_offset + np.array([0.0, 0.0, height])
            q_target = self.solve_support_tcp_q(target, support_rot, label="support-vision-hold")
            self.move_arm_to_for(self.support_qadrs, self.support_actids, q_target, duration=0.22, viewer=viewer)
            actual_tcp = self.data.site_xpos[self.support_tcp_site].copy()
            actual_hold = self.data.site_xpos[self.port_hold_site].copy()
            visual_err = np.linalg.norm(estimate.port_error)
            print(
                f"[support-vision] descend {idx:02d}: height={height * 1000.0:4.0f} mm, "
                f"port marker cams={estimate.port_detections}, "
                f"vision_port_err={visual_err * 1000.0:.1f} mm, "
                f"tcp_to_hold={np.linalg.norm(actual_tcp - (actual_hold + hold_tcp_offset)) * 1000.0:.1f} mm"
            )

        actual_tcp = self.data.site_xpos[self.support_tcp_site].copy()
        actual_hold = self.data.site_xpos[self.port_hold_site].copy()
        final_err = float(np.linalg.norm(actual_tcp - (actual_hold + hold_tcp_offset)))
        if final_err > 0.025:
            print(
                f"[support-vision] Cartesian descent residual is {final_err * 1000.0:.1f} mm; "
                "replanning a short ArUco-corrected vertical approach from above"
            )
            descent_plan = self.plan_port_hold(hover_height=0.0)
            self.move_support_arm_path(descent_plan["q_plan"], viewer)

    def move_arm_path_for(self, q_plan: np.ndarray, actids: np.ndarray, viewer=None):
        segment_duration = 0.26
        segment_steps = max(1, int(segment_duration / self.model.opt.timestep))
        for k in range(q_plan.shape[1] - 1):
            start = q_plan[:, k]
            goal = q_plan[:, k + 1]
            for i in range(segment_steps):
                alpha = smoothstep((i + 1) / segment_steps)
                self.hold_base()
                self.ctrl[actids] = (1.0 - alpha) * start + alpha * goal
                self.step(viewer)
        self.ctrl[actids] = q_plan[:, -1]
        for _ in range(int(0.45 / self.model.opt.timestep)):
            self.hold_base()
            self.step(viewer)

    def close_gripper(self, viewer=None):
        print(f"[gripper] closing {self.arm.prefix}_gripper")
        self.close_gripper_for(self.gripper_actids, viewer)
        self.print_contact_summary(
            "[gripper] plug contact",
            [f"{self.arm.prefix}_gripper_", "finger_pad_collision"],
            ["rj45_plug", "rj45_grip_"],
        )
        return self.engage_grasp_weld_if_contact()

    def close_support_gripper(self, viewer=None):
        print(f"[support] firmly closing {self.support_arm.prefix}_gripper on the groove housing")
        support_target = np.array([0.120, 0.0825, 0.120, 0.120, 0.0825, 0.120])
        self.close_gripper_for(self.support_gripper_actids, viewer, support_target)
        self.print_contact_summary(
            "[support] groove contact",
            [f"{self.support_arm.prefix}_gripper_", "finger_pad_collision"],
            ["port_slot_", "port_lip_", "port_module_"],
        )

    def right_plug_grip_contact_ok(self) -> tuple[bool, float, list[str]]:
        force, pairs = self.contact_normal_force(
            [f"{self.right_arm.prefix}_gripper_", "finger_pad_collision"],
            ["rj45_plug_", "rj45_grip_"],
        )
        left_pad = any(f"{self.right_arm.prefix}_gripper_left_inner_finger_pad" in pair for pair in pairs)
        right_pad = any(f"{self.right_arm.prefix}_gripper_right_inner_finger_pad" in pair for pair in pairs)
        grip_feature = any(("rj45_grip_" in pair or "rj45_plug_body" in pair) for pair in pairs)
        pad_contact = left_pad or right_pad
        return force > 12.0 and pad_contact and grip_feature, force, pairs

    def right_gripper_closed(self, viewer=None):
        print("[unplug] closing right gripper on the RJ45 latch")
        right_target = np.array([0.22, 0.15, 0.22, 0.22, 0.15, 0.22])
        self.close_gripper_for(self.right_gripper_actids, viewer, right_target)
        self.print_contact_summary(
            "[unplug] right grip contact",
            [f"{self.right_arm.prefix}_gripper_", "finger_pad_collision"],
            ["rj45_plug", "rj45_grip_"],
        )
        return self.right_plug_grip_contact_ok()

    def engage_right_grasp_weld_if_contact(self) -> bool:
        ok, force, pairs = self.right_plug_grip_contact_ok()
        if not ok:
            print(
                "[unplug] right grip failed: need firm contact on the RJ45 latch/body "
                f"before extraction; contact force = {force:.2f} N"
            )
            return False

        mujoco.mj_forward(self.model, self.data)
        body_pos = self.data.xpos[self.right_gripper_body].copy()
        body_rot = self.data.xmat[self.right_gripper_body].reshape(3, 3).copy()
        grasp_pos = self.data.site_xpos[self.plug_grasp_site].copy()
        grasp_rot = self.data.site_xmat[self.plug_grasp_site].reshape(3, 3).copy()

        local_pos = body_rot.T @ (grasp_pos - body_pos)
        local_rot = body_rot.T @ grasp_rot
        local_quat = np.zeros(4)
        mujoco.mju_mat2Quat(local_quat, np.ascontiguousarray(local_rot.reshape(9)))

        self.model.site_pos[self.right_grasp_lock_site] = local_pos
        self.model.site_quat[self.right_grasp_lock_site] = local_quat
        self.data.eq_active[self.right_grasp_weld_eq] = 1
        self.right_grasp_weld_active = True
        mujoco.mj_forward(self.model, self.data)

        print(
            "[unplug] contact-verified right grasp lock engaged "
            f"(force {force:.2f} N, {len(pairs)} contacts); plug will move with the right arm"
        )
        return True

    def release_right_grasp_weld(self):
        if not self.right_grasp_weld_active:
            return
        self.data.eq_active[self.right_grasp_weld_eq] = 0
        self.right_grasp_weld_active = False
        mujoco.mj_forward(self.model, self.data)
        print("[unplug] right grasp lock released")

    def release_port_latch(self):
        if not self.plug_latched:
            return
        self.plug_latched = False
        self.latched_plug_qpos = None
        self.data.qvel[self.plug_free_dofadr : self.plug_free_dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        print("[latch] RJ45 clip popped open; plug is now free to be withdrawn")

    def open_grippers(self, viewer=None):
        print("[release] opening both grippers")
        if not self.plug_latched:
            self.release_grasp_weld()
        left = np.array([self._act_id(name) for name in ARMS["left"].gripper_actuators])
        right = np.array([self._act_id(name) for name in ARMS["right"].gripper_actuators])
        actids = np.concatenate([left, right])
        start = self.ctrl[actids].copy()
        target = np.zeros_like(start)
        steps = max(1, int(0.45 / self.model.opt.timestep))
        for i in range(steps):
            alpha = (i + 1) / steps
            self.ctrl[actids] = (1.0 - alpha) * start + alpha * target
            self.hold_base()
            self.step(viewer)

    def open_right_gripper(self, viewer=None):
        print("[release] opening right gripper")
        start = self.ctrl[self.right_gripper_actids].copy()
        target = np.zeros_like(start)
        steps = max(1, int(0.45 / self.model.opt.timestep))
        for i in range(steps):
            alpha = (i + 1) / steps
            self.ctrl[self.right_gripper_actids] = (1.0 - alpha) * start + alpha * target
            self.hold_base()
            self.step(viewer)

    def plug_grip_contact_ok(self) -> tuple[bool, float, list[str]]:
        force, pairs = self.contact_normal_force(
            [f"{self.arm.prefix}_gripper_", "finger_pad_collision"],
            ["rj45_plug", "rj45_grip_"],
        )
        left_pad = any(f"{self.arm.prefix}_gripper_left_inner_finger_pad" in pair for pair in pairs)
        right_pad = any(f"{self.arm.prefix}_gripper_right_inner_finger_pad" in pair for pair in pairs)
        grip_feature = any(("rj45_grip_" in pair or "rj45_plug_body" in pair) for pair in pairs)
        return force > 3.0 and left_pad and right_pad and grip_feature, force, pairs

    def engage_grasp_weld_if_contact(self) -> bool:
        ok, force, pairs = self.plug_grip_contact_ok()
        if not ok:
            print(
                "[gripper] grip failed: need both finger pads on the RJ45 sleeve/body "
                f"before transport; contact force = {force:.2f} N"
            )
            return False

        mujoco.mj_forward(self.model, self.data)
        body_pos = self.data.xpos[self.gripper_body].copy()
        body_rot = self.data.xmat[self.gripper_body].reshape(3, 3).copy()
        grasp_pos = self.data.site_xpos[self.plug_grasp_site].copy()
        grasp_rot = self.data.site_xmat[self.plug_grasp_site].reshape(3, 3).copy()

        local_pos = body_rot.T @ (grasp_pos - body_pos)
        local_rot = body_rot.T @ grasp_rot
        local_quat = np.zeros(4)
        mujoco.mju_mat2Quat(local_quat, np.ascontiguousarray(local_rot.reshape(9)))

        self.model.site_pos[self.grasp_lock_site] = local_pos
        self.model.site_quat[self.grasp_lock_site] = local_quat
        self.data.eq_active[self.grasp_weld_eq] = 1
        self.grasp_weld_active = True
        mujoco.mj_forward(self.model, self.data)

        print(
            "[gripper] contact-verified grasp lock engaged "
            f"(force {force:.2f} N, {len(pairs)} contacts); plug remains a colliding free body"
        )
        return True

    def release_grasp_weld(self):
        if not self.grasp_weld_active:
            return
        self.data.eq_active[self.grasp_weld_eq] = 0
        self.grasp_weld_active = False
        mujoco.mj_forward(self.model, self.data)
        print("[gripper] grasp lock released")

    def close_gripper_for(self, actids: np.ndarray, viewer=None, target: np.ndarray | None = None):
        start = self.ctrl[actids].copy()
        if target is None:
            target = np.array([0.26, 0.18, 0.26, 0.26, 0.18, 0.26])
        lo = self.model.actuator_ctrlrange[actids, 0]
        hi = self.model.actuator_ctrlrange[actids, 1]
        target = np.clip(target, lo, hi)
        steps = max(1, int(0.55 / self.model.opt.timestep))
        for i in range(steps):
            alpha = (i + 1) / steps
            self.ctrl[actids] = (1.0 - alpha) * start + alpha * target
            self.hold_base()
            self.step(viewer)

    def _geom_name(self, geom_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""

    def _contact_matches(self, g1: str, g2: str, needles_a: list[str], needles_b: list[str]) -> bool:
        a1 = all(s in g1 for s in needles_a)
        b2 = any(s in g2 for s in needles_b)
        a2 = all(s in g2 for s in needles_a)
        b1 = any(s in g1 for s in needles_b)
        return (a1 and b2) or (a2 and b1)

    def contact_normal_force(self, needles_a: list[str], needles_b: list[str]) -> tuple[float, list[str]]:
        total = 0.0
        pairs = []
        wrench = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1 = self._geom_name(con.geom1)
            g2 = self._geom_name(con.geom2)
            if not self._contact_matches(g1, g2, needles_a, needles_b):
                continue
            mujoco.mj_contactForce(self.model, self.data, i, wrench)
            total += abs(float(wrench[0]))
            pairs.append(f"{g1}<->{g2}")
        return total, pairs

    def print_contact_summary(self, label: str, needles_a: list[str], needles_b: list[str]):
        force, pairs = self.contact_normal_force(needles_a, needles_b)
        if pairs:
            shown = ", ".join(sorted(set(pairs))[:4])
            print(f"{label}: {len(pairs)} contact(s), normal force {force:.2f} N [{shown}]")
        else:
            print(f"{label}: no physical contact detected")

    def port_bottom_force(self) -> float:
        force, _ = self.contact_normal_force(["rj45_plug_nose"], ["port_slot_backstop"])
        return force

    def port_socket_force(self) -> tuple[float, list[str]]:
        return self.contact_normal_force(["rj45_plug_nose"], ["port_slot_", "port_lip_"])

    def plug_port_alignment_metrics(self, use_vision: bool = False) -> dict[str, float | np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        if use_vision:
            estimate = self.estimate_visual_pose()
            port_target = estimate.port_target.copy()
            port_rot = estimate.port_rot.copy()
            plug_tip = estimate.plug_tip.copy()
            plug_rot = estimate.plug_rot.copy()
        else:
            port_target = self.data.site_xpos[self.port_site].copy()
            port_rot = self.data.site_xmat[self.port_site].reshape(3, 3).copy()
            plug_tip = self.data.site_xpos[self.plug_tip_site].copy()
            plug_rot = self.data.xmat[self.plug_body].reshape(3, 3).copy()
        err_world = plug_tip - port_target
        err_port = port_rot.T @ err_world
        lateral_error = float(np.linalg.norm(err_port[1:]))
        return {
            "tip": plug_tip,
            "target": port_target,
            "err_world": err_world,
            "err_port": err_port,
            "tip_error": float(np.linalg.norm(err_world)),
            "axial_error": float(err_port[0]),
            "lateral_error": lateral_error,
            "vertical_error": float(err_port[2]),
            "x_dot": float(plug_rot[:, 0] @ port_rot[:, 0]),
            "y_dot": float(plug_rot[:, 1] @ port_rot[:, 1]),
            "z_dot": float(plug_rot[:, 2] @ port_rot[:, 2]),
        }

    def strict_insert_alignment_ok(self, metrics: dict[str, float | np.ndarray] | None = None) -> bool:
        if metrics is None:
            metrics = self.plug_port_alignment_metrics()
        return (
            abs(float(metrics["axial_error"])) <= STRICT_INSERT_MAX_AXIAL_ERROR
            and float(metrics["lateral_error"]) <= STRICT_INSERT_MAX_LATERAL_ERROR
            and abs(float(metrics["vertical_error"])) <= STRICT_INSERT_MAX_VERTICAL_ERROR
            and float(metrics["tip_error"]) <= STRICT_INSERT_MAX_TIP_ERROR
            and float(metrics["x_dot"]) >= STRICT_INSERT_MIN_X_DOT
            and float(metrics["y_dot"]) >= STRICT_INSERT_MIN_YZ_DOT
            and float(metrics["z_dot"]) >= STRICT_INSERT_MIN_YZ_DOT
        )

    def print_insert_alignment(self, label: str, metrics: dict[str, float | np.ndarray] | None = None):
        if metrics is None:
            metrics = self.plug_port_alignment_metrics()
        err_port = np.asarray(metrics["err_port"], dtype=float)
        print(
            f"[align] {label}: "
            f"tip_err={float(metrics['tip_error']) * 1000.0:.1f} mm, "
            f"port_xyz_err=[{err_port[0] * 1000.0:.1f}, {err_port[1] * 1000.0:.1f}, {err_port[2] * 1000.0:.1f}] mm, "
            f"lateral={float(metrics['lateral_error']) * 1000.0:.1f} mm, "
            f"axis_dot xyz=[{float(metrics['x_dot']):.4f}, {float(metrics['y_dot']):.4f}, {float(metrics['z_dot']):.4f}]"
        )

    def latch_plug_to_port(self) -> bool:
        metrics = self.plug_port_alignment_metrics()
        if not self.strict_insert_alignment_ok(metrics):
            self.print_insert_alignment("latch refused", metrics)
            print("[latch] refused: plug and port are not strictly aligned, so the gripper will keep holding the plug")
            return False
        port_target, port_rot = self.current_port_pose()
        plug_tip_local = np.array([0.043, 0.0, 0.0])
        qpos = self.data.qpos[self.plug_free_qadr : self.plug_free_qadr + 7].copy()
        qpos[:3] = port_target - port_rot @ plug_tip_local
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(port_rot.reshape(9)))
        qpos[3:7] = quat
        self.latched_plug_qpos = qpos
        self.plug_latched = True
        self.release_grasp_weld()
        self.enforce_plug_latch()
        self.events["insert_success"] = self._sim_time()
        self.print_insert_alignment("latched")
        print("[latch] RJ45 plug locked in the port after strict XYZ alignment")
        return True

    def enforce_plug_latch(self):
        if self.latched_plug_qpos is None:
            return
        self.data.qpos[self.plug_free_qadr : self.plug_free_qadr + 7] = self.latched_plug_qpos
        self.data.qvel[self.plug_free_dofadr : self.plug_free_dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _orientation_error(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 0.5 * (
            np.cross(current[:, 0], target[:, 0])
            + np.cross(current[:, 1], target[:, 1])
            + np.cross(current[:, 2], target[:, 2])
        )

    def solve_tcp_q(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        label: str = "tcp",
        rot_weight: float = 0.15,
        pos_tol: float = 0.006,
        rot_tol: float = 0.18,
    ) -> np.ndarray:
        return self._solve_tcp_q_for(
            self.arm_qadrs,
            self.arm_jids,
            self.arm_dofadrs,
            self.tcp_site,
            target_pos,
            target_rot,
            label=label,
            rot_weight=rot_weight,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
        )

    def solve_right_tcp_q(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        label: str = "right_tcp",
        rot_weight: float = 0.15,
        pos_tol: float = 0.006,
        rot_tol: float = 0.18,
        flip_wrist: bool = False,
    ) -> np.ndarray:
        if flip_wrist:
            target_rot = target_rot @ np.diag([-1.0, -1.0, 1.0])
        q_target = self._solve_tcp_q_for(
            self.right_qadrs,
            self.right_jids,
            self.right_dofadrs,
            self.right_tcp_site,
            target_pos,
            target_rot,
            label=label,
            rot_weight=rot_weight,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
        )
        return q_target

    def move_arm_to_for(self, qadrs: np.ndarray, actids: np.ndarray, q_target: np.ndarray, duration: float, viewer=None):
        q_start = self.data.qpos[qadrs].copy()
        steps = max(1, int(duration / self.model.opt.timestep))
        for i in range(steps):
            alpha = smoothstep((i + 1) / steps)
            self.hold_base()
            self.ctrl[actids] = (1.0 - alpha) * q_start + alpha * q_target
            self.step(viewer)

    def move_arm_to(self, q_target: np.ndarray, duration: float, viewer=None):
        self.move_arm_to_for(self.arm_qadrs, self.arm_actids, q_target, duration, viewer)

    def move_right_arm_to(self, q_target: np.ndarray, duration: float, viewer=None):
        self.move_arm_to_for(self.right_qadrs, self.right_actids, q_target, duration, viewer)

    def move_right_tcp_pose(self, target_pos: np.ndarray, target_rot: np.ndarray, duration: float, label: str, viewer=None):
        print(f"[right] {label} tcp -> {np.round(target_pos, 4)}")
        q_target = self.solve_right_tcp_q(
            target_pos,
            target_rot,
            label=label,
            rot_weight=0.35,
            pos_tol=0.008,
            rot_tol=0.10,
            flip_wrist=True,
        )
        self.move_right_arm_to(q_target, duration=duration, viewer=viewer)

    def reset_arms(self, viewer=None):
        print("[reset] moving both arms back to home")
        left_actids = np.array([self._act_id(name) for name in ARMS["left"].actuators])
        right_actids = np.array([self._act_id(name) for name in ARMS["right"].actuators])
        left_qadrs = np.array([self.model.jnt_qposadr[self._joint_id(name)] for name in ARMS["left"].joints])
        right_qadrs = np.array([self.model.jnt_qposadr[self._joint_id(name)] for name in ARMS["right"].joints])
        left_start = self.data.qpos[left_qadrs].copy()
        right_start = self.data.qpos[right_qadrs].copy()
        left_home = np.array(ARMS["left"].home_q)
        right_home = np.array(ARMS["right"].home_q)
        steps = max(1, int(2.0 / self.model.opt.timestep))
        for i in range(steps):
            alpha = smoothstep((i + 1) / steps)
            self.hold_base()
            self.ctrl[left_actids] = (1.0 - alpha) * left_start + alpha * left_home
            self.ctrl[right_actids] = (1.0 - alpha) * right_start + alpha * right_home
            self.step(viewer)

    def _port_aligned_tcp_pose_for_tip(self, desired_tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        current_tip = self.data.site_xpos[self.plug_tip_site].copy()
        current_tcp = self.data.site_xpos[self.tcp_site].copy()
        current_tcp_rot = self.data.site_xmat[self.tcp_site].reshape(3, 3).copy()
        current_plug_rot = self.data.xmat[self.plug_body].reshape(3, 3).copy()
        port_rot = self.data.site_xmat[self.port_site].reshape(3, 3).copy()

        tcp_offset_in_plug = current_plug_rot.T @ (current_tcp - current_tip)
        tcp_rot_in_plug = current_plug_rot.T @ current_tcp_rot
        desired_tcp = desired_tip + port_rot @ tcp_offset_in_plug
        desired_tcp_rot = port_rot @ tcp_rot_in_plug
        return desired_tcp, desired_tcp_rot

    def solve_q_for_desired_plug_tip(self, desired_tip: np.ndarray, approach_offset: float = 0.0) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        _, port_rot = self.current_port_pose()
        desired_tip = desired_tip - port_rot[:, 0] * approach_offset
        desired_tcp, desired_tcp_rot = self._port_aligned_tcp_pose_for_tip_for_site(self.tcp_site, desired_tip)
        return self.solve_tcp_q(
            desired_tcp,
            desired_tcp_rot,
            label="insert",
            rot_weight=0.42,
            pos_tol=0.0035,
            rot_tol=0.055,
        )

    def solve_right_q_for_desired_plug_tip(self, desired_tip: np.ndarray, approach_offset: float = 0.0) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        _, port_rot = self.current_port_pose()
        desired_tip = desired_tip - port_rot[:, 0] * approach_offset
        desired_tcp, desired_tcp_rot = self._port_aligned_tcp_pose_for_tip_for_site(self.right_tcp_site, desired_tip)
        return self.solve_right_tcp_q(
            desired_tcp,
            desired_tcp_rot,
            label="right-unplug",
            rot_weight=0.22,
            pos_tol=0.007,
            rot_tol=0.14,
        )

    def move_plug_tip_waypoint(self, desired_tip: np.ndarray, duration: float, label: str, viewer=None):
        print(f"[insert] {label} tip -> {np.round(desired_tip, 4)}")
        q_target = self.solve_q_for_desired_plug_tip(desired_tip, approach_offset=0.0)
        self.move_arm_to(q_target, duration=duration, viewer=viewer)
        actual_tip = self.data.site_xpos[self.plug_tip_site].copy()
        _, port_rot = self.current_port_pose()
        port_axis = port_rot[:, 0].copy()
        plug_axis = self.data.xmat[self.plug_body].reshape(3, 3)[:, 0].copy()
        horizontal_err = abs(float(plug_axis[2]))
        axis_dot = float(plug_axis @ port_axis)
        print(
            f"[insert] {label} actual tip = {np.round(actual_tip, 4)}, "
            f"error = {np.linalg.norm(actual_tip - desired_tip):.3f} m, "
            f"axis-dot = {axis_dot:.3f}, vertical-axis-leak = {horizontal_err:.3f}"
        )

    def move_right_plug_tip_waypoint(self, desired_tip: np.ndarray, duration: float, label: str, viewer=None):
        print(f"[unplug] {label} tip -> {np.round(desired_tip, 4)}")
        q_target = self.solve_right_q_for_desired_plug_tip(desired_tip, approach_offset=0.0)
        self.move_right_arm_to(q_target, duration=duration, viewer=viewer)
        actual_tip = self.data.site_xpos[self.plug_tip_site].copy()
        _, port_rot = self.current_port_pose()
        port_axis = port_rot[:, 0].copy()
        plug_axis = self.data.xmat[self.plug_body].reshape(3, 3)[:, 0].copy()
        horizontal_err = abs(float(plug_axis[2]))
        axis_dot = float(plug_axis @ port_axis)
        print(
            f"[unplug] {label} actual tip = {np.round(actual_tip, 4)}, "
            f"error = {np.linalg.norm(actual_tip - desired_tip):.3f} m, "
            f"axis-dot = {axis_dot:.3f}, vertical-axis-leak = {horizontal_err:.3f}"
        )

    def insert_plug_into_port(self, viewer=None):
        if not self.grasp_weld_active:
            print("[insert] abort: plug is not locked in the gripper; refusing to fake transport")
            return False
        port_target, port_rot = self.current_port_pose()
        port_entry = self.current_port_entry()
        insertion_axis = port_rot[:, 0].copy()
        insertion_bias_port = np.zeros(3)
        horizontal_z = float(port_target[2])
        tip_before = self.data.site_xpos[self.plug_tip_site].copy()
        print(f"[insert] tip before insertion = {np.round(tip_before, 4)}")
        print(f"[insert] port entry           = {np.round(port_entry, 4)}")
        print(f"[insert] port target          = {np.round(port_target, 4)}")
        print("[insert] strict latch requires plug tip position and plug XYZ axes to match the socket frame")

        def current_lateral_bias_world() -> np.ndarray:
            return port_rot @ insertion_bias_port

        def update_insertion_bias(metrics: dict[str, float | np.ndarray]):
            err_port = np.asarray(metrics["err_port"], dtype=float)
            insertion_bias_port[1:] -= 0.72 * err_port[1:]
            insertion_bias_port[1:] = np.clip(insertion_bias_port[1:], -0.020, 0.020)
            print(f"[align] next Y/Z compensation in port frame = {np.round(insertion_bias_port[1:] * 1000.0, 1)} mm")

        lift_tip = port_entry - insertion_axis * 0.055
        lift_tip[1] = port_target[1]
        lift_tip[2] = horizontal_z
        self.move_plug_tip_waypoint(lift_tip, duration=0.8, label="Z-align", viewer=viewer)

        preinsert_tip = port_entry - insertion_axis * 0.045 + current_lateral_bias_world()
        preinsert_tip[2] = horizontal_z
        self.move_plug_tip_waypoint(preinsert_tip, duration=1.8, label="horizontal pre-insert", viewer=viewer)
        update_insertion_bias(self.plug_port_alignment_metrics(use_vision=True))

        bottom_force = 0.0
        inserted = False
        start_gap = max(0.030, float(np.dot(preinsert_tip - port_target, -insertion_axis)))
        approach = start_gap
        max_cycles = 42
        print(f"[vision-servo] starting closed-loop insertion from {approach * 1000.0:.1f} mm before the socket")
        for cycle in range(max_cycles):
            vision_metrics = self.plug_port_alignment_metrics(use_vision=True)
            actual_metrics = self.plug_port_alignment_metrics(use_vision=False)
            vision_err_port = np.asarray(vision_metrics["err_port"], dtype=float)
            lateral_norm = float(np.linalg.norm(vision_err_port[1:]))

            # Visual servoing keeps the plug centered in the socket frame before it advances.
            correction = np.clip(0.58 * vision_err_port[1:], -0.0035, 0.0035)
            insertion_bias_port[1:] -= correction
            insertion_bias_port[1:] = np.clip(insertion_bias_port[1:], -0.022, 0.022)
            if lateral_norm < 0.0030 and abs(float(vision_err_port[2])) < 0.0025:
                advance = 0.0030
            elif lateral_norm < 0.0060:
                advance = 0.0018
            else:
                advance = 0.0007
            approach = max(-0.0100, approach - advance)

            port_target, port_rot = self.current_port_pose()
            insertion_axis = port_rot[:, 0].copy()
            desired_tip = port_target - insertion_axis * approach + port_rot @ insertion_bias_port
            q_insert = self.solve_q_for_desired_plug_tip(desired_tip, approach_offset=0.0)
            self.move_arm_to(q_insert, duration=0.22, viewer=viewer)
            for _ in range(int(0.05 / self.model.opt.timestep)):
                self.hold_base()
                self.step(viewer)

            bottom_force = self.port_bottom_force()
            socket_force, socket_pairs = self.port_socket_force()
            tip_now = self.data.site_xpos[self.plug_tip_site].copy()
            actual_port_target = self.data.site_xpos[self.port_site].copy()
            err_now = np.linalg.norm(tip_now - actual_port_target)
            seated_force = max(bottom_force, socket_force) if err_now < 0.015 else bottom_force
            if cycle % 3 == 0 or approach <= 0.002:
                print(
                    f"[vision-servo] cycle {cycle + 1:02d}: approach={approach * 1000.0:5.1f} mm, "
                    f"vision_yz_err=[{vision_err_port[1] * 1000.0:5.1f}, {vision_err_port[2] * 1000.0:5.1f}] mm, "
                    f"bias_yz=[{insertion_bias_port[1] * 1000.0:5.1f}, {insertion_bias_port[2] * 1000.0:5.1f}] mm, "
                    f"force={seated_force:.2f} N"
                )
            if socket_pairs:
                print(f"[force] socket contacts: {', '.join(sorted(set(socket_pairs))[:4])}")

            vision_metrics = self.plug_port_alignment_metrics(use_vision=True)
            actual_metrics = self.plug_port_alignment_metrics(use_vision=False)
            if cycle % 3 == 0 or approach <= 0.002:
                self.print_insert_alignment("vision servo", vision_metrics)
                self.print_insert_alignment("actual servo", actual_metrics)
            if self.strict_insert_alignment_ok(vision_metrics) and self.strict_insert_alignment_ok(actual_metrics):
                inserted = True
                break
            if approach <= -0.005 and lateral_norm > 0.007:
                print("[vision-servo] plug is seated axially but lateral visual error is still high; holding the gripper and retrying alignment")

        tip_after = self.data.site_xpos[self.plug_tip_site].copy()
        final_metrics = self.plug_port_alignment_metrics()
        print(
            f"[insert] tip after insertion  = {np.round(tip_after, 4)}, "
            f"error = {np.linalg.norm(tip_after - port_target):.3f} m, "
            f"bottom force = {bottom_force:.2f} N"
        )
        if inserted:
            print("[insert] strict alignment reached; engaging the RJ45 latch")
            inserted = self.latch_plug_to_port()
        else:
            self.print_insert_alignment("final strict check failed", final_metrics)
            print("[insert] strict insertion failed; keeping the plug gripped instead of releasing it")
        return inserted

    def set_plug_free_pose(self, tip_pos: np.ndarray, rot: np.ndarray):
        plug_tip_local = np.array([0.043, 0.0, 0.0])
        qpos = self.data.qpos[self.plug_free_qadr : self.plug_free_qadr + 7].copy()
        qpos[:3] = tip_pos - rot @ plug_tip_local
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot.reshape(9)))
        qpos[3:7] = quat
        self.data.qpos[self.plug_free_qadr : self.plug_free_qadr + 7] = qpos
        self.data.qvel[self.plug_free_dofadr : self.plug_free_dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def unplug_plug_from_port(self, viewer=None):
        self.release_grasp_weld()
        self.open_right_gripper(viewer)
        port_target = self.data.site_xpos[self.port_site].copy()
        port_rot = self.data.site_xmat[self.port_site].reshape(3, 3).copy()
        insertion_axis = port_rot[:, 0].copy()
        table_tip = port_target + np.array([0.0, 0.19, -0.045])
        table_tip[2] = 0.835
        latch_top = port_target - insertion_axis * 0.040 + np.array([0.0, 0.0, 0.026])
        latch_press = latch_top.copy()
        latch_press[2] -= 0.012
        tcp_down_rot = rotation_from_axes(-insertion_axis, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]))
        tcp_offset_above_latch = np.array([0.0, 0.0, 0.0])

        print(f"[unplug] pressing RJ45 latch with right TCP horizontal to table")
        self.open_right_gripper(viewer)
        self.move_right_tcp_pose(latch_top + tcp_offset_above_latch, tcp_down_rot, duration=0.65, label="latch hover", viewer=viewer)
        self.move_right_tcp_pose(latch_press + tcp_offset_above_latch, tcp_down_rot, duration=0.45, label="latch press", viewer=viewer)
        for _ in range(int(0.18 / self.model.opt.timestep)):
            self.hold_base()
            self.step(viewer)
        self.release_port_latch()
        latch_grip_tcp = latch_press + tcp_offset_above_latch + np.array([0.0, 0.0, 0.004])
        self.move_right_tcp_pose(latch_grip_tcp, tcp_down_rot, duration=0.30, label="latch horizontal pinch", viewer=viewer)

        ok, force, _ = self.right_gripper_closed(viewer)
        if ok:
            self.engage_right_grasp_weld_if_contact()
        else:
            print("[unplug] proceeding with a best-effort withdrawal after partial contact")
            self.engage_right_grasp_weld_if_contact()

        pull_schedule = np.linspace(0.0, 0.085, 7)
        for retreat in pull_schedule:
            desired_tip = port_target - insertion_axis * retreat
            desired_tip[2] = port_target[2]
            self.move_right_plug_tip_waypoint(desired_tip, duration=0.35, label=f"withdraw {retreat:.3f} m", viewer=viewer)
        self.events["unplug_success"] = self._sim_time()

        drop_q = self.solve_right_q_for_desired_plug_tip(table_tip, approach_offset=0.0)
        self.move_right_arm_to(drop_q, duration=0.65, viewer=viewer)
        self.release_right_grasp_weld()
        self.set_plug_free_pose(table_tip, port_rot)
        self.open_right_gripper(viewer)
        for _ in range(int(0.5 / self.model.opt.timestep)):
            self.hold_base()
            self.step(viewer)
        print(f"[unplug] plug placed on table at {np.round(table_tip, 4)}")
        return True

    def _run(self, viewer=None, execute_if_unreachable: bool = False):
        table_side_base = np.array([0.0, 0.085, 0.0])
        if np.linalg.norm(self._current_base_pose() - table_side_base) > 0.004:
            print("[stage] moving mobile base from the initial stand-off pose to the table side")
            self.move_base(table_side_base, viewer)
        else:
            self.base_hold_target = table_side_base.copy()
            self.base_locked = True
            self.lock_base_pose()

        plan = self.plan_grasp()
        if not plan["strict_ok"] and not execute_if_unreachable:
            print("[abort] Not executing a fake grasp. Re-run with --execute-best-effort to visualize the closest safe plan.")
            return

        self.base_hold_target = self._current_base_pose()

        support_plan = self.plan_port_hold(hover_height=0.10)
        if support_plan["strict_ok"]:
            self.move_support_arm_path(support_plan["q_plan"], viewer)
            self.move_support_to_port_hold_with_vision(viewer)
            self.close_support_gripper(viewer)
        else:
            print("[support] groove-center pose is not reachable with the base locked; hovering only to avoid penetrating the socket.")
            self.move_support_arm_path(support_plan["q_plan"], viewer)

        plan = self.plan_grasp()
        if plan["strict_ok"] or execute_if_unreachable:
            self.base_hold_target = self._current_base_pose()
            self.move_arm_path(plan["q_plan"], viewer)
            if not self.close_gripper(viewer):
                print("[done] plug was contacted poorly, so insertion was not attempted.")
                return
            for _ in range(int(0.5 / self.model.opt.timestep)):
                self.hold_base()
                self.step(viewer)

            inserted = self.insert_plug_into_port(viewer)
            if inserted:
                for _ in range(int(0.5 / self.model.opt.timestep)):
                    self.hold_base()
                    self.step(viewer)
                self.unplug_plug_from_port(viewer)
                self.reset_arms(viewer)
                port_target = self.data.site_xpos[self.port_site].copy()
                tip_after_release = self.data.site_xpos[self.plug_tip_site].copy()
                release_force = self.port_bottom_force()
                print(
                    "[release] tip after unplug/reset = "
                    f"{np.round(tip_after_release, 4)}, port = {np.round(port_target, 4)}, "
                    f"error = {np.linalg.norm(tip_after_release - port_target):.3f} m, "
                    f"bottom force = {release_force:.2f} N"
                )
                print("[done] plug inserted, unplugged by the right arm, placed on the table, and arms reset")
            else:
                print("[done] strict insertion was not reached; plug remains gripped for viewer inspection")
                for _ in range(int(2.0 / self.model.opt.timestep)):
                    self.hold_base()
                    self.step(viewer)
        else:
            print("[done] best-effort trajectory shown; gripper was not closed because the plug was not reached.")

        for _ in range(int(2.0 / self.model.opt.timestep)):
            self.step(viewer)

    def run(self, viewer=None, execute_if_unreachable: bool = False):
        try:
            return self._run(viewer=viewer, execute_if_unreachable=execute_if_unreachable)
        finally:
            self.save_run_outputs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xml",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "mujoco" / "mobile_dual_ur3_scene.xml"),
        help="Path to the MJCF XML scene.",
    )
    parser.add_argument("--arm", choices=["nearest", "left", "right"], default="nearest")
    parser.add_argument("--execute-best-effort", action="store_true", help="Visualize the closest safe plan even if grasp is infeasible.")
    parser.add_argument("--scene-only", action="store_true", help="Only load the MuJoCo scene and keep the viewer open.")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless for testing; this only writes CSV/plot outputs.")
    parser.add_argument("--no-vision", action="store_true", help="Disable simulated ArUco visual servoing and use ground-truth scene sites.")
    parser.add_argument("--camera-view", action="store_true", help="Open the OpenCV window with the three camera images.")
    parser.add_argument("--ros-camera-topics", action="store_true", help="Publish ROS2 image and ArUco detection topics.")
    args = parser.parse_args()

    demo = CasadiInsertDemo(Path(args.xml).expanduser().resolve(), arm_name=args.arm, use_vision=not args.no_vision)
    if args.no_viewer:
        print("[viewer] --no-viewer was requested, so no MuJoCo window will be opened.")
        demo.run(viewer=None, execute_if_unreachable=args.execute_best_effort)
        return

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError(
            "No graphical display was detected. Run this from a desktop terminal/IDE with DISPLAY or WAYLAND_DISPLAY set, "
            "or use --no-viewer only for headless testing."
        )

    recorder = ViewerRecorder(demo.model, Path(args.xml).expanduser().resolve().parent / "recordings")
    demo.recorder = recorder
    if args.camera_view or args.ros_camera_topics:
        demo.camera_panel = CameraPanel(
            demo.model,
            [
                ("Head D405", "head_d405_camera", "head_d405"),
                ("Right wrist", "left_wrist_camera", "right_wrist"),
                ("Left wrist", "right_wrist_camera", "left_wrist"),
            ],
            enable_ros=args.ros_camera_topics,
            show_window=args.camera_view,
            key_callback=recorder.handle_key,
        )
        print("[camera] camera rendering enabled; use this only when checking ArUco camera images/topics.")
    else:
        print("[camera] camera rendering is off by default so the MuJoCo motion can run faster.")

    with mujoco.viewer.launch_passive(demo.model, demo.data, key_callback=recorder.handle_key) as viewer:
        recorder.set_viewer(viewer)
        configure_viewer(viewer)
        show_viewer_before_planning(demo, viewer)
        print(
            "[viewer] MuJoCo scene loaded. Adjust the view, then press R to record and start; "
            "press Space to start without recording."
        )
        if args.scene_only:
            while viewer.is_running():
                demo.lock_base_pose()
                viewer.sync()
                recorder.capture(demo.data)
                if demo.camera_panel is not None:
                    demo.camera_panel.update(demo.data)
                time.sleep(1.0 / 60.0)
            if recorder.recording:
                recorder.stop()
            if demo.camera_panel is not None:
                demo.camera_panel.close()
            return

        while viewer.is_running() and not recorder.run_requested:
            demo.lock_base_pose()
            viewer.sync()
            recorder.capture(demo.data)
            if demo.camera_panel is not None:
                demo.camera_panel.update(demo.data)
            time.sleep(1.0 / 60.0)
        if not viewer.is_running():
            recorder.stop()
            if demo.camera_panel is not None:
                demo.camera_panel.close()
            return

        demo.run(viewer=viewer, execute_if_unreachable=args.execute_best_effort)
        while viewer.is_running():
            demo.step(viewer)
        if recorder.recording:
            recorder.stop()
        if demo.camera_panel is not None:
            demo.camera_panel.close()


if __name__ == "__main__":
    main()
