# 移动双臂 RJ45 网线插拔仿真

这个包里主要放了机器人模型、MuJoCo 场景，以及一个双臂协同插拔 RJ45 网线的 Python 演示程序。工程是基于 ROS2 Humble 开发的，运行前需要先自己把 ROS2 Humble 环境配置好。

## 环境准备

先进入 ROS2 工作空间，然后编译这个包：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select robot
source install/setup.bash
```

MuJoCo 仿真脚本还会用到一些 Python 库，可以这样安装：

```bash
python3 -m pip install --user mujoco casadi numpy matplotlib opencv-python pillow
```

如果要用 ArUco 检测，建议安装带 `aruco` 模块的 OpenCV：

```bash
python3 -m pip install --user opencv-contrib-python
```

主要依赖大概是这些：

- ROS2 Humble：用来管理包、模型、launch 和 RViz 可视化。
- MuJoCo：用来做物理仿真和交互式可视化窗口。
- CasADi/IPOPT：用来做机械臂轨迹优化和逆运动学求解。
- OpenCV ArUco：用来识别网口和网线支架上的 ArUco 标记。
- Matplotlib：用来画误差、力传感、视觉识别偏差和关节角曲线。



## 运行 MuJoCo 演示

正常打开 MuJoCo 可视化窗口，默认使用 ArUco 视觉定位流程：

```bash
cd src/robot
python3 scripts/run_left_arm_insert_demo.py
```

MuJoCo 窗口里的操作方式：

- 先把视角调到你想录制的位置。
- 按 `R` 开始录制，同时开始跑插拔流程。
- 按 `E` 停止录制并保存视频。
- 按 `Space` 可以不录制，直接开始跑流程。

如果只是想重新生成数据和图片，不需要打开可视化窗口：

```bash
cd src/robot
python3 scripts/run_left_arm_insert_demo.py --no-viewer
```

如果想退回原来的确定位置控制，可以加 `--no-vision`：

```bash
python3 scripts/run_left_arm_insert_demo.py --no-viewer --no-vision
```

如果只想加载 MuJoCo 场景看看模型：

```bash
python3 scripts/run_left_arm_insert_demo.py --scene-only
```

## ArUco 视觉定位

现在场景里一共有三个相机：

- 头部 D405：负责全局粗定位，同时能看到网口和网线上的 marker。
- 插线机械臂腕部相机：靠近网线头时给网线 marker 更高权重。
- 扶持机械臂腕部相机：靠近网口时给网口 marker 更高权重。

网口凹槽上贴了 `aruco-582`，尺寸是 `0.08 m`。网线头上不能直接贴码，所以在线缆侧加了一个小旗子支架，把 `aruco-120` 固定在支架上，尺寸同样是 `0.08 m`，避免支架和插头、夹爪发生碰撞。脚本里会模拟三个相机发布的 ArUco 位姿检测结果，然后按相机距离和角色权重融合出网口、网线头的估计位姿，再用这个估计位姿完成双臂协作插入和拔出。

后面如果要接真实 ROS2 ArUco 检测，只需要把脚本里的仿真检测函数替换成订阅到的 marker 位姿，后面的融合、规划和误差评估可以继续复用。

为了让 MuJoCo 主窗口跑得更快，三路相机图像窗口和 ROS2 图像 topic 默认都是关闭的。正常录制插拔流程时直接运行即可：

```bash
python3 scripts/run_left_arm_insert_demo.py
```

如果要调试相机画面，可以打开三路相机检测窗口：

```bash
python3 scripts/run_left_arm_insert_demo.py --camera-view
```

如果要发布 ROS2 相机图像和 ArUco 检测 topic，可以这样跑：

```bash
python3 scripts/run_left_arm_insert_demo.py --ros-camera-topics
```

两个选项也可以一起用：

```bash
python3 scripts/run_left_arm_insert_demo.py --camera-view --ros-camera-topics
```

然后查看 topic：

```bash
ros2 topic list | grep mujoco_cameras
ros2 run rqt_image_view rqt_image_view
```

常用 topic：

- `/mujoco_cameras/head_d405/image_raw`：头部 D405 原始图像。
- `/mujoco_cameras/head_d405/aruco/image`：头部 D405 叠加 ArUco 检测框后的图像。
- `/mujoco_cameras/head_d405/aruco/detections`：识别到的 marker ID、像素中心、四个角点和相机坐标系下的估计位姿。
- `/mujoco_cameras/right_wrist/aruco/image`：机器人本体右臂腕部相机检测图像。
- `/mujoco_cameras/left_wrist/aruco/image`：机器人本体左臂腕部相机检测图像。

现在两个 marker 已经分开：网口是 `582`，线缆支架是 `120`。线缆侧 marker 已经绑在线缆本体上，会跟着网线一起运动；D405 的检测图像里应该能同时看到这两个 ID。

## 输出结果

仿真结束后，图片会保存到：

```text
mujoco/insert_demo_logs/
```

主要图片包括：

- `left_arm_insert_demo_error.png` / `.pdf`：网线头到网口凹槽的 X/Y/Z 误差曲线。
- `right_arm_tcp_force.png` / `.pdf`：右臂末端力传感曲线。
- `right_arm_joint_tracking.png` / `.pdf`：右臂目标关节角和实际关节角曲线。
- `left_arm_insert_demo_tcp_trajectory.png`：双臂末端轨迹曲线。
- `visual_pose_error.png` / `.pdf`：ArUco 视觉估计造成的网口和网线头位姿偏差曲线。

画图用到的数据会保存到：

```text
data/
```

主要数据文件包括：

- `left_arm_insert_demo_trace.csv`：完整仿真轨迹记录。
- `plug_socket_error_xyz.csv`：网线头和网口 X/Y/Z 误差数据。
- `right_arm_tcp_force_0_47s.csv`：右臂末端力数据。
- `right_arm_tcp_force_peak_triggers.csv`：力峰值触发的插入成功和拔出成功时刻。
- `right_arm_joint_tracking.csv`：右臂目标关节角和实际关节角。
- `dual_arm_tcp_trajectory.csv`：双臂末端轨迹数据。
- `visual_pose_error.csv`：视觉估计误差数据，包括网口 target 误差、网线 tip 误差和 marker 被几个相机看到。
- `plot_events.csv`：插入成功、拔出成功等事件时间。

## ROS2 可视化

如果想看 ROS2/RViz 里的模型，先确认工作空间已经编译并 source 过：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```


然后按需要启动对应的 launch 文件：

```bash
ros2 launch robot <launch_file_name>
```
