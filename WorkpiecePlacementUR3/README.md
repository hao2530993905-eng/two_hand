# WorkpiecePlacementUR3 项目操作教程

本文档对应当前工作目录：

```text
/home/sjh/WorkpiecePlacementUR3
```

设备与标定配置：

| 项目 | 当前配置 |
| --- | --- |
| 机械臂 | Universal Robots UR3 CB3 |
| 机械臂 IP | `192.168.1.4` |
| 相机 | Intel RealSense D455，眼在手外 |
| 相机序列号 | `105322250617` |
| 标定板 | Original ArUco，ID 582，实测边长 48 mm |
| ROS | Ubuntu + ROS Noetic，Catkin 工作空间 |
| YOLO 环境 | `/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone` |
| 夹爪 | Robotiq，默认端口 `63352` |

> 安全提示：凡是会运动机械臂的命令，都应先确认工作空间无人、无障碍物、急停可用，并使用较低速度。视觉流程第一次运行时不要加 `--execute`。

## 1. 项目功能拆分

项目主要分为四层：

```text
设备层
├── UR3 通信、RTDE/URScript 控制、Robotiq 夹爪
└── D455 彩色、深度、相机内参和 TF

标定与视觉层
├── ArUco 检测与眼在手外手眼标定
├── 深度背景差分与自动 OBB 数据采集
└── YOLO-OBB 训练、离线验证和实时检测

坐标与动作层
├── 相机像素 + 深度 → 相机三维点
├── 相机三维点 → ROS base_link
├── base_link → UR 控制器 base
└── moveJ、moveL、servoL、夹爪开合

任务层
├── 固定示教点抓取/放置
└── YOLO 检测后移动到目标表面上方 3 cm
```

主要目录：

| 目录 | 职责 |
| --- | --- |
| `catkin_ws/src/ur3_d455_handeye` | 本项目的 D455、UR3、ArUco、手眼标定启动配置 |
| `catkin_ws/src/Universal_Robots_ROS_Driver` | 官方 UR ROS 驱动 |
| `catkin_ws/src/universal_robot` | UR 描述、运动学和 MoveIt 配置 |
| `catkin_ws/src/aruco_ros` | ArUco 检测 |
| `catkin_ws/src/easy_handeye` | 手眼标定和结果发布 |
| `workpiece_demo` | 轻量 UR3/Robotiq 控制封装与基础测试 |
| `complete_process` | 固定点抓放和视觉定位完整流程 |
| `complete_process/utils/object_detective` | 深度差分、数据采集、YOLO 训练与验证 |

## 2. 环境准备与通用约定

### 2.1 编译 ROS 工作空间

源码或 ROS 包修改后执行：

```bash
cd /home/sjh/WorkpiecePlacementUR3/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

每个新的 ROS 终端至少执行：

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/WorkpiecePlacementUR3/catkin_ws/devel/setup.bash
```

### 2.2 启动 ROS Master

终端 1：

```bash
source /opt/ros/noetic/setup.bash
roscore
```

后续 ROS 节点必须连接到同一个 `ROS_MASTER_URI`，本机默认是：

```text
http://localhost:11311
```

### 2.3 坐标和单位

UR TCP 位姿格式统一为：

```text
[x, y, z, rx, ry, rz]
```

- `x/y/z`：米。
- `rx/ry/rz`：旋转向量，单位弧度，不是欧拉角。
- 关节位置和关节速度：弧度、弧度每秒。
- `base_link` 是 ROS REP-103 坐标系。
- RTDE 接收 UR 控制器 `base` 坐标。官方 UR 模型规定 `base_link → base` 绕 Z 轴旋转 π，因此点坐标满足：

```text
Xbase = -Xbase_link
Ybase = -Ybase_link
Zbase =  Zbase_link
```

`detect_and_pick.py` 已自动完成这一步，禁止在外部再次手工取反。

## 3. 硬件连通与只读测试

### 3.1 检查 UR3 网络

```bash
ping -c 3 192.168.1.4
```

示教器应满足：

- 机械臂已上电并松开抱闸。
- 安全状态正常，没有保护停止或急停。
- 已安装 External Control URCap。
- 如果使用官方 ROS 驱动，示教器应运行 External Control 程序。

### 3.2 读取当前 TCP 位姿

此命令只读取，不运动：

```bash
cd /home/sjh/WorkpiecePlacementUR3
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  workpiece_demo/read_current_pose.py --host 192.168.1.4
```

读取 TCP、关节和 TCP 速度：

```bash
cd /home/sjh/WorkpiecePlacementUR3
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  workpiece_demo/tests/test_read_state.py --host 192.168.1.4
```

### 3.3 启动官方 UR ROS 驱动

手眼标定和 RViz 需要机器人 TF：

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/WorkpiecePlacementUR3/catkin_ws/devel/setup.bash
roslaunch ur3_d455_handeye robot.launch
```

该启动文件使用：

- UR3 地址 `192.168.1.4`；
- 实机运动学校准文件 `complete_process/ur3_calibration.yaml`；
- 本机脚本命令端口 `50014`。

检查：

```bash
rostopic echo -n 1 /joint_states
rosrun tf tf_echo base_link tool0
```

## 4. D455 相机功能

### 4.1 启动相机

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/WorkpiecePlacementUR3/catkin_ws/devel/setup.bash
roslaunch ur3_d455_handeye camera.launch
```

默认开启彩色、深度、同步和深度对齐。检查：

```bash
rostopic hz /camera/color/image_raw
rostopic hz /camera/aligned_depth_to_color/image_raw
rostopic hz /camera/color/camera_info
```

查看图像：

```bash
rqt_image_view /camera/color/image_raw
rqt_image_view /camera/aligned_depth_to_color/image_raw
```

如果检测程序提示等待同步图像，优先检查上述三个话题；不能使用关闭深度的旧相机进程。

## 5. ArUco 与手眼标定

### 5.1 单独启动 ArUco 检测

相机运行后执行：

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/WorkpiecePlacementUR3/catkin_ws/devel/setup.bash
roslaunch ur3_d455_handeye aruco.launch
```

默认配置：

```text
dictionary  = Original ArUco / ARUCO
marker_id   = 582
marker_size = 0.048 m
marker TF   = aruco_marker_frame
```

查看检测结果：

```bash
rqt_image_view /aruco_single/result
rosrun tf tf_echo camera_color_optical_frame aruco_marker_frame
```

### 5.2 执行眼在手外手眼标定

标定板固定在末端，相机固定在工作区外：

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/WorkpiecePlacementUR3/catkin_ws/devel/setup.bash
roslaunch ur3_d455_handeye calibrate.launch
```

该命令一次启动：

- D455；
- UR3 官方驱动和机器人 TF；
- ArUco 检测；
- Easy Handeye 采样窗口；
- RViz；
- ArUco 图像窗口。

采样步骤：

1. 确认 ArUco 边框和坐标轴稳定显示。
2. 在示教器中手动改变末端位置和姿态。
3. 每次停止运动并等待约 1 秒，再点击 `Take Sample`。
4. 至少 15 组，建议 20～25 组；必须包含多个旋转轴和明显不同姿态。
5. 点击 `Compute`，检查结果，再点击 `Save`。

采样前检查：

```bash
rosrun ur3_d455_handeye check_setup.py
```

保存位置：

```text
~/.ros/easy_handeye/ur3_d455_handeye_eye_on_base.yaml
```

当前已验证结果方向是 `base_link <- camera_link`：

```yaml
translation: [-0.0612435729, 0.3843179341, 0.8338538643]
quaternion_xyzw: [-0.4910288272, 0.5123107314, 0.4766665622, 0.5188616326]
```

相机位置改变后必须重新标定。

### 5.3 发布和验证保存的标定结果

```bash
roslaunch ur3_d455_handeye publish.launch
rosrun tf tf_echo base_link camera_color_optical_frame
```

打印 Marker 相机坐标、机械臂基坐标以及当前末端姿态：

```bash
rosrun ur3_d455_handeye marker_to_base.py
```

只读取一次：

```bash
rosrun ur3_d455_handeye marker_to_base.py --once
```

## 6. 深度差分检测和 YOLO 数据采集

详细参数说明也见 `complete_process/utils/object_detective/README.md`。

### 6.1 启动深度差分

先启动 D455，然后：

```bash
cd /home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective
source /opt/ros/noetic/setup.bash
python3 depth_background_rect_detector.py \
  _target_color:=red \
  _min_area:=3000
```

检测条件是：

```text
深度高度差 AND 颜色条件 AND 最小轮廓面积
```

支持颜色：

```text
none red orange yellow green blue purple
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `_min_height_mm` | 12 | 相对背景最小高度 |
| `_max_height_mm` | 150 | 相对背景最大高度 |
| `_target_color` | none | 颜色预设 |
| `_min_area` | 3000 | 最小像素面积 |
| `_color_min_saturation` | 80 | HSV 最小饱和度 |
| `_color_min_value` | 50 | HSV 最小亮度 |
| `_max_objects` | 10 | 最大输出目标数 |
| `_show_gui` | true | 是否显示 OpenCV 窗口 |

窗口快捷键：

| 按键 | 功能 |
| --- | --- |
| `B` | 清空工作区后采集深度背景 |
| `R` | ROI 恢复为全图 |
| `S` | 保存正样本和当前 OBB 标签 |
| `N` | 保存负样本，标签为空 |
| `D` | 撤销本次运行最后保存的样本 |

### 6.2 采集训练集

```bash
cd /home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective
source /opt/ros/noetic/setup.bash
python3 depth_background_rect_detector.py \
  _target_color:=red \
  _min_area:=3000 \
  _collect_dataset:=true \
  _dataset_name:=red_workpiece_dataset \
  _dataset_split:=train
```

清空工件并按 `B` 建背景；放置工件并确认绿色框正确后按 `S`。改变位置、方向、数量和光照后重复采集。

### 6.3 采集验证集

验证集必须是独立场景：

```bash
python3 depth_background_rect_detector.py \
  _target_color:=red \
  _min_area:=3000 \
  _collect_dataset:=true \
  _dataset_name:=red_workpiece_dataset \
  _dataset_split:=val
```

数据结构：

```text
red_workpiece_dataset/
├── dataset.yaml
├── images/{train,val}/
├── labels/{train,val}/
└── previews/{train,val}/
```

训练前必须检查 `previews`。错误框会直接降低模型质量。

## 7. YOLO-OBB 训练、部署和验证

### 7.1 当前模型

```text
complete_process/utils/object_detective/models/red_box.pt    红色工件，默认模型
complete_process/utils/object_detective/models/white_box.pt  白色工件
```

### 7.2 训练红色工件模型

```bash
cd /home/sjh/WorkpiecePlacementUR3
MPLCONFIGDIR=/tmp/matplotlib-red-box \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -c \
"from ultralytics import YOLO; YOLO('yolo11n-obb.pt').train(data='complete_process/utils/object_detective/red_workpiece_dataset/dataset.yaml', epochs=120, patience=35, batch=4, imgsz=640, device=0, workers=2, project='/home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/runs', name='red_box')"
```

显存不足时降低 `batch`。训练后的最佳模型为：

```text
complete_process/utils/object_detective/runs/red_box/weights/best.pt
```

部署：

```bash
cp complete_process/utils/object_detective/runs/red_box/weights/best.pt \
   complete_process/utils/object_detective/models/red_box.pt
```

### 7.3 离线指标验证

```bash
cd /home/sjh/WorkpiecePlacementUR3
MPLCONFIGDIR=/tmp/matplotlib-red-box-val \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -c \
"from ultralytics import YOLO; YOLO('complete_process/utils/object_detective/models/red_box.pt').val(data='complete_process/utils/object_detective/red_workpiece_dataset/dataset.yaml', imgsz=640, device=0, project='/home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/runs', name='red_box_validation')"
```

关注 `Precision`、`Recall`、`mAP50` 和 `mAP50-95`。验证图片太少时，高指标不代表泛化能力好。

### 7.4 生成验证集预测图

```bash
cd /home/sjh/WorkpiecePlacementUR3
MPLCONFIGDIR=/tmp/matplotlib-red-box-predict \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -c \
"from ultralytics import YOLO; YOLO('complete_process/utils/object_detective/models/red_box.pt').predict(source='complete_process/utils/object_detective/red_workpiece_dataset/images/val', imgsz=640, conf=0.45, device=0, save=True, project='/home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/runs', name='red_box_predictions')"
```

检查 `runs/red_box_predictions` 中的框中心、边界和旋转方向。

### 7.5 D455 实时模型测试

相机运行后，启动红色模型：

```bash
cd /home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective
source /opt/ros/noetic/setup.bash
MPLCONFIGDIR=/tmp/matplotlib-red-live \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -u \
test_yolo_obb_live.py \
  _model:=/home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/models/red_box.pt \
  _device:=0 \
  _confidence:=0.45 \
  _show_gui:=false
```

查看：

```bash
rqt_image_view /test_yolo_obb_live/debug_image
rostopic echo /test_yolo_obb_live/result
```

白色模型只需替换路径：

```text
_model:=/home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/models/white_box.pt
```

实时验证应覆盖不同位置、角度、数量、光照、遮挡和相似干扰物。

## 8. 机械臂基础动作与夹爪测试

以下命令不依赖 ROS 图像，但依赖 UR RTDE/URScript 和相应 Python 环境。

### 8.1 夹爪开合

打开：

```bash
cd /home/sjh/WorkpiecePlacementUR3
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  complete_process/utils/test_gripper.py \
  --host 192.168.1.4 --port 63352 --action open
```

关闭并监控寄存器：

```bash
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  complete_process/utils/test_gripper.py \
  --host 192.168.1.4 --port 63352 --action close \
  --force 100 --monitor --interval 0.2
```

`force` 是 0～255 的设置值，不是牛顿。

### 8.2 小距离 moveL 往返测试

默认沿 UR 控制器基坐标 Z 上升 1 mm，再返回：

```bash
cd /home/sjh/WorkpiecePlacementUR3
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  workpiece_demo/tests/test_small_move.py \
  --host 192.168.1.4 --dz 0.001 --speed 0.01
```

### 8.3 moveL 指定 Z 距离

负值下降，例如 5 mm：

```bash
cd /home/sjh/WorkpiecePlacementUR3
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  complete_process/utils/basic_move/move_l_distance.py \
  --host 192.168.1.4 --distance -0.005 --speed 0.01
```

### 8.4 moveJ 到关节角

六个关节角均为弧度：

```bash
cd /home/sjh/WorkpiecePlacementUR3
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  workpiece_demo/move_to_joints.py \
  0 -1.57 1.57 -1.57 -1.57 0 \
  --host 192.168.1.4 --speed 0.1 --acc 0.2
```

示例值不保证适合当前现场，执行前必须在示教器中验证。

### 8.5 moveJ/IK 到 TCP 位姿

```bash
cd /home/sjh/WorkpiecePlacementUR3
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  complete_process/utils/basic_move/move_j_to_pose.py \
  --host 192.168.1.4 \
  --pose X Y Z RX RY RZ \
  --speed 0.1 --acc 0.2
```

### 8.6 servoL 轨迹测试

1 mm 生成轨迹往返：

```bash
cd /home/sjh/WorkpiecePlacementUR3
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  workpiece_demo/tests/test_servo_trajectory.py \
  --host 192.168.1.4 --dz 0.001 --steps 50 --dt 0.02
```

运行 JSON 轨迹：

```bash
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  workpiece_demo/follow_trajectory.py \
  --host 192.168.1.4 \
  --trajectory workpiece_demo/config/example_trajectory.json
```

JSON 中每项均为 `[x,y,z,rx,ry,rz]`。示例配置只是格式模板，不能未经现场检查直接执行。

### 8.7 不连接实机的单元测试

```bash
cd /home/sjh/WorkpiecePlacementUR3
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -m unittest \
  discover -s workpiece_demo/tests -p 'test_*unit.py'

/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -m unittest \
  discover -s workpiece_demo/tests -p 'test_pick_from_pose.py'
```

## 9. 固定示教点抓取和放置

### 9.1 单点下降抓取

```bash
cd /home/sjh/WorkpiecePlacementUR3
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  workpiece_demo/pick_from_pose.py \
  --host 192.168.1.4 \
  --pose X Y Z RX RY RZ \
  --down 0.01 \
  --speed 0.01 \
  --gripper-force 100 \
  --lift-after
```

流程：打开夹爪 → 到接近位姿 → 沿 base -Z 下降 → 闭爪 → 可选抬升。

### 9.2 完整固定点抓放

```bash
cd /home/sjh/WorkpiecePlacementUR3
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python \
  complete_process/pick_and_place.py \
  --host 192.168.1.4 \
  --pick-pose PICK_X PICK_Y PICK_Z PICK_RX PICK_RY PICK_RZ \
  --pick-distance -0.01 \
  --place-pose PLACE_X PLACE_Y PLACE_Z PLACE_RX PLACE_RY PLACE_RZ \
  --place-distance -0.01 \
  --joint-speed 0.1 --joint-acc 0.2 \
  --speed 0.01 --linear-acc 0.05 \
  --gripper-force 100 \
  --retreat-after-place
```

流程：

```text
开夹爪
→ moveJ/IK 到抓取接近位姿
→ moveL 下降
→ 闭夹爪
→ moveL 返回抓取接近位姿
→ moveJ/IK 到放置接近位姿
→ moveL 下降
→ 开夹爪
→ 可选撤回
```

也可以编辑 `workpiece_demo/config/example_pick_place.json`，再用 `place_workpiece_demo.py` 执行一组命名位姿；该脚本全程使用 moveL，适合已验证的短路径，不适合跨越大范围障碍物。

## 10. 视觉定位并移动到目标上方 3 cm

这是当前项目的主要视觉集成入口：

```text
complete_process/detect_and_pick.py
```

### 10.1 集成原理

```text
D455 彩色图
  → red_box.pt YOLO-OBB
  → 随机选择并连续跟踪一个目标
  → OBB 中心像素 (u,v)

对齐深度 + color camera_info
  → 反投影得到 camera_color_optical_frame 三维点

内嵌手眼标定 base_link <- camera_link
+ RealSense 内部 TF camera_link <- camera_color_optical_frame
  → 物体在 base_link 下的表面中心

base_link 的 Z + 0.03 m
  → 接近点

base_link → UR 控制器 base 的 X/Y 取反
  → RTDE moveJ_IK 目标位姿
```

程序默认使用：

- `models/red_box.pt`；
- 8 帧稳定检测；
- OBB 中心附近深度中值；
- 已内嵌的 `base_link <- camera_link` 标定结果；
- 固定向下 TCP 旋转向量；
- 目标表面上方 0.03 m。

### 10.2 第一步：只计算，不运动

先启动 `roscore` 和 D455，然后：

```bash
cd /home/sjh/WorkpiecePlacementUR3
source /opt/ros/noetic/setup.bash
source catkin_ws/devel/setup.bash

MPLCONFIGDIR=/tmp/matplotlib-detect-pick \
ROS_HOME=/tmp/ros-detect-pick \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -u \
complete_process/detect_and_pick.py \
  --model complete_process/utils/object_detective/models/red_box.pt \
  --device 0 \
  --confidence 0.45 \
  --above-height 0.03 \
  --workspace -0.45 0.45 0.10 0.55 0.02 0.55
```

查看检测框：

```bash
rqt_image_view /detect_and_pick/debug_image
```

检查终端中的：

- 相机光学坐标点；
- `base_link` 表面中心；
- `base_link` 上方 3 cm 点；
- 实际发送 RTDE 的 UR `base` TCP 位姿。

不带 `--execute` 时机械臂不会连接或运动。

### 10.3 第二步：低速实际移动

坐标确认无误后：

```bash
MPLCONFIGDIR=/tmp/matplotlib-detect-pick \
ROS_HOME=/tmp/ros-detect-pick \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -u \
complete_process/detect_and_pick.py \
  --host 192.168.1.4 \
  --model complete_process/utils/object_detective/models/red_box.pt \
  --device 0 \
  --confidence 0.45 \
  --above-height 0.03 \
  --joint-speed 0.10 \
  --joint-acc 0.20 \
  --motion-timeout 60 \
  --max-start-distance 0.15 \
  --workspace -0.45 0.45 0.10 0.55 0.02 0.55 \
  --execute
```

程序打印目标后还会要求输入：

```text
MOVE
```

只有输入完全匹配才运动。`--yes` 会跳过确认，不建议在调试阶段使用。

常用补偿参数：

```text
--target-offset DX DY DZ
```

它在 ROS `base_link` 下补偿视觉点，单位米。必须通过多次低速测量后再设置。

### 10.4 当前功能边界

当前 `detect_and_pick.py`：

- 已完成目标检测、深度定位、手眼变换、`base_link/base` 转换和移动到上方 3 cm；
- 尚未根据 OBB 短边自动计算夹爪绕 Z 的姿态；
- 尚未自动下降、闭合夹爪和搬运到放置点。

因此现阶段不能把该脚本描述为“自动抓取完成”。夹爪方向功能还需要标定 TCP 到两指连线的固定安装角，再把 OBB 短边方向转换到机械臂基座。

## 11. 两条完整工作流如何组合

### 11.1 重新标定工作流

```text
启动 UR3 + D455 + ArUco
→ Easy Handeye 多姿态采样
→ Compute/Save
→ marker_to_base.py 验证
→ 将确认后的外参同步到 detect_and_pick.py
```

只有相机固定位置变化时才需重复。

### 11.2 新颜色/新工件训练工作流

```text
启动 D455
→ 深度差分建空背景
→ target_color/min_area 过滤
→ S/N 采集 train 和 val
→ 检查 previews
→ YOLO-OBB 训练
→ val 指标 + predictions 图像
→ D455 实时验证
→ best.pt 部署到 models/<name>.pt
→ detect_and_pick.py --model 指定
```

### 11.3 当前实际视觉使用工作流

```text
roscore
→ camera.launch
→ detect_and_pick.py（不带 --execute）检查
→ rqt_image_view 核对 OBB
→ 核对 base_link 和 UR base 坐标
→ 加 --execute 低速移动到上方 3 cm
```

视觉脚本通过 RTDE 直接控制 UR3，因此正常检测移动不要求同时启动 `ur_robot_driver`；但相机 ROS 节点和 ROS Master 必须运行。手眼标定阶段则必须启动 `ur_robot_driver`，因为 Easy Handeye 需要机器人 TF。

## 12. 常见故障排查

### 12.1 `Unable to register with master node`

没有 `roscore`：

```bash
source /opt/ros/noetic/setup.bash
roscore
```

### 12.2 找不到 launch 或 ROS 包

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/WorkpiecePlacementUR3/catkin_ws/devel/setup.bash
rospack find ur3_d455_handeye
```

### 12.3 找不到 Python 脚本

相对路径取决于当前目录。推荐先：

```bash
cd /home/sjh/WorkpiecePlacementUR3
```

或者直接使用绝对路径。

### 12.4 等待彩色、对齐深度和 camera_info 超时

```bash
rostopic hz /camera/color/image_raw
rostopic hz /camera/aligned_depth_to_color/image_raw
rostopic hz /camera/color/camera_info
```

关闭旧相机进程并重新启动 `camera.launch`。

### 12.5 RViz 提示相机到 base_link 没有 TF

标定阶段应启动完整 `calibrate.launch`；标定后可启动：

```bash
roslaunch ur3_d455_handeye publish.launch
```

并检查：

```bash
rosrun tf tf_echo base_link camera_link
```

### 12.6 视觉点正确但机械臂走到镜像位置

这是混用了 ROS `base_link` 与 UR 控制器 `base`。当前 `detect_and_pick.py` 已自动转换；不要删除 `base_link_point_to_ur_base()`，也不要再对输出二次取反。

### 12.7 YOLO 没有检测框

- 用 `rqt_image_view` 确认目标清晰可见；
- 临时将置信度从 0.45 降到 0.25；
- 确认选中了正确模型；
- 增加对应颜色、角度和光照样本后重新训练。

### 12.8 深度差分误检很多

- 清空工作区并重新按 `B`；
- 缩小 ROI；
- 提高 `_min_height_mm` 或 `_min_area`；
- 调整 `_target_color`、饱和度和亮度阈值；
- 确保相机和工作台在建背景后没有移动。

### 12.9 RTDE 控制连接失败

- 检查机械臂 IP 和网络；
- 检查机器人运行和安全状态；
- 确认没有其他程序占用 RTDE 输入寄存器；
- 必要时停止官方驱动或其他 RTDE 控制进程后重试。

## 13. 推荐的安全验收顺序

新环境或重大修改后按以下顺序恢复功能：

1. `ping` UR3。
2. 只读 TCP/关节状态。
3. 测试 D455 三个话题。
4. 测试 ArUco 和 TF。
5. 用 `marker_to_base.py --once` 验证手眼结果。
6. 测试夹爪开合。
7. 执行 1 mm 的 moveL 往返。
8. 实时验证 YOLO，不运动。
9. 运行 `detect_and_pick.py` 干运行，核对四组坐标。
10. 设置低速、较小 `max-start-distance` 后移动到目标上方。
11. 最后才测试下降、闭爪和完整抓放。

这套顺序能把相机、模型、坐标变换、机械臂控制和夹爪故障分层定位，避免在完整流程中同时排查多个问题。




python complete_process/white_workpiece_to_robot.py \
  --model /home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/models/white_box.pt \
  --cloud white_workpiece.ply \
  --result white_rectangle_result.ply \
  --length-mm 330 \
  --width-mm 90 \
  --device cpu \
  --conf 0.1 \
  --height 0.08 \
  --speed 0.5 \
  --acc 0.5 \
  --execute

生成点云

conda activate camera

python complete_process/red_workpiece_pointcloud.py \
  --output red_workpiece.ply \
  --conf 0.35 \
  --frames 30 \
  --capture-frames 30 \
  --voxel-mm 0.4 \
  --model /home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/models/white_box.pt


查看最低点

conda activate camera

python complete_process/highlight_lowest_points.py \
  --input red_workpiece.ply \
  --output red_lowest_10_percent.ply \
  --percentile 90



conda activate camera

python complete_process/search_known_rectangle.py \
  --input /home/sjh/WorkpiecePlacementUR3/red_workpiece.ply\
  --output white_rectangle_result.ply \
  --length-mm 330\
  --width-mm 90 \
  --angle-range-deg 45 \
  --coarse-angle-deg 0.5 \
  --edge-band-mm 8 \
  --min-edge-points 10 \
  --top-candidates 10


slot 1: distance_to_white=0.6101 m state=AVAILABLE base_link=[-0.284147, 0.467098, -0.006854]
slot 2: distance_to_white=0.4953 m state=AVAILABLE base_link=[-0.166504, 0.45934, -0.006854]
slot 3: distance_to_white=0.3738 m state=AVAILABLE base_link=[-0.039812, 0.450986, -0.006854]
slot 4: distance_to_white=0.2497 m state=SELECTED (nearest unused) base_link=[0.09593, 0.442034, -0.006854]


slot 1: distance_to_white=0.5649 m state=AVAILABLE base_link=[-0.237617, 0.451239, -0.002887]
slot 2: distance_to_white=0.4471 m state=AVAILABLE base_link=[-0.117081, 0.442736, -0.002887]
slot 3: distance_to_white=0.3229 m state=AVAILABLE base_link=[0.012727, 0.43358, -0.002887]
slot 4: distance_to_white=0.1984 m state=SELECTED (nearest unused) base_link=[0.151807, 0.423769, -0.002887]

slot 1: distance_to_white=0.4179 m state=SELECTED (nearest unused) base_link=[-0.112988, 0.277161, -0.002832]
slot 2: distance_to_white=0.4224 m state=AVAILABLE base_link=[-0.108556, 0.394612, -0.002832]
slot 3: distance_to_white=0.4620 m state=AVAILABLE base_link=[-0.103783, 0.521098, -0.002832]
slot 4: distance_to_white=0.5352 m state=AVAILABLE base_link=[-0.098669, 0.656619, -0.002832]

slot 1: distance_to_white=0.5880 m state=AVAILABLE base_link=[-0.265635, 0.45307, -0.003503]
slot 2: distance_to_white=0.4731 m state=AVAILABLE base_link=[-0.148017, 0.446316, -0.003503]
slot 3: distance_to_white=0.3516 m state=AVAILABLE base_link=[-0.02135, 0.439043, -0.003503]
slot 4: distance_to_white=0.2279 m state=SELECTED (nearest unused) base_link=[0.114364, 0.431251, -0.003503]
