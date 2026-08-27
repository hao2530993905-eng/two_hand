# UR3 + RealSense D435 眼在手上标定

本目录是独立 ROS Noetic catkin 工作空间，仅用于以下实机：

- UR3 CB3：`192.168.1.5`，序列号 `2020330223`；
- D435：序列号 `135622076024`；
- 机器人 TF：`base_link -> tool0`；
- 相机安装：D435 刚性固定在末端，标定后不可移动；
- 手眼 marker：Original ArUco（`ARUCO`），ID 582，实测黑框边长 0.048 m；
- 手眼类型：eye-on-hand，结果方向为 `tool0 <- d435_link`；
- D435 专属话题与 TF 前缀：`/d435/...`、`d435_link`，不与 D455 冲突。

必须严格先完成第一阶段内参，再开始第二阶段手眼标定。项目没有复制 D455
内参；在实测 D435 内参文件不存在时，第二阶段相机入口会启动失败。

## 当前实机状态

UR3 的运动学校准已从 `192.168.1.5` 通过官方 `ur_calibration` 只读提取，
并已验证驱动报告 `Calibration checked successfully`。文件为：

```text
catkin_ws/src/d435_eye_on_hand/config/ur3_2020330223_calibration.yaml
```

D435 当前被系统识别为 USB 2.1。该链路不支持目标配置 `1280x720@30`，驱动会
退回 `640x480@15`，实测还没有图像帧。内参标定前必须改用可靠的 USB 3.x
端口和线缆，并用下面的命令确认 `Usb Type Descriptor` 不再显示 2.1：

```bash
rs-enumerate-devices | grep "Usb Type Descriptor"
```

## 构建与环境

```bash
cd /home/sjh/second_hand/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

工作空间内固定了与参考项目相同提交的 `aruco_ros 3.1.4` 和
`easy_handeye 0.4.3`，并保留了 Original ArUco 字典及样本管理修补。

## 第一阶段：D435 彩色相机内参

### 1. 打印 A4 ChArUco 板

打印：

```text
calibration_targets/charuco_5x5_250_7x10_A4.pdf
```

参数为 `5x5_250` 字典、7x10 方格。当前打印件的 100 mm 校验线实测为
96.31 mm，因此配置使用方格边长 24.0775 mm、marker 边长 17.3358 mm。
打印对话框必须选择“实际大小”或 `100%`，禁止“适合页面”。将纸张
平整贴到硬质平板，不能有翘曲。

打印后测量 PDF 下方标为 100 mm 的校验线。若实测为 `L` mm，则启动参数为：

```text
square_size = 0.025 * L / 100   (m)
marker_size = 0.018 * L / 100   (m)
```

如果校验线正好是 100 mm，直接使用启动文件默认值。

### 2. 确认相机 USB 3.x 和目标视频模式

```bash
rs-enumerate-devices | grep "Usb Type Descriptor"
roslaunch d435_eye_on_hand d435_raw.launch
```

另开终端检查，必须是 1280x720 且接近 30 Hz：

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/second_hand/catkin_ws/devel/setup.bash
rostopic echo -n 1 /d435/color/camera_info
rostopic hz /d435/color/image_raw
```

检查完按 `Ctrl+C` 停止相机，避免重复占用设备。

### 3. 采集并计算内参

校验线是 100 mm 时：

```bash
roslaunch d435_eye_on_hand intrinsic_calibration.launch
```

若打印有缩放，传入上面计算出的两个米制值：

```bash
roslaunch d435_eye_on_hand intrinsic_calibration.launch \
  square_size:=0.025 marker_size:=0.018
```

在 GUI 中让板覆盖画面中心、四角、四边，改变远近和倾角；保持板平整、清晰且
静止后采样。等 X、Y、Size、Skew 覆盖充分后点击 `CALIBRATE`，计算完成后点击
`SAVE`。不要点击 `COMMIT`，RealSense 驱动不接受写入该内参。

`SAVE` 会生成 `/tmp/calibrationdata.tar.gz`。将其校验并安装到项目：

```bash
rosrun d435_eye_on_hand import_intrinsics.py \
  /tmp/calibrationdata.tar.gz \
  --expected-width 1280 --expected-height 720
```

该命令会生成：

```text
catkin_ws/src/d435_eye_on_hand/config/d435_color_intrinsics.yaml
```

并把原始图像与计算结果归档到 `calibration_results/`。

### 4. 验证实测内参正在使用

```bash
roslaunch d435_eye_on_hand verify_intrinsics.launch
```

另开终端：

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/second_hand/catkin_ws/devel/setup.bash
rosrun d435_eye_on_hand check_intrinsics.py
```

只有看到 `D435 measured intrinsics are active and resolution-consistent` 才进入
第二阶段。

## 第二阶段：眼在手上手眼标定

这里使用原来的单个 ID 582 marker，不使用内参阶段的 ChArUco 板。把 ID 582
刚性固定在工作台上且全程不动；D435 必须已经刚性固定在 `tool0` 上。

启动完整标定：

```bash
cd /home/sjh/second_hand
source /opt/ros/noetic/setup.bash
source catkin_ws/devel/setup.bash
roslaunch d435_eye_on_hand calibrate_eye_on_hand.launch
```

该启动文件采用 freehand 模式，Easy Handeye 不会控制机械臂。它还会默认启动
Park 误差监视器：从第 3 组样本开始，每当样本数变化，都会强制使用
`OpenCV/Park` 重新计算并在启动终端输出误差表和至少一个离群点候选。另开终端
检查输入：

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/second_hand/catkin_ws/devel/setup.bash
rosrun d435_eye_on_hand check_handeye_inputs.py
```

在 `/d435_aruco_582/result` 中确认 ID 582 边框和坐标轴稳定。用示教器改变末端位置
与姿态，每次完全停止并等待约 1 秒再点 `Take Sample`。建议 20--25 组，绕不同
轴充分旋转，避免只有平移或姿态近似平行。每次取样后查看启动终端中的 Park
误差；离群点编号同时给出从 0 开始的索引和从 1 开始的样本号。候选只用于人工
复查，脚本不会自动删样本。

如果没有随启动文件运行监视器，或需要手动启动持续评估：

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/second_hand/catkin_ws/devel/setup.bash
rosrun d435_eye_on_hand evaluate_handeye_error.py --watch
```

脚本每次都强制 Park 算法计算 `tool0 <- d435_link`，不依赖 GUI 下拉框当前选项，
再检查各样本得到的固定
`base_link <- d435_aruco_marker_frame` 是否一致。终端会给出平移误差（mm）、旋转
误差（度）及每个样本的误差，并且每次至少给出一个离群点候选。候选不会被自动
删除；应先回到 Easy Handeye 样本列表核对，再决定是否删除和重新计算。详细报告
保存到：

```text
/home/sjh/second_hand/calibration_results/d435_handeye_park_error_时间戳.yaml
```

确认最终样本后，在标定 launch 仍运行时执行下面的收尾命令。它会再次强制使用
Park、输出最终误差，并把这一次结果保存为项目的当前矩阵：

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/second_hand/catkin_ws/devel/setup.bash
rosrun d435_eye_on_hand evaluate_handeye_error.py \
  --min-samples 15 \
  --save-latest
```

规范保存文件固定为
`~/.ros/easy_handeye/d435_ur3_eye_on_hand.yaml`。`publish_eye_on_hand.launch`、
`marker_to_base.py` 和 `complete_process/white_rectangle_pick.py` 默认都直接读取
这一文件，因此不需要手工复制矩阵；重新启动这些节点/脚本后就会加载刚保存的
最新结果。抓取脚本启动时还会打印读取路径、文件修改时间和平移量，便于排除仍在
使用旧矩阵。保存前脚本会把旧文件备份为同目录的 `.bak.时间戳` 文件，保存后再
回读核对矩阵和坐标系元数据。不要在少量或退化姿态下使用 `--save-latest`。

保存标定矩阵后，可参照 D455 项目的 marker-to-base 方法做动态验证：保持 582
固定，缓慢改变机器人位置和姿态，观察计算出的 `base_link <- 582` 是否保持不变：

```bash
rosrun d435_eye_on_hand marker_to_base.py
```

脚本以第一次有效测量为参考，持续显示平移漂移（mm）和旋转漂移（度）。只检查
一次可添加 `--once`。该脚本严格读取 D435 专用文件
`~/.ros/easy_handeye/d435_ur3_eye_on_hand.yaml`，不会读取或修改 D455 的结果。

最终结果保存为：

```text
~/.ros/easy_handeye/d435_ur3_eye_on_hand.yaml
```

发布并检查结果：

```bash
roslaunch d435_eye_on_hand publish_eye_on_hand.launch
rosrun tf tf_echo tool0 d435_link
```

相机相对 `tool0` 的安装发生任何变化后，必须重新做手眼标定；分辨率改变后必须
重新做相机内参标定。

## 主要文件

```text
calibration_targets/                 A4 ChArUco PDF/PNG
calibration_results/                 camera_calibration 原始归档
catkin_ws/src/d435_eye_on_hand/       本项目 launch、脚本和实机配置
catkin_ws/src/aruco_ros/              固定版本及本机必要修补
catkin_ws/src/easy_handeye/           固定版本及本机必要修补
complete_process/utils/base/          UR3 与 Robotiq 底层控制
complete_process/utils/basic_move/    基础机械臂动作
complete_process/utils/object_detective/  人工拖框、深度差分与 YOLO 数据采集
tools/generate_charuco_a4.py          A4 板可复现生成器
third_party.repos                     第三方源码固定提交
```

## 机械臂基础操作

项目现已按 `WorkpiecePlacementUR3` 的风格加入 `complete_process/utils/base` 和
`complete_process/utils/basic_move`，包含 UR3 状态读取、RTDE/URScript 控制、
moveJ、moveL、servoL 以及 Robotiq 夹爪基础接口。另已在
`complete_process/utils/object_detective` 中加入面向 D435 的人工拖框、白色矩形
深度差分及 YOLO 数据采集功能。

安装、安全检查及全部运行命令见 [项目操作说明.md](项目操作说明.md)。
深度差分、自动标注和 YOLO 训练说明见
[complete_process/utils/object_detective/README.md](complete_process/utils/object_detective/README.md)。
