# UR3 + RealSense D455 眼在手外标定

本包只适用于当前实测设备：UR3 CB3 (`192.168.1.4`) 和 RealSense D455
(`105322250617`)。标定板是 Original ArUco ID 582，黑色方形外边长 48 mm，
固定在机械臂末端。标定结果方向为 `base_link <- camera_link`；RealSense 驱动
再提供 `camera_link <- camera_color_frame <- camera_color_optical_frame`，因此
整个相机 TF 子树都能正确连接到机器人基座。
彩色原图会先由 `image_proc` 使用 D455 的五项畸变参数校正，然后才进行 ArUco
位姿估计。

## 启动

```bash
cd /home/sjh/WorkpiecePlacementUR3/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch ur3_d455_handeye calibrate.launch
```

启动后会出现三个窗口：Easy Handeye 采样窗口、以 `base_link` 为固定坐标系的
RViz，以及显示 `/aruco_single/result` 的实时 ArUco 识别图。识别图中必须能看到
标定板边框和坐标轴，才可以点击 `Take Sample`。

该配置采用 freehand 模式，Easy Handeye 不会自动移动机械臂。保持标定板完整、
清晰地出现在彩色画面内，在示教器上改变末端姿态，稳定后点击 `Take Sample`。
至少采集 15 组，建议 20--25 组；绕不同轴充分改变姿态，避免只做平移或所有姿态
近似平行。每次采样前先停止运动并等待约 1 秒。

采样前可在另一个终端检查输入：

```bash
source /home/sjh/WorkpiecePlacementUR3/catkin_ws/devel/setup.bash
rosrun ur3_d455_handeye check_setup.py
```

采样完成后在 GUI 中依次点击 `Compute` 和 `Save`。结果保存在：

```text
~/.ros/easy_handeye/ur3_d455_handeye_eye_on_base.yaml
```

## 发布保存的结果

```bash
roslaunch ur3_d455_handeye publish.launch
rosrun tf tf_echo base_link camera_color_optical_frame
```

相机或标定板的固定位置一旦改变，必须重新标定。Marker 尺寸必须使用实测的
0.048 m，而不是生成网页中填写的 0.050 m。

## Marker 坐标转换与末端位姿

相机、UR3 驱动和 ArUco 节点运行时，持续打印 Marker 在相机/基座下的位姿及
当前 `tool0` 位姿：

```bash
rosrun ur3_d455_handeye marker_to_base.py
```

只读取一帧后退出：

```bash
rosrun ur3_d455_handeye marker_to_base.py --once
```
