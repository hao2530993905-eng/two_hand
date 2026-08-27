# Complete process

位姿均为 `[x, y, z, rx, ry, rz]`，位置单位为米，旋转向量和关节速度单位为弧度。
`--distance`、`--pick-distance` 和 `--place-distance` 是基坐标系 Z 方向的有符号距离：
负数下降，正数上升。

## 1. moveJ 到 TCP 位姿

脚本先求逆运动学，再通过 `moveJ` 移动：

```bash
python3 complete_process/utils/move_j_to_pose.py \
  --host 192.168.1.4 \
  --pose 0.30 -0.20 0.25 3.14159 0 0
```

## 2. moveL 指定距离

从当前 TCP 位姿沿基坐标系 Z 轴下降 5 cm：

```bash
python3 complete_process/utils/move_l_distance.py \
  --host 192.168.1.4 --distance -0.05
```

## 3. 完整抓取放置

```bash
python3 complete_process/pick_and_place.py \
  --host 192.168.1.4 \
  --pick-pose -0.258145 0.332829 0.032587 -0.028553 3.090451 -0.127509 \
  --pick-distance -0.065 \
  --place-pose 0.056649 -0.368066 0.090006 -0.070462 3.010281 -0.028414 \
  --place-distance -0.0 \
  --speed 0.03 \
  --joint-speed 0.3 \
  --gripper-force 255 \
  --linear-speed 0.05 \
  --joint-speed 0.4 \
  --retreat-after-place
```

流程为：打开夹爪 → moveJ 到抓取接近位姿 → moveL 到工件 → 关闭夹爪 →
moveL 返回抓取接近位姿 →
moveJ 到放置接近位姿 → moveL 到放置点 → 打开夹爪 → 可选撤回。

抓取后固定沿直线路径返回 `--pick-pose`，完成该步骤后才会前往放置区域，不需要额外开关参数。

机械臂的直线移动速度通过 `--speed` 设置，单位为 m/s；`--linear-speed` 是其兼容别名。
关节移动速度通过 `--joint-speed` 设置，单位为 rad/s。

首次运行请降低速度、缩短下降距离，并确认所有位姿、逆解和运动路径均安全。

## 4. YOLO 识别并移动到随机工件上方 3 cm

`detect_and_pick.py` 已包含 YOLO11n-OBB 实时推理，不需要另外启动
YOLO 测试节点或深度差分检测器。多目标出现时程序随机锁定一个，并在连续帧
中跟踪同一个目标。

先启动 RealSense：

```bash
source /opt/ros/noetic/setup.bash
roslaunch realsense2_camera rs_camera.launch \
  serial_no:=105322250617 \
  enable_color:=true \
  enable_depth:=true \
  enable_sync:=true \
  align_depth:=true
```

程序启动时会自动读取
`~/.ros/easy_handeye/ur3_d455_handeye_eye_on_base.yaml` 中最新保存的
`base_link <- camera_link` 标定结果；RealSense 节点发布的
`camera_link <- camera_color_optical_frame` 内部 TF 会在运行时自动拼接，因此
默认不需要另外启动手眼标定 TF publisher。YAML 不存在时才使用代码中的备用值。

先进行不运动的坐标检查：

```bash
cd /home/sjh/WorkpiecePlacementUR3
source /opt/ros/noetic/setup.bash

MPLCONFIGDIR=/tmp/matplotlib-obb \
ROS_HOME=/tmp/ros-obb \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -u \
complete_process/detect_and_pick.py \
  --device 0 \
  --confidence 0.45 \
  --workspace -0.45 0.45 0.10 0.55 0.02 0.55 \
  --above-height 0.03
```

需要查看实时检测框时，另开终端运行：

```bash
source /opt/ros/noetic/setup.bash
rqt_image_view /detect_and_pick/debug_image
```

确认打印的中心像素、相机坐标、基坐标及 TCP 目标位姿均正确。不带
`--execute` 时程序会持续发布实时 YOLO 图像，按 `Ctrl+C` 才结束。确认安全后，
加入 `--execute`；程序会再次要求输入 `MOVE` 才移动：

```bash
cd /home/sjh/WorkpiecePlacementUR3
source /opt/ros/noetic/setup.bash

MPLCONFIGDIR=/tmp/matplotlib-obb \
ROS_HOME=/tmp/ros-obb \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -u \
complete_process/detect_and_pick.py \
  --host 192.168.1.4 \
  --device 0 \
  --confidence 0.45 \
  --workspace -0.45 0.45 0.10 0.55 0.02 0.55 \
  --above-height 0.03 \
  --joint-speed 0.20 \
  --joint-acc 0.35 \
  --motion-timeout 60 \
  --execute
```

本程序只执行一次 `moveJ` 到工件表面中心上方 3 cm，不下降、不闭合夹爪。
调试图发布在 `/detect_and_pick/debug_image`，最终三维结果发布在
`/detect_and_pick/result`。若需要补偿 TCP 与视觉点之间的固定偏差，可使用
`--target-offset DX DY DZ`（基坐标系、单位米）。

不要依赖 `--show-gui` 打开 OpenCV 窗口：Conda OpenCV 与 ROS Qt 可能产生
`QObject::moveToThread` 冲突。该兼容参数现在会被忽略，可视化统一通过上述
`rqt_image_view` 命令查看。

### 深度差分自动采集 YOLO-OBB 数据

1. 启动 RealSense，并确认两个话题都有稳定频率：

```bash
rostopic hz /camera/color/image_raw
rostopic hz /camera/aligned_depth_to_color/image_raw
```

2. 启动带采集功能的检测程序：

```bash
source /opt/ros/noetic/setup.bash

roslaunch realsense2_camera rs_camera.launch \
  serial_no:=105322250617 \
  enable_color:=true \
  enable_depth:=true \
  enable_sync:=true \
  align_depth:=true
```
```bash
cd /home/sjh/WorkpiecePlacementUR3
source /opt/ros/noetic/setup.bash

/usr/bin/python3 -u \
  complete_process/utils/object_detective/depth_background_rect_detector.py \
  _color_topic:=/camera/color/image_raw \
  _depth_topic:=/camera/aligned_depth_to_color/image_raw \
  _background_samples:=20 \
  _min_height_mm:=12 _max_height_mm:=300 \
  _median_size:=5 _open_size:=3 _close_size:=11 \
  _min_area:=3000 _max_area:=1000000 \
  _min_aspect:=1.8 _max_aspect:=6.0 \
  _min_rectangularity:=0.45 \
  _max_objects:=10 \
  _collect_dataset:=true \
  _dataset_dir:=/home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/obb_dataset \
  _dataset_split:=train \
  _show_gui:=true
```

3. 清空检测区域，点击 `Depth detection` 窗口并按 `B`，等待空背景采集完成。
4. 只放入目标工件，确认所有绿色旋转框正确后按 `S` 保存正样本。
5. 改变工件位置、方向和数量，重复按 `S`。每次按键只保存当前一帧。
6. 移走所有目标，放入容易误检的其他物品，按 `N` 保存负样本（空标签）。
7. 若刚保存的样本有误，立即按 `D` 删除本次运行中最近保存的一组文件。

按键说明：

- `B`：手动采集空深度背景。
- `S`：保存原始 RGB、当前所有旋转框标签和预览图；没有检测框时拒绝保存。
- `N`：保存负样本，标签文件为空。
- `D`：撤销本次运行中最近一次保存。
- `R`：恢复全图 ROI。

数据保存在：

```text
obb_dataset/
  dataset.yaml
  images/train/
  labels/train/
  previews/train/
```

`images` 是无框原图，`labels` 是归一化 YOLO-OBB 四角标签，`previews` 只供
人工检查。训练前浏览所有预览图，删除框不完整、框粘连或包含非目标物体的
图片及其同名标签。

验证集需要单独采集，不要复制训练集。重新启动程序时改为：

```text
_dataset_split:=val
```

建议采集 300～500 张正样本和 100～200 张负样本。不要连续保存大量没有改变
的画面；每次保存前改变位置、角度、工件数量或光照。

在 `rqt_image_view` 中选择下面这个彩色话题：

```text
/depth_background_rect_detector/debug_image
```

`/depth_background_rect_detector/mask` 是二值掩膜，没有候选时全黑是正常的。
若 `Depth detection` 也没有彩色画面，并且终端显示
`Waiting for synchronized color and aligned depth images`，说明相机话题名或
RGB/深度同步配置不正确。



source ~/catkin_ws/devel/setup.bash
