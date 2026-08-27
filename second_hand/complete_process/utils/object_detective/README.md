# D435 白色矩形物品深度差分与 YOLO-OBB 数据采集

本目录仿照 `WorkpiecePlacementUR3` 的深度背景差分流程，但只接受同时满足以下
条件的候选：

```text
相对空工作台有有效高度差
AND HSV 低饱和度、高亮度（白色）
AND 面积、长宽比、矩形度、实心度过滤
```

程序将候选拟合为旋转矩形，并可保存 Ultralytics YOLO-OBB 所需的原始图片和
四角标签。

## 人工拖框采集（当前推荐）

如果深度差分自动框不稳定，直接运行人工标注入口。它会自动启动或复用
`roscore`，并在需要时启动 D435 彩色相机：

```bash
cd /home/sjh/second_hand
./complete_process/utils/object_detective/start_manual_annotation.sh \
  --split train
```

不需要提前运行 `roscore` 或 `roslaunch`。如果相机已经在发布
`/d435/color/image_raw`，工具会复用现有相机，不会重复启动。

窗口操作：

| 操作 | 功能 |
| --- | --- |
| 第一次左键拖动 | 自动冻结；沿物体长轴从一端中点拖到另一端中点 |
| 松开后移动鼠标 | 垂直于长轴移动，实时调节旋转框宽度 |
| 再单击一次左键 | 确认当前倾斜框 |
| 重复以上操作 | 在同一帧继续添加其他目标框 |
| 鼠标右键点击框内 | 删除点击到的框 |
| `Space` | 手动冻结/恢复实时画面；恢复时清空未保存框 |
| `S` 或 `Enter` | 保存带框正样本，随后自动恢复实时画面 |
| `N` | 保存当前画面为负样本，标签为空 |
| `U` 或退格 | 删除最后一个未保存框 |
| `C` | 清空当前帧的全部框 |
| `D` | 撤销本次运行最后保存的一组文件 |
| `Q` 或 `Esc` | 退出，并关闭工具自己启动的相机/roscore |

默认数据集独立保存到：

```text
manual_white_rectangle_dataset/
├── dataset.yaml
├── images/{train,val}/
├── labels/{train,val}/
└── previews/{train,val}/
```

`images` 是训练原图，`labels` 是标签，`previews` 仅用于检查框。默认
`--format obb` 和旋转框模式，四个实际倾斜角点会写入 YOLO-OBB 标签，能直接
沿用本项目 `yolo11n-obb.pt` 训练流程。长轴拖反方向不影响标签；宽度调节阶段
可以向长轴任一侧移动。

采集独立验证集：

```bash
./complete_process/utils/object_detective/start_manual_annotation.sh \
  --split val
```

训练人工数据：

```bash
cd /home/sjh/second_hand
MPLCONFIGDIR=/tmp/matplotlib-manual-white-obb \
python3 -c "from ultralytics import YOLO; YOLO('yolo11n-obb.pt').train(data='/home/sjh/second_hand/complete_process/utils/object_detective/manual_white_rectangle_dataset/dataset.yaml', epochs=120, patience=35, batch=4, imgsz=640, device=0, workers=2, project='/home/sjh/second_hand/complete_process/utils/object_detective/runs', name='manual_white_rectangle')"
```

当前人工数据集已经完成一次 YOLO11n-OBB 训练。固定模型路径为：

```text
models/manual_white_rectangle_best.pt  最佳验证权重，部署时使用
models/manual_white_rectangle_last.pt  早停前最后一轮权重，用于续训
```

使用最佳模型测试一张图片：

```bash
cd /home/sjh/second_hand
python3 -c "from ultralytics import YOLO; YOLO('complete_process/utils/object_detective/models/manual_white_rectangle_best.pt').predict(source='待测试图片.jpg', imgsz=640, conf=0.25, save=True, project='complete_process/utils/object_detective/runs', name='predict')"
```

本次训练使用 13 张训练图和 4 张验证图，在第 60 轮触发早停，最佳结果位于
第 30 轮。验证集过小且画面相似时，指标会明显偏乐观；实际部署前应补充不同
距离、角度、光照和背景下的数据。

### 通过 rqt_image_view 查看实时 YOLO-OBB

先使用 `d435_rgbd.launch` 启动相机，再在新终端运行：

```bash
cd /home/sjh/second_hand
./complete_process/utils/object_detective/start_yolo_obb_ros.sh
```

识别节点订阅 `/d435/color/image_raw`，并将带旋转框的画面发布到：

```text
/white_rectangle_yolo_obb/debug_image
```

第三个终端查看识别画面：

```bash
source /opt/ros/noetic/setup.bash
rqt_image_view /white_rectangle_yolo_obb/debug_image
```

置信度、模型或输入话题可直接通过 ROS 私有参数调整：

```bash
./complete_process/utils/object_detective/start_yolo_obb_ros.sh \
  _conf:=0.35 \
  _iou:=0.45 \
  _imgsz:=640 \
  _device:=cpu \
  _image_topic:=/d435/color/image_raw
```

默认模型是 `models/manual_white_rectangle_best.pt`。该节点直接转换 ROS 图像，
不依赖 Conda 环境中存在二进制兼容问题的 `cv_bridge`。

如果明确要训练普通水平框 YOLO，可以在一个全新的数据集目录中使用
`--format detect`：

```bash
./complete_process/utils/object_detective/start_manual_annotation.sh \
  --dataset-name manual_white_detect_dataset \
  --format detect \
  --split train
```

同一数据集禁止混用 `obb` 和 `detect` 标签格式。

如果临时仍想使用水平拖框：

```bash
./complete_process/utils/object_detective/start_manual_annotation.sh \
  --format obb \
  --box-mode axis \
  --split train
```

## 1. 启动 D435 RGB-D

终端 1：

```bash
source /opt/ros/noetic/setup.bash
roscore
```

终端 2：

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/second_hand/catkin_ws/devel/setup.bash
roslaunch d435_eye_on_hand d435_rgbd.launch
```

该启动文件开启彩色、深度、同步和深度到彩色对齐，并保留项目实测 D435 彩色
内参。确认三个话题有数据：

```bash
rostopic hz /d435/color/image_raw
rostopic hz /d435/aligned_depth_to_color/image_raw
rostopic hz /d435/color/camera_info
```

深度和彩色图必须宽高一致。相机、支架或工作台移动后，旧背景不可继续使用。

## 2. 只运行检测

终端 3：

```bash
cd /home/sjh/second_hand
source /opt/ros/noetic/setup.bash
python3 complete_process/utils/object_detective/depth_background_rect_detector.py
```

首次运行：

1. 清空工作台，画面内不要保留工件、机械臂或移动物。
2. 点击 `Depth detection` 窗口，使其获得键盘焦点。
3. 按 `B` 采集 20 帧空工作台深度背景并取中值。
4. 放入白色矩形物品，绿色旋转框应紧贴物品边缘。

背景默认保存为：

```text
complete_process/utils/object_detective/depth_background.npy
```

节点输出：

```text
/depth_background_rect_detector/debug_image  带检测框的彩色图
/depth_background_rect_detector/mask         深度与白色条件相交后的掩膜
/depth_background_rect_detector/result       JSON 检测结果
/depth_background_rect_detector/corners      所有 OBB 四角坐标
```

也可以通过 ROS 查看调试图：

```bash
rqt_image_view /depth_background_rect_detector/debug_image
```

## 3. 采集 YOLO-OBB 数据

采集训练集：

```bash
cd /home/sjh/second_hand
source /opt/ros/noetic/setup.bash
python3 complete_process/utils/object_detective/depth_background_rect_detector.py \
  _collect_dataset:=true \
  _dataset_name:=white_rectangle_dataset \
  _dataset_split:=train
```

快捷键：

| 按键 | 功能 |
| --- | --- |
| `B` | 清空工作台后重新采集深度背景 |
| `R` | ROI 恢复到整幅图像 |
| `S` | 保存当前正样本、全部 OBB 标签和预览图 |
| `N` | 保存负样本，标签文件为空 |
| `D` | 撤销本次运行最后保存的一组文件 |

保存结构：

```text
white_rectangle_dataset/
├── dataset.yaml
├── images/{train,val}/
├── labels/{train,val}/
└── previews/{train,val}/
```

- `images` 保存无框的 95% JPEG 原图，供 YOLO 训练。
- `labels` 保存归一化的 YOLO-OBB 四角标签。
- `previews` 保存带绿色框的检查图，不参与训练。
- 正样本没有检测框时会拒绝保存。
- `N` 会保存原图和空标签，适合加入白色背景、机械臂、反光等困难负样本。
- `D` 只撤销当前进程启动后保存的最后一组文件。

标签格式：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

当前只有一个类别：`0: white_rectangle`。

采集验证集时重新启动程序，将 split 改为 `val`：

```bash
python3 complete_process/utils/object_detective/depth_background_rect_detector.py \
  _collect_dataset:=true \
  _dataset_name:=white_rectangle_dataset \
  _dataset_split:=val
```

验证集必须换位置、角度、光照或场景单独采集，不能把训练集连续帧复制过去。

## 4. 参数调节

常用参数及默认值：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `_min_height_mm` | 12 | 相对背景的最小凸起高度 |
| `_max_height_mm` | 0 | 最大深度差；0 表示不限制，适合当前远墙背景 |
| `_background_samples` | 20 | 背景中值帧数 |
| `_median_size` | 5 | 深度中值滤波核 |
| `_open_size` | 3 | 开运算去除小噪声 |
| `_close_size` | 11 | 闭运算连接物品区域 |
| `_white_max_saturation` | 120 | HSV 白色最大饱和度 |
| `_white_min_value` | 90 | HSV 白色最小亮度 |
| `_min_area` | 1200 | 最小轮廓面积，像素 |
| `_max_area` | 0 | 最大面积，0 表示不限制 |
| `_min_aspect` | 1.0 | 最小长短边比，允许接近正方形 |
| `_max_aspect` | 6.0 | 最大长短边比 |
| `_min_rectangularity` | 0.55 | 轮廓面积/最小外接矩形面积 |
| `_min_solidity` | 0.60 | 轮廓面积/凸包面积 |
| `_max_objects` | 10 | 单帧最大输出数量 |
| `_roi_x/y/width/height` | 全图 | 检测区域 |

示例：

```bash
python3 complete_process/utils/object_detective/depth_background_rect_detector.py \
  _collect_dataset:=true \
  _dataset_split:=train \
  _min_height_mm:=10 \
  _max_height_mm:=100 \
  _white_max_saturation:=75 \
  _white_min_value:=135 \
  _min_area:=2500 \
  _min_rectangularity:=0.75
```

调参顺序建议：

1. 先用 ROI 排除机械臂、桌边和线缆。
2. 调整高度范围，只留下高于工作台的区域。
3. 阴影造成白色物品断裂时，适当降低 `_white_min_value` 或提高
   `_white_max_saturation`。
4. 非白物体进入掩膜时，反向收紧这两个白色阈值。
5. 框被切碎时适当增大 `_close_size`；相邻物品粘连时减小它。
6. 最后微调面积和矩形度。

## 5. 数据质量检查

深度差分生成的是自动初始标签，不应不检查就开始训练。逐张浏览
`previews/{train,val}`：

- 框必须完整覆盖物品，四角顺序连续；
- 一个物品不能被拆成多个框；
- 相邻物品不能粘成一个框；
- 非白色或非矩形物品不能被标注；
- 遮挡严重、深度空洞或反光导致错误的样本应删除或人工修正；
- 删除错误预览图时，也要删除 `images` 和 `labels` 中的同名文件。

建议覆盖单个/多个物品、不同旋转角、画面中心和边缘、不同间距、不同光照、轻微
遮挡以及困难负样本。不要连续保存大量完全相同的静止画面。

## 6. 训练 YOLO-OBB

安装或进入已有的 Ultralytics 环境后：

```bash
cd /home/sjh/second_hand
MPLCONFIGDIR=/tmp/matplotlib-white-obb \
python3 -c "from ultralytics import YOLO; YOLO('yolo11n-obb.pt').train(data='complete_process/utils/object_detective/white_rectangle_dataset/dataset.yaml', epochs=120, patience=35, batch=4, imgsz=640, device=0, workers=2, project='complete_process/utils/object_detective/runs', name='white_rectangle')"
```

训练结果中的推荐权重：

```text
complete_process/utils/object_detective/runs/white_rectangle/weights/best.pt
```

显存不足时降低 `batch`；没有 CUDA 时将 `device=0` 改为 `device='cpu'`。训练前
必须确保同时存在独立的 train 和 val 数据，且 `dataset.yaml` 中的路径指向当前
项目。

## 7. 常见问题

### 一直等待同步图像

检查 `/d435/color/image_raw` 和
`/d435/aligned_depth_to_color/image_raw`，并确认使用的是
`d435_rgbd.launch`，不是默认关闭深度的标定相机入口。

### 整个白色桌面都进入颜色掩膜

这是正常的颜色结果，但最终掩膜还必须满足高度差。清空工作台重新按 `B`；若相机
或桌面发生过移动，删除旧背景或直接重新采集。

### 白色物品只有一部分进入掩膜

适当降低 `_white_min_value`、提高 `_white_max_saturation` 或增大
`_close_size`。调完后检查是否引入背景误检。

### 有轮廓但没有绿色框

候选没有通过面积、长宽比、矩形度或实心度过滤。查看 mask 后逐项放宽对应参数，
但不要为了增加框数量而接受明显错误的自动标签。
