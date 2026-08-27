# 工件检测：深度差分采集与 YOLO-OBB 训练

本目录提供两套互补功能：

- `depth_background_rect_detector.py`：利用 RealSense 对齐深度图与空工作台背景做差，检测凸出工作台的矩形工件，并可自动生成 YOLO-OBB 数据集。
- `test_yolo_obb_live.py`：加载训练完成的 YOLO-OBB 权重，对 ROS 彩色图像进行实时检测。

当前数据集只有一个类别：`workpiece`，类别编号为 `0`。以下命令默认项目位于：

```text
/home/sjh/WorkpiecePlacementUR3
```

## 1. 启动相机

先启动 ROS Master：

```bash
source /opt/ros/noetic/setup.bash
roscore
```

另开终端启动 D455。深度差分必须同时开启彩色、深度、同步和深度对齐：

```bash
source /opt/ros/noetic/setup.bash
source /home/sjh/WorkpiecePlacementUR3/catkin_ws/devel/setup.bash
roslaunch ur3_d455_handeye camera.launch
```

确认以下话题均有数据：

```bash
rostopic hz /camera/color/image_raw
rostopic hz /camera/aligned_depth_to_color/image_raw
rostopic hz /camera/color/camera_info
```

## 2. 使用深度差分检测工件

运行：

```bash
cd /home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective
source /opt/ros/noetic/setup.bash
python3 depth_background_rect_detector.py
```

首次建立背景：

1. 移走工作台上的全部工件，确保视野内只剩静态背景。
2. 鼠标点击 `Depth detection` 窗口，使其获得键盘焦点。
3. 按 `B`，程序采集 20 帧深度并取中值。
4. 背景默认保存为本目录下的 `depth_background.npy`。
5. 放回工件，绿色旋转框应包围凸出背景的矩形区域。

界面快捷键：

| 按键 | 功能 |
| --- | --- |
| `B` | 重新采集空工作台背景 |
| `R` | 将 ROI 恢复到整幅图像 |
| `S` | 数据采集模式下保存当前正样本及 OBB 标签 |
| `N` | 保存负样本，标签文件为空 |
| `D` | 撤销本次运行中最后保存的样本 |

常用 ROS 参数：

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

- `min_height_mm/max_height_mm`：工件相对背景的有效高度范围。
- `target_color`：颜色预设，可选 `none`、`red`、`orange`、`yellow`、`green`、`blue`、`purple`；`none` 表示不限制颜色。
- `min_area`：候选轮廓的最小像素面积，默认 `3000`，用于排除小噪声。
- `roi_x/roi_y/roi_width/roi_height`：限制检测区域，避开机械臂、电缆等干扰物。

当前版本不按长宽比、最大面积或矩形度过滤，但会应用最小面积限制。指定颜色后，候选区域必须同时满足深度高度差和对应的 HSV 颜色条件。仍保留小于 2 像素的退化轮廓检查和 `max_objects` 数量上限。建议先限制 ROI，再调高度、颜色和形态学去噪参数。相机或工作台位置改变后必须重新按 `B` 采集背景。

切换颜色时只需修改参数，例如：

```bash
python3 depth_background_rect_detector.py _target_color:=blue _min_area:=3000
python3 depth_background_rect_detector.py _target_color:=green _min_area:=3000
python3 depth_background_rect_detector.py _target_color:=yellow _min_area:=3000
```

节点输出：

```text
/depth_background_rect_detector/debug_image  标注后的彩色图
/depth_background_rect_detector/mask         深度高度差二值图
/depth_background_rect_detector/result       JSON 检测结果
/depth_background_rect_detector/corners      所有 OBB 角点
```

## 3. 用深度差分自动采集 YOLO-OBB 数据

采集训练集：

```bash
cd /home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective
source /opt/ros/noetic/setup.bash
python3 depth_background_rect_detector.py \
  _collect_dataset:=true \
  _dataset_name:=red_workpiece_dataset \
  _dataset_split:=train
```

采集验证集时改为：

```bash
python3 depth_background_rect_detector.py \
  _collect_dataset:=true \
  _dataset_name:=red_workpiece_dataset \
  _dataset_split:=val
```

`dataset_name` 是保存在当前脚本目录下的数据集文件夹名称。例如上述命令会写入：

```text
complete_process/utils/object_detective/red_workpiece_dataset/
```

名称只能是单个文件夹名，不能包含 `/`。如果需要保存到其他位置，可以用完整路径覆盖：

```bash
python3 depth_background_rect_detector.py \
  _collect_dataset:=true \
  _dataset_dir:=/home/sjh/datasets/red_workpiece \
  _dataset_split:=train
```

操作流程：

1. 清空工作台并按 `B` 建立背景。
2. 放置一个或多个工件，检查绿色 OBB 是否准确。
3. 按 `S` 保存正样本。
4. 改变工件数量、位置、旋转角度、间距和光照，重复保存。
5. 适量加入没有工件、机械臂进入画面或存在易混淆物体的场景，按 `N` 保存负样本。
6. 独立采集验证集；不要把同一静止场景的连续帧同时分到训练集和验证集。

文件将保存到：

```text
obb_dataset/
├── dataset.yaml
├── images/{train,val}/
├── labels/{train,val}/
└── previews/{train,val}/
```

标签采用 Ultralytics YOLO-OBB 格式：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

八个坐标均按图像宽高归一化到 `[0,1]`。训练前应查看 `previews`，删除框错、漏框严重或深度噪声生成的样本。

## 4. 训练 YOLO-OBB

项目当前使用以下 Conda 环境：

```text
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone
```

确认数据配置：

```bash
sed -n '1,80p' \
  /home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/red_workpiece_dataset/dataset.yaml
```

开始训练：

```bash
cd /home/sjh/WorkpiecePlacementUR3
MPLCONFIGDIR=/tmp/matplotlib-obb \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -c \
"from ultralytics import YOLO; YOLO('yolo11n-obb.pt').train(data='complete_process/utils/object_detective/red_workpiece_dataset/dataset.yaml', epochs=120, patience=35, batch=4, imgsz=640, device=0, workers=2, project='/home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/runs', name='red_box')"
```

如果显存不足，把 `batch=4` 改为 `batch=2` 或 `batch=1`。没有可用 CUDA 时将 `device=0` 改为 `device='cpu'`，但训练会明显变慢。

训练结果位于：

```text
complete_process/utils/object_detective/runs/red_box/
├── weights/best.pt
├── weights/last.pt
├── results.csv
├── results.png
└── confusion_matrix.png
```

部署时优先使用验证指标最好的 `best.pt`：

```bash
cp complete_process/utils/object_detective/runs/red_box/weights/best.pt \
   complete_process/utils/object_detective/models/red_box.pt
```

`detect_and_pick.py` 默认读取红色工件模型 `models/red_box.pt`。原白色工件模型保存为 `models/white_box.pt`，需要切换时可通过 `--model` 或 ROS `_model` 参数显式指定。

## 5. 验证模型效果

建议按“离线指标 → 验证集预测图 → 相机实时测试”的顺序验证。最终判断应以未参与训练的真实场景为准，不能只看训练指标。

### 5.1 离线验证红色模型指标

```bash
cd /home/sjh/WorkpiecePlacementUR3
MPLCONFIGDIR=/tmp/matplotlib-red-box-val \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -c \
"from ultralytics import YOLO; YOLO('complete_process/utils/object_detective/models/red_box.pt').val(data='complete_process/utils/object_detective/red_workpiece_dataset/dataset.yaml', imgsz=640, device=0, project='/home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/runs', name='red_box_validation')"
```

重点查看输出中的：

- `Precision`：预测出的目标有多少是真的，低表示误检多。
- `Recall`：真实目标有多少被找到，低表示漏检多。
- `mAP50`：IoU 0.5 下的总体检测指标。
- `mAP50-95`：更严格的框位置与角度综合指标，OBB 模型更应关注这一项。

结果保存在：

```text
complete_process/utils/object_detective/runs/red_box_validation/
```

当前验证集只有 3 张图片，因此即使指标很高，也不能证明模型已经具有良好泛化能力。

### 5.2 生成验证集预测图

```bash
cd /home/sjh/WorkpiecePlacementUR3
MPLCONFIGDIR=/tmp/matplotlib-red-box-predict \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -c \
"from ultralytics import YOLO; YOLO('complete_process/utils/object_detective/models/red_box.pt').predict(source='complete_process/utils/object_detective/red_workpiece_dataset/images/val', imgsz=640, conf=0.45, device=0, save=True, project='/home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/runs', name='red_box_predictions')"
```

逐张查看：

```text
complete_process/utils/object_detective/runs/red_box_predictions/
```

确认旋转框完整包围工件、中心位置正确、长短边方向正确，并且背景中没有多余检测框。

### 5.3 使用 D455 实时验证

先确保 `roscore` 和 D455 相机节点正在运行，并确认彩色话题有数据：

```bash
source /opt/ros/noetic/setup.bash
rostopic hz /camera/color/image_raw
```

运行红色模型：

```bash
cd /home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective
source /opt/ros/noetic/setup.bash
MPLCONFIGDIR=/tmp/matplotlib-obb \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -u \
test_yolo_obb_live.py \
  _model:=/home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/models/red_box.pt \
  _device:=0 \
  _confidence:=0.45 \
  _show_gui:=false
```

另开终端查看实时检测图：

```bash
source /opt/ros/noetic/setup.bash
rqt_image_view /test_yolo_obb_live/debug_image
```

检测 JSON 发布在：

```bash
rostopic echo /test_yolo_obb_live/result
```

切换到白色模型时只修改模型路径：

```bash
cd /home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective
source /opt/ros/noetic/setup.bash
MPLCONFIGDIR=/tmp/matplotlib-white-box \
/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python -u \
test_yolo_obb_live.py \
  _model:=/home/sjh/WorkpiecePlacementUR3/complete_process/utils/object_detective/models/white_box.pt \
  _device:=0 \
  _confidence:=0.45 \
  _show_gui:=false
```

实时测试时至少覆盖以下情况：

1. 单个工件位于画面中心、边缘和四角。
2. 工件以多个不同角度旋转。
3. 两个或多个工件相邻摆放。
4. 改变光照亮度并制造轻微阴影。
5. 加入颜色相近但不是目标的物体，检查误检。
6. 加入部分遮挡，检查框是否仍稳定。

记录每种场景的目标总数、正确检测数、漏检数和误检数。实际使用时应重点观察旋转框的中心和短边方向是否连续稳定，因为它们会直接影响机械臂抓取点和夹爪角度。

如果漏检较多，可先把 `_confidence` 降至 `0.25` 判断是阈值问题还是训练数据问题；如果降低后出现大量误检，应补充训练数据，而不是继续降低阈值。若旋转框方向不稳定，应增加不同角度、遮挡和相邻摆放的训练样本。

## 6. 常见问题

### 一直提示等待同步图像

检查对齐深度话题：

```bash
rostopic hz /camera/aligned_depth_to_color/image_raw
```

必须使用开启 `enable_depth`、`align_depth` 和 `enable_sync` 的相机配置。

### 深度差分把背景也识别为工件

- 清空工作台后重新按 `B`。
- 确保相机和工作台没有发生位移。
- 提高 `min_height_mm`。
- 缩小 ROI，排除机械臂和线缆。

### 自动标签框不准确

深度差分标签只是初始标签。训练质量取决于标签质量，必须检查 `previews`；错误标签应删除或使用支持 OBB 的标注工具重新修正。

### YOLO 检测框正确但抓取方向错误

YOLO 的 OBB 角度位于相机图像坐标系，不能直接作为 UR 腕关节角度。抓取程序还需完成相机到机械臂基座的方向变换，并补偿 TCP 到夹爪两指连线的固定安装角。
