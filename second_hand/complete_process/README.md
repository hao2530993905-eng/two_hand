# 机械臂基础操作层

本目录沿用 `WorkpiecePlacementUR3/complete_process` 的分层方式。当前包含
机械臂基础能力和面向 D435 的白色矩形深度差分数据采集，不包含任何 D455、
红色工件或已有 YOLO 模型。

```text
utils/base/              UR3、RTDE/URScript、Robotiq 通信封装
utils/basic_move/        可独立运行的基础动作脚本
utils/object_detective/  D435 人工拖框、深度差分及 YOLO 数据采集
```

默认设备是 `192.168.1.5` 的 UR3 CB3。TCP 位姿统一表示为
`[x, y, z, rx, ry, rz]`：位置单位为米，姿态为弧度制旋转向量。

完整的安装、安全检查和运行命令见项目根目录的 `项目操作说明.md`。
