# two_hand 双臂协同

本工程把两套 UR3 流程组合为一个自包含任务：

1. `192.168.1.5` 的 `second_hand` 机械臂用 D435 + YOLO-OBB 识别并抓取白色工件；
2. `second_hand` 把工件送到固定交接位姿；
3. `192.168.1.4` 的放置机械臂到达交接位并夹紧；
4. 等待 `put_second_hand_time` 后，`second_hand` 松爪并立即返回观察位姿；
5. 从松爪完成起等待 `wait_second_hand`，放置臂开始撤离并将工件放入深度差分识别出的红盒目标槽。

主入口是 `complete_process/two_hand_pick_and_place.py`。不带 `--execute`
时只识别和规划，不会发送运动或夹爪命令；带 `--execute` 后默认仍要求一次
`MOVE` 安全确认，`--full-auto` 才会取消该确认。

完整的 ROS、相机、识别预览、背景采集、干运行、实机启动和参数说明见
[操作指南](操作指南.md)。

常用入口：

```bash
cd /home/sjh/two_hand
./scripts/start_cameras.sh       # 两台相机
./scripts/start_white_preview.sh # 白色工件 YOLO 预览
./scripts/start_red_preview.sh   # 红盒深度差分预览
./scripts/start_two_hand.sh      # 干运行
./scripts/start_two_hand.sh --execute
```

工程没有从 `/home/sjh/second_hand` 或 `/home/sjh/WorkpiecePlacementUR3`
导入模块、模型或运行时文件；所需代码、模型、背景和标定副本均位于本目录。
