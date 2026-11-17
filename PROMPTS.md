# Huawei 双臂数据与训练 Prompt 说明

整理日期：2026 年 11 月 18 日。



## 当前训练实际使用的五种指令

表中的 T1 至 T5 仅用于本文对应不同指令，不是数据文件中的任务编号。

| 编号 | 中文含义 | 训练 prompt 原文 |
| --- | --- | --- |
| T1 | 左臂拿起盒盖，右臂扶稳盒底，然后盖盒 | `Pick up the lid with the left arm, hold the box base steady with the right arm, and close the box.` |
| T2 | 右臂拿起手机并放进盒子 | `Pick up the phone with the right arm and place it into the box.` |
| T3 | 左臂拿起盒盖，对齐并盖上盒子 | `Pick up the lid with the left arm, align and close the box.` |
| T4 | 左臂取下放在盒底上的盒盖，并放到一旁 | `Remove the lid resting on the box base with the left arm and set it aside.` |
| T5 | 左臂拿起手机并放进盒子 | `Pick up the phone with the left arm and place it into the box.` |



## 原始数据中的四种任务描述

以下文本来自原始数据各自的 `meta/tasks.jsonl`，同一数据集的 `all` 和 `success` 目录中任务文本相同。O1 至 O4 是本文编号。

| 编号 | 原始数据集 | 原始 prompt 原文 |
| --- | --- | --- |
| O1 | `close_box_1112_2` | `Close_the_phone_box_with_both_hands.` |
| O2 | `open_the_box_1109_60` | `open_the_box_with_left_arm` |
| O3 | `place_phone_1111_60_2`、`place_phone_1112_10` | `Place_the_phone_back_into_the_phone_box_with_left_arm` |
| O4 | `hw_close_box_1111` | 见下方完整文本 |

O4 的完整原文为：

```text
Pick up the phone with the right arm and place it into the box. Adjust the camera to fully view the side of the box. Pick up the lid with the left arm, align and close the box, then place the closed box onto the weighing pad.
```

中文含义：右臂拿起手机并放入盒中，调整相机以完整看到盒子的侧面；左臂拿起盒盖，对齐并盖盒，然后将盖好的盒子放到称重垫上。

