# LeRobot 数据转换与可视化

将单臂或双臂 LeRobot v2.x 数据转换为 HDF5，并直接在 LeRobot 视频中显示 TCP 坐标轴和未来轨迹。
以下命令均在脚本所在目录执行：本地为 `data_upload_yw`，克隆仓库后为 `yw/lerobot_hdf5_tools`。

```bash
git clone https://github.com/winkysky/yw.git
cd yw/lerobot_hdf5_tools
```

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `hw_single_arm_data_to_h5_tcp_joint.py` | 单/双臂 LeRobot → HDF5，支持采样和多进程 |
| `visualize_lerobot.py` | 直接查看 LeRobot 三维位姿、主/腕视角及 TCP 投影 |
| `visualize_h5_pose_3d.py` | 查看转换后 HDF5 的单/双臂三维位姿 |
| `camera_params.json` | 相机内外参和左右 TCP 偏移 |
| `camera_param_calculate.py` | 从脚本内已有标定常量导出参数 JSON |
| `pose_utils.py` | 输入字段拆分、单位转换、左右臂位姿计算，由入口脚本调用 |
| `visualization_utils.py` | 图像投影、未来轨迹绘制、同步多视角绘图，由入口脚本调用 |
| `environment.yml` | Conda 环境依赖 |

## 安装环境

```bash
conda env create -f environment.yml
conda activate data_yw
```

转换器使用 `lerobot==0.3.3` 的 `lerobot.datasets` API。已安装这些依赖的环境可以直接使用。
原始数据与输出通过 CLI 路径传入，不需要修改脚本中的路径。

## 输入格式

数据集应包含 `meta/info.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl` 等标准 LeRobot v2 metadata，
并由 metadata 中的路径模板指向每个 episode 的 parquet 和 MP4。不支持 LeRobot v3 共用分片。

| 字段 | 单臂 | 双臂：先左后右 |
| --- | --- | --- |
| `observation.state` | 7 维：6 关节角 + 夹爪 | 14 维：左 7 + 右 7 |
| `observation.tcp_pose` | 6 维法兰位姿 | 12 维：左 6 + 右 6；也接受 `(2,6)` |
| `observation.force_tcp` | 6 维力/力矩，或末尾附带夹爪的 7 维 | 左右各 6 维或各 7 维；各取前 6 维 |
| `timestamp`、`task_index` | 时间戳与任务索引 | 时间戳与任务索引 |
| 视频 key | `observation.images.head_image`、`observation.images.left_wrist_image` | 再加 `observation.images.right_wrist_image` |

双臂位姿也可以使用 `observation.left_tcp_pose`、`observation.right_tcp_pose` 两列，每列 6 维。
数据只有关节角而没有法兰位姿时，需要先补充位姿；本转换器不执行 FK。

虽然字段名为 `tcp_pose`，脚本将它解释为**法兰位姿**：
`[x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]`，旋转为 `Rz(rz) @ Ry(ry) @ Rx(rx)`。
如果源字段已是施加工具偏移后的 TCP，不能再次套用法兰到 TCP 的变换。
关节角单位为度，夹爪值原样保存，力数据不做坐标变换。

## 配置标定参数

默认读取脚本旁的 `camera_params.json`。转换和 LeRobot 可视化均可用
`--camera_params ./calibration.json` 指定自己的参数文件。

| key | 含义与单位 |
| --- | --- |
| `LEFT_T_FLANGE_TCP`、`RIGHT_T_FLANGE_TCP` | 各 TCP 在各自法兰中的 4×4 位姿，平移 mm |
| `LEFT_CAMINTOOLPOSE`、`RIGHT_CAMINTOOLPOSE` | 各腕相机在法兰中的 4×4 位姿，平移 m |
| `LEFT_CAMERA_K`、`RIGHT_CAMERA_K`、`HEAD_CAMERA_K` | 3×3 相机内参，像素 |
| `HEAD_CAMINBASEPOSE` | 默认主相机外参，4×4，平移 m |
| `HEAD_CAMINROBOTBASE`（可选） | 主相机在当前机器人共同基坐标中的位姿，优先于默认主相机外参，平移 m |
| `LEFT_BASE_IN_ROBOTBASE`、`RIGHT_BASE_IN_ROBOTBASE`（可选） | 各臂记录基坐标到共同机器人基坐标的变换，平移 m |
| `LEFT_T_TCP_CAM`、`RIGHT_T_TCP_CAM` | 实验相机在 TCP 中的位姿，平移暂按 mm 解释 |

当前左右 TCP 默认都为**法兰局部 Z +230 mm、绕局部 Z 约 +22.5°**。
右臂值是左臂的临时副本，标记为 `provisional_copy_of_left`；左右配置独立，正式使用时按实际安装修改。
只有工具安装和坐标轴定义一致时，两臂的工具变换才适合共用。

默认计算：

```text
robot_T_tcp = robot_T_flange @ flange_T_tcp
robot_T_wrist_camera = robot_T_flange @ flange_T_camera
```

两臂必须使用同一个机器人基坐标。若分别记录在各自的臂基坐标中，填入上述两组可选基坐标变换；
否则脚本认为原始位姿已在同一个基坐标中。主相机外参也必须与该坐标系匹配。

默认 `--wrist_extrinsic_source flange_calibration`。实验选项 `tcp_calibration` 改为
`robot_T_tcp @ tcp_T_camera`，需要提供每只手的 `T_TCP_CAM`；当前右臂该参数为 `null`，不能直接用于实验链。
`--compare_wrist_extrinsics` 可打印两条可用标定链的差异。

重新导出内置参数：

```bash
python camera_param_calculate.py --save_path ./outputs/camera_params.json
```

这只导出已有常量，不执行新的手眼标定；生成器会将右 TCP 重置为左臂的临时副本。
修改后的正式标定建议另存 JSON，并通过 `--camera_params` 使用。

## 转换为 HDF5

先转换一个 episode：

```bash
python hw_single_arm_data_to_h5_tcp_joint.py \
  --src_path ./data/input --tgt_path ./outputs/h5 \
  --episode_index 0 --sample_scale 1 --skip_saved \
  --log_file ./outputs/convert.log
```

递归批量转换：

```bash
python hw_single_arm_data_to_h5_tcp_joint.py \
  --src_path ./data/input --tgt_path ./outputs/h5 \
  -a --num_workers 8 --skip_saved --log_file ./outputs/convert.log
```

- `--single_id task_name`：只处理 `src_path` 下指定的相对数据集目录。
- `--sample_scale 0.5`：每 2 帧取 1 帧；间隔为 `max(round(1 / sample_scale), 1)`，不插值。
- `--rgb_storage video_path`（默认）：保存相对原数据集根目录的 MP4 路径，之后仍需要原视频。
- `--rgb_storage jpeg`：将采样帧以质量 90 的 JPEG 内嵌到 H5。
- `--skip_saved`：跳过已有输出。失败或没有可处理 episode 时返回非零退出码，详情见日志。

视频帧与 parquet 行按同一 episode 内的序号对应。采样后读取 MP4 时，要使用对应的原帧号/时间戳，
不能把 H5 采样序号直接当作原视频帧号。

### HDF5 字段

| key | 单臂 | 双臂 |
| --- | --- | --- |
| `timestamp` | `(T,)` | `(T,)` |
| `ee_pose` | `(T,1,4,4)` | `(T,2,4,4)`，左、右 TCP，平移 m |
| `qpos` | `(T,1,6)` | `(T,2,6)`，度 |
| `gripper` | `(T,1)` | `(T,2)` |
| `force_tcp` | `(T,6)` | `(T,2,6)` |
| 相机 group | `external_cam`、`wrist_cam` | `head_cam`、`lh_cam`、`rh_cam` |

每个相机包含 `pose=(T,4,4)`、`K=(3,3)`、`rgb`。`rgb` 为 `(1,)` MP4 路径或 `(T,)` JPEG 字节数组。
TCP 保存为 `ee_pose`，不另存 `tcp_pose`。
属性记录任务描述、`arm_names`、`qpos_unit`、工具变换、相机链和图像编码方式；双臂额外记录右 TCP 标定状态。

## 直接可视化 LeRobot

显示三维轨迹、主视角与所有可用腕视角，并叠加 TCP 和未来轨迹：

```bash
python visualize_lerobot.py \
  --dataset ./data/input --episode_index 0 --frame 0 \
  --project_tcp --future_frames 30 --axis_length 0.03
```

| 参数 | 用法 |
| --- | --- |
| `--project_tcp` | 投影当前 TCP 坐标轴和未来轨迹，同时显示视频 |
| `--camera all` | 默认显示主相机及所有可用腕相机 |
| `--camera head` | 只显示主视角 |
| `--camera head left_wrist` | 指定多个视角；还支持 `right_wrist`，`wrist` 是左腕别名 |
| `--future_frames 30` | 后续最多 30 个原始帧；0 表示只画当前 TCP |
| `--future_step 2` | 未来轨迹每隔 2 帧取一点 |
| `--axis_length 0.03` | 坐标轴长度，m |
| `--calibration_size 640 480` | 内参对应的图像尺寸；视频尺寸变化时按比例缩放 K |
| `--show_video` | 只显示视频，不加投影；不加视频相关选项时只显示三维图 |

用滑块、左右方向键或 A/D 同步切换各视角帧号。X/Y/Z 轴为红/绿/蓝；左、右未来轨迹分别为黄、青色。
三维图展示整个 episode，图像曲线只展示选定的未来区间，使用真实后续数据而非模型预测。
所有未来点都投影到**当前帧**的相机坐标系；相机后方点会过滤，画面外线段会裁剪。
采用针孔相机模型，当前配置不包含畸变参数，图像应与内参匹配。

无窗口保存：

```bash
python visualize_lerobot.py \
  --dataset ./data/input --episode_index 0 --frame 30 \
  --project_tcp --future_frames 30 \
  --no_show --save_path ./outputs/projection.png
```

## 可视化 HDF5

```bash
python visualize_h5_pose_3d.py \
  --h5_path ./outputs/h5/00000.h5 --frame 0

python visualize_h5_pose_3d.py \
  --h5_path ./outputs/h5/00000.h5 --frame 0 \
  --no_show --save_path ./outputs/h5_pose.png
```

H5 查看器显示 TCP 和腕部相机三维位姿；图像投影请使用 `visualize_lerobot.py`。
LeRobot 查看器的 `--frame` 是原始帧号，H5 查看器的 `--frame` 是采样后的帧号。
各入口的完整参数可通过 `python <脚本名> --help` 查看。
