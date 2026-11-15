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
| `visualize_lerobot.py` | 直接查看 LeRobot 三维位姿、主/腕视角及 TCP 投影，保存图片或 MP4 |
| `visualize_h5_pose_3d.py` | 查看转换后 HDF5 的单/双臂三维位姿 |
| `camera_params.json` | 相机内外参、左右 TCP 偏移及 HW 左右基座变换 |
| `camera_param_calculate.py` | 从脚本内已有标定常量导出参数 JSON |
| `estimate_hw_base_transform.py` | 从 URDF、关节角和记录法兰位姿联合估计左右基座关系 |
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
| `HEAD_CAMINBASEPOSE` | 默认主相机在共同基坐标中的外参；当前按左臂记录 base 使用，4×4，平移 m |
| `HEAD_CAMINROBOTBASE`（可选） | 主相机在当前机器人共同基坐标中的位姿，优先于默认主相机外参，平移 m |
| `LEFT_BASE_IN_ROBOTBASE`、`RIGHT_BASE_IN_ROBOTBASE`（可选） | 各臂记录基坐标到共同机器人基坐标的变换，平移 m |
| `LEFT_T_TCP_CAM`、`RIGHT_T_TCP_CAM` | 实验相机在 TCP 中的位姿，平移暂按 mm 解释 |

当前左右 TCP 默认都为**法兰局部 Z +230 mm、绕局部 Z 约 +22.5°**。
右臂值是左臂的临时副本，标记为 `provisional_copy_of_left`；左右配置独立，正式使用时按实际安装修改。
只有工具安装和坐标轴定义一致时，两臂的工具变换才适合共用。

默认计算：

```text
robot_T_flange = robot_T_recorded_base @ recorded_base_T_flange
robot_T_tcp = robot_T_flange @ flange_T_tcp
robot_T_wrist_camera = robot_T_flange @ flange_T_camera
```

输出的两臂位姿必须使用同一个机器人基坐标。当前随附配置以左臂记录 base 为共同参考，
`LEFT_BASE_IN_ROBOTBASE = I`，`RIGHT_BASE_IN_ROBOTBASE` 使用下文的 HW robot2 估计结果。
转换器和 LeRobot 可视化都会自动使用它，同时旋转和平移右臂的法兰、TCP 和腕相机。
如果输入已经统一到同一个 base，必须另用两组 base 变换均为单位矩阵的配置，避免重复变换。
其他机器人或安装方式应提供各自的 `--camera_params`；这组估计不是所有双臂机器人的通用参数。
旧配置缺少 base key 时仍按单位矩阵处理。主相机外参也必须与共同坐标系匹配。

### 当前 HW 左右 base 的关系

配置保存了完整精度的 `left_base_T_right_base`，近似为：

```text
p_left = R @ p_right + t
R ≈ diag(1, -1, -1) = Rx(180°)
t ≈ [0.000221830, -0.000030121, -0.230034851] m
```

即先旋转，再加上以左 base 表示的平移；姿态也左乘同一个 R。
这里左 base 的 +Y 向上，右 base 的 +Y 近似朝下。
URDF 两侧第一关节安装原点相距约 347 mm，但它们不等于数据记录 base；
拟合得到的记录 base 间距约 230 mm，不能直接将安装间距当成该平移。

该关系通过 `robot2_mk.urdf` 的 FK 与记录位姿联合拟合得到：10 条 episode、每条 25 帧用于拟合，
另 30 条 episode、每条 25 帧用于验证。验证集左/右臂位置 RMS 为 1.67/0.66 mm，
旋转 RMS 为 0.166/0.084°。这衡量的是 URDF 与记录位姿的一致性，不是相机或工具标定的独立精度。
`BASE_ALIGNMENT_STATUS` 明确标记为估计值。头相机沿用原参数并按左 base 使用，经过图像检查但未重新标定；
右手工具 TCP 仍是上文所述的暂定左手副本，base 拟合不能确定真实工具坐标轴。

默认 `--wrist_extrinsic_source flange_calibration`。实验选项 `tcp_calibration` 改为
`robot_T_tcp @ tcp_T_camera`，需要提供每只手的 `T_TCP_CAM`；当前右臂该参数为 `null`，不能直接用于实验链。
`--compare_wrist_extrinsics` 可打印两条可用标定链的差异。

重新导出内置参数：

```bash
python camera_param_calculate.py --save_path ./outputs/camera_params.json
```

这只导出已有常量，包含上述左右 base 关系，不执行新的拟合或手眼标定；
生成器会将右 TCP 重置为左臂的临时副本。
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
| `--label_tcp_axes` | 与 `--project_tcp` 合用，标出 L/R X/Y/Z 轴端点，便于跨视角核对 |
| `--camera all` | 默认显示主相机及所有可用腕相机 |
| `--camera head` | 只显示主视角 |
| `--camera head left_wrist` | 指定多个视角；还支持 `right_wrist`，`wrist` 是左腕别名 |
| `--future_frames 30` | 后续最多 30 个原始帧；0 表示只画当前 TCP |
| `--future_step 2` | 未来轨迹每隔 2 帧取一点 |
| `--axis_length 0.03` | 坐标轴长度，m |
| `--base_axis_length 0.15` | 三维图中左右基座的 XYZ 箭头长度，m，默认 0.15 |
| `--vertical_axis y` | 三维图默认以 +Y 为竖直向上；可选 x/y/z，仅改变观察视角 |
| `--calibration_size 640 480` | 内参对应的图像尺寸；视频尺寸变化时按比例缩放 K |
| `--show_video` | 只显示视频，不加投影；不加视频相关选项时只显示三维图 |

用滑块、左右方向键或 A/D 同步切换各视角帧号。X/Y/Z 轴为红/绿/蓝；左、右未来轨迹分别为黄、青色。
三维图中的左右 TCP 轨迹分别为金黄实线、青色虚线。
三维图展示整个 episode，图像曲线只展示选定的未来区间，使用真实后续数据而非模型预测。
三维图默认标出左臂基座 `left_base` 的原点和 XYZ 轴（红/绿/蓝），显示范围包含基座原点及箭头。
本批 HW 数据的左臂基座 +Y 为物理竖直向上，LeRobot 查看器默认使用 Y 向上的观察视角；
这不会交换坐标分量或修改外参，绕 Y 轴旋转仍按原坐标定义计算。
没有设置 `LEFT_BASE_IN_ROBOTBASE` 时，该坐标架位于 `(0, 0, 0)`；设置后按该矩阵显示。
这只是参考坐标架的显示，不会额外变换轨迹或图像投影。图下方注明右臂基座变换是否已配置；
未配置时仍使用单位矩阵，不能据此认为左右臂的真实基座重合。
投影函数要求点和相机位姿已经处于同一个参考坐标系，计算 `camera_T_point = inverse(base_T_camera) @ base_T_point`，
不会自动推断左右基座的关系。
若配置了 `RIGHT_BASE_IN_ROBOTBASE`，三维图也会显示该变换定义的 `right_base` 坐标架。
所有未来点都投影到**当前帧**的相机坐标系；相机后方点会过滤，画面外线段会裁剪。
采用针孔相机模型，当前配置不包含畸变参数，图像应与内参匹配。

无窗口保存单帧图片：

```bash
python visualize_lerobot.py \
  --dataset ./data/input --episode_index 0 --frame 30 \
  --project_tcp --future_frames 30 \
  --no_show --save_path ./outputs/projection.png
```

### 导出可视化视频

将整个 episode 的三维图、主相机与腕相机画面一起保存为 MP4，叠加 TCP 坐标轴和未来轨迹：

```bash
python visualize_lerobot.py \
  --dataset ./data/input --episode_index 0 \
  --project_tcp --label_tcp_axes --future_frames 60 --axis_length 0.03 \
  --base_axis_length 0.08 --vertical_axis y \
  --no_show --save_video ./outputs/episode_000.mp4
```

只导出第 30～89 帧的三维图和主视角，以 15 FPS 播放：

```bash
python visualize_lerobot.py \
  --dataset ./data/input --episode_index 0 --frame 30 --end_frame 90 \
  --project_tcp --camera head --fps 15 \
  --no_show --save_video ./outputs/head_clip.mp4
```

- `--save_video`：输出 `.mp4`，保存三维图和所选相机的组合画面，不包含交互滑块与音频。
- `--frame`：起始帧，默认 0；`--end_frame`：结束帧（不包含），默认 episode 末尾。
- `--fps`：输出帧率，默认使用 `meta/info.json` 的 `fps`。逐帧导出，不丢帧、不插值；修改此值会改变播放速度。输出为固定帧率，不按不规则时间戳重采样。
- 相机和投影选项与交互查看一致；仅使用 `--save_video` 时只导出三维图，加入 `--show_video` 显示相机，加入 `--project_tcp` 同时显示投影。
- 可同时传入 `--save_path` 保存起始帧图片。使用 `--no_show` 可在无窗口环境运行；不加时导出完成后打开交互查看器。

未来轨迹仍取自原 episode，片段末尾可以显示 `end_frame` 之后的未来点。
导出使用现有 OpenCV 的 `mp4v` 编码支持，无新增依赖；逐帧绘图可能比原视频播放慢。
成功后替换同名输出文件；请将生成的视频保存在本地输出目录，不提交到代码仓库。

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

## 从 URDF 和记录位姿估计左右基座关系

`estimate_hw_base_transform.py` 使用关节角计算 URDF 的 `base_link → l/r_Link6` FK，
并与记录的法兰位姿共同拟合每只手的两项固定变换：

```text
world_T_link6(q) @ link6_T_flange = world_T_recorded_base @ recorded_T_flange
left_base_T_right_base = inverse(world_T_left_base) @ world_T_right_base
```

同时拟合 `link6_T_flange`，避免将 URDF 第六连杆坐标误当成控制器记录的法兰坐标。
每只手臂共 12 个未知参数：两组旋转向量和两组平移。最小二乘残差为
`[5 * translation_error_m, rotation_error_rotvec_rad]`，使用 4 个初值并记录收敛及 Jacobian 条件。
不使用图像像素，不优化相机外参或 flange 到工具 TCP 的变换。
假定关节角为度、记录位姿为 mm 和外旋 xyz 欧拉角（度），URDF 长度单位为 m。
输入清单为 JSON 列表，每条包含 `dataset` 与 `episode_index`。每个采集目录中清单所含的
最小、最大 episode 用于拟合，其余 episode 留作验证；默认每条均匀取 25 帧。

```bash
python estimate_hw_base_transform.py \
  --urdf ./robot2_mk.urdf --manifest ./outputs/all_episodes.json \
  --camera_params ./camera_params.json --output ./outputs/base_estimate
```

输出 `base_estimate.json` 记录变换、拟合条件及留出样本误差；`estimated_camera_params.json`
以左臂记录基座为共同参考系，加入估计的右臂基座变换。头相机及腕相机标定保持输入值。
结果依赖 URDF 和记录位姿约定，应先通过 `--camera_params` 加载新配置检查少量样本。
