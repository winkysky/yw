# 华沿 E03 双臂 ROS2 推理客户端

本客户端对应 **一个双臂模型同时预测两臂**，连接服务端项目 `eevla_force` 中的
`infer_utils/remote_service.py`。该服务端及模型权重需单独部署。
客户端代码位于 `clients/e03_ros2.py`，在机器人电脑运行。
实际话题、TCP、夹爪参数需要根据机器人电脑上的 ROS2 与 E03 驱动配置填写；
示例中的话题名不是已经核实的设备配置。客户端没有启动、使能或标定机器人驱动的功能。

## 接口与执行方式

- 读取一个合并的或多个独立的 `sensor_msgs/msg/JointState` 话题。
- 读取 `sensor_msgs/msg/Image`、`CameraInfo`，通过 TF 查询图像时刻的相机光学位姿。
- 按配置 `arms` 的顺序组装 `ee_pose: (2,4,4)`、`gripper: (2,)`。
- 通过随客户端附带的 `shm_transport` 连接模型，依次调用 `reset`、`set_prompt`、
  `add_obs_frame`、`get_action`。每次推理返回 `(T,2,4,4)` 位姿、`(T,2)` 夹爪和 `(T,)` 时间戳。
- 默认只采集和推理。`--plan-only` 提交双臂目标，但不执行。
- `--execute` 使用 **一个 MoveIt2 双臂规划组**，在同一目标约束集合中要求两个 TCP
  同时到达目标；MoveIt 负责运动学、规划、轨迹时间参数化和执行。
- 双臂动作完成后，分别向两个标准 `control_msgs/action/GripperCommand` 服务提交夹爪目标，
  等待两个目标完成，再开始下一轮推理。不是机械臂和夹爪整段轨迹的同步流式控制。

这是同步单步闭环：每轮选中预测序列的一个时间步，执行后重新观测。
`--action-index 0` 表示选择第一个预测时间步；选中的左右臂目标来自同一时间步。
**不会按照模型的预测时间戳连续跟踪整段轨迹，也不提供 RTC 或力控。**
MoveIt 的规划/执行时间决定真实运行频率，可能低于训练采样频率。

## 机器人端前提

面向 ROS2 Humble 或更新版本的 Python 接口编写，需要：

1. 已运行 E03 双臂驱动；两臂使用不同的关节名、TCP frame，反馈时间戳有效。
2. 完整的双臂 URDF/SRDF、TF 树，以及能够规划和执行两臂的 MoveIt2 combined group。
   两个彼此独立的单臂 MoveIt 服务不能直接替代这个双臂规划组。
3. 已标定两臂基座之间、外部相机与机器人之间、腕部相机与 TCP 之间的变换，并发布到 TF。
4. 若控制夹爪，需要其反馈关节及标准 `GripperCommand` action；自定义 Modbus/SDK/话题接口
   需要先由机器人端转换为该标准接口。固定夹爪/固定工具可显式选择 `fixed` 模式。
5. 模型 checkpoint 和配置确实支持双臂，`ee_indices=[0,1]`。客户端不修改服务端数据配置。
   服务端项目的 HWTOP 示例为单臂，不能仅把配置改成 `[0,1]` 就将单臂权重变成双臂策略。

参考接口定义：[华沿 ROS2 仓库](https://github.com/huayan-robotics/elfin_robot_ros2)、
[MoveIt MotionPlanRequest](https://github.com/moveit/moveit_msgs/blob/ros2/msg/MotionPlanRequest.msg)、
[MoveIt PlanningOptions](https://github.com/moveit/moveit_msgs/blob/ros2/msg/PlanningOptions.msg)、
[GripperCommand](https://github.com/ros-controls/control_msgs/blob/master/control_msgs/action/GripperCommand.action)。
华沿不同版本驱动的启动方式可能不同，本客户端不假定其控制器名称。

## 环境与配置

在机器人电脑克隆仓库并进入客户端目录；后续命令均在该目录运行：

```bash
git clone https://github.com/winkysky/yw.git
cd yw/e03_dual_ros2_client
```

也可以将本目录的 `clients/` 和 `shm_transport/` 一起复制到机器人电脑上的同一目录。
客户端不需要模型权重、CUDA、OpenPI 或服务器训练依赖。

先 source 已安装的 ROS2 和机器人工作区，例如：

```bash
source /opt/ros/humble/setup.bash
source /你的机器人工作区/install/setup.bash
```

使用能导入这些模块的同一个 Python：`rclpy`、`tf2_ros`、`cv_bridge`、`sensor_msgs`、
`geometry_msgs`、`shape_msgs`、`action_msgs`、`control_msgs`、`moveit_msgs`、`numpy`、`scipy`、`Pyro4`。
ROS 模块通常由 ROS 安装提供，避免直接使用不含 ROS 模块的服务器 Conda Python。
如需虚拟环境，可基于 ROS 使用的系统 Python 创建 `--system-site-packages` 环境，
并安装 `Pyro4`；保持 NumPy 与系统 `cv_bridge` 的二进制版本兼容。

```bash
cp clients/e03_ros2.example.json clients/e03_ros2.local.json
```

编辑 `clients/e03_ros2.local.json`，必填 `null` 项以及各臂的六个真实关节名。

| 字段 | 含义 |
|---|---|
| `server.ns_host / ns_port / uri` | 模型的名称服务地址、端口和注册名；`ns_port` 不是模型 daemon 端口 |
| `reference_frame` | **训练数据的共同参考坐标系**；两臂位姿和所有相机位姿统一到该坐标系，平移单位为米 |
| `arms` 列表顺序 | 必须与训练时末端顺序相同；示例采用索引 0=左臂、1=右臂；按实际训练顺序调整 |
| `arms[i].tcp_frame` | 与训练末端定义相同、且已存在于 MoveIt 机器人模型中的 TCP link；不能混用法兰和工具尖端，单独发布一个未加入 URDF 的 TF frame 不足以作为规划 link |
| `arms[i].joint_names` | 各臂六个实际关节名，两臂不能重名 |
| `joint_states_topics` | 如 `['/joint_states']`，或左右臂各自发布的两个话题；JSON 字符串用双引号 |
| `planning_group` | 同时包含两臂的 MoveIt 规划组；从双臂 SRDF/MoveIt 配置获取 |
| `move_group_action` | `moveit_msgs/action/MoveGroup` 服务名称，常见示例是 `/move_action` |
| `cameras` 的键 | 必须与模型 `get_config()['camera_names']` 完全匹配；示例的三相机布局仅供替换 |
| 相机 `image_topic / camera_info_topic` | 同一分辨率、同一光学 frame 的图像和内参；`rectified=true` 使用 P 与 R，未校正图像只接受零畸变内参 |
| `arms[i].workspace` | 每个 TCP 在共同参考坐标系下的目标范围，单位米；用于目标检查，不是完整轨迹的碰撞模型 |
| `motion.velocity_scale / acceleration_scale` | MoveIt 限速比例，必须处于 `(0,1]` |

相机无需是特定型号，只要在 ROS2 发布上述标准消息。图像统一转换为 RGB uint8，
保留原始分辨率与匹配内参，由服务端做模型输入缩放。
TF 中缺少相机外参时会报错；程序不会用单位矩阵代替标定结果。
同时发布两臂和相机的设备需要使用一致的 ROS 时间来源。

每个夹爪独立配置：

- `mode="action"`：读取 `joint_name` 的反馈，`state_closed/state_open` 按反馈原始单位填写；
  `command_closed_m/command_open_m` 按 action 驱动要求的夹爪间隙（米）填写，
  不要把单侧指关节行程与总开口宽度混淆。`max_effort_n` 为驱动使用的力参数。
- `model_closed/model_open`：对应训练数据中的夹爪编码，例如 `[0,1]` 或 `[-1,1]`；
  客户端按端点映射，不对所有 checkpoint 假定同一种编码。
- `allow_stall`：是否将夹爪报告的堵转当作可接受的抓取结束，默认 false。
- `mode="fixed"`：需要显式填写 `fixed_model_value`；固定上报这个值，并忽略该臂预测的夹爪动作。
  仅用于确实固定的夹爪/工具，不可据此宣称已经接入未知夹爪。

在机器人电脑核对接口：

```bash
ros2 topic list -t
ros2 action list -t
ros2 control list_controllers
ros2 topic echo /joint_states --once
ros2 param get /move_group robot_description_semantic
```

若有命名空间，把上面 `/joint_states` 和 `/move_group` 换成实际名称。
用 `ros2 run tf2_ros tf2_echo 参考frame 左臂TCPframe` 和对应右臂/相机 frame 验证 TF 连通。

## 运行

先运行一轮采集与推理，保存结果，不发送任何运动目标：

```bash
python -m clients.e03_ros2 \
  --config clients/e03_ros2.local.json \
  --prompt "你的双臂任务指令" \
  --steps 1 --output-dir results/e03_preview
```

检查双臂规划，但不执行：

```bash
python -m clients.e03_ros2 \
  --config clients/e03_ros2.local.json \
  --prompt "你的双臂任务指令" --steps 1 --plan-only
```

执行一轮双臂目标：

```bash
python -m clients.e03_ros2 \
  --config clients/e03_ros2.local.json \
  --prompt "你的双臂任务指令" --steps 1 --execute
```

`--steps 0 --execute` 持续闭环，Ctrl-C 退出并尝试取消本客户端未完成的目标。
程序检查反馈新鲜度、相机同步误差、预测年龄、目标位移和旋转幅度；参数位于 JSON。
这些检查不能替代机器人驱动的限制或完整的 MoveIt 场景；场景中的双臂模型与障碍物仍由部署端提供。

启动时会清空该模型服务的观测历史，因此同一个有状态的服务注册名只能供这一组机器人使用。
预热阶段按 checkpoint 的历史长度积累观测；执行过程中 ROS 线程继续采集，供下一次推理使用。
时间戳以本回合第一帧为零点，避免服务器 float32 时间戳处理丢失 Unix 时间的小数精度。

## 验证范围

本地包含 NumPy 协议、双臂顺序、目标检查、模拟 ROS 消息/action 控制逻辑测试，
以及使用真实 localhost socket 的 Pyro4 通信测试。均使用合成观测，不连接模型服务或机器人。
尚未在真实 ROS2 graph、MoveIt 双臂配置、相机或 E03 硬件上联调。

```bash
python -m pytest -q tests/test_e03_client.py tests/test_e03_ros2_logic.py tests/test_e03_rpc.py
```
