"""Run with: python -m clients.e03_ros2 --config ... --prompt '...'."""

import argparse
import json
import math
from pathlib import Path
import signal
import threading

import numpy as np

from .e03_protocol import connect_policy


def validate_config(config, motion_enabled=False):
    def text(mapping, key):
        if not isinstance(mapping.get(key), str) or not mapping[key].strip():
            raise ValueError(f"Fill in config field: {key}")

    def number(mapping, key, positive=False):
        value = mapping.get(key)
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or (positive and value <= 0)):
            raise ValueError(f"{key} must be a finite {'positive ' if positive else ''}number")

    server = config["server"]
    for key in ("ns_host", "uri"):
        text(server, key)
    if not isinstance(server.get("ns_port"), int) or not 1 <= server["ns_port"] <= 65535:
        raise ValueError("ns_port must be an integer in 1..65535")
    number(server, "timeout_s", positive=True)
    for key in ("reference_frame", "move_group_action"):
        text(config, key)
    topics = config["joint_states_topics"]
    if not isinstance(topics, list) or not topics or any(not isinstance(topic, str) or not topic.strip() for topic in topics):
        raise ValueError("joint_states_topics must list one merged topic or the separate left/right feedback topics")
    if not isinstance(config["arms"], list) or len(config["arms"]) != 2:
        raise ValueError("Configure two E03 arms in the same order as training ee_indices=(0,1)")
    all_joints, arm_names, tcp_frames, gripper_actions = [], [], [], []
    for arm in config["arms"]:
        for key in ("name", "tcp_frame"):
            text(arm, key)
        arm_names.append(arm["name"])
        tcp_frames.append(arm["tcp_frame"])
        joints = arm["joint_names"]
        if (not isinstance(joints, list) or len(joints) != 6
                or any(not isinstance(name, str) or not name.strip() for name in joints)):
            raise ValueError("Each E03 joint_names must list six joint names")
        all_joints.extend(joints)
        gripper = arm["gripper"]
        for key in ("model_closed", "model_open"):
            number(gripper, key)
        if gripper["model_closed"] == gripper["model_open"]:
            raise ValueError("Model gripper endpoints must differ")
        if gripper["mode"] == "fixed":
            number(gripper, "fixed_model_value")
            if not min(gripper["model_closed"], gripper["model_open"]) <= gripper["fixed_model_value"] <= max(gripper["model_closed"], gripper["model_open"]):
                raise ValueError("fixed_model_value must lie between model gripper endpoints")
        elif gripper["mode"] == "action":
            for key in ("joint_name", "action_name"):
                text(gripper, key)
            gripper_actions.append(gripper["action_name"])
            for key in ("state_closed", "state_open", "command_closed_m", "command_open_m"):
                number(gripper, key)
            number(gripper, "max_effort_n", positive=True)
            if gripper["state_closed"] == gripper["state_open"] or gripper["command_closed_m"] == gripper["command_open_m"]:
                raise ValueError("Gripper state/command endpoints must differ")
            if min(gripper["command_closed_m"], gripper["command_open_m"]) < 0:
                raise ValueError("Gripper command gap must be nonnegative metres")
            if not isinstance(gripper["allow_stall"], bool):
                raise ValueError("gripper.allow_stall must be boolean")
        else:
            raise ValueError("gripper.mode must be 'fixed' or 'action'")
    if len(set(all_joints)) != 12 or len(set(arm_names)) != 2 or len(set(tcp_frames)) != 2:
        raise ValueError("Left/right arms must have distinct names, TCP frames and joint names")
    if len(set(gripper_actions)) != len(gripper_actions):
        raise ValueError("Left/right grippers must use distinct action servers")
    for key in ("max_sensor_age_s", "max_sensor_skew_s", "observation_timeout_s"):
        number(config, key, positive=True)
    if not isinstance(config["cameras"], dict) or not config["cameras"]:
        raise ValueError("Configure at least one camera")
    for name, camera in config["cameras"].items():
        if name in ("ee_pose", "gripper", "timestamp"):
            raise ValueError(f"Reserved camera name: {name}")
        for key in ("image_topic", "camera_info_topic"):
            text(camera, key)
        if not isinstance(camera["rectified"], bool):
            raise ValueError("camera.rectified must be a boolean")
    if motion_enabled:
        text(config, "planning_group")
        motion = config["motion"]
        for key in ("velocity_scale", "acceleration_scale", "planning_time_s", "action_timeout_s",
                    "position_tolerance_m", "orientation_tolerance_rad", "max_translation_step_m",
                    "max_rotation_step_deg", "max_prediction_age_s"):
            number(motion, key, positive=True)
        if not 0 < motion["velocity_scale"] <= 1 or not 0 < motion["acceleration_scale"] <= 1:
            raise ValueError("MoveIt velocity/acceleration scales must be in (0, 1]")
        for arm in config["arms"]:
            bounds = [np.asarray(arm["workspace"].get(key), dtype=float)
                      for key in ("workspace_min_m", "workspace_max_m")]
            if (any(bound.shape != (3,) for bound in bounds) or not np.isfinite(bounds).all()
                    or np.any(bounds[0] >= bounds[1])):
                raise ValueError(f"Fill workspace bounds for arm {arm['name']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Dual E03 ROS2 -> one dual-arm E2VLA Pyro4 model (default: inference without motion)")
    parser.add_argument("--config", required=True, help="JSON configuration file")
    parser.add_argument("--prompt", required=True, help="Task instruction matching the training task")
    parser.add_argument("--steps", type=int, default=1, help="Inference cycles; 0 runs until Ctrl-C")
    parser.add_argument("--action-index", type=int, default=0, help="Which predicted waypoint to use per cycle")
    parser.add_argument("--output-dir", type=Path, help="Optionally save every prediction as NPZ")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true", help="Ask MoveIt to plan the target without executing")
    modes.add_argument("--execute", action="store_true", help="Execute one target per cycle through MoveIt2")
    args = parser.parse_args(argv)
    if args.steps < 0 or args.action_index < 0 or not args.prompt.strip():
        parser.error("steps/action-index must be nonnegative and prompt must not be empty")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    validate_config(config, args.execute or args.plan_only)
    # These dependencies must come from the ROS2 Python environment. --help and
    # the protocol/config tests intentionally do not import them.
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.signals import SignalHandlerOptions
        from .e03_ros2_node import E03Node
    except ImportError as exc:
        raise RuntimeError("Source ROS2 and the E03 workspace; use their Python with rclpy, cv_bridge and moveit_msgs") from exc

    policy = connect_policy(config["server"], config["cameras"], len(config["arms"]))
    # shm_transport installs its own exit handlers. Route signals through our
    # finally block first, while the ROS executor can still cancel action goals.
    old_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in old_signals:
        signal.signal(sig, signal.default_int_handler)
    node = executor = worker = None
    spin_errors = []
    try:
        if args.action_index >= policy.config["num_future_states"]:
            raise ValueError("action-index exceeds server prediction horizon")
        print(f"Server cameras={policy.config['camera_names']}, capture_dt={policy.period:.4f}s, "
              f"history={policy.history_seconds:.3f}s", flush=True)
        policy.start(args.prompt)
        # Keep the executor alive during Ctrl-C cancellation and cleanup.
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        node = E03Node(config, policy.period, policy.history_seconds)
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        def spin():
            try:
                executor.spin()
            except BaseException as exc:
                spin_errors.append(exc)

        worker = threading.Thread(target=spin, name="e03-ros2", daemon=True)
        worker.start()
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        mode = "EXECUTE" if args.execute else "PLAN ONLY" if args.plan_only else "INFERENCE ONLY"
        print(f"Mode: {mode}. Press Ctrl-C to exit.", flush=True)
        sequence, cycle = 0, 0
        while rclpy.ok() and (args.steps == 0 or cycle < args.steps):
            if spin_errors:
                raise RuntimeError("ROS executor failed") from spin_errors[0]
            sequence, frames = node.wait_frames(sequence, policy.history_seconds if cycle == 0 else 0.0)
            prediction = policy.infer(frames)
            index = args.action_index
            if index >= len(prediction.times):
                raise ValueError("action-index exceeds returned prediction horizon")
            pose, gripper = prediction.poses[index], prediction.grippers[index]
            print(f"cycle={cycle} observed_t={frames[-1]['timestamp']:.3f} "
                  f"predicted_t={prediction.times[index]:.3f}", flush=True)
            for arm, target, value in zip(config["arms"], pose, gripper):
                print(f"  {arm['name']}: target_xyz_m={target[:3, 3].round(5).tolist()} "
                      f"gripper={value:.4f}", flush=True)
            if args.output_dir:
                np.savez_compressed(args.output_dir / f"prediction_{cycle:06d}.npz",
                                    poses=prediction.poses, grippers=prediction.grippers,
                                    times=prediction.times, observation_time=frames[-1]["timestamp"])
            if args.execute or args.plan_only:
                node.execute_target(pose, gripper, frames[-1]["timestamp"], plan_only=args.plan_only)
            cycle += 1
    except KeyboardInterrupt:
        print("Stopping; requesting cancellation of this client's active goals.", flush=True)
    finally:
        if node is not None:
            node.stop()
        if executor is not None:
            executor.shutdown(timeout_sec=3.0)
        if worker is not None:
            worker.join(timeout=3.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        policy.proxy._pyroRelease()
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
