import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clients.e03_protocol import Prediction, RemotePolicy, check_target, model_timing, observation_frame, remap_gripper
from clients.e03_ros2 import parse_args, validate_config


def configured_robot():
    config = json.loads((Path(__file__).resolve().parents[1] / "clients/e03_ros2.example.json").read_text())
    config["server"]["ns_host"] = "127.0.0.1"
    config["reference_frame"] = "world"
    config["planning_group"] = "both_arms"
    for arm in config["arms"]:
        name = arm["name"]
        arm["tcp_frame"] = name + "_tcp"
        arm["joint_names"] = [f"{name}_joint_{index}" for index in range(6)]
        arm["workspace"] = {"workspace_min_m": [-1, -1, -1], "workspace_max_m": [1, 1, 1]}
        arm["gripper"].update(mode="fixed", fixed_model_value=1.0)
    return config


def model_config():
    return dict(camera_names=["external_cam"], ee_indices=[0, 1], record_dt=0.1, sample_dt=0.1,
                num_history_cameras=2, num_history_states=3, sample_camera_gaps=2,
                sample_state_gaps=1, num_future_states=4)


def make_frame(timestamp=0.0):
    poses = np.stack([np.eye(4), np.eye(4)])
    poses[:, 1, 3] = [0.2, -0.2]
    return observation_frame(poses, [0.2, 0.8], timestamp, {
        "external_cam": dict(rgb=np.zeros((24, 32, 3), dtype=np.uint8),
                             K=np.array([[100, 0, 16], [0, 100, 12], [0, 0, 1]]),
                             wcT=np.eye(4), timestamp=timestamp),
    })


class FakePolicy:
    def __init__(self):
        self.frames = []
        self.calls = []

    def get_config(self):
        return model_config()

    def reset(self):
        self.frames.clear()
        self.calls.append("reset")

    def set_prompt(self, prompt):
        self.calls.append(prompt)

    def add_obs_frame(self, frame):
        self.frames.append(frame)

    def get_action(self, draw_traj=False, compress_traj_img=False):
        frame = self.frames[-1]
        poses = np.repeat(frame["ee_pose"][None], 4, axis=0)
        grips = np.repeat(frame["gripper"][None], 4, axis=0)
        times = frame["timestamp"] + np.arange(1, 5) * 0.1
        return poses, grips, times, None


def test_dual_arm_round_trip_keeps_left_right_order_and_history():
    server = FakePolicy()
    policy = RemotePolicy(server, ["external_cam"], 2)
    policy.start("move both arms")
    prediction = policy.infer([make_frame(0.0), make_frame(0.1), make_frame(0.2)])
    assert prediction.poses.shape == (4, 2, 4, 4)
    np.testing.assert_allclose(prediction.poses[:, :, 1, 3], [[0.2, -0.2]] * 4)
    np.testing.assert_allclose(prediction.grippers, [[0.2, 0.8]] * 4)
    np.testing.assert_allclose(prediction.times, [0.3, 0.4, 0.5, 0.6])
    assert server.calls == ["reset", "move both arms"]
    assert len(server.frames) == 3
    assert server.frames[-1]["ee_pose"].dtype == np.float32
    with pytest.raises(ValueError, match="increasing"):
        policy.infer([make_frame(0.2)])


def test_single_arm_model_is_rejected_before_reset():
    server = FakePolicy()
    server.get_config = lambda: dict(model_config(), ee_indices=[0])
    with pytest.raises(ValueError, match="2 arms"):
        RemotePolicy(server, ["external_cam"], 2)
    assert server.calls == []


def test_camera_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Camera names"):
        RemotePolicy(FakePolicy(), ["different_camera"], 2)


@pytest.mark.parametrize("fault", ["force", "one_arm", "nan", "reflection", "past", "unordered"])
def test_invalid_prediction_is_rejected(fault):
    server = FakePolicy()
    server.add_obs_frame(make_frame(0.1))
    response = list(server.get_action())
    if fault == "force":
        response += [None] * 3
    elif fault == "one_arm":
        response[0] = response[0][:, :1]
    elif fault == "nan":
        response[1][0, 0] = np.nan
    elif fault == "reflection":
        response[0][0, 0, 0, 0] = -1
    elif fault == "past":
        response[2][0] = 0.0
    elif fault == "unordered":
        response[2][1] = response[2][0]
    with pytest.raises(ValueError):
        Prediction.from_response(response, 0.1, 2)


def test_training_timebase_is_preserved():
    assert model_timing(model_config()) == (0.1, 0.2)
    for key in ("sample_dt", "record_dt", "num_history_states"):
        with pytest.raises(ValueError):
            model_timing(dict(model_config(), **{key: 0}))


def test_gripper_units_can_be_reversed_and_do_not_assume_model_range():
    assert remap_gripper(1005, 1886, 1005, -1, 1) == 1
    assert remap_gripper(1886, 1886, 1005, -1, 1) == -1
    assert remap_gripper(0.5, 0, 1, 0, 0.08) == pytest.approx(0.04)
    assert remap_gripper(2.0, 0, 1, 0, 0.08) == pytest.approx(0.08)


def test_workspace_and_rotation_limits_reject_instead_of_clipping():
    config = configured_robot()
    limits = dict(config["motion"], **config["arms"][0]["workspace"])
    current = np.eye(4)
    for xyz, angle in [([0.04, 0, 0], 0), ([0, 0, 0], 20), ([2, 0, 0], 0)]:
        target = np.eye(4)
        target[:3, 3] = xyz
        target[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
        original = target.copy()
        with pytest.raises(ValueError):
            check_target(current, target, limits)
        np.testing.assert_array_equal(target, original)


def test_config_distinguishes_two_arms_and_motion_modes():
    config = configured_robot()
    validate_config(config, motion_enabled=True)
    config["motion"]["velocity_scale"] = 0
    with pytest.raises(ValueError, match="velocity_scale"):
        validate_config(config, motion_enabled=True)
    validate_config(config, motion_enabled=False)
    config["arms"][1]["joint_names"] = config["arms"][0]["joint_names"]
    with pytest.raises(ValueError, match="distinct"):
        validate_config(config)


def test_cli_defaults_to_inference_and_disallows_conflicting_modes():
    args = parse_args(["--config", "robot.json", "--prompt", "task"])
    assert not args.execute and not args.plan_only and args.steps == 1
    with pytest.raises(SystemExit):
        parse_args(["--config", "robot.json", "--prompt", "task", "--execute", "--plan-only"])
