"""ROS-independent observation/action handling for remote_service.Service.

All transforms map coordinates from the named child frame into the model's
reference frame. Translation is in metres; matrix/quaternion order is explicit.
"""

import math
from dataclasses import dataclass

import numpy as np


def transform_matrix(value, name="transform"):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite (4, 4) matrix")
    rotation = value[:3, :3]
    if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3)):
        raise ValueError(f"{name} is not a rigid transform")
    return value


def remap_gripper(value, source_closed, source_open, target_closed, target_open):
    values = np.asarray([value, source_closed, source_open, target_closed, target_open], dtype=float)
    if not np.isfinite(values).all() or source_closed == source_open:
        raise ValueError("Gripper endpoints must be finite and source endpoints must differ")
    openness = np.clip((value - source_closed) / (source_open - source_closed), 0.0, 1.0)
    return float(target_closed + openness * (target_open - target_closed))


def observation_frame(ee_pose, gripper, timestamp, cameras):
    """cameras[name] contains rgb (uint8 HWC), K (3x3), wcT (4x4), timestamp."""
    ee_pose = np.asarray(ee_pose, dtype=float)
    if ee_pose.shape == (4, 4):
        ee_pose = ee_pose[None]
    gripper = np.atleast_1d(np.asarray(gripper, dtype=float))
    if ee_pose.ndim != 3 or ee_pose.shape[1:] != (4, 4) or len(ee_pose) == 0:
        raise ValueError("ee_pose must have shape (number_of_arms, 4, 4)")
    if gripper.shape != (len(ee_pose),):
        raise ValueError("There must be one gripper state per arm")
    if not np.isfinite(gripper).all() or not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError("Observation gripper/time must be finite and time nonnegative")
    for index, pose in enumerate(ee_pose):
        transform_matrix(pose, f"ee_pose[{index}]")
    frame = {
        "ee_pose": ee_pose.astype(np.float32),
        "gripper": gripper.astype(np.float32),
        "timestamp": float(timestamp),
    }
    if not cameras:
        raise ValueError("At least one camera is required")
    for name, camera in cameras.items():
        if name in frame:
            raise ValueError(f"Camera name conflicts with observation field: {name}")
        rgb = np.asarray(camera["rgb"])
        K = np.asarray(camera["K"], dtype=np.float32)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape:
            raise ValueError(f"{name}: expected uint8 RGB image (H, W, 3)")
        if (K.shape != (3, 3) or not np.isfinite(K).all()
                or K[0, 0] <= 0 or K[1, 1] <= 0):
            raise ValueError(f"{name}: invalid camera intrinsics")
        frame[name] = {
            "model": "pinhole",
            "camera": {"width": rgb.shape[1], "height": rgb.shape[0], "K": K},
            "data": {
                "color": np.ascontiguousarray(rgb),
                "encoding": "rgb",
                "wcT": transform_matrix(camera["wcT"], f"{name}.wcT").astype(np.float32),
                "seg": None,
                "timestep": float(camera["timestamp"]),
            },
        }
    return frame


def model_timing(config):
    """Keep the checkpoint's timebase; do not replace it with an arbitrary Hz."""
    sample_dt = config.get("sample_dt")
    if sample_dt is None or not math.isfinite(sample_dt) or sample_dt <= 0:
        raise ValueError("Server config.sample_dt must be a positive interval in seconds")
    record_dt = config.get("record_dt")
    record_dt = sample_dt if record_dt is None else record_dt
    if not math.isfinite(record_dt) or record_dt <= 0:
        raise ValueError("Server config.record_dt must be positive")
    for key in ("num_history_cameras", "num_history_states", "sample_camera_gaps", "sample_state_gaps", "num_future_states"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"Server config.{key} must be a positive integer")
    history = max(
        (config["num_history_cameras"] - 1) * config["sample_camera_gaps"],
        (config["num_history_states"] - 1) * config["sample_state_gaps"],
    ) * sample_dt
    return float(record_dt), max(float(history), float(record_dt))


@dataclass
class Prediction:
    poses: np.ndarray       # (T, number_of_arms, 4, 4), configured arm order
    grippers: np.ndarray    # (T, number_of_arms), model's gripper convention
    times: np.ndarray      # (T,), seconds since observation origin

    @classmethod
    def from_response(cls, response, observation_time, num_arms):
        if not isinstance(response, (tuple, list)) or len(response) != 4:
            raise ValueError("Expected remote_service.py's 4 outputs; force/RTC services are unsupported")
        poses, grippers, times, _ = response
        poses, grippers, times = (np.asarray(x, dtype=float) for x in (poses, grippers, times))
        count = len(times) if times.ndim == 1 else 0
        if count == 0 or poses.shape != (count, num_arms, 4, 4) or grippers.shape != (count, num_arms):
            raise ValueError(f"Expected poses (T,{num_arms},4,4), grippers (T,{num_arms}), times (T,)")
        if not all(np.isfinite(x).all() for x in (poses, grippers, times)):
            raise ValueError("Prediction contains NaN/Inf")
        if times[0] <= observation_time or np.any(np.diff(times) <= 0):
            raise ValueError("Prediction timestamps must increase and follow the observation")
        for pose in poses.reshape(-1, 4, 4):
            transform_matrix(pose, "predicted pose")
        return cls(poses, grippers, times)


def check_target(current, target, limits):
    """Reject out-of-range targets instead of silently changing model output."""
    current = transform_matrix(current, "current TCP")
    target = transform_matrix(target, "target TCP")
    lower, upper = (np.asarray(limits[key], dtype=float) for key in ("workspace_min_m", "workspace_max_m"))
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite([lower, upper]).all() or np.any(lower >= upper)):
        raise ValueError("Configure finite workspace_min_m < workspace_max_m in the model reference frame")
    if np.any(target[:3, 3] < lower) or np.any(target[:3, 3] > upper):
        raise ValueError("Predicted target is outside the configured workspace")
    distance = np.linalg.norm(target[:3, 3] - current[:3, 3])
    relative = current[:3, :3].T @ target[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    if distance > limits["max_translation_step_m"]:
        raise ValueError(f"Predicted translation step {distance:.4f} m exceeds limit")
    if angle > limits["max_rotation_step_deg"]:
        raise ValueError(f"Predicted rotation step {angle:.2f} deg exceeds limit")


class RemotePolicy:
    def __init__(self, proxy, camera_names, num_arms):
        self.proxy = proxy
        self.num_arms = num_arms
        self.config = proxy.get_config()
        if tuple(self.config["ee_indices"]) != tuple(range(num_arms)):
            raise ValueError(f"Client has {num_arms} arms but server ee_indices={self.config['ee_indices']}. "
                             "Use a checkpoint/config trained for these arms in the same order.")
        if set(self.config["camera_names"]) != set(camera_names):
            raise ValueError(f"Camera names must match server: {self.config['camera_names']}")
        self.period, self.history_seconds = model_timing(self.config)
        self.last_time = -1.0

    def start(self, prompt):
        self.proxy.reset()
        self.proxy.set_prompt(prompt)
        self.last_time = -1.0

    def infer(self, frames):
        if not frames:
            raise ValueError("No fresh observations available")
        for frame in frames:
            if frame["timestamp"] <= self.last_time:
                raise ValueError("Observations must have strictly increasing timestamps")
            self.proxy.add_obs_frame(frame)
            self.last_time = frame["timestamp"]
        response = self.proxy.get_action(draw_traj=False, compress_traj_img=False)
        return Prediction.from_response(response, self.last_time, self.num_arms)


def connect_policy(config, camera_names, num_arms):
    # Lazy imports let protocol tests run without ROS, PyTorch or a GPU.
    import Pyro4
    from shm_transport import get_shm_proxy

    Pyro4.config.COMMTIMEOUT = config["timeout_s"]
    proxy = get_shm_proxy(
        config["uri"], ns_host=config["ns_host"], ns_port=config["ns_port"],
        force_use_socket=True,
    )
    proxy._pyro_proxy._pyroTimeout = config["timeout_s"]
    try:
        return RemotePolicy(proxy, camera_names, num_arms)
    except BaseException:
        proxy._pyroRelease()
        raise
