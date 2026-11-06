"""Shared single/dual-arm schema and flange/TCP conventions."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_CAMERA_PARAMS = Path(__file__).resolve().with_name("camera_params.json")
WRIST_SOURCES = ("flange_calibration", "tcp_calibration")


def load_camera_params(path: Path | str = DEFAULT_CAMERA_PARAMS) -> dict:
    """Read calibration; translations retain their documented source units."""
    with Path(path).open(encoding="utf-8") as file:
        params = json.load(file)
    for key in (key for key in params if key.endswith("_CAMERA_K")):
        value = np.asarray(params[key], dtype=float)
        if value.shape != (3, 3) or not np.isfinite(value).all():
            raise ValueError(f"{key} must be a finite 3x3 matrix")
    matrix_keys = ("LEFT_CAMINTOOLPOSE", "RIGHT_CAMINTOOLPOSE", "HEAD_CAMINBASEPOSE",
                   "HEAD_CAMINROBOTBASE", "T_FLANGE_TCP", "T_TCP_CAM",
                   "LEFT_T_FLANGE_TCP", "RIGHT_T_FLANGE_TCP", "LEFT_T_TCP_CAM", "RIGHT_T_TCP_CAM",
                   "LEFT_BASE_IN_ROBOTBASE", "RIGHT_BASE_IN_ROBOTBASE")
    for key in matrix_keys:
        if key not in params or params[key] is None:
            continue
        value = np.asarray(params[key], dtype=float)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError(f"{key} must be a finite 4x4 matrix")
        if not np.allclose(value[3], [0, 0, 0, 1]):
            raise ValueError(f"{key} has an invalid homogeneous bottom row")
    return params


def T_mm2m(transform: np.ndarray) -> np.ndarray:
    """Convert only the translation of a 4x4 transform from mm to m."""
    result = np.array(transform, dtype=np.float64, copy=True)
    result[:3, 3] /= 1000.0
    return result


def rpy_to_matrix(xyzrpy: np.ndarray) -> np.ndarray:
    """[x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg] -> ^B T_F; R = Rz Ry Rx."""
    values = np.asarray(xyzrpy, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all():
        raise ValueError("observation.tcp_pose must have finite shape (T, 6)")
    result = np.repeat(np.eye(4)[None], len(values), axis=0)
    result[:, :3, :3] = Rotation.from_euler("xyz", values[:, 3:], degrees=True).as_matrix()
    result[:, :3, 3] = values[:, :3] / 1000.0
    return result


def arm_transform(params: dict, arm: str, name: str) -> np.ndarray:
    """Resolve an arm's mm calibration; sharing TCP offsets must be explicit."""
    value = params.get(f"{arm.upper()}_{name}")
    if value is None and (arm == "left" or (name == "T_FLANGE_TCP" and params.get("SHARE_T_FLANGE_TCP", False))):
        value = params.get(f"LEFT_{name}", params.get(name))
    if value is None:
        raise ValueError(f"Set {arm.upper()}_{name} in calibration JSON; right-arm calibration is not inferred")
    return T_mm2m(value)


def compute_poses(raw_pose: np.ndarray, params: dict, source: str, arm: str = "left") -> dict:
    """Return flange, TCP and wrist-camera poses in the recorded base frame."""
    flange = rpy_to_matrix(raw_pose)
    # Optional recorded arm-base -> shared robot-base transform; translations in m.
    flange = np.asarray(params.get(f"{arm.upper()}_BASE_IN_ROBOTBASE", np.eye(4))) @ flange
    tcp = flange @ arm_transform(params, arm, "T_FLANGE_TCP")
    if source == "flange_calibration":
        wrist = flange @ np.asarray(params[f"{arm.upper()}_CAMINTOOLPOSE"])
    elif source == "tcp_calibration":
        wrist = tcp @ arm_transform(params, arm, "T_TCP_CAM")
    else:
        raise ValueError(f"Unknown wrist calibration source: {source}")
    return {"flange": flange, "tcp": tcp, "wrist_camera": wrist}


def episode_arrays(table) -> dict:
    """Read packed HW fields. Arm order is left, right; poses are flange mm/deg."""
    state = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in table["observation.state"]])
    if state.shape[1] not in (7, 14):
        raise ValueError(f"Expected 7/14 state values, got {state.shape}")
    count, width = state.shape
    arms = ("left",) if width == 7 else ("left", "right")
    state = state.reshape(count, len(arms), 7)
    columns = table.column_names if hasattr(table, "column_names") else table.keys()
    if "observation.tcp_pose" in columns:
        raw_pose = np.stack([np.asarray(value).reshape(-1) for value in table["observation.tcp_pose"]])
        if raw_pose.shape != (count, 6 * len(arms)):
            raise ValueError(f"Expected {6 * len(arms)} flange pose values per row, got {raw_pose.shape}")
        raw_pose = raw_pose.reshape(count, len(arms), 6)
    else:
        keys = [f"observation.{arm}_tcp_pose" for arm in arms]
        if not all(key in columns for key in keys):
            raise ValueError("Missing recorded flange poses: need observation.tcp_pose (6/12 values) "
                             "or observation.left_tcp_pose/right_tcp_pose (6 each). Joint-only data needs FK first.")
        raw_pose = np.stack([np.stack(table[key]) for key in keys], axis=1)
    return {"arms": arms, "qpos": state[:, :, :6], "gripper": state[:, :, 6], "raw_pose": raw_pose}


def episode_poses(arrays: dict, params: dict, source: str) -> dict:
    """Compute per-arm poses with the same implementation for H5 and plots."""
    return {arm: compute_poses(arrays["raw_pose"][:, index], params, source, arm)
            for index, arm in enumerate(arrays["arms"])}
