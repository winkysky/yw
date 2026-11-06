"""Convert local single/dual-arm LeRobot v2 data with the configured flange-to-TCP offset."""

import argparse
import json
import logging
import socket
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, load_dataset
from tqdm import tqdm

from scipy.spatial.transform import Rotation

from pose_utils import arm_transform, episode_arrays, episode_poses, load_camera_params

SCRIPT_DIR = Path(__file__).resolve().parent
CAMERA_PARAMS_PATH = SCRIPT_DIR / "camera_params.json"
WRIST_EXTRINSIC_SOURCES = ("flange_calibration", "tcp_calibration")
RGB_STORAGE_MODES = ("video_path", "jpeg")
JPEG_QUALITY = 90

cam_param = load_camera_params(CAMERA_PARAMS_PATH)


def log_wrist_extrinsic_configuration(
    logger: logging.Logger,
    wrist_extrinsic_source: str,
    compare_wrist_extrinsics: bool,
    trans_tol_mm: float,
    rot_tol_deg: float,
):
    logger.info(f"Wrist extrinsic source: {wrist_extrinsic_source}")
    if wrist_extrinsic_source == "tcp_calibration":
        logger.warning("Experimental T_TCP_CAM translations are interpreted in mm")
    if not compare_wrist_extrinsics:
        return
    for arm in ("left", "right"):
        try:
            candidate = arm_transform(cam_param, arm, "T_FLANGE_TCP") @ arm_transform(cam_param, arm, "T_TCP_CAM")
        except ValueError:
            logger.info(f"{arm}: comparison unavailable; calibration not supplied")
            continue
        flange_camera = np.asarray(cam_param[f"{arm.upper()}_CAMINTOOLPOSE"])
        distance_mm = np.linalg.norm(candidate[:3, 3] - flange_camera[:3, 3]) * 1000
        angle_deg = np.rad2deg(Rotation.from_matrix(flange_camera[:3, :3].T @ candidate[:3, :3]).magnitude())
        logger.info(f"{arm}: calibration difference {distance_mm:.3f} mm, {angle_deg:.3f} deg")
        if distance_mm > trans_tol_mm or angle_deg > rot_tol_deg:
            logger.warning(f"{arm}: calibration difference exceeds tolerance; keeping selected source")


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("convert_single_arm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_sampled_indices(total_length: int, sample_scale: float) -> np.ndarray:
    if sample_scale <= 0:
        raise ValueError(f"sample_scale must be positive, got {sample_scale}")
    sample_dt = max(int(round(1 / sample_scale)), 1)
    return np.arange(0, total_length, sample_dt)


def extract_frames_jpeg(
    video_path: Path,
    frame_indices: np.ndarray,
    jpeg_quality: int = JPEG_QUALITY,
) -> list[np.ndarray]:
    """Sequentially decode selected frames and encode standard BGR JPEG bytes."""
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if frame_indices.ndim != 1 or frame_indices.size == 0:
        raise ValueError(f"frame_indices must be a non-empty 1D array, got {frame_indices.shape}")
    if np.any(frame_indices < 0) or np.any(np.diff(frame_indices) <= 0):
        raise ValueError("frame_indices must be strictly increasing and non-negative")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_target = int(frame_indices[-1])
    if reported_frame_count > 0 and max_target >= reported_frame_count:
        cap.release()
        raise IndexError(
            f"Requested frame {max_target} from {video_path}, but video reports "
            f"{reported_frame_count} frames"
        )

    target_set = set(frame_indices.tolist())
    encoded_by_index: dict[int, np.ndarray] = {}
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    frame_idx = 0

    try:
        while frame_idx <= max_target:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx in target_set:
                encoded_ok, encoded = cv2.imencode(".jpg", frame_bgr, encode_params)
                if not encoded_ok:
                    raise RuntimeError(f"JPEG encoding failed for frame {frame_idx}: {video_path}")
                encoded_by_index[frame_idx] = np.asarray(encoded, dtype=np.uint8).reshape(-1)
            frame_idx += 1
    finally:
        cap.release()

    missing = [int(i) for i in frame_indices if int(i) not in encoded_by_index]
    if missing:
        raise RuntimeError(
            f"Failed to decode {len(missing)} requested frames from {video_path}; "
            f"first missing indices: {missing[:10]}"
        )
    return [encoded_by_index[int(i)] for i in frame_indices]


def to_scalar(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def get_prompt_text(ds_meta: LeRobotDatasetMetadata, hf_dataset) -> str:
    task_index = to_scalar(hf_dataset["task_index"][0])
    return str(ds_meta.tasks[int(task_index)])


def resolve_video_path(
    ds_meta: LeRobotDatasetMetadata,
    ep_idx: int,
    video_root: Path,
    video_key: str,
) -> str:
    video_path = str(ds_meta.get_video_file_path(ep_idx, video_key))
    if not (video_root / video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_root / video_path}")
    return video_path


def load_local_dataset(
    ds_meta: LeRobotDatasetMetadata,
    hf_dataset,
    ep_idx: int,
    video_root: Path,
    sample_scale: float = 0.33,
    wrist_extrinsic_source: str = "flange_calibration",
    rgb_storage: str = "video_path",
) -> tuple[dict, dict]:
    ref_idxs = get_sampled_indices(len(hf_dataset), sample_scale)
    timestamp = np.array(
        [to_scalar(hf_dataset[int(i)]["timestamp"]) for i in ref_idxs],
        dtype=np.float64,
    )

    prompt_text = get_prompt_text(ds_meta, hf_dataset)
    print(f"Prompt text: {prompt_text}")

    sampled = hf_dataset.select(ref_idxs)
    arrays = episode_arrays(sampled)
    arms = arrays["arms"]
    poses = episode_poses(arrays, cam_param, wrist_extrinsic_source)
    dual = len(arms) == 2
    count = len(ref_idxs)

    force = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value in sampled["observation.force_tcp"]])
    if force.shape[1] not in (6 * len(arms), 7 * len(arms)):
        raise ValueError("force_tcp must contain 6 or 7 values per arm, ordered left then right")
    force = force.reshape(count, len(arms), -1)[:, :, :6]
    data = {
        "timestamp": timestamp,
        "ee_pose": np.stack([poses[arm]["tcp"] for arm in arms], axis=1),
        "qpos": np.ascontiguousarray(arrays["qpos"]),
        "gripper": np.ascontiguousarray(arrays["gripper"]),
        "force_tcp": np.ascontiguousarray(force if dual else force[:, 0]),
    }
    # Head calibration is only aligned if an explicit camera-in-recorded-base matrix is supplied.
    head_pose = np.asarray(cam_param.get("HEAD_CAMINROBOTBASE", cam_param["HEAD_CAMINBASEPOSE"]))
    cameras = [("head_cam" if dual else "external_cam", "head", "HEAD", np.repeat(head_pose[None], count, axis=0))]
    for arm in arms:
        name = ("lh_cam" if arm == "left" else "rh_cam") if dual else "wrist_cam"
        cameras.append((name, f"{arm}_wrist", arm.upper(), poses[arm]["wrist_camera"]))
    for name, video_name, prefix, camera_pose in cameras:
        video = resolve_video_path(ds_meta, ep_idx, video_root, f"observation.images.{video_name}_image")
        if rgb_storage == "video_path":
            rgb = [video]
        elif rgb_storage == "jpeg":
            rgb = extract_frames_jpeg(video_root / video, ref_idxs)
        else:
            raise ValueError(f"Unknown rgb_storage={rgb_storage!r}")
        data[f"{name}/rgb"] = rgb
        data[f"{name}/pose"] = np.ascontiguousarray(camera_pose)
        data[f"{name}/K"] = np.asarray(cam_param[f"{prefix}_CAMERA_K"])

    offsets = np.stack([arm_transform(cam_param, arm, "T_FLANGE_TCP") for arm in arms])
    attr = {
        "prompt_text": prompt_text,
        "compress": "mp4" if rgb_storage == "video_path" else True,
        "is_bgr": rgb_storage == "jpeg",
        "episode_index": ep_idx,
        "select_grasp_index": np.arange(len(arms)),
        "arm_names": list(arms),
        "qpos_unit": "degree",
        "wrist_extrinsic_source": wrist_extrinsic_source,
        "T_FLANGE_TCP_m": offsets if dual else offsets[0],
        "external_cam_base_aligned": "HEAD_CAMINROBOTBASE" in cam_param,
    }
    if dual:
        attr["right_tcp_calibration_status"] = cam_param.get("RIGHT_T_FLANGE_TCP_STATUS", "user_configured")
    return data, attr


def find_data_directories(root_path: Path) -> list[tuple[Path, str]]:
    valid_dirs = []

    def _is_valid_data_dir(path: Path) -> bool:
        return path.is_dir() and (path / "meta" / "tasks.jsonl").is_file()

    def _recurse(current_path: Path, rel_path: str):
        if not current_path.exists() or not current_path.is_dir():
            return
        if _is_valid_data_dir(current_path):
            valid_dirs.append((current_path, rel_path or "."))
            return
        try:
            for item in current_path.iterdir():
                if item.is_dir() and not item.is_symlink():
                    new_rel_path = f"{rel_path}/{item.name}" if rel_path else item.name
                    _recurse(item, new_rel_path)
        except PermissionError:
            pass

    _recurse(root_path, "")
    return valid_dirs


def save_to_h5(path: str, data_dict: dict, attr_dict: dict):
    output_path = Path(path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    try:
        with h5py.File(temp_path, "w") as h5:
            for k, v in data_dict.items():
                if isinstance(v, np.ndarray):
                    h5.create_dataset(k, data=v)
                else:
                    if isinstance(v[0], np.ndarray):
                        dtype = h5py.vlen_dtype(v[0].dtype)
                    elif isinstance(v[0], str):
                        dtype = h5py.string_dtype(encoding="utf-8")
                    else:
                        raise ValueError(f"Unsupported data type for key {k}, value type: {type(v[0])}")
                    dset = h5.create_dataset(k, shape=(len(v),), dtype=dtype)
                    for i, vi in enumerate(v):
                        dset[i] = vi
            for ak, av in attr_dict.items():
                h5.attrs[ak] = av
        temp_path.replace(output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    print(f"Saved to {output_path}")


def _process_one_episode(args: tuple):
    socket.setdefaulttimeout(60)
    (
        episode_id_full,
        sample_idx,
        src_path,
        tgt_path,
        skip_saved,
        sample_scale,
        wrist_extrinsic_source,
        rgb_storage,
        camera_params_path,
    ) = args
    try:
        global cam_param
        cam_param = load_camera_params(camera_params_path)
        if episode_id_full == ".":
            data_repo_id = Path(src_path).name
            data_root = Path(src_path)
        else:
            data_repo_id = Path(episode_id_full).name
            data_root = Path(src_path) / episode_id_full

        tgt_file = Path(tgt_path) / episode_id_full / f"{sample_idx:05}.h5"
        if skip_saved and tgt_file.exists():
            return episode_id_full, sample_idx, "skipped", None

        ds_meta = LeRobotDatasetMetadata(repo_id=data_repo_id, root=data_root, force_cache_sync=False)
        video_root = data_root

        parquet_file = ds_meta.get_data_file_path(sample_idx)
        hf_dataset = load_dataset("parquet", data_files=str(data_root / parquet_file), split="train")

        data, attr = load_local_dataset(
            ds_meta=ds_meta,
            hf_dataset=hf_dataset,
            ep_idx=sample_idx,
            sample_scale=sample_scale,
            video_root=video_root,
            wrist_extrinsic_source=wrist_extrinsic_source,
            rgb_storage=rgb_storage,
        )
        save_to_h5(str(tgt_file), data, attr)
        return episode_id_full, sample_idx, "success", None
    except Exception:
        return episode_id_full, sample_idx, "error", traceback.format_exc()


def main(
    src_path: str,
    tgt_path: str,
    single_id: Optional[str] = None,
    process_all: bool = False,
    skip_saved: bool = False,
    num_workers: int = 1,
    log_file: str = "convert_single_arm_errors.log",
    sample_scale: float = 0.33,
    episode_index: Optional[int] = None,
    wrist_extrinsic_source: str = "flange_calibration",
    compare_wrist_extrinsics: bool = False,
    wrist_compare_trans_tol_mm: float = 5.0,
    wrist_compare_rot_tol_deg: float = 5.0,
    rgb_storage: str = "video_path",
    camera_params_path: str = str(CAMERA_PARAMS_PATH),
):
    global cam_param
    cam_param = load_camera_params(camera_params_path)
    logger = setup_logger(log_file)
    socket.setdefaulttimeout(60)
    log_wrist_extrinsic_configuration(
        logger=logger,
        wrist_extrinsic_source=wrist_extrinsic_source,
        compare_wrist_extrinsics=compare_wrist_extrinsics,
        trans_tol_mm=wrist_compare_trans_tol_mm,
        rot_tol_deg=wrist_compare_rot_tol_deg,
    )

    src_root = Path(src_path)
    if process_all:
        dataset_name_list = [rel_path for _, rel_path in find_data_directories(src_root)]
        logger.info(f"Found {len(dataset_name_list)} data directories via recursive search")
    else:
        dataset_name_list = [single_id or "."]
        single_path = src_root if dataset_name_list[0] == "." else src_root / dataset_name_list[0]
        if not single_path.exists():
            raise FileNotFoundError(f"single_id path does not exist: {single_path}")
        logger.info(f"Processing single task: {dataset_name_list[0]}")

    work_items = []
    metadata_errors = 0
    for episode_id_full in tqdm(dataset_name_list, desc="Scanning metadata"):
        try:
            data_root = src_root if episode_id_full == "." else src_root / episode_id_full
            info_path = data_root / "meta" / "info.json"
            with info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            if not str(info.get("codebase_version", "")).startswith("v2."):
                raise ValueError("Expected LeRobot v2.x with one parquet/MP4 per episode")
            total_episodes = info["total_episodes"]
            if episode_index is None:
                sample_indices = range(total_episodes)
            else:
                if episode_index < 0 or episode_index >= total_episodes:
                    raise IndexError(
                        f"episode_index={episode_index} out of range for {episode_id_full}: "
                        f"total_episodes={total_episodes}"
                    )
                sample_indices = [episode_index]
            for sample_idx in sample_indices:
                work_items.append(
                    (
                        episode_id_full,
                        sample_idx,
                        src_path,
                        tgt_path,
                        skip_saved,
                        sample_scale,
                        wrist_extrinsic_source,
                        rgb_storage,
                        camera_params_path,
                    )
                )
        except Exception:
            metadata_errors += 1
            logger.error(f"Failed to load metadata for {episode_id_full}:\n{traceback.format_exc()}")

    logger.info(f"Total episodes to process: {len(work_items)}")
    logger.info(f"RGB storage mode: {rgb_storage}")
    n_success, n_skipped, n_error = 0, 0, metadata_errors

    if num_workers <= 1:
        for args in tqdm(work_items, desc="Processing episodes"):
            episode_id_full, sample_idx, status, tb = _process_one_episode(args)
            if status == "success":
                n_success += 1
            elif status == "skipped":
                n_skipped += 1
            else:
                n_error += 1
                logger.error(f"Error [{episode_id_full} ep={sample_idx}]:\n{tb}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_one_episode, args): args for args in work_items}
            with tqdm(total=len(futures), desc="Processing episodes") as pbar:
                for future in as_completed(futures):
                    episode_id_full, sample_idx, status, tb = future.result()
                    if status == "success":
                        n_success += 1
                    elif status == "skipped":
                        n_skipped += 1
                    else:
                        n_error += 1
                        logger.error(f"Error [{episode_id_full} ep={sample_idx}]:\n{tb}")
                    pbar.update(1)
                    pbar.set_postfix(ok=n_success, skip=n_skipped, err=n_error)

    logger.info(f"Done. success={n_success}, skipped={n_skipped}, errors={n_error}")
    if n_error or not work_items:
        raise RuntimeError("Conversion incomplete; inspect the conversion log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert single/dual-arm LeRobot datasets to HDF5.")
    parser.add_argument("--camera_params", default=str(CAMERA_PARAMS_PATH), help="Camera/TCP calibration JSON")
    parser.add_argument("--src_path", type=str, required=True, help="Dataset root path")
    parser.add_argument("--tgt_path", type=str, required=True, help="Target HDF5 root")
    parser.add_argument("--single_id", type=str, default=None, help="One dataset path relative to src_path")
    parser.add_argument("-a", action="store_true", default=False, help="Process all datasets under src_path")
    parser.add_argument("--skip_saved", action="store_true", default=False, help="Skip already saved h5 files")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log_file", type=str, default="convert_single_arm_errors.log", help="Error log path")
    parser.add_argument("--sample_scale", type=float, default=1, help="target_hz/original_hz, 0.33=10/30")
    parser.add_argument("--episode_index", type=int, default=None, help="Optional single episode index for testing")
    parser.add_argument(
        "--wrist_extrinsic_source",
        choices=WRIST_EXTRINSIC_SOURCES,
        default="flange_calibration",
        help="Calibration chain used for wrist_cam/pose (default: flange_calibration)",
    )
    parser.add_argument(
        "--compare_wrist_extrinsics",
        action="store_true",
        help="Log the static disagreement between flange and TCP wrist-camera calibrations",
    )
    parser.add_argument(
        "--wrist_compare_trans_tol_mm",
        type=float,
        default=5.0,
        help="Translation warning threshold for wrist extrinsic comparison in mm",
    )
    parser.add_argument(
        "--wrist_compare_rot_tol_deg",
        type=float,
        default=5.0,
        help="Rotation warning threshold for wrist extrinsic comparison in degrees",
    )
    parser.add_argument(
        "--rgb_storage",
        choices=RGB_STORAGE_MODES,
        default="video_path",
        help="Store one MP4 path per camera or sampled JPEG bytes (default: video_path), video_path or jpeg",
    )
    args = parser.parse_args()

    main(
        src_path=args.src_path,
        tgt_path=args.tgt_path,
        single_id=args.single_id,
        process_all=args.a,
        skip_saved=args.skip_saved,
        num_workers=args.num_workers,
        log_file=args.log_file,
        sample_scale=args.sample_scale,
        episode_index=args.episode_index,
        wrist_extrinsic_source=args.wrist_extrinsic_source,
        compare_wrist_extrinsics=args.compare_wrist_extrinsics,
        wrist_compare_trans_tol_mm=args.wrist_compare_trans_tol_mm,
        wrist_compare_rot_tol_deg=args.wrist_compare_rot_tol_deg,
        rgb_storage=args.rgb_storage,
        camera_params_path=args.camera_params,
    )
