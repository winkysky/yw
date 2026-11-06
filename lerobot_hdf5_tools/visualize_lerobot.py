"""Visualize single/dual-arm LeRobot v2 poses, with optional TCP/video projection."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pose_utils import DEFAULT_CAMERA_PARAMS, WRIST_SOURCES, episode_arrays, episode_poses, load_camera_params
from visualization_utils import draw_tcp_projection, plot_poses

if TYPE_CHECKING:
    from pandas import DataFrame


def episode_path(root: Path, info: dict, episode: int, kind: str, video_key: str = "") -> Path:
    """Resolve the v2 metadata template, requiring files inside the dataset root."""
    relative = info[kind].format(
        episode_chunk=episode // int(info["chunks_size"]),
        episode_index=episode,
        video_key=video_key,
    )
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{kind} must point inside the dataset root")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def load_episode(root: Path, episode: int) -> tuple[DataFrame, dict]:
    import pandas as pd

    with (root / "meta" / "info.json").open(encoding="utf-8") as file:
        info = json.load(file)
    if not str(info.get("codebase_version", "")).startswith("v2."):
        raise ValueError("This viewer supports LeRobot v2.x (one parquet/MP4 per episode)")
    if not 0 <= episode < int(info["total_episodes"]):
        raise ValueError("episode_index out of range")
    table = pd.read_parquet(episode_path(root, info, episode, "data_path"))
    for key in ("observation.state", "timestamp"):
        if key not in table:
            raise KeyError(f"Missing required column: {key}")
    if table.empty:
        raise ValueError("Episode is empty")
    if "episode_index" in table and not (table["episode_index"] == episode).all():
        raise ValueError("Parquet contains rows from a different episode")
    if "frame_index" in table and not np.array_equal(table["frame_index"], np.arange(len(table))):
        raise ValueError("Expected ordered consecutive frame_index values starting at zero")
    return table, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="One local LeRobot v2 dataset root")
    parser.add_argument("--episode_index", type=int, default=0, help="Episode number")
    parser.add_argument("--frame", type=int, default=0, help="Initial original episode row/video frame")
    parser.add_argument("--camera_params", type=Path, default=DEFAULT_CAMERA_PARAMS, help="Calibration JSON")
    parser.add_argument("--wrist_extrinsic_source", choices=WRIST_SOURCES, default="flange_calibration",
                        help="Use the same wrist-camera chain as the converter")
    parser.add_argument("--show_video", action="store_true", help="Show the matching MP4 frame alongside 3D poses")
    parser.add_argument("--camera", nargs="+", choices=("all", "wrist", "left_wrist", "right_wrist", "head"),
                        default=["all"], help="Views to show (default: head and available wrists); wrist aliases left_wrist")
    parser.add_argument("--project_tcp", action="store_true", help="Overlay current TCP axes and future trajectory; implies show_video")
    parser.add_argument("--future_frames", type=int, default=30, help="Future original frames to project (0: only current TCP)")
    parser.add_argument("--future_step", type=int, default=1, help="Stride between future trajectory points")
    parser.add_argument("--calibration_size", type=int, nargs=2, default=(640, 480), metavar=("WIDTH", "HEIGHT"),
                        help="Image size for the stored intrinsics; scale K if video size differs")
    parser.add_argument("--axis_length", type=float, default=0.05, help="Coordinate axis length in metres")
    parser.add_argument("--save_path", type=Path, help="Optional output plot, outside the upload directory")
    parser.add_argument("--save_video", type=Path, help="Export the displayed panels to an MP4, starting at frame")
    parser.add_argument("--end_frame", type=int, help="Exclusive video end frame (default: episode length)")
    parser.add_argument("--fps", type=float, help="Output video FPS (default: dataset FPS); changes playback speed")
    parser.add_argument("--no_show", action="store_true", help="Headless rendering; requires save_path or save_video")
    args = parser.parse_args()
    args.show_video = args.show_video or args.project_tcp
    if args.future_frames < 0 or args.future_step < 1 or min(args.calibration_size) < 1:
        parser.error("future_frames must be >= 0; future_step and calibration_size must be positive")

    if (args.end_frame is not None or args.fps is not None) and args.save_video is None:
        parser.error("--end_frame and --fps require --save_video")
    if args.save_video is not None and args.save_path is not None and args.save_video.resolve() == args.save_path.resolve():
        parser.error("--save_video and --save_path must be different files")

    if args.show_video or args.save_video is not None:
        # Load the optional video backend before Arrow's native libraries.
        import cv2

    table, info = load_episode(args.dataset, args.episode_index)
    video_fps = args.fps if args.fps is not None else info.get("fps", 0)
    params = load_camera_params(args.camera_params)
    arrays = episode_arrays(table)
    arm_poses = episode_poses(arrays, params, args.wrist_extrinsic_source)
    poses = {f"{arm}_{name}": sequence for arm, values in arm_poses.items() for name, sequence in values.items()}
    tcp = np.stack([arm_poses[arm]["tcp"] for arm in arrays["arms"]], axis=1)
    if "all" in args.camera and len(args.camera) != 1:
        parser.error("Use --camera all alone, or list individual cameras")
    cameras = (["head"] + [f"{arm}_wrist" for arm in arrays["arms"]] if args.camera == ["all"]
               else list(dict.fromkeys("left_wrist" if camera == "wrist" else camera for camera in args.camera)))
    if "right_wrist" in cameras and "right" not in arm_poses:
        parser.error("right_wrist is unavailable for single-arm data")
    if args.wrist_extrinsic_source == "tcp_calibration":
        warnings.warn("Experimental T_TCP_CAM route; its translation is assumed to be in mm.")
    # Reuse the head calibration stored in the original label/tap H5 files.
    # An explicit camera-in-current-robot-base matrix can override it for other setups.
    head_key = "HEAD_CAMINROBOTBASE" if "HEAD_CAMINROBOTBASE" in params else "HEAD_CAMINBASEPOSE"
    head_pose = np.asarray(params[head_key])
    captures = {}
    image_loaders = {}

    def load_camera_frame(camera: str, index: int) -> tuple[np.ndarray, str]:
        # All views use the same episode row/frame, as in conversion.
        capture = captures[camera]
        if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode {camera} video frame {index}")
        if args.project_tcp:
            camera_pose = (arm_poses[camera.split("_")[0]]["wrist_camera"][index]
                           if camera != "head" else head_pose)
            key = f"{camera.split('_')[0].upper()}_CAMERA_K"
            intrinsic = np.array(params[key], dtype=float)
            intrinsic[0] *= image.shape[1] / args.calibration_size[0]
            intrinsic[1] *= image.shape[0] / args.calibration_size[1]
            image = draw_tcp_projection(image, tcp, camera_pose, intrinsic,
                                        index, args.future_frames, args.future_step, args.axis_length)
        title = f"{'head (main)' if camera == 'head' else camera} camera / frame {index}"
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), title

    try:
        if args.show_video:
            for camera in cameras:
                path = episode_path(args.dataset, info, args.episode_index, "video_path",
                                    f"observation.images.{camera}_image")
                if args.save_video is not None and args.save_video.resolve() == path:
                    parser.error("--save_video must not overwrite an input video")
                captures[camera] = cv2.VideoCapture(str(path))
                if not captures[camera].isOpened():
                    raise RuntimeError(f"Cannot open {camera} video: {path}")
                image_loaders[camera] = lambda index, camera=camera: load_camera_frame(camera, index)
        plot_poses(poses, table["timestamp"].to_numpy(), args.frame, args.axis_length,
                   args.save_path, args.no_show, image_loaders,
                   save_video=args.save_video, video_fps=video_fps, end_frame=args.end_frame)
    finally:
        for capture in captures.values():
            capture.release()


if __name__ == "__main__":
    main()
