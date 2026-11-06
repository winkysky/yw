"""View single/dual-arm HDF5 poses exported by this folder's converter."""

import argparse
from pathlib import Path

import h5py
import numpy as np

from visualization_utils import plot_poses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5_path", type=Path, required=True, help="Converted HDF5 episode")
    parser.add_argument("--frame", type=int, default=0, help="Initial sampled HDF5 frame")
    parser.add_argument("--axis_length", type=float, default=0.05, help="Coordinate axis length in metres")
    parser.add_argument("--save_path", type=Path, help="Optional output plot, outside the upload directory")
    parser.add_argument("--no_show", action="store_true", help="Headless rendering; requires save_path")
    args = parser.parse_args()
    with h5py.File(args.h5_path, "r") as file:
        tcp = np.asarray(file["ee_pose"])
        if tcp.ndim != 4 or tcp.shape[1] not in (1, 2) or tcp.shape[2:] != (4, 4):
            raise ValueError(f"Expected ee_pose shape (T, 1/2, 4, 4), got {tcp.shape}")
        arms = ("left",) if tcp.shape[1] == 1 else ("left", "right")
        cameras = ("wrist_cam",) if len(arms) == 1 else ("lh_cam", "rh_cam")
        poses = {}
        for index, (arm, camera) in enumerate(zip(arms, cameras)):
            poses[f"{arm}_tcp"] = tcp[:, index]
            poses[f"{arm}_wrist_camera"] = np.asarray(file[f"{camera}/pose"])
        timestamps = np.asarray(file["timestamp"])
    plot_poses(poses, timestamps, args.frame, args.axis_length, args.save_path, args.no_show)


if __name__ == "__main__":
    main()
