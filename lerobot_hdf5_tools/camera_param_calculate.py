"""Export single/dual-arm calibration from the existing calibration values.

This exports existing results; it does not run a new hand-eye calibration.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


LEFT_CAMERA_K = [[392.3834955, 0, 316.884], [0, 392.3277108, 236.26], [0, 0, 1]]
RIGHT_CAMERA_K = [[392.5541209, 0, 318.032], [0, 392.4855422, 238.947], [0, 0, 1]]
HEAD_CAMERA_K = [[389.3139881, 0, 321.565], [0, 389.560241, 248.887], [0, 0, 1]]
# HALCON pose type 2 (abg): translation in m, extrinsic XYZ angles in degrees.
LEFT_CAM_IN_FLANGE_RPY = [-0.0846545, -0.0269479, 0.1148732, 335.147, 0.214637, 295.065]
RIGHT_CAM_IN_FLANGE_RPY = [0.0867175, 0.0283009, 0.1140642, 335.107, 359.389, 114.261]
# Preserved from the original converter; relation to the recorded base is unresolved.
HEAD_CAMINBASEPOSE = [
    [-0.007527, 0.876441, -0.48145, -0.126936],
    [-0.001166, -0.481471, -0.876461, 0.253658],
    [-0.9999971, -0.006035, 0.004646, -0.124553],
    [0, 0, 0, 1],
]
# Translation in mm; rotation about local Z is approximately +22.5 degrees.
T_FLANGE_TCP = [
    [0.92388, -0.382683, 0, 0],
    [0.382683, 0.92388, 0, 0],
    [0, 0, 1, 230],
    [0, 0, 0, 1],
]
# Experimental camera-in-TCP calibration; translation provisionally treated as mm.
T_TCP_CAM = [
    [0.042917, 0.906315, 0.420417, -0.088514],
    [-0.999074, 0.040096, 0.015551, 0.007496],
    [-0.002763, -0.420696, 0.907198, -229.935127],
    [0, 0, 0, 1],
]


def camera_in_flange(pose: list[float]) -> list:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", pose[3:], degrees=True
    ).as_matrix()
    transform[:3, 3] = pose[:3]
    return np.round(transform, 6).tolist()


def build_camera_params() -> dict:
    return {
        "LEFT_CAMERA_K": LEFT_CAMERA_K,
        "RIGHT_CAMERA_K": RIGHT_CAMERA_K,
        "HEAD_CAMERA_K": HEAD_CAMERA_K,
        "LEFT_CAMINTOOLPOSE": camera_in_flange(LEFT_CAM_IN_FLANGE_RPY),
        "RIGHT_CAMINTOOLPOSE": camera_in_flange(RIGHT_CAM_IN_FLANGE_RPY),
        "HEAD_CAMINBASEPOSE": HEAD_CAMINBASEPOSE,
        "LEFT_T_FLANGE_TCP": T_FLANGE_TCP,
        # Provisional right TCP offset copied from left by request; independently editable.
        "RIGHT_T_FLANGE_TCP": [row[:] for row in T_FLANGE_TCP],
        "RIGHT_T_FLANGE_TCP_STATUS": "provisional_copy_of_left",
        "LEFT_T_TCP_CAM": T_TCP_CAM,
        "RIGHT_T_TCP_CAM": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save_path", type=Path, default=Path(__file__).with_name("camera_params.json"),
                        help="Output JSON path (default: camera_params.json beside this script)")
    parser.add_argument("--no_save", action="store_true", help="Print JSON without writing")
    args = parser.parse_args()
    text = json.dumps(build_camera_params(), ensure_ascii=False, indent=2) + "\n"
    if args.no_save:
        print(text, end="")
    else:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        args.save_path.write_text(text, encoding="utf-8")
        print(f"Saved calibration: {args.save_path}")


if __name__ == "__main__":
    main()
