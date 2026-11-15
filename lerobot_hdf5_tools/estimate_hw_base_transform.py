"""Estimate recorded arm bases from URDF FK and HW flange poses.

Fit world_T_link6(q) @ link6_T_flange = world_T_recorded_base @ recorded_T_flange.
The two constant transforms per arm are fitted jointly so Link6 is not mistaken
for the controller flange frame. Units: input joints/RPY degrees, XYZ mm;
URDF, fitted transforms, and output calibration translations use metres.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from pose_utils import episode_arrays, load_camera_params, rpy_to_matrix
from visualize_lerobot import load_episode


def joint_chain(urdf: Path, prefix: str) -> list[tuple[np.ndarray, np.ndarray]]:
    root = ET.parse(urdf).getroot()
    chain = []
    parent = "base_link"
    for index in range(1, 7):
        joint = root.find(f"joint[@name='{prefix}_joint{index}']")
        if joint is None or joint.get("type") not in ("revolute", "continuous"):
            raise ValueError(f"Missing revolute joint {prefix}_joint{index}")
        if joint.find("parent").get("link") != parent:
            raise ValueError("Expected a direct six-joint chain from base_link")
        parent = joint.find("child").get("link")
        origin = joint.find("origin")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
        transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        chain.append((transform, axis / np.linalg.norm(axis)))
    return chain


def fk(chain: list, joint_degrees: np.ndarray) -> np.ndarray:
    angles = np.deg2rad(joint_degrees)
    poses = np.tile(np.eye(4), (len(angles), 1, 1))
    for index, (origin, axis) in enumerate(chain):
        motion = np.tile(np.eye(4), (len(angles), 1, 1))
        motion[:, :3, :3] = Rotation.from_rotvec(angles[:, index, None] * axis).as_matrix()
        poses = poses @ origin @ motion
    return poses


def transform(vector: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    result[:3, 3] = vector[3:]
    return result


def errors(p: np.ndarray, q: np.ndarray, base: np.ndarray, tip: np.ndarray) -> tuple:
    predicted, recorded = p @ tip, base @ q
    translation = predicted[:, :3, 3] - recorded[:, :3, 3]
    rotation = Rotation.from_matrix(
        np.swapaxes(predicted[:, :3, :3], 1, 2) @ recorded[:, :3, :3]
    ).as_rotvec()
    return translation, rotation


def stats(p: np.ndarray, q: np.ndarray, base: np.ndarray, tip: np.ndarray) -> dict:
    translation, rotation = errors(p, q, base, tip)
    distances = np.linalg.norm(translation, axis=1) * 1000
    angles = np.rad2deg(np.linalg.norm(rotation, axis=1))
    return {"samples": len(p), "position_rms_mm": float(np.sqrt(np.mean(distances ** 2))),
            "position_p95_mm": float(np.percentile(distances, 95)),
            "position_max_mm": float(distances.max()),
            "rotation_rms_deg": float(np.sqrt(np.mean(angles ** 2))),
            "rotation_p95_deg": float(np.percentile(angles, 95)),
            "rotation_max_deg": float(angles.max())}


def fit(p: np.ndarray, q: np.ndarray) -> tuple:
    initial_bases = p @ np.linalg.inv(q)
    initial = np.r_[Rotation.from_matrix(initial_bases[:, :3, :3]).mean().as_rotvec(),
                    initial_bases[:, :3, 3].mean(axis=0), np.zeros(6)]

    def residual(vector: np.ndarray) -> np.ndarray:
        translation, rotation = errors(p, q, transform(vector[:6]), transform(vector[6:]))
        # Balance 0.2 m translation and 1 rad rotation; report unweighted errors separately.
        return np.r_[(translation * 5).ravel(), rotation.ravel()]

    solutions = []
    for seed in range(4):
        start = initial.copy()
        if seed:
            start[6:9] = np.random.default_rng(seed).normal(size=3) * 2
        solution = least_squares(residual, start, max_nfev=500,
                                 ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if solution.success:
            solutions.append(solution)
    if not solutions:
        raise RuntimeError("No optimizer run converged")
    best = min(solutions, key=lambda solution: solution.cost)
    singular = np.linalg.svd(best.jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-8))
    if rank != 12:
        raise ValueError(f"Insufficient motion excitation: Jacobian rank {rank}/12")
    diagnostics = {"cost": best.cost, "jacobian_rank": rank,
                   "jacobian_condition": float(singular[0] / singular[-1]),
                   "converged_multistart_costs": [solution.cost for solution in solutions]}
    return transform(best.x[:6]), transform(best.x[6:]), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True, help="HW robot2 URDF with base_link and l/r_joint1..6")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON episode list with dataset and episode_index")
    parser.add_argument("--camera_params", type=Path, required=True, help="Original viewer calibration to extend")
    parser.add_argument("--output", type=Path, required=True, help="Directory for estimates and validation report")
    parser.add_argument("--samples_per_episode", type=int, default=25, help="Evenly spaced fitting/validation samples")
    args = parser.parse_args()
    if args.samples_per_episode < 2:
        parser.error("samples_per_episode must be at least 2")
    jobs = json.loads(args.manifest.read_text())
    sources = {}
    for job in jobs:
        sources.setdefault(job["dataset"], []).append(job["episode_index"])
    # First and last selected episodes per collection fit the model. Others are held out.
    training = {(source, episode) for source, episodes in sources.items()
                for episode in (min(episodes), max(episodes))}
    chains = {arm: joint_chain(args.urdf, arm[0]) for arm in ("left", "right")}
    records = []
    for job in jobs:
        table, _ = load_episode(Path(job["dataset"]), job["episode_index"])
        indices = np.unique(np.linspace(0, len(table) - 1, min(len(table), args.samples_per_episode)).round().astype(int))
        arrays = episode_arrays(table.iloc[indices])
        if arrays["arms"] != ("left", "right"):
            raise ValueError("Expected dual-arm recordings")
        record = {"dataset": job["dataset"], "episode_index": job["episode_index"],
                  "split": "train" if (job["dataset"], job["episode_index"]) in training else "held_out",
                  "frame_indices": indices.tolist(), "arms": {}}
        for i, arm in enumerate(arrays["arms"]):
            record["arms"][arm] = (fk(chains[arm], arrays["qpos"][:, i]),
                                    rpy_to_matrix(arrays["raw_pose"][:, i]))
        records.append(record)
    report = {"urdf": str(args.urdf.resolve()), "manifest": str(args.manifest.resolve()),
              "equation": "world_T_link6(q) @ link6_T_flange = world_T_recorded_base @ recorded_T_flange",
              "translation_unit": "m", "assumptions": ["URDF geometry and joint angle conventions are applicable",
              "recorded XYZ is mm and RPY is extrinsic xyz in degrees", "constant bases and flange offsets"],
              "arms": {}, "episodes": []}
    fitted = {}
    for arm in chains:
        train_p, train_q = [np.concatenate([r["arms"][arm][i] for r in records if r["split"] == "train"]) for i in (0, 1)]
        base, tip, diagnostics = fit(train_p, train_q)
        fitted[arm] = (base, tip)
        report["arms"][arm] = {"world_T_recorded_base": base.tolist(), "link6_T_recorded_flange": tip.tolist(),
                                "fit": diagnostics, "train": stats(train_p, train_q, base, tip)}
        for split in ["held_out"]:
            selected = [r for r in records if r["split"] == split]
            if selected:
                p, q = [np.concatenate([r["arms"][arm][i] for r in selected]) for i in (0, 1)]
                report["arms"][arm][split] = stats(p, q, base, tip)
    for record in records:
        result = {key: value for key, value in record.items() if key != "arms"}
        result["errors"] = {arm: stats(*record["arms"][arm], *fitted[arm]) for arm in chains}
        report["episodes"].append(result)
    relative = np.linalg.inv(fitted["left"][0]) @ fitted["right"][0]
    report["left_recorded_base_T_right_recorded_base"] = relative.tolist()
    params = load_camera_params(args.camera_params)
    if not np.allclose(params.get("LEFT_BASE_IN_ROBOTBASE", np.eye(4)), np.eye(4)):
        raise ValueError("Output uses left recorded base as reference; input left base must be identity")
    params["LEFT_BASE_IN_ROBOTBASE"] = np.eye(4).tolist()
    params["RIGHT_BASE_IN_ROBOTBASE"] = relative.tolist()
    params["ROBOT_BASE_REFERENCE"] = "left_recorded_arm_base"
    params["BASE_ALIGNMENT_STATUS"] = "estimated_from_urdf_and_recorded_poses; see base_estimate.json"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "base_estimate.json").write_text(json.dumps(report, indent=2))
    (args.output / "estimated_camera_params.json").write_text(json.dumps(params, indent=2))
    print("left recorded base <- right recorded base:\n", np.round(relative, 6))
    for arm in chains:
        print(arm, "held_out", report["arms"][arm].get("held_out"), flush=True)


if __name__ == "__main__":
    main()
