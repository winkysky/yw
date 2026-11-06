"""Interactive and headless pose plots; no robot meshes or image assets needed."""

from collections.abc import Callable
from pathlib import Path

import numpy as np


def project_points(points: np.ndarray, camera_pose: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Project base-frame points; points behind the camera become NaN."""
    camera_from_base = np.linalg.inv(camera_pose)
    camera_points = np.asarray(points) @ camera_from_base[:3, :3].T + camera_from_base[:3, 3]
    pixels = np.full((len(camera_points), 2), np.nan)
    visible = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-6)
    homogeneous = camera_points[visible] @ intrinsic.T
    pixels[visible] = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels


def draw_tcp_projection(
    image: np.ndarray,
    tcp: np.ndarray,
    camera_pose: np.ndarray,
    intrinsic: np.ndarray,
    frame: int,
    future_frames: int = 30,
    future_step: int = 1,
    axis_length: float = 0.05,
) -> np.ndarray:
    """Overlay each TCP's XYZ axes and future positions in the CURRENT camera."""
    import cv2

    output = image.copy()
    height, width = output.shape[:2]

    def pixel(point):
        return tuple(np.rint(np.clip(point, -1e7, 1e7)).astype(int))

    def line(start, end, color):
        if np.isfinite([start, end]).all():
            visible, p0, p1 = cv2.clipLine((0, 0, width, height), pixel(start), pixel(end))
            if visible:
                cv2.line(output, p0, p1, color, 2, cv2.LINE_AA)

    for arm in range(tcp.shape[1]):
        current = tcp[frame, arm]
        origin = current[:3, 3]
        axes = np.vstack([origin, origin + axis_length * current[:3, :3].T])
        uv = project_points(axes, camera_pose, intrinsic)
        for endpoint, color in zip(uv[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            line(uv[0], endpoint, color)
        color = (0, 200, 255) if arm == 0 else (255, 200, 0)
        if np.isfinite(uv[0]).all() and 0 <= uv[0, 0] < width and 0 <= uv[0, 1] < height:
            cv2.circle(output, pixel(uv[0]), 4, color, -1)
            cv2.putText(output, "L TCP" if arm == 0 else "R TCP", pixel(uv[0] + [5, -5]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        indices = np.r_[frame, np.arange(frame + 1, min(len(tcp), frame + future_frames + 1), future_step)]
        future_uv = project_points(tcp[indices, arm, :3, 3], camera_pose, intrinsic)
        for start, end in zip(future_uv[:-1], future_uv[1:]):
            line(start, end, color)
    return output


def plot_poses(
    poses: dict[str, np.ndarray],
    timestamps: np.ndarray,
    frame: int = 0,
    axis_length: float = 0.05,
    save_path: Path | None = None,
    no_show: bool = False,
    image_loaders: dict[str, Callable] | None = None,
) -> None:
    """Plot trajectories and current axes; arrow keys/slider navigate samples."""
    import matplotlib

    if no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    timestamps = np.asarray(timestamps).reshape(-1)
    count = len(timestamps)
    if not 0 <= frame < count:
        raise ValueError(f"frame={frame} outside [0, {count})")
    if axis_length <= 0:
        raise ValueError("axis_length must be positive")
    if no_show and save_path is None:
        raise ValueError("--no_show requires --save_path")
    for name, sequence in poses.items():
        if sequence.shape != (count, 4, 4) or not np.isfinite(sequence).all():
            raise ValueError(f"Invalid {name} pose shape/values: {sequence.shape}")

    image_loaders = image_loaders or {}
    rows = 2 if len(image_loaders) >= 2 else 1
    columns = 2 if image_loaders else 1
    fig = plt.figure(figsize=(14, 10) if rows == 2 else ((13, 6) if image_loaders else (8, 7)))
    grid = fig.add_gridspec(rows, columns)
    ax = fig.add_subplot(grid[:, 0] if len(image_loaders) == 2 else grid[0, 0], projection="3d")
    image_axes = {}
    for index, name in enumerate(image_loaders):
        row, column = (index, 1) if len(image_loaders) == 2 else divmod(index + 1, columns)
        image_axes[name] = fig.add_subplot(grid[row, column])
    fig.subplots_adjust(bottom=0.12, top=0.92, hspace=0.30, wspace=0.15)
    points = np.concatenate([value[:, :3, 3] for value in poses.values()])
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) / 2 + axis_length, 0.1)
    colors = {"flange": "gray", "tcp": "tab:orange", "wrist_camera": "tab:purple"}

    def draw(index: int) -> None:
        ax.clear()
        for name, sequence in poses.items():
            xyz = sequence[:, :3, 3]
            color = colors.get(name.removeprefix("left_").removeprefix("right_"))
            ax.plot(*xyz.T, color=color, alpha=0.65, label=name,
                    linestyle="--" if name.startswith("right_") else "-")
            pose = sequence[index]
            origin = pose[:3, 3]
            ax.scatter(*origin, color=color, s=25)
            ax.text(*origin, name, fontsize=8)
            for column, color in enumerate(("r", "g", "b")):
                end = origin + axis_length * pose[:3, column]
                ax.plot(*np.stack([origin, end]).T, color=color)
        ax.set(xlim=(center[0] - radius, center[0] + radius),
               ylim=(center[1] - radius, center[1] + radius),
               zlim=(center[2] - radius, center[2] + radius),
               xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"Frame {index}/{count - 1}, t={timestamps[index]:.3f}s\n"
                     "Recorded robot base; axes: X red / Y green / Z blue")
        ax.legend(loc="upper left", fontsize=8)
        for name, loader in image_loaders.items():
            rgb, title = loader(index)
            image_ax = image_axes[name]
            image_ax.clear()
            image_ax.imshow(rgb)
            image_ax.set_title(title)
            image_ax.axis("off")
        fig.canvas.draw_idle()

    draw(frame)
    if not no_show and count > 1:
        slider_ax = fig.add_axes([0.2, 0.06, 0.6, 0.03])
        slider = Slider(slider_ax, "Frame", 0, count - 1, valinit=frame, valstep=1)
        slider.on_changed(lambda value: draw(int(value)))

        def on_key(event) -> None:
            if event.key in ("right", "d"):
                slider.set_val(min(int(slider.val) + 1, count - 1))
            elif event.key in ("left", "a"):
                slider.set_val(max(int(slider.val) - 1, 0))

        fig.canvas.mpl_connect("key_press_event", on_key)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        print(f"Saved plot: {save_path}")
    if not no_show:
        plt.show()
    plt.close(fig)
