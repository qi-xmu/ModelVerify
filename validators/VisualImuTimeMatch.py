"""验证实时数据的相机和IMU数据是否匹配

读取 NaVIODB 数据库，使用 match21 分析相机姿态和 IMU 姿态的时间对齐情况。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from base.calibration.time import _get_angvels, match21
from base.datatype import PosesData
from base.interpolate import get_time_series
from base.types.navio_db import NaVIODB

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["SimSun", "Songti SC", "STSong"]
plt.rcParams["axes.unicode_minus"] = False


def _plot_zoom(
    cs1: PosesData,
    cs2: PosesData,
    t_offset_us: int,
    time_range: tuple[float, float],
    resolution: float,
    zoom_start_s: float,
    zoom_duration_s: float,
    save_path: Path,
    label1: str = "相机",
    label2: str = "IMU",
):
    """绘制局部放大的对齐前后对比图。"""
    rate = min(cs1.rate, cs2.rate)
    resolution = min(resolution, rate)
    t_new_us = get_time_series([cs1.t_us, cs2.t_us], *time_range, rate=rate)

    cs1_interp = cs1.interpolate(t_new_us)
    cs2_interp = cs2.interpolate(t_new_us)

    seq1, t1 = _get_angvels(
        cs1_interp.t_us, cs1_interp.rots, step=int(rate / resolution)
    )
    seq2, t2 = _get_angvels(
        cs2_interp.t_us, cs2_interp.rots, step=int(rate / resolution)
    )

    t1_arr = np.array(t1) / 1e6
    t2_arr = np.array(t2) / 1e6
    t2_fix_arr = t2_arr + t_offset_us / 1e6

    zoom_end = zoom_start_s + zoom_duration_s

    fig, (ax_before, ax_after) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # 对齐前
    ax_before.plot(t1_arr, seq1, linewidth=1.0, label=label1, alpha=0.85)
    ax_before.plot(t2_arr, seq2, linewidth=1.0, label=label2, alpha=0.85)
    ax_before.set_xlim(zoom_start_s, zoom_end)
    ax_before.legend(fontsize=10)
    ax_before.set_ylabel("角速度（rad/s）", fontsize=11)
    ax_before.set_title(
        f"对齐前（{zoom_start_s:.0f}s - {zoom_end:.0f}s 局部放大）", fontsize=12
    )
    ax_before.grid(True, alpha=0.3)

    # 对齐后
    ax_after.plot(t1_arr, seq1, linewidth=1.0, label=label1, alpha=0.85)
    ax_after.plot(t2_fix_arr, seq2, linewidth=1.0, label=label2, alpha=0.85)
    ax_after.set_xlim(zoom_start_s, zoom_end)
    ax_after.legend(fontsize=10)
    ax_after.set_xlabel("时间（秒）", fontsize=11)
    ax_after.set_ylabel("角速度（rad/s）", fontsize=11)
    ax_after.set_title(f"对齐后（偏移 {t_offset_us / 1e3:.2f} ms）", fontsize=12)
    ax_after.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"局部放大图已保存至 {save_path}")


def check_camera_imu_match(
    db_path: Path,
    output_dir: Path | None = None,
    time_range: tuple[float, float] = (0, 30),
    zoom_start_s: float = 5.0,
    zoom_duration_s: float = 5.0,
):
    """检查相机和IMU数据的时间对齐情况。"""
    name = db_path.stem
    if output_dir is None:
        output_dir = Path("results") / name
    output_dir.mkdir(parents=True, exist_ok=True)

    with NaVIODB(db_path) as cdb:
        cam_pose = cdb.get_pose_data()
        assert cam_pose is not None, f"[{name}] 无相机姿态数据"

        imu_pose = cdb.get_imu_data().to_poses()

        cam_pose.t_us -= cam_pose.t_us[0]
        imu_pose.t_us -= imu_pose.t_us[0]

        print(f"[{name}] 相机帧数: {len(cam_pose)}, IMU帧数: {len(imu_pose)}")

        t_offset_us = match21(cam_pose, imu_pose, resolution=200, time_range=time_range)

        print(
            f"[{name}] 相机 → IMU 时间偏移: {t_offset_us / 1e3:.2f} ms ({t_offset_us / 1e6:.4f} s)"
        )

        _plot_zoom(
            cam_pose,
            imu_pose,
            t_offset_us,
            time_range=time_range,
            resolution=100,
            zoom_start_s=zoom_start_s,
            zoom_duration_s=zoom_duration_s,
            save_path=output_dir / "cam_imu_time_match.png",
        )

    return t_offset_us


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "用法: uv run python validators/VisualImuTimeMatch.py <db_path> [zoom_start_s] [zoom_duration_s]"
        )
        sys.exit(1)

    db_path = Path(sys.argv[1])
    kwargs = {}
    if len(sys.argv) >= 3:
        kwargs["zoom_start_s"] = float(sys.argv[2])
    if len(sys.argv) >= 4:
        kwargs["zoom_duration_s"] = float(sys.argv[3])

    check_camera_imu_match(db_path, **kwargs)
