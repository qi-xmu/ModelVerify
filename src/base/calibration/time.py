from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray
from scipy.spatial.transform import Rotation

from base.datatype import PosesData
from base.interpolate import get_time_series

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["SimSun", "Songti SC", "STSong"]
plt.rcParams["axes.unicode_minus"] = False


def _get_angvels(t_us: NDArray, rots: Rotation, step: int = 1):
    """获取角速度列表"""
    n = len(rots)
    step = max(int(step), 1)
    assert n >= 2, "At least two rotations are required"

    As: list = []
    Ts = []
    for i in range(0, n - step, step):
        drot = rots[i].inv() * rots[i + step]
        angle = float(np.linalg.norm(drot.as_rotvec()))
        dt_s = (t_us[i + step] - t_us[i]) * 1e-6
        assert dt_s > 0, "Time difference must be positive"
        ang_vel = angle / dt_s
        As.append(ang_vel)
        Ts.append(t_us[i])
    return As, Ts


def match21(
    cs1: PosesData,
    cs2: PosesData,
    *,
    time_range=(0, 50),
    resolution=100,
    check_plot_path: str | Path | None = None,
) -> int:
    # 分辨率不能大于时间序列的采样率，否则没有插值的意义
    rate = min(cs1.rate, cs2.rate)
    resolution = min(resolution, rate)
    print(f"> 频率：Rate1:{cs1.rate}, Rate2: {cs2.rate}, reso: {resolution}")

    t_new_us = get_time_series([cs1.t_us, cs2.t_us], *time_range, rate=rate)
    cs1 = cs1.interpolate(t_new_us)
    cs2 = cs2.interpolate(t_new_us)
    print(f"> 时间范围：{(cs1.t_us[-1] - cs1.t_us[0]) / 1e6} 秒, 数量 {len(cs1)}")

    seq1, t1 = _get_angvels(cs1.t_us, cs1.rots, step=int(rate / resolution))
    seq2, t2 = _get_angvels(cs2.t_us, cs2.rots, step=int(rate / resolution))
    t_new_us = t1

    corr = np.correlate(seq1, seq2, mode="full")
    lag_arr = np.arange(-len(seq2) + 1, len(seq1))
    lag = lag_arr[np.argmax(corr)]
    t21_us = lag * (t_new_us[1] - t_new_us[0])
    print("> 真值（+）时间偏移: ", t21_us / 1e6, "s")

    if check_plot_path is not None:
        t1_arr = np.array(t1)
        t2_arr = np.array(t2)
        t2_fix_arr = t2_arr + t21_us
        # 对齐前
        fig, (ax_before, ax_after) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax_before.plot(t1_arr / 1e6, seq1, label="序列1")
        ax_before.plot(t2_arr / 1e6, seq2, label="序列2")
        ax_before.legend()
        ax_before.set_ylabel("角速度（rad/s）")
        ax_before.set_title("对齐前")
        # 对齐后
        ax_after.plot(t1_arr / 1e6, seq1, label="序列1")
        ax_after.plot(t2_fix_arr / 1e6, seq2, label="序列2")
        ax_after.legend()
        ax_after.set_xlabel("时间（秒）")
        ax_after.set_ylabel("角速度（rad/s）")
        offset_s = t21_us / 1e6
        ax_after.set_title(f"对齐后（偏移: {offset_s:+.3f} s）")
        fig.tight_layout()
        plt.savefig(check_plot_path, dpi=300)
        plt.close(fig)
        print(f"> Check Plot saved to {check_plot_path}")

    return t21_us
