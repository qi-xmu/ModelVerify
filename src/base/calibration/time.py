import numpy as np
from numpy._typing import NDArray
from scipy.spatial.transform import Rotation

from base.datatype import PosesData
from base.interpolate import get_time_series

from pathlib import Path
import matplotlib.pyplot as plt

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
    resolution=200,
    save_path: Path | None = None,
    show=False,
) -> int:
    # 分辨率不能大于时间序列的采样率，否则没有插值的意义
    rate = min(cs1.rate, cs2.rate)
    resolution = min(resolution, rate)
    print(f"Rate1:{cs1.rate}, Rate2: {cs2.rate}, reso: {resolution}")

    t_new_us = get_time_series([cs1.t_us, cs2.t_us], *time_range, rate=rate)
    cs1 = cs1.interpolate(t_new_us)
    cs2 = cs2.interpolate(t_new_us)
    print(f"使用时间范围：{(cs1.t_us[-1] - cs1.t_us[0]) / 1e6} 秒, 数量 {len(cs1)}")

    seq1, t1 = _get_angvels(cs1.t_us, cs1.rots, step=int(rate / resolution))
    seq2, t2 = _get_angvels(cs2.t_us, cs2.rots, step=int(rate / resolution))
    t_new_us = t1

    corr = np.correlate(seq1, seq2, mode="full")
    lag_arr = np.arange(-len(seq2) + 1, len(seq1))
    lag = lag_arr[np.argmax(corr)]
    t21_us = lag * (t_new_us[1] - t_new_us[0])
    print("Ground time gap: ", t21_us / 1e6)

    if save_path or show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        t_new_us = t_new_us - t_new_us[0]
        ax.set_title("Best Time Offset: {:.3f}s".format(t21_us * 1e-6))
        ax.plot(t_new_us / 1e6, seq1, label="Seq1", alpha=0.5)
        ax.plot((t_new_us + t21_us) / 1e6, seq2, label="Seq2", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Velocity (rad/s)")
        ax.legend()
        ax.grid()
        if save_path:
            fig.savefig(save_path)
        if not show:
            plt.close(fig)

    return t21_us
