"""打印所有上界点并拟合绘图

基于 VisualError 的数据处理流程，汇总所有数据的上界点，使用 RANSAC 拟合并绘图显示。
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import RANSACRegressor
from VisualError import (
    MatchList,
    extract_upper_bound,
    fit_boundary_inv,
)

from base.obj import Obj

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["SimSun", "Songti SC", "STSong"]
plt.rcParams["axes.unicode_minus"] = False


def print_upper_boundary(label: str):
    matchs = MatchList(
        f"/Users/qi/Resources/实验数据整理/点云视觉数据/mapping_{label}.csv"
    )

    all_error_list = []
    all_ac_list = []
    for cdb, gt_pose in matchs.iter_object():
        output_dir = Path("results") / cdb.name
        result_path = output_dir / "visual_error_dict.pkl"
        if result_path.exists():
            print(f"[{cdb.name}] 加载结果")
            result = Obj.load(result_path)
        else:
            continue

        all_error_list.extend(result["error_list"])
        all_ac_list.extend(result["ac_list"])

    all_err_arr = np.array(all_error_list)
    all_err_arr -= np.mean(all_err_arr, axis=0)
    all_err_arr = np.abs(all_err_arr)

    all_ac_arr = np.array(all_ac_list)
    merged_ac = np.concatenate([all_ac_arr, all_ac_arr, all_ac_arr])
    merged_err = np.concatenate(
        [all_err_arr[:, 0], all_err_arr[:, 1], all_err_arr[:, 2]]
    )

    upper_boundary = extract_upper_bound(merged_ac, merged_err)
    if len(upper_boundary) == 0:
        print("无上界点")
        return

    print(f"\n上界点（共 {len(upper_boundary)} 个）:")
    print(f"{'置信度':>10s}  {'误差上界(m)':>12s}")
    print("-" * 25)
    for x, y in upper_boundary:
        print(f"{x:10.6f}  {y:12.6f}")

    from scipy.optimize import curve_fit

    # 均匀插值
    x_ub = upper_boundary[:, 0]
    y_ub = upper_boundary[:, 1]
    x_uniform = np.linspace(x_ub.min(), x_ub.max(), len(x_ub))
    y_uniform = np.interp(x_uniform, x_ub, y_ub)

    mask = y_uniform > 0
    x_fit = x_uniform[mask]
    y_fit = y_uniform[mask]

    def model_frc(x, A, B, C):
        return A / (x + B) + C

    p_frc, _ = curve_fit(model_frc, x_fit, y_fit, p0=[20, 3, 0], maxfev=10000)
    Af, Bf, Cf = p_frc
    print(f"分式拟合:  y = {Af:.4f}/(x + {Bf:.4f}) + {Cf:.4f}")

    # 绘制对比
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(merged_ac, merged_err, s=2, alpha=0.3, color="lightgray", label="数据点")
    ax.scatter(
        upper_boundary[:, 0], upper_boundary[:, 1],
        s=20, c="black", marker="^", alpha=0.8,
        zorder=5, label=f"上界点（{len(upper_boundary)}）",
    )

    x_curve = np.linspace(upper_boundary[:, 0].min(), upper_boundary[:, 0].max(), 300)
    ax.plot(x_curve, model_frc(x_curve, Af, Bf, Cf), "r-", linewidth=2,
            label=f"分式: {Af:.4f}/(x + {Bf:.4f}) + {Cf:.4f})")

    ax.axhline(2, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("置信度", fontsize=11)
    ax.set_ylabel("误差 (m)", fontsize=11)
    ax.set_title("上界点 A/(x+B)+C 拟合（误差<2）", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, np.max(upper_boundary[:, 1]) * 1.1)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) >= 2 else "sm"
    print_upper_boundary(label)
