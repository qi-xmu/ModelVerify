#!/usr/bin/env python3
# 使用公式 x = A * exp(B * y)
# 对 sub_points_X/Y/Z轴.pkl 中的点进行 RANSAC 拟合，分别绘制在三个子图中

import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor

from base.obj import Obj

# 设置中文字体为宋体
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["SimSun", "Songti SC", "STSong"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 12

AXES = ["X轴", "Y轴", "Z轴"]
COLORS = {"X轴": "r", "Y轴": "g", "Z轴": "b"}
TARGET = "Huawei"


def ransac_exp_fit(x_pos, y_pos):
    """对 x = A*exp(B*y) 做对数变换后 RANSAC 线性拟合，返回 (A, B, inlier_mask)"""
    log_y = np.log(y_pos)
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(x_pos.reshape(-1, 1), log_y)
    B = ransac.estimator_.coef_[0]  # pyright: ignore[reportAttributeAccessIssue]
    A = np.exp(ransac.estimator_.intercept_)  # pyright: ignore[reportAttributeAccessIssue]
    return A, B, ransac.inlier_mask_


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, axis in enumerate(AXES):
        data_path = Path(f"fit/data/{TARGET}/sub_points_{axis}.pkl")
        data = Obj.load(data_path)
        # if i == 0 or i == 1:
        #     ext_path = Path(f"fit/data/sub_points_{AXES[1 - i]}.pkl")

        #     ext_data = Obj.load(ext_path)
        #     # 合并到 data中
        #     data = np.vstack((data, ext_data))

        x_data = data[:, 0]
        y_data = data[:, 1]

        # 过滤 x > 0 的点（取对数要求正值）
        # mask = x_data > 0
        # x_pos = x_data[mask]
        # y_pos = y_data[mask]
        x_pos = x_data
        y_pos = y_data

        A, B, inlier_mask = ransac_exp_fit(y_pos, x_pos)

        print(
            f"[{axis}] A = {A:.6f}, B = {B:.6f}, 内点: {inlier_mask.sum()}, 离群点: {(~inlier_mask).sum()}"
        )

        color = COLORS[axis]
        ax = axes[i]

        # 原始散点（内点）
        ax.scatter(
            y_data[inlier_mask],
            x_data[inlier_mask],
            c=color,
            s=12,
            alpha=0.5,
            label="内点 (inlier)",
        )
        ax.scatter(
            y_data[~inlier_mask],
            x_data[~inlier_mask],
            c="black",
            s=12,
            alpha=0.5,
            marker="x",
            label="离群点 (outlier)",
        )

        # 拟合曲线
        y_fit = np.linspace(y_data.min(), y_data.max(), 300)
        x_fit = A * np.exp(B * y_fit)
        ax.plot(
            y_fit,
            x_fit,
            color=color,
            linewidth=2,
            label=f"拟合: {A:.4f}·exp({B:.4f}·x)",
        )

        ax.set_xlabel("不确定性")
        ax.set_ylabel("误差绝对值")
        ax.set_title(f"{axis} RANSAC 指数拟合")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("nihe_exp_ransac.png", dpi=300)
    print("图像已保存: nihe_exp_ransac.png")
    plt.show()


if __name__ == "__main__":
    main()
