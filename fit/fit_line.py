#!/usr/bin/env python3
# 使用公式 y = A*x^2 + B*x + C
# 对 sub_points_Z轴.pkl 中的点进行 RANSAC 二次拟合

import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# 设置中文字体为宋体
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["SimSun", "Songti SC", "STSong"]
mpl.rcParams["axes.unicode_minus"] = False


def main():
    data_path = Path("sub_points_X轴.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    x_data = data[:, 1]
    y_data = data[:, 0]

    # RANSAC 二次多项式拟合
    ransac = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=True),
        RANSACRegressor(random_state=42),
    )
    ransac.fit(x_data.reshape(-1, 1), y_data)

    reg = ransac.named_steps["ransacregressor"]
    coeffs = reg.estimator_.coef_  # [C, B, A] (include_bias=True)
    A, B, C = coeffs[2], coeffs[1], coeffs[0]

    inlier_mask = reg.inlier_mask_
    outlier_mask = ~inlier_mask

    print(f"A = {A:.6f}, B = {B:.6f}, C = {C:.6f}")
    print(f"内点数量: {inlier_mask.sum()}, 离群点数量: {outlier_mask.sum()}")

    # 绘图
    x_fit = np.linspace(x_data.min(), x_data.max(), 300)
    y_fit = A * x_fit**2 + B * x_fit + C

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        x_data[inlier_mask],
        y_data[inlier_mask],
        c="b",
        s=10,
        label="内点 (inlier)",
        alpha=0.6,
    )
    ax.scatter(
        x_data[outlier_mask],
        y_data[outlier_mask],
        c="r",
        s=10,
        label="离群点 (outlier)",
        alpha=0.6,
    )
    ax.plot(
        x_fit,
        y_fit,
        "g-",
        linewidth=2,
        label=f"拟合: y = {A:.4f}·x² + {B:.4f}·x + {C:.4f}",
    )
    ax.set_xlabel("不确定性")
    ax.set_ylabel("误差绝对值")
    ax.set_title("RANSAC 二次拟合: y = A·x² + B·x + C")
    ax.legend()
    # ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
