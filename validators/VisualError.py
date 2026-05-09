"""
说明文件

用来分析视觉相关因子与误差之前的相应关系


TODO List
1. 完成两个序列的时间对齐
2. 创建一个相机数据队列，每一次取出1s内的数据
3. 计算相应的误差，和不确定因子。
4. 绘制散点图，查看结果。
"""

from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from base.calibration import time
from base.datatype import GroundTruthData, Pose, PosesData
from base.obj import Obj
from base.rtab import RTABData
from base.types.navio_db import NaVIODB

# from base.draw.Poses import draw_trajectory_2d

# 设置中文绘图字体
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["SimSun", "Songti SC", "STSong"]
plt.rcParams["axes.unicode_minus"] = False


class MatchList:
    def __init__(self, mapping_file: str | Path):
        df = pd.read_csv(mapping_file)
        df = df[~df["gt"].str.startswith("#")]
        self.mapping = df.to_numpy()
        print(f"[{mapping_file}] {len(self.mapping)} groups")
        print(df)

    def __iter__(self) -> Iterator[tuple[Path, Path]]:
        for row in self.mapping:
            yield (Path(row[0]), Path(row[1]))

    def __getitem__(self, key: int):
        if key >= len(self.mapping):
            raise IndexError(f"Index {key} is out of range")
        row = self.mapping[key]
        return (Path(row[0]), Path(row[1]))

    def __len__(self):
        return len(self.mapping)

    def __get_gpose(self, path: Path):
        file_path = Path(path)
        csv_path = file_path.parent / f"{file_path.stem}.csv"
        if not csv_path.exists():
            rtab_data = RTABData(file_path)
            rtab_data.transform_local()
            rtab_data.transform_global()
            rtab_data.save_csv(csv_path)
        return GroundTruthData.from_csv(csv_path)

    def __row_to_obj(self, row: NDArray[np.str_]) -> tuple[NaVIODB, GroundTruthData]:
        gdb_path = Path(row[0])
        cdb_path = Path(row[1])
        name = cdb_path.stem
        output_dir = Path("results") / name
        output_dir.mkdir(parents=True, exist_ok=True)

        cdb = NaVIODB(cdb_path)
        # 是否复用 gt_pose
        gt_pose_path = output_dir / "gt_pose.pkl"
        if gt_pose_path.exists():
            print(f"[{name}] 加载匹配结果")
            gt_pose = Obj.load(gt_pose_path)
        else:
            gt_pose = self.__get_gpose(gdb_path)

            pose = cdb.get_pose_data()
            assert pose is not None
            imu_pose = cdb.get_imu_data().to_poses()
            imu_pose.t_us -= imu_pose.t_us[0]
            gt_pose.t_us -= gt_pose.t_us[0]

            time_gc = time.match21(
                imu_pose, gt_pose, check_plot_path=output_dir / "time_fix.png"
            )
            gt_pose.t_us += time_gc + pose.t_us[0]

            Obj.save(gt_pose, gt_pose_path)
        return (cdb, gt_pose)

    def iter_object(self) -> Iterator[tuple[NaVIODB, GroundTruthData]]:
        for row in self.mapping:
            yield self.__row_to_obj(row)


class VisualErrorResult:
    pass


def plot_results(
    output_dir: Path,
    error_list: list[NDArray],
    ac_list: list[float],
    error_idx_list: list[int],
    gap_idx_list: list[int],
    dp_list: list[float],
    name: str,
    sensor_pose: PosesData,
    match_pose: PosesData,
    show: bool = False,
):
    err_arr = np.array(error_list)
    assert len(err_arr) > 0, f"[{name}] 没有误差数据"

    plot_error_distribution(err_arr, output_dir / "error_distribution.png", show)

    positions = np.array([sensor_pose.get_pose(i).p for i in range(len(sensor_pose))])
    gt_positions = np.array([match_pose.get_pose(i).p for i in range(len(match_pose))])
    err_norms = np.linalg.norm(err_arr, axis=1)

    if len(ac_list) > 0 and len(ac_list) == len(error_idx_list):
        plot_trajectory_comparison(
            positions,
            gt_positions,
            err_norms,
            ac_list,
            error_idx_list,
            gap_idx_list,
            output_dir / f"trajectory_comparison_{label}.png",
            show,
        )

    if len(ac_list) > 0 and len(ac_list) == len(err_arr):
        plot_error_vs_confidence(
            err_arr, ac_list, output_dir / "error_vs_confidence.png", show
        )
        plot_error_merged_vs_confidence(
            err_arr, ac_list, output_dir / "error_merged_vs_confidence.png", show
        )

    print(f"[{name}] 图片已保存至 {output_dir}")


def process_(sensor_db_obj: NaVIODB, gt_pose: GroundTruthData):
    update_cam_interval = 1e6 / 10  # 10Hz

    # 找到两个序列重叠的时间段
    sensor_pose = sensor_db_obj.get_pose_data()
    timestamps = sensor_db_obj.get_cam_timestamps()
    assert sensor_pose is not None
    # 使用 sensor_pose 的时间戳
    ref_t_mask = (sensor_pose.t_us < gt_pose.t_us[-1]) & (
        sensor_pose.t_us > gt_pose.t_us[0]
    )
    ref_t_us = sensor_pose.t_us[ref_t_mask]
    match_pose = gt_pose.interpolate(ref_t_us)
    sensor_pose = sensor_pose[ref_t_mask]

    assert len(sensor_pose) == len(match_pose), (
        f"Sensor pose length {len(sensor_pose)} does not match match pose length {len(match_pose)}"
    )

    timestamps_mask = np.array(timestamps, dtype=int)[ref_t_mask]
    frames = sensor_db_obj.read_all_pointcloud_frames()

    buffer: list[tuple[int, Pose]] = []
    error_list = []
    error_angle_list = []
    error_idx_list = []
    gap_idx_list = []
    dp_list = []
    ac_list = []

    for idx in range(0, len(sensor_pose)):
        if len(buffer) > 0:
            dpose = buffer[-1][1].between(sensor_pose.get_pose(idx))
            dp = np.linalg.norm(dpose.p)
            dp_list.append(dp)

            if dp > 0.18:
                gap_idx_list.append(idx)
                buffer.clear()
                print("Gap = ", idx)

            # 正常
            if len(buffer) > 0:
                cur_idx, cur_pose = buffer[-1]
                pre_t_us = cur_pose.t_us - 1e6

                # 在 buffer 中寻找大于 pre_t_us 的第一个状态
                for i, p in buffer:
                    if p.t_us >= pre_t_us:
                        pre_idx = i
                        pre_pose = p
                        break
                else:
                    print("Not Found.")
                    continue

                assert isinstance(pre_pose, Pose)
                assert isinstance(cur_pose, Pose)
                if cur_pose.t_us - pre_pose.t_us >= update_cam_interval:
                    sensor_rel_motion = pre_pose.between(cur_pose)
                    real_rel_motion = match_pose.get_between(pre_idx, cur_idx)
                    error = sensor_rel_motion.p - real_rel_motion.p
                    error_list.append(error)

                    error_angle_list.append(
                        (sensor_rel_motion.rot * real_rel_motion.rot.inv()).as_euler(
                            "xyx"
                        )
                    )
                    error_idx_list.append(cur_idx)

                    mean_confidence = 0
                    # 在 frames里面寻找 timestamps_mask[pre_idx], timestamps_mask[cur_idx]区间内的 confidence
                    cnt = 1
                    for pf in frames:
                        if (
                            timestamps_mask[pre_idx]
                            <= pf.t_us * 1000
                            <= timestamps_mask[cur_idx]
                        ):
                            mean_confidence += pf.confidence
                            cnt += 1
                    mean_confidence /= cnt
                    ac_list.append(mean_confidence)

        buffer.append((idx, sensor_pose.get_pose(idx)))

    return {
        "error_list": error_list,
        "error_angle_list": error_angle_list,
        "error_idx_list": error_idx_list,
        "ac_list": ac_list,
        "gap_idx_list": gap_idx_list,
        "dp_list": dp_list,
        "name": cdb.name,
        "sensor_pose": sensor_pose,
        "match_pose": match_pose,
    }


def plot_error_distribution(err_arr: NDArray, save_path: Path, show: bool = False):
    """绘制三轴误差分布散点图"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["X", "Y", "Z"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.scatter(range(len(err_arr)), err_arr[:, i], s=8, alpha=0.6, color=color)
        mean_val = np.mean(err_arr[:, i])
        std_val = np.std(err_arr[:, i])
        ax.axhline(y=mean_val, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"{label}轴误差 (m)", fontsize=11)
        ax.set_title(
            f"{label}轴误差: μ={mean_val:.4f}m, σ={std_val:.4f}m",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("样本索引", fontsize=11)
    fig.suptitle("三轴误差分布", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_trajectory_comparison(
    positions: NDArray,
    gt_positions: NDArray,
    err_norms: NDArray,
    ac_list: list[float],
    error_idx_list: list[int],
    gap_idx_list: list[int],
    save_path: Path,
    show: bool = False,
    right_label: str = "点云置信度",
    right_cmap: str = "YlGnBu_r",
):
    """绘制传感器轨迹对比图，左子图按误差模长着色，右子图按置信度着色"""
    fig, (ax_err, ax_ac) = plt.subplots(1, 2, figsize=(18, 8))

    # ---- 左子图：按误差模长着色 ----
    ax_err.plot(
        gt_positions[:, 0],
        gt_positions[:, 1],
        color="steelblue",
        linewidth=1.2,
        alpha=0.6,
        zorder=0,
        label="真值轨迹",
    )
    ax_err.plot(
        positions[:, 0],
        positions[:, 1],
        color="lightgray",
        linewidth=1,
        alpha=0.5,
        zorder=1,
        label="传感器轨迹",
    )
    scatter_err = ax_err.scatter(
        positions[error_idx_list, 0],
        positions[error_idx_list, 1],
        c=err_norms,
        cmap="YlOrRd",
        s=20,
        alpha=0.8,
        zorder=2,
    )
    ax_err.scatter(
        positions[0, 0],
        positions[0, 1],
        c="green",
        marker="o",
        s=60,
        zorder=3,
        label="起点",
    )
    ax_err.scatter(
        positions[-1, 0],
        positions[-1, 1],
        c="blue",
        marker="X",
        s=60,
        zorder=3,
        label="终点",
    )
    if len(gap_idx_list) > 0:
        ax_err.scatter(
            positions[gap_idx_list, 0],
            positions[gap_idx_list, 1],
            c="red",
            marker="^",
            s=40,
            zorder=4,
            alpha=0.5,
            label="断裂点",
            edgecolors="black",
            linewidths=0.8,
        )
    cbar_err = plt.colorbar(scatter_err, ax=ax_err)
    cbar_err.set_label("误差模长 (m)", fontsize=11)
    ax_err.set_xlabel("X (m)", fontsize=11)
    ax_err.set_ylabel("Y (m)", fontsize=11)
    ax_err.set_title("传感器轨迹（按误差模长着色）", fontsize=13, fontweight="bold")
    ax_err.legend()
    ax_err.axis("equal")
    ax_err.grid(True, alpha=0.3)

    # ---- 右子图：按置信度着色 ----
    ax_ac.plot(
        positions[:, 0],
        positions[:, 1],
        color="lightgray",
        linewidth=1,
        alpha=0.5,
        zorder=1,
    )
    ac_arr = np.array(ac_list)
    scatter_ac = ax_ac.scatter(
        positions[error_idx_list, 0],
        positions[error_idx_list, 1],
        c=ac_arr,
        cmap=right_cmap,
        s=20,
        alpha=0.8,
        zorder=2,
    )
    ax_ac.scatter(
        positions[0, 0],
        positions[0, 1],
        c="green",
        marker="o",
        s=60,
        zorder=3,
        label="起点",
    )
    ax_ac.scatter(
        positions[-1, 0],
        positions[-1, 1],
        c="blue",
        marker="X",
        s=60,
        zorder=3,
        label="终点",
    )
    cbar_ac = plt.colorbar(scatter_ac, ax=ax_ac)
    cbar_ac.set_label(right_label, fontsize=11)
    ax_ac.set_xlabel("X (m)", fontsize=11)
    ax_ac.set_ylabel("Y (m)", fontsize=11)
    ax_ac.set_title(
        f"传感器轨迹（按{right_label}着色）", fontsize=13, fontweight="bold"
    )
    ax_ac.legend()
    ax_ac.axis("equal")
    ax_ac.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_error_vs_confidence(
    err_arr: NDArray,
    ac_list: list[float],
    save_path: Path,
    show: bool = False,
    x_label: str = "置信度",
):
    """绘制误差与点云置信度的相关性散点图，分三轴"""
    ac_arr = np.array(ac_list)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ["X", "Y", "Z"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.scatter(ac_arr, err_arr[:, i], s=8, alpha=0.6, color=color)

        if False:
            mean_val = np.mean(err_arr[:, i])
            ax.axhline(y=mean_val, color="black", linestyle="--", linewidth=0.8)
            ax.text(
                0.02,
                0.95,
                f"μ={mean_val:.4f}m",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )
        # ax.set_ylim(-1.05, 1.05)
        ax.set_ylim(-0.05, 2.05)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(f"{label}轴误差 (m)", fontsize=11)
        ax.set_title(f"{label}轴误差 vs {x_label}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def extract_upper_bound(
    x_arr: NDArray,
    y_arr: NDArray,
    x_bins: int = 100,
    y_bins: int = 50,
    min_points: int = 3,
):
    """提取散点图的上边界点。

    将 x 轴分为 x_bins 个区间，在每个区间内对 y 再分桶，
    找到满足 min_points 个点的子区间的最大 y 值作为上边界。

    Returns: (N, 2) array of [x_center, y_upper] points
    """
    points = np.vstack([x_arr, y_arr]).T
    x_min, x_max = np.min(x_arr), np.max(x_arr)
    x_gap = (x_max - x_min) / x_bins
    boundary_points = []

    for x in np.arange(x_min, x_max, x_gap):
        x_range = [x, x + x_gap]
        x_center = (x_range[0] + x_range[1]) / 2
        in_range = (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1])
        if not np.any(in_range):
            continue

        y_vals = points[in_range, 1]
        y_min, y_max = np.min(y_vals), np.max(y_vals)
        y_gap = (y_max - y_min) / y_bins
        if y_gap == 0:
            boundary_points.append([x_center, y_max])
            continue

        sub_points = points[in_range]
        found = False
        for y in np.linspace(y_max, y_min, y_bins + 1)[:-1]:
            y_range = (y - y_gap, y)
            mask = (sub_points[:, 1] >= y_range[0]) & (sub_points[:, 1] < y_range[1])
            if np.sum(mask) >= min_points:
                boundary_points.append([x_center, np.max(sub_points[mask, 1])])
                found = True
                break
        if not found:
            boundary_points.append([x_center, y_max])

    return np.array(boundary_points) if boundary_points else np.empty((0, 2))


def plot_error_merged_vs_confidence(
    err_arr: NDArray,
    ac_list: list[float],
    save_path: Path,
    show: bool = False,
    upper_boundary: NDArray | None = None,
    x_label: str = "置信度",
):
    """绘制三轴误差合并到一张图中，按颜色区分"""
    ac_arr = np.array(ac_list)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["X", "Y", "Z"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.scatter(
            ac_arr, err_arr[:, i], s=4, alpha=0.4, color=color, label=f"{label}轴"
        )

    if upper_boundary is not None and len(upper_boundary) > 0:
        ax.scatter(
            upper_boundary[:, 0],
            upper_boundary[:, 1],
            s=30,
            c="black",
            marker="^",
            alpha=0.8,
            zorder=5,
            label="上边界",
        )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("误差 (m)", fontsize=11)
    ax.set_title(f"三轴误差 vs {x_label}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def fit_boundary_inv(upper_boundary: NDArray) -> tuple[float, float, NDArray] | None:
    """对上边界点进行 RANSAC 拟合 y = A/(x+B)，返回 (A, B, inlier_mask)"""
    from sklearn.linear_model import RANSACRegressor

    x_pos = upper_boundary[:, 0]
    y_pos = upper_boundary[:, 1]

    mask = y_pos > 0
    if mask.sum() < 3:
        return None

    # 1/y = (x+B)/A = (1/A)*x + B/A，令 Y = 1/y
    inv_y = (1.0 / y_pos[mask]).reshape(-1, 1)
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(x_pos[mask].reshape(-1, 1), inv_y)
    m = ransac.estimator_.coef_.item()
    c = ransac.estimator_.intercept_.item()  # ty: ignore
    A = 1.0 / m
    B = c / m
    inlier_mask = ransac.inlier_mask_
    return (A, B, inlier_mask)


def plot_boundary_fit(
    upper_boundary: NDArray,
    A: float,
    B: float,
    inlier_mask: NDArray,
    save_path: Path,
    show: bool = False,
):
    """绘制上边界点与拟合曲线 y = A/(x+B)，区分离群点"""
    x_pos = upper_boundary[:, 0]
    y_pos = upper_boundary[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        x_pos[inlier_mask],
        y_pos[inlier_mask],
        s=20,
        c="black",
        marker="^",
        alpha=0.8,
        zorder=5,
        label=f"上边界(内点, {inlier_mask.sum()})",
    )
    ax.scatter(
        x_pos[~inlier_mask],
        y_pos[~inlier_mask],
        s=20,
        c="gray",
        marker="x",
        alpha=0.5,
        zorder=5,
        label=f"离群点 ({(~inlier_mask).sum()})",
    )

    x_curve = np.linspace(x_pos.min(), x_pos.max(), 300)
    y_curve = A / (x_curve + B)
    ax.plot(x_curve, y_curve, "r-", linewidth=2, label=f"拟合: {A:.4f}/(x + {B:.4f})")

    ax.set_xlabel("置信度", fontsize=11)
    ax.set_ylabel("误差 (m)", fontsize=11)
    ax.set_title("上边界 RANSAC 拟合 (y = A/(x+B))", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, np.max(y_pos) * 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_error_boundary_classify(
    err_arr: NDArray,
    ac_list: list[float],
    A: float,
    B: float,
    save_path: Path,
    show: bool = False,
):
    """绘制三轴误差散点图，按是否在拟合边界内着色区分"""
    ac_arr = np.array(ac_list)
    labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (ax, label) in enumerate(zip(axes, labels)):
        x = ac_arr
        y = err_arr[:, i]
        boundary_y = A / (x + B)
        inside = y <= boundary_y
        ax.scatter(
            x[inside],
            y[inside],
            s=4,
            alpha=0.5,
            color="#2ecc71",
            label=f"边界内 ({inside.sum()})",
        )
        ax.scatter(
            x[~inside],
            y[~inside],
            s=4,
            alpha=0.5,
            color="#e74c3c",
            label=f"边界外 ({(~inside).sum()})",
        )
        x_curve = np.linspace(x.min(), x.max(), 300)
        ax.plot(
            x_curve,
            A / (x_curve + B),
            "r-",
            linewidth=2,
            label=f"拟合: {A:.4f}/(x + {B:.4f})",
        )
        ax.set_ylim(-0.05, 3.05)
        ax.set_xlabel("置信度", fontsize=11)
        ax.set_ylabel(f"{label}轴误差 (m)", fontsize=11)
        ax.set_title(f"{label}轴 — 边界内外分布", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    label = "sm"
    fit_res_path = Path("results") / f"boundary_fit_{label}.pkl"
    matchs = MatchList(
        f"/Users/qi/Resources/实验数据整理/点云视觉数据/mapping_{label}.csv"
    )
    all_error_list = []
    all_ac_list = []
    for cdb, gt_pose in matchs.iter_object():
        output_dir = Path("results") / cdb.name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 视觉误差评估结果
        result_path = output_dir / "visual_error_dict.pkl"
        if result_path.exists():
            print(f"[{cdb.name}] 加载结果")
            result = Obj.load(result_path)
        else:
            result = process_(cdb, gt_pose)
            Obj.save(result, result_path)

        plot_results(output_dir, **result)
        all_error_list.extend(result["error_list"])
        all_ac_list.extend(result["ac_list"])

        if fit_res_path.exists():
            print(f"[{cdb.name}] 加载拟合结果")
            fit_result = Obj.load(fit_res_path)
            A, B = fit_result["A"], fit_result["B"]
            noise_list = [A / (ac + B) for ac in result["ac_list"]]
            sensor_pose = result["sensor_pose"]
            match_pose = result["match_pose"]
            positions = np.array(
                [sensor_pose.get_pose(i).p for i in range(len(sensor_pose))]
            )
            gt_positions = np.array(
                [match_pose.get_pose(i).p for i in range(len(match_pose))]
            )
            err_arr = np.array(result["error_list"])
            err_norms = np.linalg.norm(err_arr, axis=1)
            plot_trajectory_comparison(
                positions,
                gt_positions,
                err_norms,
                noise_list,
                result["error_idx_list"],
                result["gap_idx_list"],
                output_dir / "trajectory_comparison_noise.png",
                right_label="噪声估计",
                right_cmap="YlGnBu",
            )

    all_err_arr = np.array(all_error_list)
    all_err_arr -= np.mean(all_err_arr, axis=0)
    all_err_arr = np.abs(all_err_arr)
    plot_error_vs_confidence(
        all_err_arr,
        all_ac_list,
        Path(f"results/all_abs_error_vs_confidence_{label}.png"),
    )

    # 合并三轴误差与置信度，提取共同上边界
    all_ac_arr = np.array(all_ac_list)
    merged_ac = np.concatenate([all_ac_arr, all_ac_arr, all_ac_arr])
    merged_err = np.concatenate(
        [all_err_arr[:, 0], all_err_arr[:, 1], all_err_arr[:, 2]]
    )
    upper_boundary = extract_upper_bound(merged_ac, merged_err)
    if len(upper_boundary) > 0:
        Obj.save(upper_boundary, Path("results/upper_boundary_merged.pkl"))

    plot_error_merged_vs_confidence(
        all_err_arr,
        all_ac_list,
        Path(f"results/all_abs_error_merged_vs_confidence_{label}.png"),
        upper_boundary=upper_boundary,
    )

    # 拟合上边界，绘图并统计
    if len(upper_boundary) > 0:
        result = fit_boundary_inv(upper_boundary)
        if result is not None:
            A, B, inlier_mask = result
            plot_boundary_fit(
                upper_boundary,
                A,
                B,
                inlier_mask,
                Path(f"results/boundary_fit_{label}.png"),
            )
            boundary_vals = A / (all_ac_arr + B)
            merged_err_all = np.concatenate(
                [all_err_arr[:, 0], all_err_arr[:, 1], all_err_arr[:, 2]]
            )
            boundary_vals_all = np.concatenate([boundary_vals] * 3)
            total = len(merged_err_all)
            within = np.sum(merged_err_all <= boundary_vals_all)
            print(f"[统计] 拟合模型: y = {A:.4f}/(x + {B:.4f})")
            print(f"[统计] 边界内点: {within}/{total} = {within / total:.2%}")

            # 绘制图像，区别边界内外的点
            plot_error_boundary_classify(
                all_err_arr,
                all_ac_list,
                A,
                B,
                Path(f"results/boundary_classify_{label}.png"),
            )

            Obj.save({"A": A, "B": B}, fit_res_path)
