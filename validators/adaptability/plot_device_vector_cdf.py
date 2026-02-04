#!/usr/bin/env python3
"""
绘制模型在不同设备下的CDF对比图（使用 meas - gt 向量误差）

用法:
    uv run python validators/adaptability/plot_device_vector_cdf.py

输出:
    results/compare/CDF_DeviceComparison_Vector.png
"""

from pathlib import Path
import argparse
import pickle

import numpy as np
from matplotlib import pyplot as plt

from base.model import NetworkResult


def collect_vector_errors_from_result_dir(result_dir: Path) -> tuple[list, dict]:
    """
    从结果目录中收集所有轨迹的 meas-gt 向量误差

    Args:
        result_dir: 结果目录 (如 results/model_resnet_0111_96_ABR-AL60)

    Returns:
        tuple: (所有误差值列表, 汇总统计信息)
    """
    all_errors = []
    stats = {
        "trajectory_count": 0,
        "total_samples": 0,
        "mean": 0,
        "std": 0,
    }

    temp_dir = result_dir / "temp"
    if not temp_dir.exists():
        print(f"Warning: temp dir not found: {temp_dir}")
        return all_errors, stats

    pkl_files = list(temp_dir.glob("action_*.pkl"))
    print(f"  Found {len(pkl_files)} trajectory files")

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                nr_list = data[0]  # nr_list

                for nr in nr_list:
                    assert isinstance(nr, NetworkResult)
                    meas = np.array(nr.meas_list)
                    gt = np.array(nr.gt_list)

                    err = meas - gt
                    err_norm = np.linalg.norm(err, axis=1)
                    all_errors.extend(err_norm)
                    stats["total_samples"] += len(err_norm)

                stats["trajectory_count"] += 1
        except Exception as e:
            print(f"  Warning: Failed to read {pkl_path}: {e}")

    if all_errors:
        all_errors_arr = np.array(all_errors)
        stats["mean"] = float(np.mean(all_errors_arr))
        stats["std"] = float(np.std(all_errors_arr))

    return all_errors, stats


def get_cdf_from_errors(errors: list) -> dict:
    """计算误差的累积分布函数"""
    errors = np.array(errors)
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    percentiles = {
        "50%": np.percentile(errors, 50),
        "90%": np.percentile(errors, 90),
        "95%": np.percentile(errors, 95),
        "99%": np.percentile(errors, 99),
    }

    return {
        "errors": sorted_errors,
        "cdf": cdf,
        "percentiles": percentiles,
    }


def plot_device_comparison(
    datasets: list[dict],
    output_path: Path,
    title: str = "",
):
    """绘制多条CDF曲线对比"""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    linestyles = ["-", "--", "-.", ":"]

    # 存储每条曲线的百分位数值和曲线句柄
    percentile_data = []
    curve_handles = []

    for idx, ds in enumerate(datasets):
        if len(ds["errors"]) == 0:
            continue

        cdf_data = get_cdf_from_errors(ds["errors"])
        errors = cdf_data["errors"]
        cdf = cdf_data["cdf"]
        percentiles = cdf_data["percentiles"]

        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]

        line, = ax.plot(
            errors,
            cdf,
            label=f"{ds['name']} (mean={ds['stats']['mean']:.4f}, std={ds['stats']['std']:.4f})",
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
        curve_handles.append(line)

        # 标记百分位点
        for pct, marker in [("90%", "o"), ("95%", "s"), ("99%", "^")]:
            val = percentiles[pct]
            idx_closest = np.searchsorted(errors, val)
            if idx_closest < len(cdf):
                ax.plot(val, cdf[idx_closest], marker, color=color, markersize=8)

        # 收集百分位数据
        percentile_data.append({
            "name": ds["name"],
            "color": color,
            "90%": percentiles["90%"],
            "95%": percentiles["95%"],
            "99%": percentiles["99%"],
        })

    # 在图例上方绘制百分位数值表格
    ax2 = fig.add_axes([0.58, 0.55, 0.38, 0.25])  # 放在图例上方
    ax2.axis('off')

    # 表头
    table_header = ["Label", "90%", "95%", "99%"]
    cell_text = []
    text_colors = []

    for data in percentile_data:
        cell_text.append([
            data["name"],
            f"{data['90%']:.4f}",
            f"{data['95%']:.4f}",
            f"{data['99%']:.4f}",
        ])
        # 字体颜色与曲线颜色一致
        text_colors.append([data["color"], data["color"], data["color"], data["color"]])

    table = ax2.table(
        cellText=cell_text,
        colLabels=table_header,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    # 增大Label列的宽度
    col_widths = [0.40, 0.20, 0.20, 0.20]
    for i, width in enumerate(col_widths):
        table.auto_set_column_width([i])
        cells = table.get_celld()
        for (row, col), cell in cells.items():
            cell.set_width(width)
            # 设置字体颜色
            if row > 0:  # 数据行
                cell.set_text_props(color=text_colors[row - 1][col])

    # 设置表头颜色
    for i in range(len(table_header)):
        table[(0, i)].set_facecolor('lightgray')
        table[(0, i)].set_text_props(weight='bold')

    ax.set_xlabel("Error (m)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(title if title else "Device Comparison - Vector Error", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0, right=1.0)

    # 添加百分位标记图例说明
    from matplotlib.lines import Line2D
    percentile_handles = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=8, label='90th percentile'),
        Line2D([0], [0], marker='s', color='gray', linestyle='None', markersize=8, label='95th percentile'),
        Line2D([0], [0], marker='^', color='gray', linestyle='None', markersize=8, label='99th percentile'),
    ]
    ax.legend(handles=curve_handles + percentile_handles, fontsize=10, loc="lower right")

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"CDF comparison saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="绘制不同设备的向量误差CDF对比图")
    args = parser.parse_args()

    base_result_dir = Path("results")
    output_dir = Path("results/compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Device CDF Comparison (Vector Error) ===")
    datasets = []

    # ABR-AL60 (未标定)
    abr_dir = base_result_dir / "model_resnet_0111_96_ABR-AL60"
    if abr_dir.exists():
        errors, stats = collect_vector_errors_from_result_dir(abr_dir)
        datasets.append({
            "name": "ABR-AL60 (Uncalibrated)",
            "errors": errors,
            "stats": stats,
        })
        print(f"  ABR-AL60: {stats['trajectory_count']} trajectories, {stats['total_samples']} samples")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

    # Redmi-K30-Pro (未标定)
    k30_dir = base_result_dir / "model_resnet_0111_96_Redmi-K30-Pro"
    if k30_dir.exists():
        errors, stats = collect_vector_errors_from_result_dir(k30_dir)
        datasets.append({
            "name": "Redmi-K30-Pro (Uncalibrated)",
            "errors": errors,
            "stats": stats,
        })
        print(f"  Redmi-K30-Pro: {stats['trajectory_count']} trajectories, {stats['total_samples']} samples")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

    if len(datasets) >= 2:
        output_path = output_dir / "CDF_DeviceComparison_Vector.png"
        plot_device_comparison(
            datasets,
            output_path,
            title="model_resnet_0111_96 - Device Comparison (Vector Error)",
        )
    else:
        print("Warning: Not enough datasets for comparison")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
