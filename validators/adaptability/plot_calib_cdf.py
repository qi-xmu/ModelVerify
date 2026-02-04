#!/usr/bin/env python3
"""
绘制模型在相同设备下标定/非标定的CDF对比图

用法:
    # ATE对比（使用默认路径）
    uv run python validators/adaptability/plot_calib_cdf.py --error_type ATE

    # RPE对比（指定时间间隔）
    uv run python validators/adaptability/plot_calib_cdf.py --error_type RPE --interval 1
    uv run python validators/adaptability/plot_calib_cdf.py --error_type RPE --interval 5

    # 自定义数据路径
    uv run python validators/adaptability/plot_calib_cdf.py \
        --uncalib_dir results/model_resnet_0111_96_Redmi-K30-Pro \
        --calib_dir calib2/model_resnet_0111_96_Redmi-K30-Pro

参数:
    --error_type: 误差类型 (ATE/RPE/RTE/APE)，默认ATE
    --interval: RPE/RTE的时间间隔（秒），默认1
    --uncalib_dir: 未标定数据目录路径，默认 results/model_resnet_0111_96_Redmi-K30-Pro
    --calib_dir: 标定数据目录路径，默认 calib2/model_resnet_0111_96_Redmi-K30-Pro

输出:
    results/compare/CDF_CalibrationComparison_{error_type}.png
"""

from pathlib import Path
import argparse
import pickle

import numpy as np
from matplotlib import pyplot as plt

from base.evaluate import get_cdf_from_err


def collect_errors_from_result_dir(result_dir: Path, err_type: str = "ATE", interval: int = 1) -> tuple[list, dict]:
    """
    从结果目录中收集所有轨迹的误差数据

    Args:
        result_dir: 结果目录
        err_type: 误差类型 (ATE/RPE/RTE/APE)
        interval: RPE/RTE的时间间隔（秒）

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
                evaluator = data[1]

                for key, inner_data in evaluator.inner.items():
                    cdf_key = f"{err_type}_CDF"
                    if cdf_key in inner_data:
                        errors = inner_data[cdf_key]["errors"]
                        all_errors.extend(errors.tolist())
                        stats["total_samples"] += len(errors)
                        stats["trajectory_count"] += 1
                        break
        except Exception as e:
            print(f"  Warning: Failed to read {pkl_path}: {e}")

    if all_errors:
        all_errors_arr = np.array(all_errors)
        stats["mean"] = float(np.mean(all_errors_arr))
        stats["std"] = float(np.std(all_errors_arr))

    return all_errors, stats


def plot_calib_comparison(
    datasets: list[dict],
    output_path: Path,
    err_type: str = "ATE",
    interval: int = 1,
    title: str = "",
):
    """绘制标定/非标定CDF曲线对比"""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ["#ff7f0e", "#2ca02c"]
    linestyles = ["-", "--"]

    # 存储每条曲线的百分位数值和曲线句柄
    percentile_data = []
    curve_handles = []

    for idx, ds in enumerate(datasets):
        if len(ds["errors"]) == 0:
            continue

        cdf_data = get_cdf_from_err(ds["errors"], ds["name"])
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

    unit_map = {"ATE": "(m)", "RTE": "(m)"}
    unit = unit_map.get(err_type, "")

    ax.set_xlabel(f"Error {unit}", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(title if title else f"Calibration Comparison - {err_type}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)

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
    parser = argparse.ArgumentParser(description="绘制标定/非标定的CDF对比图")
    parser.add_argument("--error_type", type=str, default="ATE",
                        choices=["ATE", "APE", "RPE", "RTE"],
                        help="误差类型")
    parser.add_argument("--interval", type=int, default=1,
                        help="RPE/RTE的时间间隔（秒），默认1")
    parser.add_argument("--uncalib_dir", type=str, default="results/model_resnet_0111_96_Redmi-K30-Pro",
                        help="未标定数据目录路径")
    parser.add_argument("--calib_dir", type=str, default="calib2/model_resnet_0111_96_Redmi-K30-Pro",
                        help="标定数据目录路径")
    args = parser.parse_args()

    base_result_dir = Path(".")
    output_dir = Path("results/compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    err_type = args.error_type
    interval = args.interval

    print(f"=== Calibration CDF Comparison ({err_type}, interval={interval}s) ===")
    datasets = []

    # 未标定数据
    k30_dir = Path(args.uncalib_dir)
    if k30_dir.exists():
        errors, stats = collect_errors_from_result_dir(k30_dir, err_type, interval)
        datasets.append({
            "name": "Redmi-K30-Pro (Uncalibrated)",
            "errors": errors,
            "stats": stats,
        })
        print(f"  Uncalibrated: {stats['trajectory_count']} trajectories, {stats['total_samples']} samples")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

    # 标定数据
    k30_calib_dir = Path(args.calib_dir)
    if k30_calib_dir.exists():
        errors, stats = collect_errors_from_result_dir(k30_calib_dir, err_type, interval)
        datasets.append({
            "name": "Redmi-K30-Pro (Calibrated)",
            "errors": errors,
            "stats": stats,
        })
        print(f"  Calibrated: {stats['trajectory_count']} trajectories, {stats['total_samples']} samples")
        print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

    if len(datasets) >= 2:
        # RPE/RTE需要标注时间间隔
        if err_type in ["RPE", "RTE"]:
            title_suffix = f" (interval={interval}s)"
        else:
            title_suffix = ""
        output_path = output_dir / f"CDF_CalibrationComparison_{err_type}.png"
        plot_calib_comparison(
            datasets,
            output_path,
            err_type,
            interval,
            title=f"model_resnet_0111_96 - Calibration Comparison ({err_type}){title_suffix}",
        )
    else:
        print("Warning: Not enough datasets for comparison")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
