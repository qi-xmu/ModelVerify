#!/usr/bin/env python3
"""
绘制三个模型的 CDF 向量误差对比图

用法:
    uv run python validators/adaptability/plot_model_cdf_comparison.py

输出:
    results/compare/CDF_ModelComparison_Vector.png
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
        result_dir: 结果目录

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


def plot_model_comparison(
    models: list[dict],
    output_path: Path,
    title: str = "",
):
    """绘制多个模型的CDF曲线对比"""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    linestyles = ["-", "--", "-."]

    # 存储每条曲线的百分位数值
    percentile_data = []
    curve_handles = []

    for idx, model in enumerate(models):
        if len(model["errors"]) == 0:
            continue

        cdf_data = get_cdf_from_errors(model["errors"])
        errors = cdf_data["errors"]
        cdf = cdf_data["cdf"]
        percentiles = cdf_data["percentiles"]

        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]

        line, = ax.plot(
            errors,
            cdf,
            label=f"{model['name']} (mean={model['stats']['mean']:.4f}, std={model['stats']['std']:.4f})",
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
            "name": model["name"],
            "color": color,
            "90%": percentiles["90%"],
            "95%": percentiles["95%"],
            "99%": percentiles["99%"],
        })

    # 在图例上方绘制百分位数值表格
    # 使用 ax 的坐标系，位置为 [left, bottom, width, height]
    ax_table = ax.inset_axes([0.55, 0.38, 0.42, 0.20])
    ax_table.axis('off')

    table_header = ["Model", "90%", "95%", "99%"]
    cell_text = []
    text_colors = []

    for data in percentile_data:
        cell_text.append([
            data["name"],
            f"{data['90%']:.4f}",
            f"{data['95%']:.4f}",
            f"{data['99%']:.4f}",
        ])
        text_colors.append([data["color"], data["color"], data["color"], data["color"]])

    table = ax_table.table(
        cellText=cell_text,
        colLabels=table_header,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    col_widths = [0.40, 0.20, 0.20, 0.20]
    for i in range(len(col_widths)):
        table.auto_set_column_width([i])
        cells = table.get_celld()
        for (row, col), cell in cells.items():
            cell.set_width(col_widths[col])
            if row > 0:
                cell.set_text_props(color=text_colors[row - 1][col])

    for i in range(len(table_header)):
        table[(0, i)].set_facecolor('lightgray')
        table[(0, i)].set_text_props(weight='bold')

    ax.set_xlabel("Error (m)", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(title if title else "Model Comparison - Vector Error", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0, right=1.0)

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
    parser = argparse.ArgumentParser(description="绘制三个模型的向量误差CDF对比图")
    parser.add_argument("--output", type=str, default="results/compare",
                        help="输出目录 (default: results/compare)")
    parser.add_argument("--device", type=str, default="ABR-AL60",
                        choices=["ABR-AL60", "Redmi-K30-Pro"],
                        help="设备类型 (default: ABR-AL60)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device

    # 模型配置
    model_configs = [
        {
            "name": "model_imot_0111_64",
            "path": Path(f"/Users/qi/Resources/results/model_imot_0111_64_{device}"),
        },
        {
            "name": "model_lstm_0105",
            "path": Path(f"/Users/qi/Resources/results/model_lstm_0105_{device}"),
        },
        {
            "name": "model_resnet_0111_96",
            "path": Path(f"/Users/qi/Resources/results/model_resnet_0111_96_{device}"),
        },
    ]

    print(f"=== Model CDF Comparison (Vector Error) - {device} ===")
    models = []

    for config in model_configs:
        result_dir = config["path"]
        print(f"\n[{config['name']}]")
        if result_dir.exists():
            errors, stats = collect_vector_errors_from_result_dir(result_dir)
            models.append({
                "name": config["name"],
                "errors": errors,
                "stats": stats,
            })
            print(f"  Trajectories: {stats['trajectory_count']}, Samples: {stats['total_samples']}")
            print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        else:
            print(f"  Warning: Directory not found: {result_dir}")

    if len(models) >= 2:
        output_path = output_dir / f"CDF_ModelComparison_Vector_{device}.png"
        plot_model_comparison(
            models,
            output_path,
            title=f"Model Comparison - Vector Error ({device})",
        )
    else:
        print("Warning: Not enough models for comparison")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
