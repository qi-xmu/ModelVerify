#!/usr/bin/env python3
"""多模型 + 多数据集推理与跨维度汇总。

支持两种模式：
1) 交叉模式（默认）：所有模型评估所有数据集，并输出跨模型/跨数据集汇总。
2) 配对模式：
    - 组A模型仅评估组A数据集
    - 组B模型仅评估组B数据集
    - 按数据集一一配对、按同名单元绘制同图轨迹对比（GT + 组A预测 + 组B预测）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from base import device as base_device
from base.args_parser import DatasetArgsParser
from base.dataset import H5SplitDataset, H5UnitData
from base.draw.Poses import draw_trajectory_2d_compare
from base.model import InertialNetworkData, ModelLoader
from base.obj import Obj

from VaildModelFromH5atrdata import (
    _draw_metric_boxplot_by_group,
    _draw_metric_cdf_by_group,
    _draw_metric_pde_by_group,
    _evaluate_single_dataset,
    _get_err_stats,
    _save_json,
    _safe_name,
)


METRIC_KEYS: list[str] = [
    "ATE(m)",
    "RTE(m)",
    "position_rmse(m)",
    "velocity_rmse(m/s)",
    "heading_mae(deg)",
    "disp_error_norm(m/s)",
]


def _save_ranking_text(path: Path, ranking: list[dict[str, float | str]]) -> None:
    lines = ["=" * 80, "Global Ranking by ATE mean", "=" * 80]
    for idx, row in enumerate(ranking, start=1):
        lines.append(
            (
                f" {idx}. {row['model']} | "
                f"ATE={row['ATE_mean(m)']:.4f}m, "
                f"RTE={row['RTE_mean(m)']:.4f}m, "
                f"PosRMSE={row['position_rmse_mean(m)']:.4f}m, "
                f"VelRMSE={row['velocity_rmse_mean(m/s)']:.4f}m/s, "
                f"HeadingMAE={row['heading_mae_mean(deg)']:.4f}deg"
            )
        )
    lines.append("=" * 80)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_nets(model_names: list[str], models_path: str, selected_device: str):
    loader = ModelLoader(models_path)
    nets = loader.get_by_names(model_names)
    for net in nets:
        net.set_device(selected_device)
        net.warmup()
    return nets


def _evaluate_group_datasets(
    group_tag: str,
    model_names: list[str],
    dataset_paths: list[Path],
    models_path: str,
    selected_device: str,
    split: str,
    unit_filter: str | None,
    config_path: str | None,
    eval_dir: Path,
    no_plots: bool,
) -> tuple[list[Any], list[dict[str, Any]], Path]:
    data_cfg = InertialNetworkData.set_step(20).no_rerun()
    nets = _load_nets(model_names, models_path, selected_device)

    model_tag = "_VS_".join([n.name for n in nets])
    dataset_eval_root = eval_dir / "dataset_eval" / group_tag
    dataset_reports: list[dict[str, Any]] = []

    for dataset_path in dataset_paths:
        report = _evaluate_single_dataset(
            dataset_path=dataset_path,
            split=split,
            nets=nets,
            data_cfg=data_cfg,
            unit_filter=unit_filter,
            config_path=config_path,
            dataset_out_root=dataset_eval_root,
            no_plots=no_plots,
        )
        dataset_reports.append(report)

    cross_root = eval_dir / "cross_model_dataset" / f"{group_tag}_{split}_{model_tag}"
    cross_root.mkdir(parents=True, exist_ok=True)
    _save_group_cross_summary(
        split=split,
        group_tag=group_tag,
        nets=nets,
        dataset_reports=dataset_reports,
        out_root=cross_root,
    )
    return nets, dataset_reports, cross_root


def _save_group_cross_summary(
    split: str,
    group_tag: str,
    nets: list[Any],
    dataset_reports: list[dict[str, Any]],
    out_root: Path,
) -> None:
    by_model: dict[str, Any] = {}
    for net in nets:
        model_name = net.name
        per_metric_values: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}

        for report in dataset_reports:
            metric_lists = report["model_metric_lists"][model_name]
            for metric_key in METRIC_KEYS:
                per_metric_values[metric_key].extend(metric_lists[metric_key])

        by_model[model_name] = {
            metric_key: _get_err_stats(values)
            for metric_key, values in per_metric_values.items()
        }

    by_dataset: dict[str, Any] = {}
    for report in dataset_reports:
        dataset_key = f"{report['dataset_name']}_{report['split']}"
        model_table: dict[str, dict[str, float]] = {}

        for net in nets:
            model_name = net.name
            model_summary = report["compare_summary"]["models"][model_name]
            model_table[model_name] = {
                "ATE_mean(m)": model_summary["ATE(m)"]["mean"],
                "RTE_mean(m)": model_summary["RTE(m)"]["mean"],
                "position_rmse_mean(m)": model_summary["position_rmse(m)"]["mean"],
                "velocity_rmse_mean(m/s)": model_summary["velocity_rmse(m/s)"]["mean"],
                "heading_mae_mean(deg)": model_summary["heading_mae(deg)"]["mean"],
                "disp_error_norm_mean(m/s)": model_summary["disp_error_norm(m/s)"]["mean"],
            }

        by_dataset[dataset_key] = {
            "dataset": report["dataset"],
            "sample_count": report["compare_summary"]["sample_count"],
            "models": model_table,
        }

    global_ranking: list[dict[str, float | str]] = []
    for model_name, stats in by_model.items():
        global_ranking.append(
            {
                "model": model_name,
                "ATE_mean(m)": stats["ATE(m)"]["mean"],
                "RTE_mean(m)": stats["RTE(m)"]["mean"],
                "position_rmse_mean(m)": stats["position_rmse(m)"]["mean"],
                "velocity_rmse_mean(m/s)": stats["velocity_rmse(m/s)"]["mean"],
                "heading_mae_mean(deg)": stats["heading_mae(deg)"]["mean"],
            }
        )
    global_ranking.sort(key=lambda x: float(x["ATE_mean(m)"]))

    summary = {
        "mode": "paired_group_cross",
        "group": group_tag,
        "split": split,
        "model_count": len(nets),
        "dataset_count": len(dataset_reports),
        "datasets": [
            {
                "dataset": report["dataset"],
                "dataset_name": report["dataset_name"],
                "res_root": report["res_root"],
            }
            for report in dataset_reports
        ],
        "by_model": by_model,
        "by_dataset": by_dataset,
        "global_ranking_by_ATE_mean": global_ranking,
    }

    _save_json(out_root / "CrossModelDatasetSummary.json", summary)
    _save_ranking_text(out_root / "GlobalRanking.txt", global_ranking)


def _save_paired_global_metric_plots(
    split: str,
    datasets_a: list[Path],
    datasets_b: list[Path],
    reports_a: list[dict[str, Any]],
    reports_b: list[dict[str, Any]],
    nets_a: list[Any],
    nets_b: list[Any],
    out_root: Path,
    no_plots: bool,
) -> dict[str, Any]:
    global_root = out_root / "global_metrics"
    global_root.mkdir(parents=True, exist_ok=True)

    model_metric_values: dict[str, dict[str, list[float]]] = {
        **{f"A:{n.name}": {metric_key: [] for metric_key in METRIC_KEYS} for n in nets_a},
        **{f"B:{n.name}": {metric_key: [] for metric_key in METRIC_KEYS} for n in nets_b},
    }

    pair_details: list[dict[str, Any]] = []
    total_common = 0
    total_strict_matched = 0

    for dataset_a, dataset_b, report_a, report_b in zip(datasets_a, datasets_b, reports_a, reports_b):
        datas_a = H5SplitDataset(dataset_a, split=split)
        datas_b = H5SplitDataset(dataset_b, split=split)

        common_units = sorted(set(datas_a.sequence_names).intersection(set(datas_b.sequence_names)))
        total_common += len(common_units)

        strict_units: list[str] = []
        missing_units: list[str] = []

        for unit_name in common_units:
            has_all_cache = True
            for net in nets_a:
                cache_path = Path(report_a["res_root"]) / "temp" / f"{net.name}_{unit_name}.pkl"
                if not cache_path.exists():
                    has_all_cache = False
                    break
            if has_all_cache:
                for net in nets_b:
                    cache_path = Path(report_b["res_root"]) / "temp" / f"{net.name}_{unit_name}.pkl"
                    if not cache_path.exists():
                        has_all_cache = False
                        break

            if has_all_cache:
                strict_units.append(unit_name)
            else:
                missing_units.append(unit_name)

        total_strict_matched += len(strict_units)

        for unit_name in strict_units:
            for net in nets_a:
                cache_path = Path(report_a["res_root"]) / "temp" / f"{net.name}_{unit_name}.pkl"
                nr, _eval_json, unit_metrics = Obj.load(cache_path)
                model_label = f"A:{net.name}"
                model_metric_values[model_label]["ATE(m)"].append(float(unit_metrics["ATE(m)"]))
                model_metric_values[model_label]["RTE(m)"].append(float(unit_metrics["RTE(m)"]))
                model_metric_values[model_label]["position_rmse(m)"].append(float(unit_metrics["position_rmse(m)"]))
                model_metric_values[model_label]["velocity_rmse(m/s)"].append(float(unit_metrics["velocity_rmse(m/s)"]))
                model_metric_values[model_label]["heading_mae(deg)"].append(float(unit_metrics["heading_mae(deg)"]))

                meas = np.asarray(nr.meas_list, dtype=np.float64)
                gt = np.asarray(nr.gt_list, dtype=np.float64)
                if meas.size > 0 and gt.size > 0 and meas.shape == gt.shape:
                    err_norm = np.linalg.norm(meas - gt, axis=1)
                    model_metric_values[model_label]["disp_error_norm(m/s)"].extend(err_norm.astype(float).tolist())

            for net in nets_b:
                cache_path = Path(report_b["res_root"]) / "temp" / f"{net.name}_{unit_name}.pkl"
                nr, _eval_json, unit_metrics = Obj.load(cache_path)
                model_label = f"B:{net.name}"
                model_metric_values[model_label]["ATE(m)"].append(float(unit_metrics["ATE(m)"]))
                model_metric_values[model_label]["RTE(m)"].append(float(unit_metrics["RTE(m)"]))
                model_metric_values[model_label]["position_rmse(m)"].append(float(unit_metrics["position_rmse(m)"]))
                model_metric_values[model_label]["velocity_rmse(m/s)"].append(float(unit_metrics["velocity_rmse(m/s)"]))
                model_metric_values[model_label]["heading_mae(deg)"].append(float(unit_metrics["heading_mae(deg)"]))

                meas = np.asarray(nr.meas_list, dtype=np.float64)
                gt = np.asarray(nr.gt_list, dtype=np.float64)
                if meas.size > 0 and gt.size > 0 and meas.shape == gt.shape:
                    err_norm = np.linalg.norm(meas - gt, axis=1)
                    model_metric_values[model_label]["disp_error_norm(m/s)"].extend(err_norm.astype(float).tolist())

        pair_details.append(
            {
                "dataset_a": str(dataset_a),
                "dataset_b": str(dataset_b),
                "common_unit_count": len(common_units),
                "strict_matched_unit_count": len(strict_units),
                "missing_cache_unit_count": len(missing_units),
            }
        )

    summary: dict[str, Any] = {
        "mode": "paired",
        "stats_scope": "paired_common_units_strict",
        "split": split,
        "metric_keys": METRIC_KEYS,
        "pair_count": len(pair_details),
        "total_common_unit_count": total_common,
        "total_strict_matched_unit_count": total_strict_matched,
        "pair_details": pair_details,
        "models": {},
    }
    for model_label, metric_map in model_metric_values.items():
        summary["models"][model_label] = {
            metric_key: _get_err_stats(values)
            for metric_key, values in metric_map.items()
        }

    for metric_key in METRIC_KEYS:
        metric_by_model = {
            model_label: metric_map[metric_key]
            for model_label, metric_map in model_metric_values.items()
        }
        metric_tag = _safe_name(metric_key)
        if not no_plots:
            _draw_metric_cdf_by_group(
                metric_by_group=metric_by_model,
                save_path=global_root / f"{metric_tag}_CDF_by_model.png",
                title=f"{metric_key} CDF by Model (Paired)",
                x_label=metric_key,
            )
            _draw_metric_pde_by_group(
                metric_by_group=metric_by_model,
                save_path=global_root / f"{metric_tag}_PDE_by_model.png",
                title=f"{metric_key} PDE by Model (Paired)",
                x_label=metric_key,
            )
            _draw_metric_boxplot_by_group(
                metric_by_group=metric_by_model,
                save_path=global_root / f"{metric_tag}_Boxplot_by_model.png",
                title=f"{metric_key} Boxplot by Model (Paired)",
                y_label=metric_key,
            )

    _save_json(global_root / "PairedGlobalMetricsSummary.json", summary)
    return summary


def _paired_plot_one_dataset_pair(
    dataset_a: Path,
    dataset_b: Path,
    split: str,
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    nets_a: list[Any],
    nets_b: list[Any],
    out_root: Path,
    no_plots: bool,
) -> dict[str, Any]:
    datas_a = H5SplitDataset(dataset_a, split=split)
    datas_b = H5SplitDataset(dataset_b, split=split)

    names_a = set(datas_a.sequence_names)
    names_b = set(datas_b.sequence_names)
    common_names = sorted(names_a.intersection(names_b))

    pair_key = f"{datas_a.name}_{split}_VS_{datas_b.name}_{split}"
    pair_root = out_root / pair_key
    pair_root.mkdir(parents=True, exist_ok=True)

    drawn = 0
    missing_cache: list[str] = []
    for unit_name in common_names:
        seq_a = datas_a.get_sequence(unit_name)
        if seq_a is None:
            continue
        ud_a = H5UnitData(seq_a)
        ud_a.load_data(using_opt=True)

        poses = [ud_a.gt_data]
        labels = ["GT"]

        all_cache_hit = True
        for net in nets_a:
            cache_path = Path(report_a["res_root"]) / "temp" / f"{net.name}_{unit_name}.pkl"
            if not cache_path.exists():
                all_cache_hit = False
                continue
            nr, _eval_json, _unit_metrics = Obj.load(cache_path)
            poses.append(nr.poses)
            labels.append(f"A:{net.name}")

        for net in nets_b:
            cache_path = Path(report_b["res_root"]) / "temp" / f"{net.name}_{unit_name}.pkl"
            if not cache_path.exists():
                all_cache_hit = False
                continue
            nr, _eval_json, _unit_metrics = Obj.load(cache_path)
            poses.append(nr.poses)
            labels.append(f"B:{net.name}")

        if not all_cache_hit:
            missing_cache.append(unit_name)

        if len(poses) <= 1 or no_plots:
            continue

        draw_trajectory_2d_compare(
            poses_data_list=poses,
            labels=labels,
            title=f"Trajectory_Compare_Paired_{unit_name}",
            save_path=pair_root / "units" / unit_name / "Trajectory_Compare_Paired.png",
            show=False,
        )
        drawn += 1

    pair_summary = {
        "pair_key": pair_key,
        "dataset_a": str(dataset_a),
        "dataset_b": str(dataset_b),
        "split": split,
        "common_unit_count": len(common_names),
        "drawn_unit_count": drawn,
        "missing_cache_unit_count": len(missing_cache),
        "missing_cache_units": missing_cache,
    }
    _save_json(pair_root / "PairedPlotSummary.json", pair_summary)
    return pair_summary


def _run_paired_mode(dap: DatasetArgsParser, eval_dir: Path, selected_device: str) -> None:
    models_path = dap.args.models_path or "models"
    datasets_a = [Path(path) for path in dap.args.datasets_a]
    datasets_b = [Path(path) for path in dap.args.datasets_b]

    if len(datasets_a) != len(datasets_b):
        raise ValueError("--datasets-a 与 --datasets-b 数量必须一致（按位置一一配对）")

    nets_a, reports_a, _cross_root_a = _evaluate_group_datasets(
        group_tag="group_a",
        model_names=dap.args.models_a,
        dataset_paths=datasets_a,
        models_path=models_path,
        selected_device=selected_device,
        split=dap.args.split,
        unit_filter=dap.unit,
        config_path=dap.args.config,
        eval_dir=eval_dir,
        no_plots=dap.args.no_plots,
    )
    nets_b, reports_b, _cross_root_b = _evaluate_group_datasets(
        group_tag="group_b",
        model_names=dap.args.models_b,
        dataset_paths=datasets_b,
        models_path=models_path,
        selected_device=selected_device,
        split=dap.args.split,
        unit_filter=dap.unit,
        config_path=dap.args.config,
        eval_dir=eval_dir,
        no_plots=dap.args.no_plots,
    )

    pair_root = eval_dir / "paired_compare" / dap.args.split
    pair_root.mkdir(parents=True, exist_ok=True)

    paired_summaries: list[dict[str, Any]] = []
    for dataset_a, dataset_b, report_a, report_b in zip(datasets_a, datasets_b, reports_a, reports_b):
        paired_summaries.append(
            _paired_plot_one_dataset_pair(
                dataset_a=dataset_a,
                dataset_b=dataset_b,
                split=dap.args.split,
                report_a=report_a,
                report_b=report_b,
                nets_a=nets_a,
                nets_b=nets_b,
                out_root=pair_root,
                no_plots=dap.args.no_plots,
            )
        )

    paired_global_summary = _save_paired_global_metric_plots(
        split=dap.args.split,
        datasets_a=datasets_a,
        datasets_b=datasets_b,
        reports_a=reports_a,
        reports_b=reports_b,
        nets_a=nets_a,
        nets_b=nets_b,
        out_root=pair_root,
        no_plots=dap.args.no_plots,
    )

    summary = {
        "mode": "paired",
        "split": dap.args.split,
        "models_a": [n.name for n in nets_a],
        "models_b": [n.name for n in nets_b],
        "datasets_a": [str(p) for p in datasets_a],
        "datasets_b": [str(p) for p in datasets_b],
        "pair_count": len(paired_summaries),
        "pairs": paired_summaries,
        "group_a_reports": reports_a,
        "group_b_reports": reports_b,
        "paired_global_metrics_summary": paired_global_summary,
    }
    summary_path = pair_root / "PairedCompareSummary.json"
    _save_json(summary_path, summary)
    print(f"配对模式完成，总结: {summary_path}", flush=True)
    print(f"配对轨迹输出目录: {pair_root}", flush=True)


def _run_cross_mode(dap: DatasetArgsParser, eval_dir: Path, selected_device: str) -> None:
    model_names = dap.args.models or ["model_resnet_0111_96"]
    models_path = dap.args.models_path or "models"

    nets = _load_nets(model_names, models_path, selected_device)
    data_cfg = InertialNetworkData.set_step(20).no_rerun()

    dataset_paths_raw = dap.args.datasets or ([] if dap.dataset is None else [dap.dataset])
    if len(dataset_paths_raw) == 0:
        raise ValueError("Dataset path is not provided, use -d/--dataset or --datasets")
    dataset_paths = [Path(path) for path in dataset_paths_raw]

    model_tag = "_VS_".join([n.name for n in nets])
    dataset_eval_root = eval_dir / "dataset_eval"

    dataset_reports: list[dict[str, Any]] = []
    for dataset_path in dataset_paths:
        report = _evaluate_single_dataset(
            dataset_path=dataset_path,
            split=dap.args.split,
            nets=nets,
            data_cfg=data_cfg,
            unit_filter=dap.unit,
            config_path=dap.args.config,
            dataset_out_root=dataset_eval_root,
            no_plots=dap.args.no_plots,
        )
        dataset_reports.append(report)

    cross_root = eval_dir / "cross_model_dataset" / f"{dap.args.split}_{model_tag}"
    cross_root.mkdir(parents=True, exist_ok=True)

    by_model: dict[str, Any] = {}
    for net in nets:
        model_name = net.name
        per_metric_values: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}

        for report in dataset_reports:
            metric_lists = report["model_metric_lists"][model_name]
            for metric_key in METRIC_KEYS:
                per_metric_values[metric_key].extend(metric_lists[metric_key])

        by_model[model_name] = {
            metric_key: _get_err_stats(values)
            for metric_key, values in per_metric_values.items()
        }

    by_dataset: dict[str, Any] = {}
    per_dataset_ranking: dict[str, list[dict[str, float | str]]] = {}
    for report in dataset_reports:
        dataset_key = f"{report['dataset_name']}_{report['split']}"
        model_table: dict[str, dict[str, float]] = {}

        ranking_rows: list[dict[str, float | str]] = []
        for net in nets:
            model_name = net.name
            model_summary = report["compare_summary"]["models"][model_name]
            row = {
                "ATE_mean(m)": model_summary["ATE(m)"]["mean"],
                "RTE_mean(m)": model_summary["RTE(m)"]["mean"],
                "position_rmse_mean(m)": model_summary["position_rmse(m)"]["mean"],
                "velocity_rmse_mean(m/s)": model_summary["velocity_rmse(m/s)"]["mean"],
                "heading_mae_mean(deg)": model_summary["heading_mae(deg)"]["mean"],
            }
            model_table[model_name] = row

            ranking_rows.append(
                {
                    "model": model_name,
                    "ATE_mean(m)": row["ATE_mean(m)"],
                    "RTE_mean(m)": row["RTE_mean(m)"],
                    "position_rmse_mean(m)": row["position_rmse_mean(m)"],
                    "velocity_rmse_mean(m/s)": row["velocity_rmse_mean(m/s)"],
                    "heading_mae_mean(deg)": row["heading_mae_mean(deg)"],
                }
            )

        ranking_rows.sort(key=lambda x: float(x["ATE_mean(m)"]))
        by_dataset[dataset_key] = {
            "dataset": report["dataset"],
            "sample_count": report["compare_summary"]["sample_count"],
            "models": model_table,
        }
        per_dataset_ranking[dataset_key] = ranking_rows

    metric_matrix: dict[str, dict[str, dict[str, float]]] = {
        metric_key: {} for metric_key in METRIC_KEYS
    }
    for report in dataset_reports:
        dataset_key = f"{report['dataset_name']}_{report['split']}"
        for metric_key in METRIC_KEYS:
            metric_matrix[metric_key][dataset_key] = {}
            for net in nets:
                model_name = net.name
                model_summary = report["compare_summary"]["models"][model_name]
                metric_matrix[metric_key][dataset_key][model_name] = model_summary[metric_key]["mean"]

    global_ranking: list[dict[str, float | str]] = []
    for model_name, stats in by_model.items():
        global_ranking.append(
            {
                "model": model_name,
                "ATE_mean(m)": stats["ATE(m)"]["mean"],
                "RTE_mean(m)": stats["RTE(m)"]["mean"],
                "position_rmse_mean(m)": stats["position_rmse(m)"]["mean"],
                "velocity_rmse_mean(m/s)": stats["velocity_rmse(m/s)"]["mean"],
                "heading_mae_mean(deg)": stats["heading_mae(deg)"]["mean"],
            }
        )
    global_ranking.sort(key=lambda x: float(x["ATE_mean(m)"]))

    cross_summary: dict[str, Any] = {
        "mode": "cross",
        "split": dap.args.split,
        "models": model_names,
        "model_count": len(model_names),
        "dataset_count": len(dataset_reports),
        "datasets": [
            {
                "dataset": report["dataset"],
                "dataset_name": report["dataset_name"],
                "res_root": report["res_root"],
            }
            for report in dataset_reports
        ],
        "by_model": by_model,
        "by_dataset": by_dataset,
        "metric_matrix_mean": metric_matrix,
        "global_ranking_by_ATE_mean": global_ranking,
        "dataset_ranking_by_ATE_mean": per_dataset_ranking,
    }

    summary_path = cross_root / "CrossModelDatasetSummary.json"
    ranking_path = cross_root / "GlobalRanking.txt"
    _save_json(summary_path, cross_summary)
    _save_ranking_text(ranking_path, global_ranking)

    print(f"跨模型+跨数据集汇总: {summary_path}", flush=True)
    print(f"全局排名文本: {ranking_path}", flush=True)
    print(f"完成，输出目录: {cross_root}", flush=True)


def main() -> None:
    """执行评估并输出汇总（交叉模式或配对模式）。"""
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models-a", nargs="+", help="配对模式：组A模型")
    dap.parser.add_argument("--models-b", nargs="+", help="配对模式：组B模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parser.add_argument("-c", "--config", type=str, help="配置文件")
    dap.parser.add_argument("--split", type=str, default="test", help="h5 split")
    dap.parser.add_argument("--datasets", nargs="+", help="多个数据集路径")
    dap.parser.add_argument("--datasets-a", nargs="+", help="配对模式：组A数据集")
    dap.parser.add_argument("--datasets-b", nargs="+", help="配对模式：组B数据集")
    dap.parser.add_argument("--no-plots", action="store_true", help="跳过绘图以提速/排障")
    dap.parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="推理设备：auto/cuda/cpu/mps",
    )
    dap.parse()
    eval_dir = Path(dap.output) if dap.output else Path("results")

    selected_device = base_device.set_default_device(dap.args.device)
    print(f"推理设备: {selected_device}", flush=True)

    paired_args = [
        dap.args.models_a,
        dap.args.models_b,
        dap.args.datasets_a,
        dap.args.datasets_b,
    ]
    has_paired_any = any(v is not None for v in paired_args)
    has_paired_all = all(v is not None for v in paired_args)

    if has_paired_any and not has_paired_all:
        raise ValueError(
            "配对模式参数需要同时提供：--models-a --models-b --datasets-a --datasets-b"
        )

    if has_paired_all:
        _run_paired_mode(dap, eval_dir, selected_device)
    else:
        _run_cross_mode(dap, eval_dir, selected_device)


if __name__ == "__main__":
    main()
