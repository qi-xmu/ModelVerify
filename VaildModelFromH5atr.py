#!/usr/bin/env python3
"""从 H5 数据集做模型验证并输出完整评估结果。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import base.rerun_ext as bre
from base.args_parser import DatasetArgsParser
from base.dataset import H5SplitDataset, H5UnitData
from base.datatype import UnitData
from base.draw.Bar import Bar
from base.draw.CDF import plot_one_cdf
from base.draw.Poses import draw_trajectory_2d_compare
from base.draw.Scatter import Scatter
from base.evaluate import get_cdf_from_err
from base.interpolate import get_time_series
from base.model import DataRunner, InertialNetworkData, ModelLoader, NetworkResult
from base.obj import Obj
from base.utils import angle_between_vectors, angle_with_x_axis


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _get_err_stats(errors: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(errors, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "rmse": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": _to_float(arr.size),
        "mean": _to_float(np.mean(arr)),
        "std": _to_float(np.std(arr)),
        "rmse": _to_float(np.sqrt(np.mean(arr**2))),
        "median": _to_float(np.percentile(arr, 50)),
        "p90": _to_float(np.percentile(arr, 90)),
        "p95": _to_float(np.percentile(arr, 95)),
        "p99": _to_float(np.percentile(arr, 99)),
        "min": _to_float(np.min(arr)),
        "max": _to_float(np.max(arr)),
    }


def _rmse_from_vectors(diff: np.ndarray) -> float:
    if diff.size == 0:
        return 0.0
    return _to_float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _pose_error_norm(
    ref_rot: Rotation,
    ref_pos: np.ndarray,
    eva_rot: Rotation,
    eva_pos: np.ndarray,
) -> tuple[float, float, float]:
    delta_rot = ref_rot.inv() * eva_rot
    rot_vec = delta_rot.as_rotvec()
    rot_err = float(np.linalg.norm(rot_vec))

    trans_global = np.asarray(eva_pos) - np.asarray(ref_pos)
    trans_local = ref_rot.inv().apply(trans_global)
    trans_err = float(np.linalg.norm(trans_local))

    se3_like = float(np.sqrt(trans_err**2 + rot_err**2))
    return trans_err, rot_err, se3_like


def _safe_eval_metrics(pred_poses, gt_poses, rel_duration: int = 1) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rate = int(min(max(pred_poses.rate, 1.0), max(gt_poses.rate, 1.0)))
    t_new_us = get_time_series([pred_poses.t_us, gt_poses.t_us], rate=rate)
    pred_i = pred_poses.interpolate(t_new_us)
    gt_i = gt_poses.interpolate(t_new_us)

    ape_vals = []
    ate_vals = []
    for i in range(len(t_new_us)):
        trans_err, _rot_err, se3_like = _pose_error_norm(
            gt_i.rots[i],
            gt_i.ps[i],
            pred_i.rots[i],
            pred_i.ps[i],
        )
        ate_vals.append(trans_err)
        ape_vals.append(se3_like)
    ape_errs = np.asarray(ape_vals, dtype=np.float64)
    ate_errs = np.asarray(ate_vals, dtype=np.float64)

    ape_cdf = get_cdf_from_err(ape_errs, "APE")
    ate_cdf = get_cdf_from_err(ate_errs, "ATE")

    size = len(ate_errs)
    gap = int(rate * rel_duration)
    rpe_errs = np.array([], dtype=np.float64)
    rte_errs = np.array([], dtype=np.float64)
    if size > gap:
        rpe_vals = []
        rte_vals = []
        for i in range(size - gap):
            ref_rel_rot = gt_i.rots[i].inv() * gt_i.rots[i + gap]
            eva_rel_rot = pred_i.rots[i].inv() * pred_i.rots[i + gap]

            ref_rel_pos = gt_i.rots[i].inv().apply(gt_i.ps[i + gap] - gt_i.ps[i])
            eva_rel_pos = pred_i.rots[i].inv().apply(pred_i.ps[i + gap] - pred_i.ps[i])

            trans_err, _rot_err, se3_like = _pose_error_norm(
                ref_rel_rot,
                ref_rel_pos,
                eva_rel_rot,
                eva_rel_pos,
            )
            rte_vals.append(trans_err)
            rpe_vals.append(se3_like)
        rte_errs = np.asarray(rte_vals, dtype=np.float64)
        rpe_errs = np.asarray(rpe_vals, dtype=np.float64)

    rpe_cdf = get_cdf_from_err(rpe_errs, "RPE") if len(rpe_errs) > 0 else {
        "tag": "RPE",
        "errors": np.array([], dtype=np.float64),
        "cdf": np.array([], dtype=np.float64),
        "percentiles": {"50%": 0.0, "90%": 0.0, "95%": 0.0, "99%": 0.0},
    }
    rte_cdf = get_cdf_from_err(rte_errs, "RTE") if len(rte_errs) > 0 else {
        "tag": "RTE",
        "errors": np.array([], dtype=np.float64),
        "cdf": np.array([], dtype=np.float64),
        "percentiles": {"50%": 0.0, "90%": 0.0, "95%": 0.0, "99%": 0.0},
    }

    eval_json = {
        "rate": rate,
        "size": int(size),
        "APE(rad_like)": _to_float(np.mean(ape_errs)) if len(ape_errs) else 0.0,
        "APE_CDF": ape_cdf["percentiles"],
        "ATE(m)": _to_float(np.mean(ate_errs)) if len(ate_errs) else 0.0,
        "ATE_CDF": ate_cdf["percentiles"],
        "RPE(rad_like)": _to_float(np.mean(rpe_errs)) if len(rpe_errs) else 0.0,
        "RPE_CDF": rpe_cdf["percentiles"],
        "RTE(m)": _to_float(np.mean(rte_errs)) if len(rte_errs) else 0.0,
        "RTE_CDF": rte_cdf["percentiles"],
    }
    return eval_json, {
        "APE_CDF": ape_cdf,
        "ATE_CDF": ate_cdf,
        "RPE_CDF": rpe_cdf,
        "RTE_CDF": rte_cdf,
    }


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda 但当前环境不可用 CUDA")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps 但当前环境不可用 MPS")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _calc_position_rmse(pred_poses, gt_poses) -> float:
    rate = int(min(max(pred_poses.rate, 1.0), max(gt_poses.rate, 1.0)))
    t_new_us = get_time_series([pred_poses.t_us, gt_poses.t_us], rate=rate)
    pred_i = pred_poses.interpolate(t_new_us)
    gt_i = gt_poses.interpolate(t_new_us)
    diff = pred_i.ps - gt_i.ps
    return _rmse_from_vectors(diff)


def _calc_unit_metrics(nr: NetworkResult, eval_main: dict[str, Any], gt_poses) -> dict[str, Any]:
    meas = np.asarray(nr.meas_list, dtype=np.float64)
    gt_disp = np.asarray(nr.gt_list, dtype=np.float64)

    vel_rmse = 0.0
    speed_rmse = 0.0
    heading_errors: list[float] = []
    heading_gt_angles: list[float] = []
    unc_norm_list: list[float] = []
    axis_err_abs_mean = [0.0, 0.0, 0.0]

    if meas.size > 0 and gt_disp.size > 0 and meas.shape == gt_disp.shape:
        vel_diff = meas - gt_disp
        vel_rmse = _rmse_from_vectors(vel_diff)

        meas_speed = np.linalg.norm(meas, axis=1)
        gt_speed = np.linalg.norm(gt_disp, axis=1)
        speed_rmse = _to_float(np.sqrt(np.mean((meas_speed - gt_speed) ** 2)))

        axis_err_abs_mean = np.mean(np.abs(vel_diff), axis=0).astype(float).tolist()

        for pred_v, gt_v in zip(meas, gt_disp):
            heading_gt_angles.append(_to_float(angle_with_x_axis(gt_v)))
            heading_errors.append(_to_float(angle_between_vectors(pred_v, gt_v)))

        cov = np.asarray(nr.meas_cov_list, dtype=np.float64)
        if cov.ndim == 2 and cov.shape[1] >= 3 and len(cov) == len(meas):
            cov_var = np.exp(cov[:, :3])
            unc_norm = np.linalg.norm(cov_var, axis=1)
            unc_norm_list = unc_norm.astype(float).tolist()

    ate = _to_float(eval_main.get("ATE(m)", 0.0))
    rte = _to_float(eval_main.get("RTE(m)", 0.0))
    ape = _to_float(eval_main.get("APE(rad_like)", 0.0))
    rpe = _to_float(eval_main.get("RPE(rad_like)", 0.0))

    heading_arr = np.asarray(heading_errors, dtype=np.float64)
    heading_mae = _to_float(np.mean(np.abs(heading_arr))) if heading_arr.size else 0.0
    heading_rmse = _to_float(np.sqrt(np.mean(heading_arr**2))) if heading_arr.size else 0.0

    return {
        "ATE(m)": ate,
        "RTE(m)": rte,
        "APE(rad_like)": ape,
        "RPE(rad_like)": rpe,
        "trajectory_mean_error_ATE(m)": ate,
        "position_rmse(m)": _calc_position_rmse(nr.poses, gt_poses),
        "velocity_rmse(m/s)": vel_rmse,
        "speed_rmse(m/s)": speed_rmse,
        "heading_mae(deg)": heading_mae,
        "heading_rmse(deg)": heading_rmse,
        "heading_errors(deg)": heading_errors,
        "heading_gt_angles(deg)": heading_gt_angles,
        "uncertainty_norm": unc_norm_list,
        "axis_abs_err_mean(m/s)": {
            "x": _to_float(axis_err_abs_mean[0]),
            "y": _to_float(axis_err_abs_mean[1]),
            "z": _to_float(axis_err_abs_mean[2]),
        },
        "latency_mean(s)": _to_float(np.mean(nr.eval_t_list)) if len(nr.eval_t_list) else 0.0,
        "latency_p95(s)": _to_float(np.percentile(nr.eval_t_list, 95)) if len(nr.eval_t_list) else 0.0,
    }


def _draw_compare_cdf(model_errors: dict[str, list[float]], save_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for model_name, errors in model_errors.items():
        if len(errors) == 0:
            continue
        sorted_err = np.sort(np.asarray(errors, dtype=np.float64))
        cdf = np.arange(1, len(sorted_err) + 1, dtype=np.float64) / len(sorted_err)
        plt.plot(sorted_err, cdf, label=model_name)

    plt.xlabel("Error (m/s)")
    plt.ylabel("CDF")
    plt.title("Model Error CDF Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _save_ranking_text(res_root: Path, ranking: list[dict[str, float | str]]) -> Path:
    lines = []
    lines.append("=" * 70)
    lines.append("Ranking by ATE mean")
    lines.append("=" * 70)
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
    lines.append("=" * 70)
    lines.append("")
    ranking_txt = res_root / "Ranking.txt"
    ranking_txt.write_text("\n".join(lines), encoding="utf-8")
    return ranking_txt


def main() -> None:
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parser.add_argument("-c", "--config", type=str, help="配置文件")
    dap.parser.add_argument("--split", type=str, default="test", help="h5 split")
    dap.parser.add_argument("--step", type=int, default=20, help="推理步长")
    dap.parser.add_argument("--no-plots", action="store_true", help="跳过绘图以提速")
    dap.parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="推理设备",
    )
    dap.parse()

    model_names = dap.args.models or ["model_resnet_0111_96"]
    models_path = dap.args.models_path or "models"
    eval_dir = Path(dap.output) if dap.output else Path("results")

    loader = ModelLoader(models_path)
    data_cfg = InertialNetworkData.set_step(dap.args.step).no_rerun()
    nets = loader.get_by_names(model_names)

    infer_device = _resolve_device(dap.args.device)
    print(f"[Device] Use device: {infer_device}", flush=True)
    for net in nets:
        net.device = infer_device
        net.device_name = str(infer_device)
        net.model = net.model.to(infer_device)
        net.warmup()

    assert dap.dataset is not None, "Dataset path is not provided"
    dataset_path = Path(dap.dataset)
    datas = H5SplitDataset(dataset_path, split=dap.args.split)

    model_tag = "_VS_".join([n.name for n in nets])
    res_root = eval_dir / f"{datas.name}_{dap.args.split}" / model_tag
    res_root.mkdir(parents=True, exist_ok=True)

    if dap.unit:
        selected_names = [n for n in datas.sequence_names if dap.unit in n]
        if len(selected_names) == 0:
            raise ValueError(f"No sequence name contains: {dap.unit}")
    else:
        selected_names = list(datas.sequence_names)

    model_error_pool: dict[str, list[float]] = {n.name: [] for n in nets}
    model_heading_pool: dict[str, dict[str, list[float]]] = {
        n.name: {"angle": [], "error": []} for n in nets
    }
    model_unc_pool: dict[str, dict[str, list[float]]] = {
        n.name: {"unc": [], "err": []} for n in nets
    }
    model_unit_metrics: dict[str, list[dict[str, Any]]] = {n.name: [] for n in nets}

    def run_one_unit(
        ud: UnitData,
        sample_index: int,
        sample_total: int,
    ) -> dict[str, tuple[NetworkResult, dict[str, Any], dict[str, Any]]]:
        started = time.perf_counter()
        print(f"[START] {sample_index}/{sample_total} {ud.name}", flush=True)

        out_unit_root = res_root / "units" / ud.name
        out_unit_root.mkdir(parents=True, exist_ok=True)

        cache_dir = res_root / "temp"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_map = {net.name: cache_dir / f"{net.name}_{ud.name}.pkl" for net in nets}
        result_map: dict[str, tuple[NetworkResult, dict[str, Any], dict[str, Any]]] = {}

        if all(path.exists() for path in cache_map.values()):
            for net in nets:
                nr, eval_json, unit_metrics = Obj.load(cache_map[net.name])
                result_map[net.name] = (nr, eval_json, unit_metrics)
            elapsed = time.perf_counter() - started
            print(f"[CACHE] {sample_index}/{sample_total} {ud.name} ({elapsed:.2f}s)", flush=True)
            return result_map

        ud.load_data(using_opt=True)
        if dap.args.config:
            print(f"> 加载配置文件：{dap.args.config}", flush=True)
            ud.imu_data.calibrate_with(dap.args.config)

        if data_cfg.using_rerun:
            bre.RerunView().add_spatial_view().send(ud.name)
            bre.send_pose_data(ud.gt_data, "Groundtruth", color=[192, 72, 72])

        dr = DataRunner(ud, data_cfg, has_init_rerun=True)
        nr_list = dr.predict_batch(nets)

        unit_pred_poses = []
        unit_labels = []
        for net, nr in zip(nets, nr_list):
            eval_json, eval_inner = _safe_eval_metrics(nr.poses, ud.gt_data, rel_duration=1)

            unit_model_dir = out_unit_root / net.name
            unit_model_dir.mkdir(parents=True, exist_ok=True)

            _save_json(unit_model_dir / "Eval.json", eval_json)

            if not dap.args.no_plots:
                plot_one_cdf(eval_inner["ATE_CDF"], unit_model_dir / "CDF.png", show=False)
                draw_trajectory_2d_compare(
                    [nr.poses, ud.gt_data],
                    labels=[net.name, "GT"],
                    title=f"Trajectory_{net.name}_{ud.name}",
                    save_path=unit_model_dir / "Trajectory.png",
                    show=False,
                )
                if len(nr.eval_t_list) > 0:
                    Bar(
                        x=None,
                        y=nr.eval_t_list,
                        x_label="x",
                        y_label="Time(s)",
                        title=f"Inference Latency with {nr.network_device_name}",
                    ).save(unit_model_dir)

            unit_metrics = _calc_unit_metrics(nr, eval_json, ud.gt_data)
            _save_json(unit_model_dir / "Metrics.json", unit_metrics)

            result_map[net.name] = (nr, eval_json, unit_metrics)
            Obj.save((nr, eval_json, unit_metrics), cache_map[net.name])

            unit_pred_poses.append(nr.poses)
            unit_labels.append(net.name)

        if len(unit_pred_poses) > 1 and not dap.args.no_plots:
            draw_trajectory_2d_compare(
                [ud.gt_data] + unit_pred_poses,
                labels=["GT"] + unit_labels,
                title=f"Trajectory_Compare_{ud.name}",
                save_path=out_unit_root / "Trajectory_Compare.png",
                show=False,
            )

        elapsed = time.perf_counter() - started
        print(f"[END] {sample_index}/{sample_total} {ud.name} ({elapsed:.2f}s)", flush=True)
        return result_map

    total = len(selected_names)
    for index, seq_name in enumerate(tqdm(selected_names, desc="Evaluating"), start=1):
        sequence = datas.get_sequence(seq_name)
        assert sequence is not None
        ud = H5UnitData(sequence)

        res_map = run_one_unit(ud, index, total)

        for net in nets:
            model_name = net.name
            nr, _eval_json, unit_metrics = res_map[model_name]

            meas = np.asarray(nr.meas_list, dtype=np.float64)
            gt = np.asarray(nr.gt_list, dtype=np.float64)
            if meas.size > 0 and gt.size > 0 and meas.shape == gt.shape:
                err_norm = np.linalg.norm(meas - gt, axis=1)
                model_error_pool[model_name].extend(err_norm.tolist())

                model_heading_pool[model_name]["error"].extend(unit_metrics["heading_errors(deg)"])
                model_heading_pool[model_name]["angle"].extend(unit_metrics["heading_gt_angles(deg)"])

                unc_norm = unit_metrics["uncertainty_norm"]
                if len(unc_norm) == len(err_norm):
                    model_unc_pool[model_name]["unc"].extend(unc_norm)
                    model_unc_pool[model_name]["err"].extend(err_norm.tolist())

            model_unit_metrics[model_name].append(unit_metrics)

    compare_summary: dict[str, Any] = {
        "dataset": str(dataset_path),
        "split": dap.args.split,
        "model_count": len(nets),
        "sample_count": total,
        "eval_config": {
            "device": str(infer_device),
            "step": dap.args.step,
            "no_plots": dap.args.no_plots,
            "unit_filter": dap.unit,
        },
        "models": {},
    }

    for net in nets:
        model_name = net.name
        model_dir = res_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        all_errors = model_error_pool[model_name]
        if len(all_errors) > 0 and not dap.args.no_plots:
            model_cdf = get_cdf_from_err(all_errors, model_name)
            plot_one_cdf(model_cdf, model_dir / "CDF.png", show=False)

        angle_list = model_heading_pool[model_name]["angle"]
        angle_error_list = model_heading_pool[model_name]["error"]
        if not dap.args.no_plots and len(angle_list) > 5 and len(angle_list) == len(angle_error_list):
            Scatter(
                x=angle_list,
                y=angle_error_list,
                title=f"Heading Error vs Heading - {model_name}",
                x_label="Heading(deg)",
                y_label="Error(deg)",
            ).save(model_dir)

        unc_list = model_unc_pool[model_name]["unc"]
        unc_err_list = model_unc_pool[model_name]["err"]
        unc_corr = 0.0
        if len(unc_list) > 5 and len(unc_list) == len(unc_err_list):
            if np.std(unc_list) > 1e-12 and np.std(unc_err_list) > 1e-12:
                unc_corr = _to_float(np.corrcoef(unc_list, unc_err_list)[0, 1])
            if not dap.args.no_plots:
                Scatter(
                    x=unc_list,
                    y=unc_err_list,
                    title=f"Uncertainty vs Error - {model_name}",
                    x_label="Uncertainty(norm)",
                    y_label="Error(norm)",
                ).save(model_dir)

        units_metrics = model_unit_metrics[model_name]
        summary = {
            "model": model_name,
            "samples": len(units_metrics),
            "position_rmse(m)": _get_err_stats([m["position_rmse(m)"] for m in units_metrics]),
            "velocity_rmse(m/s)": _get_err_stats([m["velocity_rmse(m/s)"] for m in units_metrics]),
            "speed_rmse(m/s)": _get_err_stats([m["speed_rmse(m/s)"] for m in units_metrics]),
            "heading_mae(deg)": _get_err_stats([m["heading_mae(deg)"] for m in units_metrics]),
            "heading_rmse(deg)": _get_err_stats([m["heading_rmse(deg)"] for m in units_metrics]),
            "APE(rad_like)": _get_err_stats([m["APE(rad_like)"] for m in units_metrics]),
            "RPE(rad_like)": _get_err_stats([m["RPE(rad_like)"] for m in units_metrics]),
            "ATE(m)": _get_err_stats([m["ATE(m)"] for m in units_metrics]),
            "RTE(m)": _get_err_stats([m["RTE(m)"] for m in units_metrics]),
            "latency_mean(s)": _get_err_stats([m["latency_mean(s)"] for m in units_metrics]),
            "disp_error_norm(m/s)": _get_err_stats(all_errors),
            "uncertainty_error_corr": unc_corr,
        }
        _save_json(model_dir / "Summary.json", summary)
        compare_summary["models"][model_name] = summary

    if len(nets) > 1 and not dap.args.no_plots:
        _draw_compare_cdf(model_error_pool, res_root / "CDF_Compare.png")

    ranking = []
    for model_name, summary in compare_summary["models"].items():
        ranking.append(
            {
                "model": model_name,
                "ATE_mean(m)": summary["ATE(m)"]["mean"],
                "RTE_mean(m)": summary["RTE(m)"]["mean"],
                "position_rmse_mean(m)": summary["position_rmse(m)"]["mean"],
                "velocity_rmse_mean(m/s)": summary["velocity_rmse(m/s)"]["mean"],
                "heading_mae_mean(deg)": summary["heading_mae(deg)"]["mean"],
            }
        )
    ranking.sort(key=lambda x: x["ATE_mean(m)"])
    compare_summary["ranking_by_ATE_mean"] = ranking

    compare_summary["mean_metrics_table"] = {
        model_name: {
            "ATE_mean(m)": summary["ATE(m)"]["mean"],
            "RTE_mean(m)": summary["RTE(m)"]["mean"],
            "APE_mean(rad_like)": summary["APE(rad_like)"]["mean"],
            "RPE_mean(rad_like)": summary["RPE(rad_like)"]["mean"],
            "position_rmse_mean(m)": summary["position_rmse(m)"]["mean"],
            "velocity_rmse_mean(m/s)": summary["velocity_rmse(m/s)"]["mean"],
            "speed_rmse_mean(m/s)": summary["speed_rmse(m/s)"]["mean"],
            "heading_mae_mean(deg)": summary["heading_mae(deg)"]["mean"],
            "heading_rmse_mean(deg)": summary["heading_rmse(deg)"]["mean"],
            "latency_mean(s)": summary["latency_mean(s)"]["mean"],
        }
        for model_name, summary in compare_summary["models"].items()
    }
    compare_summary["mean"] = compare_summary["mean_metrics_table"]

    ranking_txt = _save_ranking_text(res_root, ranking)
    mean_json = res_root / "mean.json"
    _save_json(
        mean_json,
        {
            "dataset": str(dataset_path),
            "split": dap.args.split,
            "sample_count": total,
            "mean": compare_summary["mean_metrics_table"],
        },
    )

    _save_json(res_root / "CompareSummary.json", compare_summary)
    print(f"Ranking text: {ranking_txt}", flush=True)
    print(f"Mean json   : {mean_json}", flush=True)
    print(f"Compare summary: {res_root / 'CompareSummary.json'}", flush=True)
    print(f"验证完成，结果目录: {res_root}", flush=True)


if __name__ == "__main__":
    main()
