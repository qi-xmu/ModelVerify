#!/usr/bin/env python3
"""从 H5 数据集做模型验证并输出完整评估结果。


支持：
- 单模型/多模型同轮验证多h5个数据集
- 每个样本轨迹图（GT vs Pred）与多模型轨迹对比
- 位置 RMSE、速度 RMSE、航向误差（deg）、ATE/RTE（含 APE/RPE）
- 模型级 CDF、数据集级汇总、多模型 CDF 对比
- 不确定性与误差关系分析（基于 meas_cov）
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import base.rerun_ext as bre
from base import device as base_device
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

def evaluate_from_unitdata(unitdata_list, nets, data_cfg, output_dir, no_plots, max_samples=None):
    """
    入口函数：支持直接传入UnitData列表进行评估，兼容TLIO格式。
    参数：
        unitdata_list: List[UnitData]，每个单元的数据
        nets: 模型列表
        data_cfg: InertialNetworkData配置
        output_dir: 输出目录（Path对象）
        no_plots: 是否跳过绘图
        max_samples: 最大样本数（用于测试，None表示使用全部数据）
    """
    import numpy as np
    from tqdm import tqdm
    from base.datatype import ImuData, PosesData
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_error_pool = {n.name: [] for n in nets}
    model_unit_metrics = {n.name: [] for n in nets}
    model_heading_pool = {n.name: {"angle": [], "error": []} for n in nets}
    print(f"[DEBUG] 开始处理 {len(unitdata_list)} 个单元", flush=True)
    for idx, ud in enumerate(unitdata_list, start=1):
        print(f"[DEBUG] 处理单元 {idx}/{len(unitdata_list)}: {ud.name}", flush=True)
        
        # 如果指定了最大样本数，截取数据
        if max_samples is not None and ud.imu_data is not None:
            import copy
            ud_copy = copy.copy(ud)
            if len(ud_copy.imu_data.t_us) > max_samples:
                print(f"[DEBUG] 截取数据到前 {max_samples} 个样本", flush=True)
                # 截取 IMU 数据
                t_us = ud_copy.imu_data.t_us[:max_samples]
                gyro = ud_copy.imu_data.gyro[:max_samples]
                acce = ud_copy.imu_data.acce[:max_samples]
                ahrs = ud_copy.imu_data.ahrs[:max_samples]
                mag = ud_copy.imu_data.magn[:max_samples] if ud_copy.imu_data.magn is not None else np.zeros_like(acce)
                ud_copy.imu_data = ImuData(t_us, gyro, acce, ahrs, mag, frame=ud_copy.imu_data.frame)
                
                # 截取 GT 数据
                if ud_copy.gt_data is not None:
                    gt_t_us = ud_copy.gt_data.t_us[:max_samples]
                    gt_rots = ud_copy.gt_data.rots[:max_samples]
                    gt_ps = ud_copy.gt_data.ps[:max_samples]
                    ud_copy.gt_data = PosesData(gt_t_us, gt_rots, gt_ps)
            ud = ud_copy
        
        out_unit_root = output_dir / "units" / Path(ud.name).name
        out_unit_root.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] 创建输出目录: {out_unit_root}", flush=True)
        # 完全禁用Rerun
        data_cfg_no_rerun = data_cfg.no_rerun()
        dr = DataRunner(ud, data_cfg_no_rerun, has_init_rerun=False, visual=False)
        print(f"[DEBUG] 开始推理...", flush=True)
        nr_list = dr.predict_batch(nets)
        print(f"[DEBUG] 推理完成，得到 {len(nr_list)} 个结果", flush=True)
        unit_pred_poses = []
        unit_labels = []
        for net, nr in zip(nets, nr_list):
            try:
                from base.evaluate import Evaluation
                eva = Evaluation(ud.gt_data)
                eval_json, eval_inner = eva.get_eval(nr.poses, net.name)
                unit_model_dir = out_unit_root / net.name
                unit_model_dir.mkdir(parents=True, exist_ok=True)
                if not no_plots:
                    try:
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
                    except Exception as e:
                        print(f"[WARN] 绘图失败: {e}", flush=True)
                unit_metrics = {
                    "ATE(m)": eval_json.get("ATE(m)", 0.0),
                    "trajectory_mean_error_ATE(m)": eval_json.get("ATE(m)", 0.0),
                }
                _save_json(unit_model_dir / "Metrics.json", unit_metrics)
                model_error_pool[net.name].extend(eval_inner["ATE_CDF"]["errors"].tolist())
                model_unit_metrics[net.name].append(unit_metrics)
                unit_pred_poses.append(nr.poses)
                unit_labels.append(net.name)
            except Exception as e:
                print(f"[ERROR] 评估模型 {net.name} 失败: {e}", flush=True)
                import traceback
                traceback.print_exc()
        if len(unit_pred_poses) > 1 and not no_plots:
            draw_trajectory_2d_compare(
                [ud.gt_data] + unit_pred_poses,
                labels=["GT"] + unit_labels,
                title=f"Trajectory_Compare_{Path(ud.name).name}",
                save_path=out_unit_root / "Trajectory_Compare.png",
                show=False,
            )
        if hasattr(dr, 'cleanup'):
            dr.cleanup()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import matplotlib.pyplot as plt
        plt.close('all')
    compare_summary = {
        "unit_count": len(unitdata_list),
        "model_count": len(nets),
        "models": [n.name for n in nets],
        "model_errors": {k: v for k, v in model_error_pool.items()},
        "per_unit_metrics": model_unit_metrics,
    }
    if not no_plots:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        for name, errors in model_error_pool.items():
            if len(errors) == 0:
                continue
            arr = np.sort(np.asarray(errors, dtype=np.float64))
            cdf = np.arange(1, len(arr) + 1, dtype=np.float64) / len(arr)
            plt.plot(arr, cdf, label=name)
        plt.xlabel("ATE (m)")
        plt.ylabel("CDF")
        plt.title("Model ATE CDF Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        (output_dir / "compare").mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "compare" / "ATE_CDF_Compare.png", dpi=150)
        plt.close()
    _save_json(output_dir / "CompareSummary.json", compare_summary)
    print(f"全部评估结果已保存到 {output_dir / 'CompareSummary.json'}", flush=True)

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


def _draw_metric_cdf_by_group(
    metric_by_group: dict[str, list[float]],
    save_path: Path,
    title: str,
    x_label: str,
) -> None:
    plt.figure(figsize=(8, 6))
    for group_name, values in metric_by_group.items():
        if len(values) == 0:
            continue
        arr = np.sort(np.asarray(values, dtype=np.float64))
        cdf = np.arange(1, len(arr) + 1, dtype=np.float64) / len(arr)
        plt.plot(arr, cdf, label=group_name)

    plt.xlabel(x_label)
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _draw_metric_pde_by_group(
    metric_by_group: dict[str, list[float]],
    save_path: Path,
    title: str,
    x_label: str,
    bins: int = 50,
) -> None:
    plt.figure(figsize=(8, 6))
    for group_name, values in metric_by_group.items():
        if len(values) < 2:
            continue
        arr = np.asarray(values, dtype=np.float64)
        hist, edges = np.histogram(arr, bins=bins, density=True)
        x_centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(x_centers, hist, label=group_name)

    plt.xlabel(x_label)
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _draw_metric_boxplot_by_group(
    metric_by_group: dict[str, list[float]],
    save_path: Path,
    title: str,
    y_label: str,
) -> None:
    valid_items = [(name, values) for name, values in metric_by_group.items() if len(values) > 0]
    if len(valid_items) == 0:
        return

    labels = [it[0] for it in valid_items]
    data = [np.asarray(it[1], dtype=np.float64) for it in valid_items]

    plt.figure(figsize=(max(8, len(labels) * 1.2), 6))
    try:
        plt.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def _safe_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ["_", "-", "."]) else "_" for ch in name)


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


def _safe_eval_metrics(
    pred_poses,
    gt_poses,
    rel_duration: int = 1,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
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


def _calc_position_rmse(pred_poses, gt_poses) -> float:
    rate = int(min(max(pred_poses.rate, 1.0), max(gt_poses.rate, 1.0)))
    t_new_us = get_time_series([pred_poses.t_us, gt_poses.t_us], rate=rate)
    pred_i = pred_poses.interpolate(t_new_us)
    gt_i = gt_poses.interpolate(t_new_us)
    diff = pred_i.ps - gt_i.ps
    return _rmse_from_vectors(diff)


def _calc_unit_metrics(
    nr: NetworkResult,
    eval_main: dict[str, Any],
    gt_poses,
) -> dict[str, Any]:
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
    ape = _to_float(eval_main.get("APE(rad_like)", eval_main.get("APE(_)", 0.0)))
    rpe = _to_float(eval_main.get("RPE(rad_like)", eval_main.get("RPE(_)", 0.0)))

    heading_arr = np.asarray(heading_errors, dtype=np.float64)
    heading_mae = _to_float(np.mean(np.abs(heading_arr))) if heading_arr.size else 0.0
    heading_rmse = _to_float(np.sqrt(np.mean(heading_arr**2))) if heading_arr.size else 0.0

    return {
        "ATE(m)": ate,
        "RTE(m)": rte,
        "APE(rad)": ape,
        "RPE(rad)": rpe,
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


def _evaluate_single_dataset(
    dataset_path: Path,
    split: str,
    nets,
    data_cfg: InertialNetworkData,
    unit_filter: str | None,
    config_path: str | None,
    dataset_out_root: Path,
    no_plots: bool,
) -> dict[str, Any]:
    datas = H5SplitDataset(dataset_path, split=split)
    model_tag = "_VS_".join([n.name for n in nets])
    res_root = dataset_out_root / f"{datas.name}_{split}" / model_tag
    res_root.mkdir(parents=True, exist_ok=True)

    if unit_filter:
        selected_names = [n for n in datas.sequence_names if unit_filter in n]
        if len(selected_names) == 0:
            raise ValueError(f"No sequence name contains: {unit_filter}")
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
        if config_path:
            print(f"> 加载配置文件：{config_path}", flush=True)
            ud.imu_data.calibrate_with(config_path)

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

            if not no_plots:
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

        if len(unit_pred_poses) > 1 and not no_plots:
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
    for index, seq_name in enumerate(tqdm(selected_names, desc=f"Evaluating {datas.name}"), start=1):
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

                heading_errs = unit_metrics["heading_errors(deg)"]
                heading_angs = unit_metrics["heading_gt_angles(deg)"]
                model_heading_pool[model_name]["error"].extend(heading_errs)
                model_heading_pool[model_name]["angle"].extend(heading_angs)

                unc_norm = unit_metrics["uncertainty_norm"]
                if len(unc_norm) == len(err_norm):
                    model_unc_pool[model_name]["unc"].extend(unc_norm)
                    model_unc_pool[model_name]["err"].extend(err_norm.tolist())

            model_unit_metrics[model_name].append(unit_metrics)

    compare_summary: dict[str, Any] = {
        "dataset": str(dataset_path),
        "dataset_name": datas.name,
        "split": split,
        "model_count": len(nets),
        "sample_count": total,
        "models": {},
    }

    dataset_model_metric_lists: dict[str, dict[str, list[float]]] = {}
    for net in nets:
        model_name = net.name
        model_dir = res_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        all_errors = model_error_pool[model_name]
        if len(all_errors) > 0 and not no_plots:
            model_cdf = get_cdf_from_err(all_errors, model_name)
            plot_one_cdf(model_cdf, model_dir / "CDF.png", show=False)

        angle_list = model_heading_pool[model_name]["angle"]
        angle_error_list = model_heading_pool[model_name]["error"]
        if (not no_plots) and len(angle_list) > 5 and len(angle_list) == len(angle_error_list):
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
            if not no_plots:
                Scatter(
                    x=unc_list,
                    y=unc_err_list,
                    title=f"Uncertainty vs Error - {model_name}",
                    x_label="Uncertainty(norm)",
                    y_label="Error(norm)",
                ).save(model_dir)

        units_metrics = model_unit_metrics[model_name]
        pos_rmse_list = [m["position_rmse(m)"] for m in units_metrics]
        vel_rmse_list = [m["velocity_rmse(m/s)"] for m in units_metrics]
        heading_mae_list = [m["heading_mae(deg)"] for m in units_metrics]
        ate_list = [m["ATE(m)"] for m in units_metrics]
        rte_list = [m["RTE(m)"] for m in units_metrics]

        summary = {
            "model": model_name,
            "samples": len(units_metrics),
            "position_rmse(m)": _get_err_stats(pos_rmse_list),
            "velocity_rmse(m/s)": _get_err_stats(vel_rmse_list),
            "heading_mae(deg)": _get_err_stats(heading_mae_list),
            "ATE(m)": _get_err_stats(ate_list),
            "RTE(m)": _get_err_stats(rte_list),
            "disp_error_norm(m/s)": _get_err_stats(all_errors),
            "uncertainty_error_corr": unc_corr,
        }
        _save_json(model_dir / "Summary.json", summary)
        compare_summary["models"][model_name] = summary

        dataset_model_metric_lists[model_name] = {
            "ATE(m)": ate_list,
            "RTE(m)": rte_list,
            "position_rmse(m)": pos_rmse_list,
            "velocity_rmse(m/s)": vel_rmse_list,
            "heading_mae(deg)": heading_mae_list,
            "disp_error_norm(m/s)": all_errors,
        }

    if len(nets) > 1 and not no_plots:
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

    _save_json(res_root / "CompareSummary.json", compare_summary)
    print(f"数据集验证完成，结果目录: {res_root}", flush=True)

    return {
        "dataset": str(dataset_path),
        "dataset_name": datas.name,
        "split": split,
        "res_root": str(res_root),
        "compare_summary": compare_summary,
        "model_metric_lists": dataset_model_metric_lists,
    }


def main() -> None:
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parser.add_argument("-c", "--config", type=str, help="配置文件")
    dap.parser.add_argument("--split", type=str, default="test", help="h5 split")
    dap.parser.add_argument("--datasets", nargs="+", help="多个数据集路径")
    dap.parser.add_argument("--no-plots", action="store_true", help="跳过绘图以提速/排障")
    dap.parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="推理设备：auto/cuda/cpu/mps",
    )
    dap.parse()

    model_names = dap.args.models or ["model_resnet_0111_96"]
    models_path = dap.args.models_path or "models"
    eval_dir = Path(dap.output) if dap.output else Path("results")

    selected_device = base_device.set_default_device(dap.args.device)
    print(f"推理设备: {selected_device}", flush=True)

    loader = ModelLoader(models_path)
    data_cfg = InertialNetworkData.set_step(20).no_rerun()
    nets = loader.get_by_names(model_names)
    for net in nets:
        net.set_device(selected_device)
        net.warmup()
    print("已迁移模型到推理设备并完成 warmup", flush=True)

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

    if len(dataset_reports) == 1:
        print(f"验证完成，结果目录: {dataset_reports[0]['res_root']}", flush=True)
        return

    cross_root = eval_dir / "cross_dataset" / f"{dap.args.split}_{model_tag}"
    cross_root.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        "ATE(m)",
        "RTE(m)",
        "position_rmse(m)",
        "velocity_rmse(m/s)",
        "heading_mae(deg)",
        "disp_error_norm(m/s)",
    ]

    cross_summary: dict[str, Any] = {
        "split": dap.args.split,
        "models": model_names,
        "dataset_count": len(dataset_reports),
        "datasets": [
            {
                "dataset": report["dataset"],
                "dataset_name": report["dataset_name"],
                "res_root": report["res_root"],
            }
            for report in dataset_reports
        ],
        "by_model": {},
    }

    for net in nets:
        model_name = net.name
        model_cross_dir = cross_root / model_name
        model_cross_dir.mkdir(parents=True, exist_ok=True)

        metric_by_dataset: dict[str, dict[str, list[float]]] = {
            key: {} for key in metric_keys
        }
        dataset_table: dict[str, dict[str, Any]] = {}

        for report in dataset_reports:
            dataset_name = report["dataset_name"]
            dataset_key = f"{dataset_name}_{report['split']}"
            metric_lists = report["model_metric_lists"][model_name]
            for metric_key in metric_keys:
                metric_by_dataset[metric_key][dataset_key] = metric_lists[metric_key]

            model_summary = report["compare_summary"]["models"][model_name]
            dataset_table[dataset_key] = {
                "dataset": report["dataset"],
                "samples": model_summary["samples"],
                "ATE_mean(m)": model_summary["ATE(m)"]["mean"],
                "RTE_mean(m)": model_summary["RTE(m)"]["mean"],
                "position_rmse_mean(m)": model_summary["position_rmse(m)"]["mean"],
                "velocity_rmse_mean(m/s)": model_summary["velocity_rmse(m/s)"]["mean"],
                "heading_mae_mean(deg)": model_summary["heading_mae(deg)"]["mean"],
            }

        model_metric_overall: dict[str, Any] = {}
        for metric_key in metric_keys:
            merged_values: list[float] = []
            for values in metric_by_dataset[metric_key].values():
                merged_values.extend(values)

            metric_tag = _safe_name(metric_key)
            if not dap.args.no_plots:
                _draw_metric_cdf_by_group(
                    metric_by_group=metric_by_dataset[metric_key],
                    save_path=model_cross_dir / f"{metric_tag}_CDF_by_dataset.png",
                    title=f"{metric_key} CDF by Dataset - {model_name}",
                    x_label=metric_key,
                )
                _draw_metric_pde_by_group(
                    metric_by_group=metric_by_dataset[metric_key],
                    save_path=model_cross_dir / f"{metric_tag}_PDE_by_dataset.png",
                    title=f"{metric_key} PDE by Dataset - {model_name}",
                    x_label=metric_key,
                )
                _draw_metric_boxplot_by_group(
                    metric_by_group=metric_by_dataset[metric_key],
                    save_path=model_cross_dir / f"{metric_tag}_Boxplot_by_dataset.png",
                    title=f"{metric_key} Boxplot by Dataset - {model_name}",
                    y_label=metric_key,
                )
            model_metric_overall[metric_key] = {
                "overall": _get_err_stats(merged_values),
                "by_dataset": {
                    dataset_key: _get_err_stats(values)
                    for dataset_key, values in metric_by_dataset[metric_key].items()
                },
            }

        cross_summary["by_model"][model_name] = {
            "dataset_table": dataset_table,
            "metrics": model_metric_overall,
        }

    _save_json(cross_root / "CrossDatasetSummary.json", cross_summary)
    print(f"多数据集验证完成，跨数据集结果目录: {cross_root}", flush=True)


if __name__ == "__main__":
    main()
