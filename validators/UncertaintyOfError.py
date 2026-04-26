#!/usr/bin/env python3
"""
不确定性误差分析

分析神经网络预测的不确定性（不确定性估计）与实际误差之间的关系。

主要功能：
1. 对3轴数据（X、Y、Z）进行不确定性-误差分析
2. 计算皮尔逊相关系数评估不确定性与误差的相关性
3. 可视化展示不确定性与误差的散点图及统计信息

使用方法：
    uv run python validators/UncertaintyOfError.py -m <模型名称> --dataset <数据集路径>
    uv run python validators/UncertaintyOfError.py -m <模型名称> --unit <单个数据路径>

    可用数据集 /Users/qi/Resources/实验数据整理/TestDataset2/Redmi-K30-Pro

输出：
    - covariance_error_analysis_3axis.png: 3轴不确定性-误差分析可视化图
    - 结果保存在 results/<模型名>_<设备名>/ 目录下
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

import base.rerun_ext as bre
from base.args_parser import DatasetArgsParser
from base.datatype import DeviceDataset, UnitData
from base.evaluate import Evaluation
from base.model import DataRunner, InertialNetworkData, ModelLoader, NetworkResult
from base.obj import Obj

# 设置中文字体为宋体
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["SimSun", "Songti SC", "STSong"]
mpl.rcParams["axes.unicode_minus"] = False


class NetworkResultAnalysis:
    def __init__(self, nr: NetworkResult):  # noqa: F821
        self.nr = nr
        self.name = self.nr.tag

    def analyze_uc_err(self):
        """
        Analyze the relationship between covariance and error for 3-axis data

        meas_cov_list: Model predicted covariance (uncertainty estimation) - 3x3 covariance matrices
        meas_list: Model predictions - 3D vectors (x, y, z)
        gt_list: Ground truth displacements - 3D vectors (x, y, z)
        """
        # Extract 3-axis data
        measurements = np.array(self.nr.meas_list)  # (N, 3)
        ground_truth = np.array(self.nr.gt_list)  # (N, 3)
        errors = measurements - ground_truth  # (N, 3) - error for each axis

        # Extract covariance matrices (diagonal elements for each axis)
        meas_cov = np.array(self.nr.meas_cov_list)
        # 设置xyz的最小值为-4
        # meas_cov = np.maximum(meas_cov, -4)
        # meas_cov = np.exp(meas_cov)

        cov_x = meas_cov[:, 0]
        cov_y = meas_cov[:, 1]
        cov_z = meas_cov[:, 2]

        # Extract errors for each axis
        errors = np.abs(errors)
        err_x = errors[:, 0]  # Error in x
        err_y = errors[:, 1]  # Error in y
        err_z = errors[:, 2]  # Error in z

        # Create figure with 3 subplots (one for each axis)
        _fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        axis_labels = ["X轴", "Y轴", "Z轴"]
        cov_data = [cov_x, cov_y, cov_z]
        err_data = [err_x, err_y, err_z]
        colors = ["red", "green", "blue"]

        for i, (ax_label, cov, err, color) in enumerate(
            zip(axis_labels, cov_data, err_data, colors)
        ):
            # Scatter plot: Error vs Covariance
            ax = axes[i]
            ax.scatter(err, cov, alpha=0.5, s=20, color=color)
            ax.set_xlabel(f"{ax_label} 误差")
            ax.set_ylabel(f"{ax_label} 不确定性")
            ax.set_title(f"{ax_label}")
            # ax.set_xlim(-0.05, 1.05)
            # ax.set_ylim(-0.01, 0.31)
            ax.grid(True, alpha=0.3)

            # 以 Y轴为例子 计算所有绘制的点
            points = np.vstack([err, cov]).transpose()
            # 每 interval=0.02去这个区间内 cov 的均值和方差
            sub_points = []

            err_min, err_max = np.min(err), np.max(err)
            err_gap = (err_max - err_min) / 100
            for e in np.arange(err_min, err_max, err_gap):
                err_range = [e, e + err_gap]
                err_c = (err_range[0] + err_range[1]) / 2
                in_range = (points[:, 0] >= err_range[0]) & (
                    points[:, 0] < err_range[1]
                )
                if np.any(in_range):
                    cov_min, cov_max = (
                        np.min(points[in_range, 1]),
                        np.max(points[in_range, 1]),
                    )
                    gap = (cov_max - cov_min) / 50
                    if gap == 0:
                        sub_points.append([err_c, cov_min])
                        break
                    cnt = 1
                    for c in np.arange(cov_min, cov_max, gap):
                        cov_range = [c, c + gap]
                        in_points = points[
                            in_range
                            & (points[:, 1] >= cov_range[0])
                            & (points[:, 1] < cov_range[1])
                        ]
                        if len(in_points) >= 3:
                            sub_points.append([err_c, np.min(in_points)])
                            cnt -= 1
                            if cnt == 0:
                                break

            sub_points = np.array(sub_points)

            if np.any(sub_points):
                ax.errorbar(
                    sub_points[:, 0],
                    sub_points[:, 1],
                    fmt="v",
                    color="black",
                    alpha=0.8,
                )

            # 保存这些点
            Obj.save(sub_points, Path(f"sub_points_{ax_label}.pkl"))

            # Calculate correlation
            # if len(cov) > 2:
            #     corr, p_value = pearsonr(err, cov)
            #     ax.text(
            #         0.05,
            #         0.95,
            #         f"相关系数：{corr:.3f}\np值：{p_value:.2e}",
            #         transform=ax.transAxes,
            #         verticalalignment="top",
            #         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            #     )

        plt.tight_layout()

        # Save figure
        output_path = Path("covariance_error_analysis_3axis.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Analysis chart saved to: {output_path}")

        # Print statistics
        print("\n=== 三轴不确定性-误差分析 ===")
        print(f"数据点数：{len(errors)}")

        for i, (ax_label, cov, err) in enumerate(zip(axis_labels, cov_data, err_data)):
            print(f"\n{ax_label} 统计：")
            print(f"  误差 - 均值：{np.mean(err):.6f}，标准差：{np.std(err):.6f}")
            print(f"  误差 - 最小值：{np.min(err):.6f}，最大值：{np.max(err):.6f}")
            print(f"  误差 - 中位数：{np.median(err):.6f}")
            print(f"  不确定性 - 均值：{np.mean(cov):.6f}，标准差：{np.std(cov):.6f}")
            print(f"  不确定性 - 最小值：{np.min(cov):.6f}，最大值：{np.max(cov):.6f}")
            print(f"  不确定性 - 中位数：{np.median(cov):.6f}")

            if len(cov) > 2:
                corr, p_value = pearsonr(err, cov)
                print(f"  相关系数：{corr:.6f}，p值：{p_value:.2e}")

                if abs(corr) > 0.7:
                    strength = "强"
                elif abs(corr) > 0.4:
                    strength = "中等"
                elif abs(corr) > 0.2:
                    strength = "弱"
                else:
                    strength = "极弱或无"

                direction = "正" if corr > 0 else "负"
                print(f"  强度：{strength}{direction}相关")

        plt.show()


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parse()

    # regen_fusion = dap.regen
    model_names = dap.args.models
    if model_names is None or len(model_names) == 0:
        model_names = ["model_resnet_0111_96"]

    models_path = "models"
    if dap.args.models_path is not None:
        models_path = dap.args.models_path

    # 结果输出路径
    EvalDir = Path(dap.args.output) if dap.args.output else Path("results")

    loader = ModelLoader(models_path)
    Data = InertialNetworkData.set_step(20)
    nets = loader.get_by_names(model_names)

    def action(ud: UnitData, res_dir: Path):
        print(f"> Eval {ud.name}")
        # 数据保存路径
        unit_out_dir = res_dir / ud.name
        unit_out_dir.mkdir(parents=True, exist_ok=True)

        obj_path = res_dir / "temp" / f"action_{ud.name}.pkl"
        print(f"> 存储路径：{obj_path}")
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        # 如果已经计算过
        if obj_path.exists():
            print(f"> 已存在结果：{obj_path}")
            nr_list, evaluator = Obj.load(obj_path)
            assert isinstance(nr_list, list)
            assert isinstance(evaluator, Evaluation)
        else:
            # 加载数据
            ud.load_data(using_opt=True)
            # 可视化
            bre.RerunView().add_spatial_view().send(ud.name)
            bre.send_pose_data(ud.gt_data, "Groundtruth", color=[192, 72, 72])

            # 模型推理
            nr_list = DataRunner(ud, Data, has_init_rerun=True).predict_batch(nets)

            # 计算 ATE
            evaluator = Evaluation(ud.gt_data, name=ud.name, rel_duration=1)
            evaluator.get_eval(nr_list[0].poses, f"{nets[0].name}_{ud.name}")
            evaluator.print()

            # 保存结果
            Obj.save((nr_list, evaluator), obj_path)

        nr = nr_list[0]
        return nr

    if dap.unit:
        unit_path = Path(dap.unit)
        # 使用 网络名称
        res_dir = EvalDir / f"{nets[0].name}"
        res_dir.mkdir(parents=True, exist_ok=True)

        ud = UnitData(unit_path)
        action(ud, res_dir)

    elif dap.dataset:
        dataset_path = Path(dap.dataset)
        datas = DeviceDataset(dataset_path)
        # 使用 网络名称 + 设备名称
        res_dir = EvalDir / f"{nets[0].name}_{datas.device_name}"
        res_dir.mkdir(parents=True, exist_ok=True)

        all_nr = NetworkResult()

        for ud in tqdm(datas, desc="Evaluating"):
            nr = action(ud, res_dir)
            all_nr.meas_cov_list.extend(nr.meas_cov_list)
            all_nr.gt_list.extend(nr.gt_list)
            all_nr.meas_list.extend(nr.meas_list)

        nra = NetworkResultAnalysis(all_nr)
        nra.analyze_uc_err()


if __name__ == "__main__":
    main()
