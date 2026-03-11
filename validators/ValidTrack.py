#!/usr/bin/env python3
"""
轨迹误差分析

比较多个模型预测轨迹与真值的误差，绘制累积分布函数（CDF）对比图。

用法:
    # 比较单个数据单元上的多个模型
    uv run python validators/ValidTrack.py -u <unit_path> -m model1 model2

    # 比较整个数据集上的多个模型
    uv run python validators/ValidTrack.py -d <dataset_path> -m model1 model2

    # 指定模型文件夹
    uv run python validators/ValidTrack.py -u <unit_path> -m model1 model2 --models_path /path/to/models

参数:
    -u, --unit: 指定单个数据单元路径
    -d, --dataset: 指定数据集路径
    -m, --models: 指定模型文件名（支持多个）
    --models_path: 指定模型文件夹路径（默认为"models"）
    -n, --name: 指定轨迹名称（默认为 cam.csv 或 fusion_desktop.csv）

输出:
    - results/<model1>_<model2>_.../ CDF对比图
    - Eval.json: 包含各模型误差统计结果

主要功能:
    - 对多个模型的预测轨迹与真值进行误差分析
    - 计算ATE (Absolute Trajectory Error)
    - 绘制CDF对比图，直观展示不同模型的误差分布
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from base.args_parser import DatasetArgsParser
from base.calibration import space, time
from base.datatype import DeviceDataset, GroundTruthData, UnitData
from base.draw.CDF import plot_one_cdf
from base.draw.Poses import draw_trajectory_2d_compare
from base.evaluate import Evaluation

name_list = ["cam.csv", "fusion_desktop.csv"]


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parser.add_argument("-n", "--name", type=str, help="名称")
    dap.parse()

    model_names = dap.args.models
    if model_names is None or len(model_names) == 0:
        model_names = ["model_resnet_0111_96"]

    # 结果输出路径 - 使用模型名称组合
    output = Path(dap.output) if dap.output else None

    def action(ud: UnitData, res_dir: Path, name: str):
        """对单个数据执行模型推理"""
        unit_out_dir = res_dir / ud.name
        unit_out_dir.mkdir(parents=True, exist_ok=True)

        # 读取
        eval_path = ud.base_dir / f"{name}.csv"
        eval_poses = GroundTruthData.from_csv(eval_path)

        ud.load_data()

        # 对齐时间
        time_gc = time.match21(eval_poses, ud.gt_data)
        ud.gt_data.t_us += time_gc
        print("Time GC:", time_gc)

        # 对齐起点
        ud.gt_data.ps -= ud.gt_data.ps[0]
        eval_poses.ps -= eval_poses.ps[0] + np.array([-0.6, 0.7, 0])
        tf_fg = space.global12(eval_poses, ud.gt_data)
        ud.gt_data.transform_global(tf_fg)

        evaluator = Evaluation(ud.gt_data, name=ud.name, rel_duration=1)
        evaluator.get_eval(eval_poses, name)

        cdf_data = evaluator.get_cdf(name, "ATE")
        mean_ate = np.mean(cdf_data["errors"])
        print("Mean ATE:", mean_ate)

        draw_trajectory_2d_compare(
            [eval_poses, evaluator.ref_poses],
            labels=["Fusion", "GT"],
            colors=[None, "red"],
            save_path=unit_out_dir / "trajectory.png",
            show=False,
        )
        plot_one_cdf(
            cdf_data,
            unit_out_dir / f"{cdf_data['tag']}.png",
            (0, np.max(cdf_data["errors"])),
            show=False,
        )

        last_eval_pose = eval_poses.get_pose(len(eval_poses) - 1)
        last_gt_pose = ud.gt_data.get_pose(len(ud.gt_data) - 1)

        # 计算重点误差
        error = np.linalg.norm(last_eval_pose.p - last_gt_pose.p)
        print(f"Final Error: {error:.4f}m")

    if dap.unit:
        unit_path = Path(dap.unit)
        if output is None:
            output = unit_path.parent

        ud = UnitData(unit_path)
        action(ud, output, dap.args.name)

    elif dap.dataset:
        dataset_path = Path(dap.dataset)
        datas = DeviceDataset(dataset_path)

        for ud in tqdm(datas, desc="Evaluating"):
            action(ud, dataset_path, dap.args.name)


if __name__ == "__main__":
    main()
