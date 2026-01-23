#!/usr/bin/env python3
"""
轨迹误差分析

比较多个模型预测轨迹与真值的误差，绘制累积分布函数（CDF）对比图。

用法:
    # 比较单个数据单元上的多个模型
    python ValidTrack.py -u <unit_path> -m model1 model2

    # 比较整个数据集上的多个模型
    python ValidTrack.py -d <dataset_path> -m model1 model2

    # 指定模型文件夹
    python ValidTrack.py -u <unit_path> -m model1 model2 --models_path /path/to/models

参数:
    -u, --unit: 指定单个数据单元路径
    -d, --dataset: 指定数据集路径
    -m, --models: 指定模型文件名（支持多个）
    --models_path: 指定模型文件夹路径（默认为"models"）

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
    output = Path(dap.output) if dap.output else Path("results")

    def action(ud: UnitData, res_dir: Path, name: str):
        """对单个数据执行模型推理"""
        unit_out_dir = res_dir / ud.name
        unit_out_dir.mkdir(parents=True, exist_ok=True)

        # 读取
        eval_path = ud.base_dir / f"{name}.csv"
        eval_pose = GroundTruthData.from_csv(eval_path)

        ud.load_data()
        evaluator = Evaluation(ud.gt_data, name=ud.name, rel_duration=1)
        evaluator.get_eval(eval_pose, name)

        cdf_data = evaluator.get_cdf(name, "RTE")

        draw_trajectory_2d_compare(
            [eval_pose, evaluator.ref_poses],
            labels=["Fusion", "GT"],
            save_path=unit_out_dir / "trajectory.png",
            show=False,
        )
        plot_one_cdf(
            cdf_data,
            unit_out_dir / "PoseCDF.png",
            (0, np.max(cdf_data["errors"])),
            show=False,
        )

    # 输出路径
    res_dir = output
    res_dir.mkdir(parents=True, exist_ok=True)

    if dap.unit:
        unit_path = Path(dap.unit)
        ud = UnitData(unit_path)
        action(ud, res_dir, dap.args.name)

    elif dap.dataset:
        dataset_path = Path(dap.dataset)
        datas = DeviceDataset(dataset_path)

        for ud in tqdm(datas, desc="Evaluating"):
            action(ud, res_dir, dap.args.name)


if __name__ == "__main__":
    main()
