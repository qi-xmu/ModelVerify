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
from base.datatype import DeviceDataset, UnitData
from base.draw.CDF import plot_one_cdf
from base.draw.Poses import draw_trajectory_2d_compare
from base.evaluate import Evaluation
from base.model import DataRunner, InertialNetworkData, ModelLoader, NetworkResult
from base.obj import Obj

name_list = ["cam.csv", "fusion_desktop.csv"]


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parse()

    model_names = dap.args.models
    if model_names is None or len(model_names) == 0:
        model_names = ["model_resnet_0111_96"]

    models_path = "models"
    if dap.args.models_path is not None:
        models_path = dap.args.models_path

    # 结果输出路径 - 使用模型名称组合
    output = Path(dap.output) if dap.output else Path("results")

    loader = ModelLoader(models_path)
    Data = InertialNetworkData.set_step(20).no_rerun()
    nets = loader.get_by_names(model_names)

    def action(ud: UnitData, res_dir: Path):
        """对单个数据执行模型推理"""
        unit_out_dir = res_dir / ud.name
        unit_out_dir.mkdir(parents=True, exist_ok=True)

        obj_path = res_dir / "temp" / f"action_{ud.name}.pkl"
        obj_path.parent.mkdir(parents=True, exist_ok=True)

        # 加载缓存结果
        if obj_path.exists():
            nr_list, evaluator = Obj.load(obj_path)
            nr = nr_list[0]
            assert isinstance(nr, NetworkResult)
        else:
            # 加载数据
            ud.load_data(using_opt=True)
            # 模型推理
            nr_list = DataRunner(ud, Data).predict_batch(nets)
            evaluator = Evaluation(ud.gt_data, name=ud.name, rel_duration=1)
            evaluator.get_eval(nr_list[0].poses, nets[0].name)
            # 保存结果
            Obj.save((nr_list, evaluator), obj_path)

        nr = nr_list[0]
        assert isinstance(nr, NetworkResult)
        cdf_data = evaluator.get_cdf(nets[0].name, "RPE")

        draw_trajectory_2d_compare(
            [
                nr_list[0].poses,
                evaluator.ref_poses,
            ],
            labels=[
                "Fusion",
                "GT",
            ],
            save_path=unit_out_dir / "trajectory.png",
        )
        plot_one_cdf(
            cdf_data, unit_out_dir / "PoseCDF.png", (0, np.max(cdf_data["errors"]))
        )

        return nr_list, evaluator

    # 输出路径
    res_dir = output / f"{nets[0].name}"
    res_dir.mkdir(parents=True, exist_ok=True)

    if dap.unit:
        unit_path = Path(dap.unit)
        ud = UnitData(unit_path)

        nr_list, evaluator = action(ud, res_dir)

    elif dap.dataset:
        Data = Data.no_rerun()
        dataset_path = Path(dap.dataset)
        datas = DeviceDataset(dataset_path)

        # 存储每个模型的所有误差
        all_errors = {net.name: [] for net in nets}

        for ud in tqdm(datas, desc="Evaluating"):
            for net in nets:
                _nr = action(ud, res_dir)

        # 打印统计信息
        for name, errors in all_errors.items():
            print(f"{name}: {len(errors)} samples, mean error: {np.mean(errors):.4f}m")


if __name__ == "__main__":
    main()
