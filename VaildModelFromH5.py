#!/usr/bin/env python3
"""
验证模型效果

验证单个模型在数据集上的效果，生成CDF图和ATE/APE/RPE等评估指标。

用法:
    # 验证单个数据单元
    python VaildModel.py -u <unit_path> -m model_name

    # 验证整个数据集
    python VaildModel.py -d <dataset_path> -m model_name

    # 指定模型文件夹
    python VaildModel.py -u <unit_path> -m model_name --models_path /path/to/models

参数:
    -u, --unit: 指定单个数据单元路径
    -d, --dataset: 指定数据集路径
    -m, --models: 指定模型文件名（支持多个）
    --models_path: 指定模型文件夹路径（默认为"models"）

输出:
    - results/<model_name>/<unit_name>/: 单个单元的结果目录
      - CDF.png: 误差累积分布函数图
      - Eval.json: 评估指标（ATE/APE/RPE）
    - results/<model_name>_<device_name>/: 数据集结果目录
      - CDF.png: 整体误差CDF图
    - results/<model_name>/temp/: 临时结果缓存
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

import base.rerun_ext as bre
from base.args_parser import DatasetArgsParser
from base.dataset import H5SplitDataset, H5UnitData
from base.datatype import UnitData
from base.draw.Bar import Bar
from base.draw.CDF import plot_one_cdf
from base.draw.Scatter import Scatter
from base.evaluate import Evaluation, get_cdf_from_err
from base.model import DataRunner, InertialNetworkData, ModelLoader, NetworkResult
from base.obj import Obj
from base.utils import angle_between_vectors, angle_with_x_axis


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parser.add_argument("-c", "--config", type=str, help="配置文件")
    dap.parse()

    # regen_fusion = dap.regen
    model_names = dap.args.models
    if model_names is None or len(model_names) == 0:
        model_names = ["model_resnet_0111_96"]

    models_path = "models"
    if dap.args.models_path is not None:
        models_path = dap.args.models_path

    # 结果输出路径
    EvalDir = Path(dap.output) if dap.output else Path("results")

    loader = ModelLoader(models_path)
    Data = InertialNetworkData.set_step(20).no_rerun()
    nets = loader.get_by_names(model_names)

    def action(ud: UnitData, res_dir: Path):
        # 数据保存路径
        unit_out_dir = res_dir / ud.name
        unit_out_dir.mkdir(parents=True, exist_ok=True)

        obj_path = res_dir / "temp" / f"action_{ud.name}.pkl"
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        # 如果已经计算过
        if obj_path.exists():
            nr_list, evaluator = Obj.load(obj_path)
            assert isinstance(nr_list, list)
            assert isinstance(evaluator, Evaluation)

            nr = nr_list[0]
        else:
            # 加载数据
            ud.load_data(using_opt=True)
            if dap.args.config:
                print(f"> 加载配置文件：{dap.args.config}")
                ud.imu_data.calibrate_with(dap.args.config)

            # 可视化
            if Data.using_rerun:
                print("> Using rerun")
                bre.RerunView().add_spatial_view().send(ud.name)
                bre.send_pose_data(ud.gt_data, "Groundtruth", color=[192, 72, 72])

            # 模型推理
            dr = DataRunner(ud, Data, has_init_rerun=True)
            nr_list = dr.predict_batch(nets)
            nr = nr_list[0]

            # 计算 ATE
            evaluator = Evaluation(ud.gt_data, name=ud.name, rel_duration=1)
            evaluator.get_eval(nr_list[0].poses, f"{nets[0].name}_{ud.name}")
            evaluator.save(unit_out_dir / "Eval.json")

            # 保存结果
            Obj.save((nr_list, evaluator), obj_path)

            # 绘制 CDF
            model_cdf = get_cdf_from_err(nr.err_list, nets[0].name)
            plot_one_cdf(model_cdf, unit_out_dir / "CDF.png", show=False)

        if "eval_t_list" in nr.__dict__:
            Bar(
                x=None,
                y=nr.eval_t_list,
                x_label="x",
                y_label="Time(s)",
                title=f"Inference Latency with {nr_list[0].network_device_name}",
            ).save(unit_out_dir)

        return nr_list, evaluator

    assert dap.dataset is not None, "Dataset path is not provided"

    dataset_path = Path(dap.dataset)
    datas = H5SplitDataset(dataset_path, split="test")

    # 使用 网络名称 + 设备名称
    res_dir = EvalDir / f"{nets[0].name}_{datas.name}"
    res_dir.mkdir(parents=True, exist_ok=True)
    # 存储结果
    netres_list: list[NetworkResult] = []

    # 单位是-180~180度数，遵循右手螺旋定律
    angle_list = []
    angle_error_list = []

    for ud in tqdm(datas, desc="Evaluating"):
        assert isinstance(ud, H5UnitData)
        nr_list, evaluator = action(ud, res_dir)
        netres_list.extend(nr_list)
        nr = nr_list[0]

        for meas, gt in zip(nr.meas_list, nr.gt_list):
            angle = angle_with_x_axis(gt)
            angle_error = angle_between_vectors(meas, gt)

            if abs(angle_error) < 45:
                angle_list.append(angle)
                angle_error_list.append(angle_error)

    # 角度分析
    mean_angle_error = np.mean(angle_error_list)
    print(f"Mean Angle Error: {mean_angle_error:.2f} degrees")
    Scatter(
        x=angle_list,
        y=angle_error_list,
        title="Angle Error",
        x_label="Angle",
        y_label="Error",
    ).save(res_dir)

    # 合并所有netres的误差项
    all_errors = []
    for nr in netres_list:
        assert isinstance(nr, NetworkResult)
        meas = np.array(nr.meas_list)
        gt = np.array(nr.gt_list)

        err = meas - gt
        err_norm = np.linalg.norm(err, axis=1)
        all_errors.extend(err_norm)

    # 绘制总体的结果
    model_cdf = get_cdf_from_err(all_errors, nets[0].name)
    plot_one_cdf(model_cdf, res_dir / "CDF.png", show=False)


if __name__ == "__main__":
    main()
