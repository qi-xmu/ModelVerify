#!/usr/bin/env python3
"""
这是一个用于可视化惯性导航网络模型结果的Python脚本。

主要功能：
1. 加载并处理IMU、相机、融合数据和真值数据
2. 进行时间校准和坐标系对齐
3. 提供数据可视化功能
4. 支持多个模型的批量推理

使用说明：
1. 通过命令行参数指定要使用的模型和模型路径
2. 可以处理单个单元数据(--unit)或整个数据集(--dataset)
3. 可视化使用Rerun框架实现

作者：qi-xmu
版本：1.0

示例用法：
python DrawCompare.py --unit /path/to/data --models model1 model2
python DrawCompare.py --dataset /path/to/dataset --models_path /custom/models/path
"""

import numpy as np
from scipy.spatial.transform import Rotation

import base.rerun_ext as bre
from base.args_parser import DatasetArgsParser
from base.binary import SensorFusion
from base.calibration import space, time
from base.datatype import (
    CameraData,
    DeviceDataset,
    FusionData,
    GroundTruthData,
    ImuData,
    Pose,
    PosesData,
    UnitData,
)
from base.evaluate import Evaluation
from base.interpolate import get_time_series
from base.model import DataRunner, InertialNetworkData, ModelLoader

# 真值的Body坐标系和Sensor的Body坐标系存在的固定插值
DefaultBodyRotation = Rotation.from_rotvec([0, -90, 0], degrees=True)
DefaultBodyTransform = Pose(DefaultBodyRotation, np.zeros(3))
# 使用融合算法推理轨迹，指定项目路径
ProjectPath = (
    "/Users/qi/Codespace/Android/NAVIO/navio_sdk/src/main/cpp/SensorFusionAndroid"
)


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parse()

    # regen_fusion = dap.regen
    model_names = dap.args.models
    if model_names is None or len(model_names) == 0:
        model_names = ["96"]

    models_path = "models"
    if dap.args.models_path is not None:
        models_path = dap.args.models_path

    loader = ModelLoader(models_path)
    Data = InertialNetworkData.set_step(20)

    nets = loader.get_by_names(model_names)

    def action(ud: UnitData):
        # NOTE: sensor fusion 生成数据
        sf = SensorFusion(ProjectPath)
        model = nets[0].model_path.absolute()
        res_dir = sf.unit_run(ud, model=model)
        fusion_data = FusionData.from_csv(res_dir)

        imu_data = ImuData.from_csv(ud._imu_path)
        # gt_data = GroundTruthData.from_csv(ud._gt_path)
        camera_data = CameraData.from_csv(ud._cam_path)
        result_data = CameraData.from_csv(ud.base_dir / "result.csv")
        # fusion_data = result_data

        # 完成时间校准
        # time_gc = time.match21(fusion_data, gt_data)
        # gt_data.t_us += time_gc

        # 参考坐标系对齐
        # tf_fg = space.global12(fusion_data, gt_data)
        # gt_data.transform_local(DefaultBodyTransform)
        # gt_data.transform_global(tf_fg)

        # tf_fr = space.global12(fusion_data, result_data)
        # result_data.transform_global(tf_fr)
        # camera_data.transform_global(tf_fr)

        # gt_data.ps -= gt_data.ps[0]
        fusion_data.ps -= fusion_data.ps[0]
        result_data.ps -= result_data.ps[0]
        camera_data.ps -= camera_data.ps[0]

        # 获取共同的时间窗口
        ts_us = []
        # ts_us.append(gt_data.t_us)
        ts_us.append(imu_data.t_us)
        ts_us.append(camera_data.t_us)
        ts_us.append(fusion_data.t_us)
        ts_us.append(result_data.t_us)

        t_new_us = get_time_series(ts_us)
        # ud.gt_data = gt_data.interpolate(t_new_us)
        ud.imu_data = imu_data.interpolate(t_new_us)
        camera_data = camera_data.interpolate(t_new_us)
        fusion_data = fusion_data.interpolate(t_new_us)
        result_data = result_data.interpolate(t_new_us)

        # 可视化
        bre.rerun_init(ud.name)
        # bre.send_pose_data(ud.gt_data, "Groundtruth", color=[192, 72, 72])
        # bre.send_pose_data(gt_data, "__gtc", color=[192, 72, 72])
        bre.send_pose_data(fusion_data, "Fusion", color=[72, 192, 72])
        bre.send_pose_data(camera_data, "Camera", color=[72, 72, 192])
        bre.send_pose_data(result_data, "Reuslt", color=[192, 192, 72])

        # 模型推理
        # runner = DataRunner(ud, Data, using_gt=False, has_init_rerun=True)
        # net_results = runner.predict_batch(nets)
        # netres_data = PosesData.from_list(net_results[0].pose_list)

        # 计算 ATE
        # evaluator = Evaluation(ud.gt_data)
        # evaluator.get_eval(camera_data, "camera")
        # evaluator.get_eval(fusion_data, "fusion")
        # evaluator.get_eval(result_data, "result")
        # evaluator.get_eval(netres_data, "netres")
        # evaluator.print()
        # evaluator.save(ud.base_dir / "eval.json")

    if dap.unit:
        ud = UnitData(dap.unit)
        action(ud)
    elif dap.dataset:
        dataset_path = dap.dataset
        datas = DeviceDataset(dataset_path)
        for ud in datas:
            action(ud)
    else:
        pass


if __name__ == "__main__":
    main()
