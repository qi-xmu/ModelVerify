#!/usr/bin/env python3
"""
多模型对比可视化

这是一个用于可视化惯性导航网络模型结果的Python脚本。

主要功能：
1. 加载并处理IMU、相机、融合数据和真值数据
2. 进行时间校准和坐标系对齐
3. 提供数据可视化功能
4. 支持多个模型的批量推理

用法:
    # 对比单个数据单元上的多个模型
    uv run python visualizers/DrawCompare.py -u <unit_path> -m model1 model2

    # 对比整个数据集上的多个模型
    uv run python visualizers/DrawCompare.py -d <dataset_path> -m model1 model2

    # 指定模型文件夹
    uv run python visualizers/DrawCompare.py -u <unit_path> -m model1 model2 --models_path /path/to/models

参数:
    -u, --unit: 指定单个数据单元路径
    -d, --dataset: 指定数据集路径
    -m, --models: 指定模型文件名（支持多个）
    --models_path: 指定模型文件夹路径（默认为"models"）
"""

import numpy as np
from scipy.spatial.transform import Rotation

import base.rerun_ext as bre
from base.args_parser import DatasetArgsParser
from base.datatype import (
    CameraData,
    DeviceDataset,
    GroundTruthData,
    ImuData,
    Pose,
    UnitData,
)

from base.interpolate import get_time_series

# 真值的Body坐标系和Sensor的Body坐标系存在的固定插值
DefaultBodyRotation = Rotation.from_rotvec([0, -90, 0], degrees=True)
DefaultBodyTransform = Pose(DefaultBodyRotation, np.zeros(3))


def main():
    dap = DatasetArgsParser()
    dap.parse()

    def action(ud: UnitData):
        imu_data = ImuData.from_csv(ud._imu_path)
        gt_data = GroundTruthData.from_csv(ud._gt_path)
        camera_data = CameraData.from_csv(ud._cam_path)

        # 获取共同的时间窗口
        ts_us = []
        ts_us.append(gt_data.t_us)
        ts_us.append(imu_data.t_us)
        ts_us.append(camera_data.t_us)

        t_new_us = get_time_series(ts_us)
        ud.gt_data = gt_data.interpolate(t_new_us)
        ud.imu_data = imu_data.interpolate(t_new_us)
        camera_data = camera_data.interpolate(t_new_us)

        # 可视化
        bre.rerun_init(ud.name)
        bre.send_pose_data(ud.gt_data, "Groundtruth", color=[192, 72, 72])
        bre.send_pose_data(camera_data, "Camera", color=[72, 72, 192])

    if dap.unit:
        ud = UnitData(dap.unit)
        action(ud)
    elif dap.dataset:
        dataset_path = dap.dataset
        datas = DeviceDataset(dataset_path)
        for ud in datas:
            action(ud)


if __name__ == "__main__":
    main()
