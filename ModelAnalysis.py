#!/usr/bin/env python3
"""
只有真值和IMU数据时，使用此脚本进行验证。
"""

import base.rerun_ext as bre
from base.args_parser import DatasetArgsParser
from base.calibration import time
from base.datatype import (
    DeviceDataset,
    GroundTruthData,
    ImuData,
    PosesData,
    UnitData,
)
from base.evaluate import Evaluation
from base.interpolate import get_time_series
from base.model import DataRunner, InertialNetworkData, ModelLoader


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parse()

    # regen_fusion = dap.regen
    model_names = dap.args.models
    if model_names is None or len(model_names) == 0:
        model_names = ["model_tlio_mi_hw_1216"]

    models_path = "models"
    if dap.args.models_path is not None:
        models_path = dap.args.models_path

    loader = ModelLoader(models_path)
    # Data = InertialNetworkData.set_step(20)
    nets = loader.get_by_names(model_names)

    for net in nets:
        net.analysis()

    def action(ud: UnitData):
        pass

    if dap.unit:
        ud = UnitData(dap.unit, using_ext=False)
        action(ud)
    elif dap.dataset:
        dataset_path = dap.dataset
        datas = DeviceDataset(dataset_path, False)
        for ud in datas:
            action(ud)


if __name__ == "__main__":
    main()
