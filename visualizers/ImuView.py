#!/usr/bin/env python3
"""
IMU数据可视化工具

该脚本用于将惯性测量单元(IMU)数据通过Rerun进行可视化展示。

功能:
- 加载单设备数据(UnitData)
- 使用Rerun可视化IMU数据
- 同时显示body坐标系和global坐标系下的IMU数据
- 支持使用地面真值(ground truth)的旋转信息进行坐标变换

参数说明:
-u/--unit: 指定要可视化的设备数据单元

使用示例:
python visualizers/ImuView.py -u <unit_path>

作者: qi-xmu
版本: 1.0
"""

from base.args_parser import DatasetArgsParser
from base.datatype import UnitData
from base.rerun_ext import RerunView, send_imu_data


def main():
    dap = DatasetArgsParser()
    dap.parse()

    if dap.unit:
        # 数据
        ud = UnitData(dap.unit)
        ud.load_data()

        RerunView().add_imu_view(visible=True, tags=["body", "global"]).send(ud.name)

        raw_imu_data = ud.imu_data

        global_imu_data = raw_imu_data.transform(ud.gt_data.rots)
        send_imu_data(raw_imu_data, "body")
        send_imu_data(global_imu_data, "global")


if __name__ == "__main__":
    main()
