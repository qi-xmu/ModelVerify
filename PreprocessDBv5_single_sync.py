#!/usr/bin/env python3
"""
PreprocessDBv5 单次时间对齐版本（外部脚本）

说明：
- 不修改 PreprocessDBv5.py 本体。
- 通过 monkey patch 覆盖 time_sync，仅执行一次 match21 对齐。
- 默认时间范围为 0-50 秒，可通过 -t1 修改，例如：-t1 0 100。
- 若传入 -t2，会提示已忽略（单次对齐不使用精对齐）。
"""

from pathlib import Path

import numpy as np

import PreprocessDBv5 as base
from base.calibration.time import match21
from base.datatype import GroundTruthData, ImuData
from base.rtab import RTABData


def time_sync_single(
    rtab_data: RTABData,
    imu_data: ImuData,
    coarse_range: tuple[float, float] | None = None,
    fine_range: tuple[float, float] | None = None,
    output_dir: Path | None = None,
) -> tuple[GroundTruthData, int]:
    """单次时间同步：只执行一次 match21。"""
    print("=== 时间同步检查（单次对齐）===")

    coarse_range = coarse_range or (0.0, 50.0)

    gt_data = GroundTruthData(rtab_data.node_t_us, rtab_data.node_rots, rtab_data.node_ps)
    imu_poses = imu_data.to_poses()

    print(f"IMU数据点数: {len(imu_data)}")
    print(f"GT数据点数: {len(gt_data)}")
    print(f"IMU采样率: {1e6 / np.mean(np.diff(imu_data.t_us)):.1f} Hz")
    print(f"GT采样率: {gt_data.rate:.1f} Hz")
    print(f"搜索范围: {coarse_range[0]}s 到 {coarse_range[1]}s")

    if fine_range is not None:
        print(
            "提示: 当前脚本为单次对齐模式，已忽略 -t2/--time-range-fine 参数，"
            "请使用 -t1 调整搜索范围。"
        )

    save_path = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "TimeDiff1.png"

    time_gc = match21(
        imu_poses,
        gt_data,
        time_range=coarse_range,
        save_path=save_path,
        resolution=100,
    )

    print(f"单次对齐结果: {time_gc / 1e6:.6f} 秒 ({time_gc} μs)")

    # 只应用一次时间偏移；起点重置仍在统一插值后处理
    gt_data.t_us += time_gc

    print("时间同步完成（单次对齐）")
    print("=" * 80)

    return gt_data, int(time_gc)


def main():
    # 外部覆盖：不改原文件，只替换 time_sync 实现
    base.time_sync = time_sync_single
    base.main()


if __name__ == "__main__":
    main()
