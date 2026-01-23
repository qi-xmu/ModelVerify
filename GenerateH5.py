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

from base.args_parser import DatasetArgsParser
from base.dataset.H5Type import (
    AlignedData,
    H5Dataset,
    Metadata,
    ResampledData,
    Sequence,
    SequenceAttributes,
)
from base.datatype import DeviceDataset, UnitData
from base.interpolate import get_time_series

# 默认结果输出路径
EvalDir = Path("/Users/qi/Resources/results")


def calculate_velocity(t_us: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    通过位移计算速度

    Args:
        t_us: 时间戳数组 [N] (微秒)
        pos: 位置数组 [N, 3]

    Returns:
        vel: 速度数组 [N, 3] (m/s)，首尾元素使用前向/后向差分
    """
    dt = np.diff(t_us.astype(np.float64)) / 1e6  # 转换为秒
    dpos = np.diff(pos, axis=0)

    # 中间点使用中心差分
    vel_mid = dpos / dt[:, np.newaxis]

    # 边界点使用前向/后向差分
    vel = np.zeros_like(pos)
    vel[1:-1] = (vel_mid[:-1] + vel_mid[1:]) / 2  # 平均相邻速度
    vel[0] = vel_mid[0]
    vel[-1] = vel_mid[-1]
    return vel


def create_aligned_data(
    t_us: np.ndarray,
    gyro: np.ndarray,
    acc: np.ndarray,
    pos: np.ndarray,
    quats: np.ndarray,
    vel: np.ndarray,
) -> AlignedData:
    """
    创建旋转对齐数据

    Args:
        t_us: 时间戳数组 [N] (微秒)
        gyro: 陀螺仪数据 [N, 3]
        acc: 加速度计数据 [N, 3]
        pos: 位置数据 [N, 3]
        quats: 四元数数据 [N, 4] (qw, qx, qy, qz)
        vel: 速度数据 [N, 3]

    Returns:
        AlignedData对象，包含[N, 17]数组: t_us, gyr(3), acc(3), pos(3), qwxyz(4), vel(3)
    """
    # 按顺序拼接数据: t_us(1), gyr(3), acc(3), pos(3), qwxyz(4), vel(3) = 17列
    t_us_col = t_us.reshape(-1, 1)  # [N, 1]

    aligned_array = np.column_stack([t_us_col, gyro, acc, pos, quats, vel])  # [N, 17]

    return AlignedData(data=aligned_array)


def get_h5type_code() -> tuple[str, str]:
    """
    读取H5Type.py文件内容

    Returns:
        tuple: (readme, code) - readme是文件头部的文档说明，code是完整的文件内容
    """
    h5type_path = Path(__file__).parent / "base" / "dataset" / "H5Type.py"
    with open(h5type_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 提取文件头部的文档（从第一行开始到第一个非注释行之前）
    readme_lines = []
    for line in lines:
        stripped = line.strip()
        # 跳过空行
        if not stripped:
            continue
        # 如果是注释行（以#开头），添加到readme
        if stripped.startswith("#"):
            readme_lines.append(line)
        # 如果遇到非注释行，停止
        else:
            break

    readme = "".join(readme_lines)
    code = "".join(lines)

    return readme, code


def unit_data_to_sequence(ud: UnitData) -> Sequence:
    """
    将UnitData转换为Sequence对象

    Args:
        ud: UnitData对象

    Returns:
        Sequence对象
    """
    # 确保数据已加载
    ud.load_data()

    # 真值和IMU插值对齐
    ts_us = []
    ts_us.append(ud.imu_data.t_us)
    ts_us.append(ud.gt_data.t_us)
    t_new_us = get_time_series(ts_us)
    ud.imu_data = ud.imu_data.interpolate(t_new_us)
    ud.gt_data = ud.gt_data.interpolate(t_new_us)

    # 计算序列属性
    num_rows = len(ud.imu_data.t_us)
    t_start_us = int(ud.imu_data.t_us[0])
    t_end_us = int(ud.imu_data.t_us[-1])
    frequency_hz = ud.gt_data.rate

    attributes = SequenceAttributes(
        num_rows=num_rows,
        t_start_us=t_start_us,
        t_end_us=t_end_us,
        frequency_hz=frequency_hz,
    )

    # 统一时间戳
    t_us = ud.imu_data.t_us.astype(np.int64)

    # 准备IMU数据 [1, N, 6]: gyr(3), acc(3)，不包含时间戳
    imu_data_array = np.column_stack([ud.imu_data.gyro, ud.imu_data.acce])
    # 转换为 [1, N, 6] 格式，1个传感器
    imu_data = imu_data_array[np.newaxis, :, :]

    # 准备磁力计数据 [1, N, 3]: mag(3)，不包含时间戳
    mag_data_array = ud.imu_data.magn
    # 转换为 [1, N, 3] 格式，1个传感器
    mag_data = mag_data_array[np.newaxis, :, :]

    # 准备真值数据 [N, 10]: pos(3), qwxyz(4), vel(3)，不包含时间戳
    # 计算速度
    vel = calculate_velocity(ud.gt_data.t_us, ud.gt_data.ps)

    # 准备四元数 (qw, qx, qy, qz)
    quats = ud.gt_data.rots.as_quat(scalar_first=True)

    ground_truth_array = np.column_stack([ud.gt_data.ps, quats, vel])

    resampled_data = ResampledData(
        t_us=t_us,  # 统一时间戳
        imu=imu_data,  # [1, N, 6] gyr(3), acc(3)
        mag=mag_data,  # [1, N, 3] mag(3)
        barom=None,  # 气压计数据为空（UnitData中没有气压计数据）
        ground_truth=ground_truth_array,  # [N, 10] pos(3), qwxyz(4), vel(3)
    )

    imu_global = ud.imu_data.transform(ud.gt_data.rots)

    # 创建旋转对齐数据
    aligned_data = create_aligned_data(
        t_us=t_us,
        gyro=imu_global.gyro,
        acc=imu_global.acce,
        pos=ud.gt_data.ps,
        quats=quats,
        vel=vel,
    )

    return Sequence(
        name=ud.name,
        attributes=attributes,
        resampled=resampled_data,
        aligned=aligned_data,
    )


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("-m", "--models", nargs="+", help="模型")
    dap.parser.add_argument("--models_path", type=str, help="模型文件夹")
    dap.parser.add_argument("-c", "--config", type=str, help="配置文件")
    dap.parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="训练集比例 (默认: 0.7)"
    )
    dap.parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="验证集比例 (默认: 0.15)"
    )
    dap.parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="测试集比例 (默认: 0.15)"
    )
    dap.parser.add_argument("--no_shuffle", action="store_true", help="不打乱序列顺序")
    dap.parser.add_argument(
        "--random_seed", type=int, default=42, help="随机种子 (默认: 42)"
    )
    dap.parse()

    assert dap.output is not None
    output_dir = Path(dap.output)

    def action(ud: UnitData):
        print(f"> Eval {ud.name}")
        ud.load_data()

        # 转换为Sequence对象
        sequence = unit_data_to_sequence(ud)
        print(f"  Created sequence: {sequence.name}")
        print(f"    Rows: {sequence.attributes.num_rows}")
        print(
            f"    Duration: {sequence.attributes.t_end_us - sequence.attributes.t_start_us} us"
        )
        print(f"    Frequency: {sequence.attributes.frequency_hz} Hz")
        if sequence.resampled and sequence.resampled.ground_truth is not None:
            print(f"    Ground truth shape: {sequence.resampled.ground_truth.shape}")

        return sequence

    sequences = []

    if dap.unit:
        unit_path = Path(dap.unit)
        # 使用 网络名称
        ud = UnitData(unit_path)
        output_path = output_dir / f"{ud.name}.h5"
        seq = action(ud)
        sequences.append(seq)

    elif dap.dataset:
        dataset_path = Path(dap.dataset)
        datas = DeviceDataset(dataset_path)

        output_path = output_dir / f"{datas.device_name}.h5"

        for ud in datas:
            seq = action(ud)
            sequences.append(seq)
    else:
        raise ValueError("No unit or dataset specified")

    # 将所有sequences保存为H5Dataset
    if len(sequences) > 0:
        print(f"\n> Saving H5Dataset to: {output_path}")
        # 计算总时长
        total_duration_us = sum(
            seq.attributes.t_end_us - seq.attributes.t_start_us for seq in sequences
        )
        total_duration_hours = total_duration_us / 1e6 / 3600

        # 创建元数据
        readme, code = get_h5type_code()
        metadata = Metadata(
            version="1.0",
            num_sequences=len(sequences),
            total_duration_hours=total_duration_hours,
            readme=readme,
            code=code,
        )

        # 创建H5文件并写入数据
        with H5Dataset(output_path, mode="w") as h5_dataset:
            # 写入元数据
            h5_dataset.create_metadata(metadata)
            print(
                f"  Created metadata: {len(sequences)} sequences, {total_duration_hours:.2f} hours"
            )

            # 写入每个序列
            for seq in sequences:
                h5_dataset.create_sequence(seq)
                print(f"  Saved sequence: {seq.name} ({seq.attributes.num_rows} rows)")

            # 自动创建并写入序列列表
            sequence_lists = h5_dataset.create_sequence_lists(
                train_ratio=dap.args.train_ratio,
                val_ratio=dap.args.val_ratio,
                test_ratio=dap.args.test_ratio,
                shuffle=not dap.args.no_shuffle,
                random_seed=dap.args.random_seed,
            )
            print("\n  Sequence split:")
            print(f"    Train: {len(sequence_lists.train)} sequences")
            print(f"    Val: {len(sequence_lists.val)} sequences")
            print(f"    Test: {len(sequence_lists.test)} sequences")

            # 自动创建并写入索引映射
            index_maps = h5_dataset.create_index_maps()
            print("\n  Index maps:")
            print(f"    Train indices: {len(index_maps.train_index)} samples")
            print(f"    Val indices: {len(index_maps.val_index)} samples")
            print(f"    Test indices: {len(index_maps.test_index)} samples")

        print(f"\n✓ Successfully saved H5Dataset with {len(sequences)} sequences")
        print(f"  Output file: {output_path}")
        print(f"  Total duration: {total_duration_hours:.2f} hours")
    else:
        print("! No sequences to save")


if __name__ == "__main__":
    main()
