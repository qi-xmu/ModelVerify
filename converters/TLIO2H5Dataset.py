#!/usr/bin/env python3
"""
将TLIO golden数据集转换为H5格式

TLIO golden数据集结构:
    /path/to/tlio_golden/
    ├── 1008221029329889/
    │   ├── calibration.json
    │   ├── imu_samples_0.csv          # Body frame IMU数据 (原始采样)
    │   ├── imu0_resampled_description.json
    │   └── imu0_resampled.npy         # Global frame数据 (200Hz重采样)
    ├── train_list.txt  
    ├── val_list.txt   
    └── test_list.txt   

数据格式:
    - imu_samples_0.csv: Body frame IMU数据 (原始采样率，非200Hz)
        - timestamp [ns], temperature, w_RS_S_x/y/z, a_RS_S_x/y/z

    - imu0_resampled.npy: Global frame数据 (200Hz重采样，已对齐)
        - ts_us: 时间戳(微秒)
        - gyr_compensated_rotated_in_World: 陀螺仪(世界坐标系, 3维)
        - acc_compensated_rotated_in_World: 加速度计(世界坐标系, 3维)
        - qxyzw_World_Device: 四元数(世界到设备, 4维, xyzw顺序)
        - pos_World_Device: 位置(世界坐标系, 3维)
        - vel_World: 速度(世界坐标系, 3维)

转换策略:
    - resampled: 使用npy的global frame IMU数据 (200Hz，已对齐)
    - aligned: 使用npy的global frame数据 (200Hz，已对齐)

用法:
    uv run python converters/TLIO2H5Dataset.py -i /path/to/tlio_golden -o output.h5

参数:
    -i, --input: 输入TLIO golden数据集目录
    -o, --output: 输出H5文件路径
    --use_splits : 使用TLIO数据集自带的train/val/test分割
    --train_ratio: 自动分割时训练集比例 (默认0.7)
    --val_ratio: 自动分割时验证集比例 (默认0.15)
    --test_ratio: 自动分割时测试集比例 (默认0.15)
    --random_seed: 自动分割时的随机种子 (默认42)
    --no_shuffle: 自动分割时不打乱序列顺序  
"""

import argparse
import json
from pathlib import Path

import numpy as np

from base.dataset.H5Type import (
    AlignedData,
    H5Dataset,
    Metadata,
    ResampledData,
    Sequence,
    SequenceAttributes,
)


def get_h5type_code() -> tuple[str, str]:
    """读取H5Type.py文件内容"""
    h5type_path = Path(__file__).parent.parent / "src" / "base" / "dataset" / "H5Type.py"
    with open(h5type_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    readme_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            readme_lines.append(line)
        else:
            break

    readme = "".join(readme_lines)
    code = "".join(lines)

    return readme, code


def convert_tlio_to_sequence(sequence_dir: Path) -> Sequence:
    """
    将TLIO序列目录转换为Sequence对象

    Args:
        sequence_dir: 序列目录路径

    Returns:
        Sequence对象
    """
    sequence_id = sequence_dir.name

    # 加载npy数据 (global frame, 200Hz, 用于resampled和aligned)
    npy_path = sequence_dir / "imu0_resampled.npy"
    npy_data = np.load(npy_path)

    # 加载描述文件
    desc_path = sequence_dir / "imu0_resampled_description.json"
    with open(desc_path, "r") as f:
        description = json.load(f)

    # 提取数据
    # 列索引: 0:ts_us, 1-3:gyr_global, 4-6:acc_global,
    #        7-10:qxyzw, 11-13:pos, 14-16:vel
    t_us = npy_data[:, 0].astype(np.int64)  # 时间戳 (微秒)
    gyr_global = npy_data[:, 1:4]  # 陀螺仪 (global)
    acc_global = npy_data[:, 4:7]  # 加速度计 (global)
    qxyzw = npy_data[:, 7:11]  # 四元数 (xyzw)
    pos = npy_data[:, 11:14]  # 位置
    vel = npy_data[:, 14:17]  # 速度

    num_rows = len(t_us)
    t_start_us = int(t_us[0])
    t_end_us = int(t_us[-1])
    frequency_hz = description.get("approximate_frequency_hz", 200.0)

    # 创建序列属性
    attributes = SequenceAttributes(
        num_rows=num_rows,
        t_start_us=t_start_us,
        t_end_us=t_end_us,
        frequency_hz=frequency_hz,
        label="tlio_golden",
    )

    # 四元数从xyzw转换为wxyz
    qwxyz = np.column_stack([qxyzw[:, 3:4], qxyzw[:, 0:3]])  # [N, 4], wxyz

    # 准备真值数据 [N, 10]: pos(3), qwxyz(4), vel(3)
    ground_truth_array = np.column_stack([pos, qwxyz, vel])

    # 准备IMU数据 [1, N, 6]: gyr(3), acc(3) (global frame)
    imu_data_array = np.column_stack([gyr_global, acc_global])  # [N, 6]
    imu_data = imu_data_array[np.newaxis, :, :]  # [1, N, 6]

    # 准备磁力计数据 (TLIO没有磁力计，设为NaN)
    mag_data = np.full((1, num_rows, 3), np.nan, dtype=np.float32)

    # 创建ResampledData (使用global frame数据)
    resampled_data = ResampledData(
        t_us=t_us,  # [N] 统一时间戳(微秒)
        imu=imu_data,  # [1, N, 6] gyr(3), acc(3) - global frame
        mag=mag_data,  # [1, N, 3] mag(3), 全为NaN
        barom=None,  # 气压计数据为空
        ground_truth=ground_truth_array,  # [N, 10] pos(3), qwxyz(4), vel(3)
    )

    # 创建AlignedData (使用global frame数据)
    aligned_array = np.column_stack(
        [
            t_us.reshape(-1, 1),  # [N, 1]
            gyr_global,  # [N, 3] - global frame
            acc_global,  # [N, 3] - global frame
            pos,  # [N, 3]
            qwxyz,  # [N, 4]
            vel,  # [N, 3]
        ]
    )  # [N, 17]

    aligned_data = AlignedData(data=aligned_array)

    return Sequence(
        name=sequence_id,
        attributes=attributes,
        resampled=resampled_data,
        aligned=aligned_data,
    )


def load_sequence_lists(tlio_dir: Path) -> dict[str, list[str]]:
    """加载训练/验证/测试序列列表"""
    splits = {}

    for split_name in ["train", "val", "test"]:
        list_file = tlio_dir / f"{split_name}_list.txt"
        if list_file.exists():
            with open(list_file, "r") as f:
                sequence_ids = [line.strip() for line in f if line.strip()]
            splits[split_name] = sequence_ids
            print(f"  Loaded {len(sequence_ids)} sequences from {split_name}_list.txt")
        else:
            print(f"  Warning: {list_file} not found")
            splits[split_name] = []

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="将TLIO golden数据集转换为H5格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="TLIO golden数据集根目录路径"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="tlio_golden.h5", help="输出H5文件路径"
    )
    parser.add_argument(
        "--use_splits", action="store_true", help="使用数据集自带的train/val/test分割"
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_shuffle", action="store_true", help="不打乱序列顺序")

    args = parser.parse_args()

    tlio_dir = Path(args.input)
    output_path = Path(args.output)

    if not tlio_dir.exists():
        raise FileNotFoundError(f"TLIO directory not found: {tlio_dir}")

    print("> Converting TLIO golden dataset")
    print(f"  Input: {tlio_dir}")
    print(f"  Output: {output_path}")

    # 获取所有序列目录
    sequence_dirs = sorted(
        [d for d in tlio_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    print(f"\n> Found {len(sequence_dirs)} sequences")

    # 加载所有序列数据
    sequences = []
    total_duration_us = 0

    for seq_dir in sequence_dirs:
        print(f"  Loading {seq_dir.name}: ", end="", flush=True)
        sequence = convert_tlio_to_sequence(seq_dir)
        sequences.append(sequence)
        total_duration_us += (
            sequence.attributes.t_end_us - sequence.attributes.t_start_us
        )
        print(f"{sequence.attributes.num_rows} rows")

    total_duration_hours = total_duration_us / 1e6 / 3600

    print(f"\n> Loaded {len(sequences)} sequences")
    print(f"  Total duration: {total_duration_hours:.2f} hours")

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
    print(f"\n> Saving H5Dataset to: {output_path}")
    with H5Dataset(output_path, mode="w") as h5_dataset:
        # 写入元数据
        h5_dataset.create_metadata(metadata)
        print(
            f"  Created metadata: {len(sequences)} sequences, {total_duration_hours:.2f} hours"
        )

        # 写入每个序列
        for seq in sequences:
            h5_dataset.create_sequence(seq)

        print(f"  Saved {len(sequences)} sequences")

        # 处理序列分割
        if args.use_splits:
            print("\n> Using dataset splits")
            splits = load_sequence_lists(tlio_dir)

            seq_id_to_name = {seq.name: seq.name for seq in sequences}
            train_list = [
                seq_id_to_name[sid] for sid in splits["train"] if sid in seq_id_to_name
            ]
            val_list = [
                seq_id_to_name[sid] for sid in splits["val"] if sid in seq_id_to_name
            ]
            test_list = [
                seq_id_to_name[sid] for sid in splits["test"] if sid in seq_id_to_name
            ]

            print("\n  Dataset splits:")
            print(f"    Train: {len(train_list)} sequences")
            print(f"    Val: {len(val_list)} sequences")
            print(f"    Test: {len(test_list)} sequences")

            from base.dataset.H5Type import SequenceLists

            sequence_lists = SequenceLists(
                train=train_list, val=val_list, test=test_list
            )

            import h5py

            dt = h5py.string_dtype(encoding="utf-8")
            assert h5_dataset._file is not None
            list_group = h5_dataset._file.create_group("sequence_lists")
            list_group.create_dataset(
                "train", data=np.array(sequence_lists.train, dtype=dt)
            )
            list_group.create_dataset(
                "val", data=np.array(sequence_lists.val, dtype=dt)
            )
            list_group.create_dataset(
                "test", data=np.array(sequence_lists.test, dtype=dt)
            )

            h5_dataset._sequence_lists = sequence_lists
        else:
            print("\n> Creating automatic split")
            sequence_lists = h5_dataset.create_sequence_lists(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                shuffle=not args.no_shuffle,
                random_seed=args.random_seed,
            )

        # 创建索引映射
        index_maps = h5_dataset.create_index_maps()
        print("\n  Index maps:")
        print(f"    Train indices: {len(index_maps.train_index)} samples")
        print(f"    Val indices: {len(index_maps.val_index)} samples")
        print(f"    Test indices: {len(index_maps.test_index)} samples")

    print("\n> Successfully converted TLIO dataset to H5 format")
    print(f"  Output file: {output_path}")


if __name__ == "__main__":
    main()
