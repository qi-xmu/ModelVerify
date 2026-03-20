#!/usr/bin/env python3
"""
将多个设备数据集汇总到一个 H5 文件。

用法:
    uv run converters/GenerateH5DatasetMulti.py -d /path/to/dataset_compress -o /path/to/output/atr.h5

    # 使用 opt.csv 作为真值
    uv run converters/GenerateH5DatasetMulti.py -d /path/to/dataset_compress -o /path/to/output/atr_opt.h5 --use_opt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 兼容直接运行脚本时的导入路径
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from base.dataset.H5Type import H5Dataset, Metadata
from base.datatype import DeviceDataset
from converters.GenerateH5Dataset import unit_data_to_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将多个设备数据集汇总成一个 H5")
    parser.add_argument(
        "-d",
        "--dataset_root",
        required=True,
        help="设备数据集根目录（包含多个设备目录）",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="输出 H5 文件路径 (例如: /path/to/atr.h5)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="训练集比例 (默认: 0.7)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="验证集比例 (默认: 0.15)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="测试集比例 (默认: 0.15)",
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="不打乱序列顺序",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--use_opt",
        action="store_true",
        help="使用 opt.csv 作为真值 (存在时)",
    )
    return parser.parse_args()


def iter_device_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def get_h5type_code() -> tuple[str, str]:
    """
    读取 H5Type.py 文件内容

    Returns:
        tuple: (readme, code)
    """
    h5type_path = ROOT_DIR / "src" / "base" / "dataset" / "H5Type.py"
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


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 遍历根目录下所有设备文件夹
    device_dirs = iter_device_dirs(dataset_root)
    if not device_dirs:
        raise ValueError(f"未在目录下找到设备数据集: {dataset_root}")

    sequences = []
    total_distance_m = 0.0

    # 汇总多个设备的数据
    for device_dir in device_dirs:
        datas = DeviceDataset(device_dir)
        print(f"\n> 设备: {datas.device_name}")

        for ud in datas:
            print(f"> 处理 {ud.name}")
            ud.load_data(using_opt=args.use_opt)
            # 单个 UnitData 转为标准 Sequence
            sequence = unit_data_to_sequence(ud)
            print(f"  已创建序列: {sequence.name}")
            print(f"    行数: {sequence.attributes.num_rows}")
            print(
                f"    时长: {sequence.attributes.t_end_us - sequence.attributes.t_start_us} us"
            )
            print(f"    频率: {sequence.attributes.frequency_hz} Hz")
            if sequence.resampled and sequence.resampled.ground_truth is not None:
                # ground_truth: [N, 10] -> pos(0:3)
                pos = sequence.resampled.ground_truth[:, 0:3]
                if len(pos) > 1:
                    diffs = np.diff(pos, axis=0)
                    seq_distance_m = float(np.linalg.norm(diffs, axis=1).sum())
                else:
                    seq_distance_m = 0.0
                total_distance_m += seq_distance_m
                print(
                    f"    真值形状: {sequence.resampled.ground_truth.shape}"
                )
                print(f"    路程: {seq_distance_m:.2f} m")
            sequences.append(sequence)

    if not sequences:
        print("! 没有可写入的序列")
        return

    print(f"\n> 写入 H5Dataset: {output_path}")
    # 总时长与总路程统计
    total_duration_us = sum(
        seq.attributes.t_end_us - seq.attributes.t_start_us for seq in sequences
    )
    total_duration_hours = total_duration_us / 1e6 / 3600

    readme, code = get_h5type_code()
    metadata = Metadata(
        version="1.0",
        num_sequences=len(sequences),
        total_duration_hours=total_duration_hours,
        readme=readme,
        code=code,
    )

    with H5Dataset(output_path, mode="w") as h5_dataset:
        h5_dataset.create_metadata(metadata)
        print(
            f"  已创建元数据: {len(sequences)} 个序列, {total_duration_hours:.2f} 小时"
        )

        for seq in sequences:
            h5_dataset.create_sequence(seq)
            print(f"  已写入序列: {seq.name} ({seq.attributes.num_rows} 行)")

        sequence_lists = h5_dataset.create_sequence_lists(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            shuffle=not args.no_shuffle,
            random_seed=args.random_seed,
        )
        print("\n  序列划分:")
        print(f"    训练: {len(sequence_lists.train)} 个序列")
        print(f"    验证: {len(sequence_lists.val)} 个序列")
        print(f"    测试: {len(sequence_lists.test)} 个序列")

        index_maps = h5_dataset.create_index_maps()
        print("\n  索引映射:")
        print(f"    训练索引: {len(index_maps.train_index)} 个样本")
        print(f"    验证索引: {len(index_maps.val_index)} 个样本")
        print(f"    测试索引: {len(index_maps.test_index)} 个样本")

    total_distance_km = total_distance_m / 1000.0
    print(f"\n✓ 已生成 H5Dataset, 共 {len(sequences)} 个序列")
    print(f"  输出文件: {output_path}")
    print(f"  总时长: {total_duration_hours:.2f} 小时")
    print(f"  总路程: {total_distance_m:.2f} m ({total_distance_km:.3f} km)")


if __name__ == "__main__":
    main()
