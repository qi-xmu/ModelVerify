#!/usr/bin/env python3
"""将 H5Type 数据集转换为 TLIO 格式目录。

输出结构示例:
    output_dir/
    ├── sequence_001/
    │   ├── imu0_resampled.npy
    │   ├── imu0_resampled.csv  (可选)
    │   └── imu0_resampled_description.json
    ├── sequence_002/
    │   ├── imu0_resampled.npy
    │   ├── imu0_resampled.csv  (可选)
    │   └── imu0_resampled_description.json
    ├── train_list.txt  (可选)
    ├── val_list.txt    (可选)
    └── test_list.txt   (可选)

TLIO npy 列定义:
    0: ts_us(1)
    1-3: gyr_compensated_rotated_in_World(3)
    4-6: acc_compensated_rotated_in_World(3)
    7-10: qxyzw_World_Device(4)
    11-13: pos_World_Device(3)
    14-16: vel_World(3)

用法:
    uv run python H5Dataset2TLIO.py -i input.h5 -o /path/to/tlio_out --export_split_lists
    uv run python H5Dataset2TLIO.py -i input.h5 -o /path/to/tlio_out --export_csv --float_decimals 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from base.dataset.H5Type import H5Dataset, Sequence

JSON_TEMPLATE = {
    "columns_name(width)": [
        "ts_us(1)",
        "gyr_compensated_rotated_in_World(3)",
        "acc_compensated_rotated_in_World(3)",
        "qxyzw_World_Device(4)",
        "pos_World_Device(3)",
        "vel_World(3)",
    ],
    "num_rows": 0,
    "approximate_frequency_hz": 0.0,
    "t_start_us": 0.0,
    "t_end_us": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 H5Type 数据集转换为 TLIO 格式")
    parser.add_argument("-i", "--input", required=True, help="输入 H5 文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出 TLIO 目录路径")
    parser.add_argument(
        "--imu_sensor_index",
        type=int,
        default=0,
        help="从 resampled.imu 中选用的传感器索引，默认 0",
    )
    parser.add_argument(
        "--export_split_lists",
        action="store_true",
        help="导出 train_list.txt / val_list.txt / test_list.txt",
    )
    parser.add_argument(
        "--skip_invalid",
        action="store_true",
        help="遇到无可用数据的序列时跳过，而不是终止",
    )
    parser.add_argument(
        "--float_decimals",
        type=int,
        default=6,
        help="浮点列统一保留的小数位数（用于 CSV，且会量化写入 npy），默认 6",
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="在每个序列目录额外导出 imu0_resampled.csv（固定小数位）",
    )
    return parser.parse_args()


def _safe_frequency_hz(t_us: np.ndarray, fallback_hz: float) -> float:
    if t_us.size < 2:
        return float(fallback_hz)

    dt_us = np.diff(t_us.astype(np.float64))
    dt_us = dt_us[dt_us > 0]
    if dt_us.size == 0:
        return float(fallback_hz)

    return float(1e6 / np.median(dt_us))


def _build_from_resampled(
    sequence: Sequence,
    imu_sensor_index: int,
) -> np.ndarray:
    assert sequence.resampled is not None
    t_us = sequence.resampled.get_t_us()
    imu = sequence.resampled.get_imu()
    gt = sequence.resampled.get_ground_truth()

    if t_us is None or imu is None or gt is None:
        raise ValueError("resampled 缺少必要字段 t_us/imu/ground_truth")

    if imu.ndim != 3 or imu.shape[2] < 6:
        raise ValueError(f"resampled.imu 形状异常: {imu.shape}, 期望 [M, N, >=6]")

    if imu_sensor_index < 0 or imu_sensor_index >= imu.shape[0]:
        raise IndexError(
            f"imu_sensor_index={imu_sensor_index} 越界, 可选范围 [0, {imu.shape[0] - 1}]"
        )

    if gt.ndim != 2 or gt.shape[1] < 10:
        raise ValueError(
            f"resampled.ground_truth 形状异常: {gt.shape}, 期望 [N, >=10]"
        )

    gyr = imu[imu_sensor_index, :, 0:3]
    acc = imu[imu_sensor_index, :, 3:6]
    pos = gt[:, 0:3]
    qwxyz = gt[:, 3:7]
    vel = gt[:, 7:10]

    # qwxyz -> qxyzw
    qxyzw = np.column_stack([qwxyz[:, 1:4], qwxyz[:, 0:1]])

    return np.column_stack([t_us.reshape(-1, 1), gyr, acc, qxyzw, pos, vel])


def _build_from_aligned(sequence: Sequence) -> np.ndarray:
    assert sequence.aligned is not None
    aligned = sequence.aligned.get_data()

    if aligned is None:
        raise ValueError("aligned 数据为空")

    if aligned.ndim != 2 or aligned.shape[1] < 17:
        raise ValueError(f"aligned 形状异常: {aligned.shape}, 期望 [N, >=17]")

    t_us = aligned[:, 0:1]
    gyr = aligned[:, 1:4]
    acc = aligned[:, 4:7]
    pos = aligned[:, 7:10]
    qwxyz = aligned[:, 10:14]
    vel = aligned[:, 14:17]

    # qwxyz -> qxyzw
    qxyzw = np.column_stack([qwxyz[:, 1:4], qwxyz[:, 0:1]])

    return np.column_stack([t_us, gyr, acc, qxyzw, pos, vel])


def sequence_to_tlio_array(sequence: Sequence, imu_sensor_index: int) -> np.ndarray:
    # 优先导出 aligned 数据
    if sequence.aligned is not None:
        return _build_from_aligned(sequence)

    # 如果没有 aligned，再导出 resampled
    if sequence.resampled is not None:
        return _build_from_resampled(sequence, imu_sensor_index)

    raise ValueError("序列既没有 aligned 也没有 resampled 数据")


def _quantize_tlio_array(tlio_data: np.ndarray, float_decimals: int) -> np.ndarray:
    """统一浮点列小数位，时间列保持整数微秒语义。"""
    if float_decimals < 0:
        raise ValueError(f"float_decimals 不能为负数: {float_decimals}")

    quantized = tlio_data.astype(np.float64, copy=True)
    quantized[:, 0] = np.rint(quantized[:, 0])
    if quantized.shape[1] > 1:
        quantized[:, 1:] = np.round(quantized[:, 1:], decimals=float_decimals)
    return quantized


def _export_tlio_csv(tlio_data: np.ndarray, csv_path: Path, float_decimals: int) -> None:
    """导出固定精度 CSV，便于表格工具稳定渲染。"""
    header = (
        "timestamp_sec,gyr_x,gyr_y,gyr_z,acc_x,acc_y,acc_z,"
        "quat_x,quat_y,quat_z,quat_w,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z"
    )

    timestamp_sec = tlio_data[:, 0:1] / 1e6
    csv_data = np.column_stack([timestamp_sec, tlio_data[:, 1:]])

    fmt = [f"%.{float_decimals}f"] * csv_data.shape[1]
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header=header,
        comments="",
        fmt=fmt,
    )


def save_tlio_sequence(
    sequence: Sequence,
    output_dir: Path,
    imu_sensor_index: int,
    float_decimals: int,
    export_csv: bool,
) -> int:
    seq_dir = output_dir / sequence.name
    seq_dir.mkdir(parents=True, exist_ok=True)

    tlio_data = sequence_to_tlio_array(sequence, imu_sensor_index)
    tlio_data = _quantize_tlio_array(tlio_data, float_decimals)
    npy_path = seq_dir / "imu0_resampled.npy"
    np.save(npy_path, tlio_data)

    if export_csv:
        csv_path = seq_dir / "imu0_resampled.csv"
        _export_tlio_csv(tlio_data, csv_path, float_decimals)

    t_us = tlio_data[:, 0]
    frequency_hz = _safe_frequency_hz(t_us, sequence.attributes.frequency_hz)

    desc = dict(JSON_TEMPLATE)
    desc["num_rows"] = int(tlio_data.shape[0])
    desc["approximate_frequency_hz"] = float(frequency_hz)
    desc["t_start_us"] = float(t_us[0])
    desc["t_end_us"] = float(t_us[-1])

    desc_path = seq_dir / "imu0_resampled_description.json"
    with open(desc_path, "w", encoding="utf-8") as f:
        json.dump(desc, f, indent=4)

    return int(tlio_data.shape[0])


def _write_split_lists(h5_dataset: H5Dataset, output_dir: Path) -> None:
    sequence_lists = h5_dataset.sequence_lists
    if sequence_lists is None:
        try:
            sequence_lists = h5_dataset.load_sequence_lists()
        except Exception:
            print("! 未找到 sequence_lists，跳过导出 train/val/test 列表")
            return

    split_map = {
        "train_list.txt": sequence_lists.train,
        "val_list.txt": sequence_lists.val,
        "test_list.txt": sequence_lists.test,
    }

    for filename, names in split_map.items():
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            for name in names:
                f.write(f"{name}\n")

    print("  已导出 split 列表: train_list.txt / val_list.txt / test_list.txt")


def main() -> None:
    args = parse_args()

    input_h5 = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.float_decimals < 0:
        raise ValueError("--float_decimals 必须 >= 0")

    if not input_h5.exists():
        raise FileNotFoundError(f"输入 H5 文件不存在: {input_h5}")

    print("> Converting H5Type dataset to TLIO format")
    print(f"  Input:  {input_h5}")
    print(f"  Output: {output_dir}")

    ok_count = 0
    total_rows = 0
    errors: list[tuple[str, str]] = []

    with H5Dataset(input_h5, mode="r") as h5_dataset:
        sequence_names = h5_dataset.list_sequences()
        print(f"\n> Found {len(sequence_names)} sequences")

        for name in sequence_names:
            print(f"  Export {name}: ", end="", flush=True)
            try:
                sequence = h5_dataset.load_sequence(name)
                row_count = save_tlio_sequence(
                    sequence=sequence,
                    output_dir=output_dir,
                    imu_sensor_index=args.imu_sensor_index,
                    float_decimals=args.float_decimals,
                    export_csv=args.export_csv,
                )
                ok_count += 1
                total_rows += row_count
                print(f"{row_count} rows")
            except Exception as exc:  # noqa: BLE001
                errors.append((name, str(exc)))
                print("failed")
                if not args.skip_invalid:
                    raise

        if args.export_split_lists:
            _write_split_lists(h5_dataset, output_dir)

    print("\n> Done")
    print(f"  Exported sequences: {ok_count}")
    print(f"  Total rows: {total_rows}")

    if errors:
        print(f"  Failed sequences: {len(errors)}")
        for seq_name, err_msg in errors:
            print(f"    - {seq_name}: {err_msg}")


if __name__ == "__main__":
    main()
