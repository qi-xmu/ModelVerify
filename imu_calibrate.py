#!/usr/bin/env python3
"""

感觉没什么用处emmmm

对 imu.csv 应用 IMU 标定（零偏 + 轴对齐/尺度），输出 imu_no.csv。

标定公式：
    acce_calibrated = Ra * acce_raw - Ba
    gyro_calibrated = Rg * gyro_raw - Bg

用法示例：
    python imu_calibrate.py --imu /path/to/imu.csv --device Redmi-K30-Pro
    python imu_calibrate.py --root-dir /path/to/dataset
    python imu_calibrate.py --imu /path/to/imu.csv --config /path/to/config.yaml --output /path/to/imu_no.csv
    python imu_calibrate.py --root-dir /path/to/dataset -p
    python imu_calibrate.py -u /path/to/unit_folder -p
    python imu_calibrate.py -u 20251105_103700_Redmi-K30-Pro
    python imu_calibrate.py -d /path/to/dataset -s Redmi-K30-Pro SM-G9900 -u unit1 unit2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import shutil

import numpy as np
import matplotlib.pyplot as plt
import rerun as rr

from base.datatype import ImuData
from base.serialize import ImuDataSerializer
from base.rerun_ext import RerunView, send_imu_data


AUX_FILES = ["gt.csv", "cam.csv", "opt.csv", "gt.png"]


def resolve_config_path(
    device: str | None,
    config_path: str | None,
    base_dir: Path,
    *,
    allow_default: bool = False,
) -> Path:
    # 优先使用手动指定的配置文件
    if config_path:
        return Path(config_path)
    if not device:
        if allow_default:
            return base_dir / "config" / "Default.yaml"
        raise ValueError("Either --device or --config must be provided.")
    return base_dir / "config" / f"{device}.yaml"


def load_raw_and_calibrated(imu_path: Path, config_path: Path) -> Tuple[ImuData, ImuData, bool]:
    # 读取原始 IMU（传感器坐标系）
    imu_sensor = ImuData.from_csv(imu_path)

    # 若存在 gt.csv，则先按 gt 时间戳插值；
    # raw：未校准 + GT 旋转到全局
    # cal：先在传感器系校准，再 GT 旋转到全局
    gt_path = imu_path.with_name("gt.csv")
    gt_used = False
    if gt_path.exists():
        from base.datatype import GroundTruthData

        gt_data = GroundTruthData.from_csv(gt_path)
        imu_sensor = imu_sensor.interpolate(gt_data.t_us)
        gt_used = True

    calibrated_sensor = ImuData(
        imu_sensor.t_us.copy(),
        imu_sensor.gyro.copy(),
        imu_sensor.acce.copy(),
        imu_sensor.ahrs,
        imu_sensor.magn.copy(),
        imu_sensor.frame,
    )
    calibrated_sensor.calibrate_with(config_path)

    if gt_used:
        raw = imu_sensor.transform(rots=gt_data.rots)
        calibrated = calibrated_sensor.transform(rots=gt_data.rots)
    else:
        raw = imu_sensor
        calibrated = calibrated_sensor

    return raw, calibrated, gt_used


def save_plot(raw: ImuData, calibrated: ImuData, output_path: Path) -> None:
    # 绘图内容保持英文，便于通用查看
    t = (raw.t_us - raw.t_us[0]) * 1e-6
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    labels = ["x", "y", "z"]
    for i, label in enumerate(labels):
        axes[0, i].plot(t, raw.gyro[:, i], linewidth=0.8, alpha=0.7, label="raw")
        axes[0, i].plot(t, calibrated.gyro[:, i], linewidth=0.8, alpha=0.8, label="cal")
        # 参考基准线与均值线
        axes[0, i].axhline(0.0, color="black", linewidth=0.8, alpha=0.5, label="baseline")
        axes[0, i].axhline(np.mean(raw.gyro[:, i]), color="tab:blue", linestyle="--", linewidth=0.8, alpha=0.7, label="raw mean")
        axes[0, i].axhline(np.mean(calibrated.gyro[:, i]), color="tab:orange", linestyle="--", linewidth=0.8, alpha=0.7, label="cal mean")
        axes[0, i].set_title(f"Gyro {label}")

    for i, label in enumerate(labels):
        axes[1, i].plot(t, raw.acce[:, i], linewidth=0.8, alpha=0.7, label="raw")
        axes[1, i].plot(t, calibrated.acce[:, i], linewidth=0.8, alpha=0.8, label="cal")
        # 加速度基准：x/y 为 0，z 为 9.81
        baseline = 9.81 if label == "z" else 0.0
        axes[1, i].axhline(baseline, color="black", linewidth=0.8, alpha=0.5, label="baseline")
        axes[1, i].axhline(np.mean(raw.acce[:, i]), color="tab:blue", linestyle="--", linewidth=0.8, alpha=0.7, label="raw mean")
        axes[1, i].axhline(np.mean(calibrated.acce[:, i]), color="tab:orange", linestyle="--", linewidth=0.8, alpha=0.7, label="cal mean")
        axes[1, i].set_title(f"Accel {label}")
        axes[1, i].set_xlabel("time (s)")

    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibrate_one(
    imu_path: Path,
    config_path: Path,
    output_path: Path,
    plot: bool,
    rerun_verify: bool,
) -> None:
    # 输出 imu_no.csv，并可选保存对比图
    raw, calibrated, gt_used = load_raw_and_calibrated(imu_path, config_path)
    ImuDataSerializer(calibrated).save(output_path)
    if plot:
        plot_path = output_path.with_name(f"{output_path.stem}_compare.png")
        save_plot(raw, calibrated, plot_path)
    if rerun_verify:
        if not gt_used:
            print("跳过 Rerun 验证：未找到 gt.csv")
        else:
            print("正在使用 Rerun 验证 raw/cal（全局坐标系）...")
            view = RerunView()
            view.add_imu_view(visible=True, tags=["raw_global", "cal_global"])
            rr.init("IMU Calibration Verification", spawn=True)
            view.send("IMU Calibration Verification")
            send_imu_data(raw, tag="raw_global")
            send_imu_data(calibrated, tag="cal_global")


def process_single(
    imu_path: Path,
    config_path: Path,
    output_path: Path | None,
    plot: bool,
    rerun_verify: bool,
) -> None:
    out_path = output_path if output_path else imu_path.with_name("imu_no.csv")
    calibrate_one(imu_path, config_path, out_path, plot, rerun_verify)
    print(f"Saved: {out_path}")


def copy_aux_files(src_unit_dir: Path, dst_unit_dir: Path) -> None:
    # 复制可选的配套文件（存在才复制）
    dst_unit_dir.mkdir(parents=True, exist_ok=True)
    for name in AUX_FILES:
        src_file = src_unit_dir / name
        if src_file.exists():
            shutil.copy2(src_file, dst_unit_dir / name)


def find_units_in_dataset(dataset_dir: Path, device_dirs: list[str] | None = None) -> list[Path]:
    # 在数据集目录中搜索所有包含 imu.csv 的单元目录
    units: list[Path] = []
    device_filter = set(device_dirs or [])
    for imu_file in dataset_dir.rglob("imu.csv"):
        try:
            rel = imu_file.relative_to(dataset_dir)
        except ValueError:
            continue
        if len(rel.parts) < 2:
            continue
        if device_filter and rel.parts[0] not in device_filter:
            continue
        units.append(imu_file.parent)
    return sorted({p.resolve() for p in units})


def build_output_unit_dir(
    unit_dir: Path,
    output_root: Path,
    dataset_roots: list[Path],
) -> Path:
    # 优先保持相对数据集根目录的层级结构
    unit_resolved = unit_dir.resolve()
    for root in dataset_roots:
        root_resolved = root.resolve()
        try:
            rel = unit_resolved.relative_to(root_resolved)
            return output_root / rel
        except ValueError:
            continue
    # 回退：仅使用单元目录名
    return output_root / unit_dir.name


def detect_device_name(root_dir: Path, imu_path: Path) -> str | None:
    # 通过数据集根目录下的一级子目录名识别设备
    try:
        rel = imu_path.relative_to(root_dir)
    except ValueError:
        return None
    if not rel.parts:
        return None
    return rel.parts[0]


def detect_device_by_unit(unit_dir: Path, base_dir: Path) -> str | None:
    # 仅从单元目录名中匹配配置文件名
    config_dir = base_dir / "config"
    if not config_dir.exists():
        return None

    candidates = []
    for config_file in config_dir.glob("*.yaml"):
        name = config_file.stem
        if name == "Default":
            continue
        if name in unit_dir.name:
            candidates.append(name)

    if not candidates:
        return None
    # 选择最长的匹配名，避免短名误匹配
    return sorted(candidates, key=len, reverse=True)[0]


def process_batch(root_dir: Path, base_dir: Path, plot: bool, rerun_verify: bool) -> None:
    imu_files = sorted(root_dir.rglob("imu.csv"))
    if not imu_files:
        print(f"No imu.csv found under: {root_dir}")
        return
    if rerun_verify:
        print("批量模式下不启动 Rerun 验证")
    for imu_path in imu_files:
        device_name = detect_device_name(root_dir, imu_path)
        config_path = resolve_config_path(
            device_name,
            None,
            base_dir,
            allow_default=True,
        )
        if not config_path.exists():
            raise FileNotFoundError(f"Calibration config not found: {config_path}")
        out_path = imu_path.with_name("imu_no.csv")
        calibrate_one(imu_path, config_path, out_path, plot, False)
        print(f"Saved: {out_path}")


def find_unit_dir_by_name(unit_name: str, search_roots: list[Path]) -> Path | None:
    # 在给定根目录列表中搜索唯一匹配
    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        matches.extend([p for p in root.rglob(unit_name) if (p / "imu.csv").exists()])
    # 去重，保持确定性排序
    uniq = sorted({p.resolve() for p in matches})
    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) > 1:
        # 优先选择当前目录下的直接匹配
        cwd_match = (Path.cwd() / unit_name).resolve()
        if cwd_match in uniq:
            return cwd_match
        raise ValueError(f"Multiple unit matches for {unit_name}: {uniq}")
    return None


def resolve_unit_dirs(
    dataset_dirs: list[Path],
    device_dirs: list[str],
    unit_inputs: list[str],
) -> list[Path]:
    unit_paths: list[Path] = []
    for unit_input in unit_inputs:
        unit_path = Path(unit_input)
        # 直接路径（绝对或包含路径分隔符）
        if unit_path.is_absolute() or ("/" in unit_input) or ("\\" in unit_input):
            resolved = (Path.cwd() / unit_path).resolve() if not unit_path.is_absolute() else unit_path
            unit_paths.append(resolved)
            continue

        # -s/-d 模式：按 dataset/device/unit 组合
        if device_dirs:
            roots = dataset_dirs if dataset_dirs else [Path.cwd()]
            for dataset_dir in roots:
                for device_dir in device_dirs:
                    unit_paths.append(dataset_dir / device_dir / unit_input)
            continue

        # 仅提供单元名：从当前目录和 dataset_compress 搜索
        search_roots = [Path.cwd(), Path.cwd() / "dataset_compress"]
        match = find_unit_dir_by_name(unit_input, search_roots)
        if match is not None:
            unit_paths.append(match)
        else:
            unit_paths.append((Path.cwd() / unit_input).resolve())

    return unit_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate imu.csv and save imu_no.csv")
    parser.add_argument("--imu", type=str, help="Path to imu.csv")
    parser.add_argument("--root-dir", type=str, help="Root directory for batch processing")
    parser.add_argument("--device", type=str, help="Device name, e.g. Redmi-K30-Pro")
    parser.add_argument("--config", type=str, help="Calibration yaml path (overrides --device)")
    parser.add_argument("--output", type=str, help="Output path for single imu.csv")
    parser.add_argument("-o", "--output-dir", type=str, help="Output dataset root directory")
    parser.add_argument("-p", "--plot", action="store_true", help="Save before/after comparison plot")
    parser.add_argument("-rv", "--rerun-verify", action="store_true", help="Use Rerun to verify raw/cal data")
    parser.add_argument("-d", "--dataset-dir", nargs="+", type=str, help="Dataset root dir")
    parser.add_argument("-s", "--device-dir", nargs="+", type=str, help="Device folder name under dataset")
    parser.add_argument("-u", "--unit-dir", nargs="+", type=str, help="Unit folder path or name")

    args = parser.parse_args()

    if args.imu and (args.root_dir or args.unit_dir or args.dataset_dir or args.device_dir):
        parser.error("Use only one input mode: --imu, --root-dir, or -u/-d/-s.")
    if args.root_dir and (args.unit_dir or args.dataset_dir or args.device_dir):
        parser.error("Use only one input mode: --root-dir or -u/-d/-s.")

    # 以项目根目录为基准（脚本位于项目根下）
    base_dir = Path(__file__).resolve().parent

    unit_dirs: list[Path] = []
    dataset_roots_for_output: list[Path] = []
    if args.unit_dir:
        dataset_dirs = [Path(p) for p in args.dataset_dir] if args.dataset_dir else []
        device_dirs = args.device_dir if args.device_dir else []
        unit_dirs = resolve_unit_dirs(dataset_dirs, device_dirs, args.unit_dir)
        dataset_roots_for_output = [p.resolve() for p in dataset_dirs]

    # -d 可直接批量处理数据集（不需要 -u）
    if args.dataset_dir and not args.unit_dir:
        dataset_dirs = [Path(p) for p in args.dataset_dir]
        device_dirs = args.device_dir if args.device_dir else []
        for ds in dataset_dirs:
            if not ds.exists():
                raise FileNotFoundError(f"Dataset dir not found: {ds}")
            unit_dirs.extend(find_units_in_dataset(ds, device_dirs))
        unit_dirs = sorted({p.resolve() for p in unit_dirs})
        dataset_roots_for_output = [p.resolve() for p in dataset_dirs]

    if args.imu:
        if args.output_dir:
            print("-o/--output-dir is ignored in --imu mode")
        config_path = resolve_config_path(args.device, args.config, base_dir)
        if not config_path.exists():
            raise FileNotFoundError(f"Calibration config not found: {config_path}")
        imu_path = Path(args.imu)
        if not imu_path.exists():
            raise FileNotFoundError(f"imu.csv not found: {imu_path}")
        output_path = Path(args.output) if args.output else None
        print(f"Using calibration: {config_path}")
        process_single(imu_path, config_path, output_path, args.plot, args.rerun_verify)
    elif args.root_dir:
        root_dir = Path(args.root_dir)
        if not root_dir.exists():
            raise FileNotFoundError(f"Root dir not found: {root_dir}")
        if args.device or args.config:
            print("--device/--config are ignored in batch mode; use device folders with matching config names.")
        if args.output:
            print("--output is ignored in batch mode; outputs are saved next to imu.csv.")
        print(f"Using calibration: auto (device match or Default)")
        if args.output_dir:
            output_root = Path(args.output_dir)
            for imu_path in sorted(root_dir.rglob("imu.csv")):
                unit_path = imu_path.parent
                device_name = detect_device_name(root_dir, imu_path)
                config_path = resolve_config_path(device_name, None, base_dir, allow_default=True)
                if not config_path.exists():
                    raise FileNotFoundError(f"Calibration config not found: {config_path}")
                out_unit = build_output_unit_dir(unit_path, output_root, [root_dir])
                copy_aux_files(unit_path, out_unit)
                out_imu = out_unit / "imu.csv"
                print(f"Using calibration: {config_path}")
                process_single(imu_path, config_path, out_imu, args.plot, False)
        else:
            process_batch(root_dir, base_dir, args.plot, args.rerun_verify)
    elif unit_dirs:
        output_path = Path(args.output) if args.output else None
        output_root = Path(args.output_dir) if args.output_dir else None
        for unit_path in unit_dirs:
            if not unit_path.exists():
                raise FileNotFoundError(f"Unit dir not found: {unit_path}")
            imu_path = unit_path / "imu.csv"
            if not imu_path.exists():
                raise FileNotFoundError(f"imu.csv not found: {imu_path}")
            device_name = detect_device_by_unit(unit_path, base_dir)
            config_path = resolve_config_path(device_name, args.config, base_dir, allow_default=True)
            if not config_path.exists():
                raise FileNotFoundError(f"Calibration config not found: {config_path}")
            print(f"Using calibration: {config_path}")
            if output_root is not None:
                out_unit = build_output_unit_dir(unit_path, output_root, dataset_roots_for_output)
                copy_aux_files(unit_path, out_unit)
                out_imu = out_unit / "imu.csv"
                process_single(imu_path, config_path, out_imu, args.plot, args.rerun_verify)
            else:
                process_single(imu_path, config_path, output_path, args.plot, args.rerun_verify)
    else:
        parser.error("Either --imu, --root-dir, or -u/-d/-s must be provided.")


if __name__ == "__main__":
    main()
