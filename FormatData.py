#!/usr/bin/env python3
"""Organize dataset002 by phone model and extract key files.

Expected input structure:
  data/dataset002/00x/xxx_xxx_xx/xxx_xxx_xx/

We copy imu-c.csv, gt.csv, cam.csv, gt.png to:
  <output>/<Model-Name>/<xxx_xxx_xx>/
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

TARGET_FILES = ("imu-c.csv", "gt.csv", "cam.csv", "gt.png")
# Map source filenames to target filenames
FILE_MAPPING = {
    "imu-c.csv": "imu.csv",  # Rename imu-c.csv to imu.csv
    "gt.csv": "gt.csv",
    "cam.csv": "cam.csv",
    "gt.png": "gt.png",
}


def normalize_folder_name(name: str) -> str:
    """Normalize folder name: convert spaces to hyphens."""
    normalized = re.sub(r"\s+", "-", name.strip())
    normalized = re.sub(r"-+", "-", normalized)
    return normalized


def extract_model(folder_name: str) -> str:
    parts = folder_name.split("_", 2)
    if len(parts) >= 3:
        model = normalize_model_name(parts[2])
        return model
    return "Unknown-Model"


def normalize_model_name(raw: str) -> str:
    name = raw.strip().strip("_")
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name or "Unknown-Model"


def collect_source_folders(root: Path) -> list[Path]:
    folders: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        # Skip Calibration folders
        if "Calibration" in path.parts:
            continue
        if any((path / fname).exists() for fname in TARGET_FILES):
            folders.append(path)
    return folders


def copy_target_files(src_dir: Path, dst_dir: Path) -> list[str]:
    copied: list[str] = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_fname, dst_fname in FILE_MAPPING.items():
        src = src_dir / src_fname
        if src.exists():
            shutil.copy2(src, dst_dir / dst_fname)
            copied.append(dst_fname)
    return copied


def organize(input_dir: Path, output_dir: Path, dry_run: bool, force_overwrite: bool = False) -> None:
    src_folders = collect_source_folders(input_dir)
    if not src_folders:
        print("未找到源文件夹。")
        return

    total_files = 0
    conflicts: list[tuple[Path, Path]] = []
    processed_targets: set[Path] = set()

    for src in sorted(src_folders):
        model = extract_model(src.name)
        # Normalize the folder name: convert spaces to hyphens
        normalized_folder_name = normalize_folder_name(src.name)
        dst = output_dir / model / normalized_folder_name

        # Check for conflicts
        if dst in processed_targets:
            conflicts.append((src, dst))
            if not force_overwrite:
                print(f"[冲突] 跳过（已存在）：{dst}")
                continue
            else:
                print(f"[冲突] 覆盖：{dst}")

        existing = [f for f in TARGET_FILES if (src / f).exists()]
        if not existing:
            continue

        if dry_run:
            print(f"[模拟运行] {src} -> {dst} :: {', '.join(existing)}")
            processed_targets.add(dst)
            continue

        copied = copy_target_files(src, dst)
        total_files += len(copied)
        processed_targets.add(dst)
        if copied:
            print(f"{src} -> {dst} :: {', '.join(copied)}")

    if conflicts:
        print(f"\n{'='*60}")
        print(f"发现冲突：{len(conflicts)} 个重复的目标文件夹")
        print(f"{'='*60}")
        for src, dst in conflicts:
            print(f"  源：{src}")
            print(f"  目标：{dst}")
            print()

    if not dry_run:
        print(f"完成。已复制 {total_files} 个文件到 {output_dir}。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="整理 dataset002 并提取 imu-c.csv、gt.csv、cam.csv、gt.png。"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="data/dataset002",
        help="输入数据集根目录（默认：data/dataset002）",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="输出目录，例如 data/dataset002_compress",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="显示将要复制的文件但不进行实际操作",
    )
    parser.add_argument(
        "-r",
        "--force-overwrite",
        action="store_true",
        help="强制覆盖重复的目标文件夹（默认：跳过）",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"输入路径不存在：{input_dir}")

    organize(input_dir, output_dir, args.dry_run, args.force_overwrite)


if __name__ == "__main__":
    main()