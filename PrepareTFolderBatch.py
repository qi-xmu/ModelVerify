#!/usr/bin/env python3
"""
批量整理 t 目录中的压缩包和数据库。

流程:
1. 解压 root 目录下所有 zip 文件，并在成功后删除 zip。
2. 调用 ReadImuCamFromDB.py 导出已解压子目录中同名 db 的 imu.csv / cam.csv。
3. 可选删除子目录中“被读取”的 db / db-journal。
4. 扫描形如 YYYYMMDD_HHMMSS_DEVICE 的设备目录。
5. 扫描 root 目录下形如 YYMMDD-HHMMSS.db 的顶层数据库，并按时间排序。
6. 按时间顺序将每个顶层 db 复制到匹配设备目录中（仅复制，不再导出）。

示例:
  uv run python PrepareTFolderBatch.py -d /remote/path/to/t

只做匹配检查，不实际改动:
  uv run python PrepareTFolderBatch.py -d /remote/path/to/t --dry-run

只解压，跳过后面:
    uv run python PrepareTFolderBatch.py -d /remote/path/to/t --extract-only
只解压+读取，跳过后面:
    uv run python PrepareTFolderBatch.py -d /remote/path/to/t --read-only

解压+读取后删除被读取的子目录 db / db-journal，跳过后面:
    uv run python PrepareTFolderBatch.py -d /remote/path/to/t --read-only --cleanup-read
解压+读取后删除被读取的子目录 db / db-journal，再做匹配复制:
    uv run python PrepareTFolderBatch.py -d /remote/path/to/t --cleanup-read

全部复制完成后删除 root 顶层 db / db-journal:
    uv run python PrepareTFolderBatch.py -d /remote/path/to/t --cleanup-root-db
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ReadImuCamFromDB import process_single_database


DEVICE_DIR_PATTERN = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_(?P<device>.+)$")
ROOT_DB_PATTERN = re.compile(r"^(?P<date>\d{6})-(?P<time>\d{6})\.db$")
DEFAULT_DEVICES = ["SM-G9900", "Redmi-K30-Pro", "ABR-AL60"]


@dataclass(frozen=True)
class DeviceFolder:
    path: Path
    device: str
    timestamp: datetime


@dataclass(frozen=True)
class RootDatabase:
    path: Path
    timestamp: datetime


@dataclass(frozen=True)
class MappingGroup:
    source_db: RootDatabase
    device_folders: list[DeviceFolder]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量整理 t 目录中的 zip 和 db")
    parser.add_argument(
        "-d",
        type=Path,
        required=True,
        help="t 目录路径",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=DEFAULT_DEVICES,
        help="参与匹配的设备名，默认是三台设备",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="跳过 zip 解压步骤",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="解压后保留 zip 文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划，不执行解压、复制、导出和删除",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="解压时允许覆盖已存在文件（默认不覆盖）",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="只执行解压（及可选删除 zip），不做后续 db 匹配、复制与导出",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="只做解压+读取导出，跳过后续匹配复制",
    )
    parser.add_argument(
        "--cleanup-read",
        action="store_true",
        help="读取导出后，删除子目录中被读取的同名 db / db-journal",
    )
    parser.add_argument(
        "--cleanup-root-db",
        action="store_true",
        help="全部复制顶层 db 完成后，删除 root 顶层 db / db-journal",
    )

    # 兼容旧参数（隐藏）
    parser.add_argument("--overwrite-extracted", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cleanup-read-db", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--keep-source-db", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--keep-copied-db", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--keep-all-db", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cleanup-db", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_device_folder(path: Path) -> DeviceFolder | None:
    match = DEVICE_DIR_PATTERN.match(path.name)
    if not match:
        return None

    timestamp = datetime.strptime(
        f"{match.group('date')}_{match.group('time')}",
        "%Y%m%d_%H%M%S",
    )
    return DeviceFolder(path=path, device=match.group("device"), timestamp=timestamp)


def parse_root_database(path: Path) -> RootDatabase | None:
    match = ROOT_DB_PATTERN.match(path.name)
    if not match:
        return None

    timestamp = datetime.strptime(
        f"20{match.group('date')}-{match.group('time')}",
        "%Y%m%d-%H%M%S",
    )
    return RootDatabase(path=path, timestamp=timestamp)


def journal_path(db_path: Path) -> Path:
    return db_path.parent / f"{db_path.name}-journal"


def remove_file_if_exists(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return

    if dry_run:
        print(f"[dry-run] 删除: {path}")
        return

    path.unlink()
    print(f"删除: {path}")


def extract_archives(
    root_dir: Path,
    keep_zip: bool,
    dry_run: bool,
    overwrite_existing: bool,
) -> list[Path]:
    zip_files = sorted(root_dir.glob("*.zip"))
    if not zip_files:
        print("未发现 zip 文件，跳过解压")
        return []

    if shutil.which("unzip") is None:
        raise RuntimeError("系统未找到 unzip 命令，请先安装 unzip")

    extracted_archives: list[Path] = []
    skipped_archives: list[Path] = []
    print(f"发现 {len(zip_files)} 个 zip 文件")
    for zip_path in zip_files:
        print(f"解压: {zip_path}")
        if dry_run:
            extracted_archives.append(zip_path)
            if not keep_zip:
                remove_file_if_exists(zip_path, dry_run=dry_run)
            continue

        if not zipfile.is_zipfile(zip_path):
            print(f"跳过非法 zip 文件: {zip_path}")
            skipped_archives.append(zip_path)
            continue

        try:
            # 先做完整性测试
            test_cmd = ["unzip", "-tqq", str(zip_path)]
            subprocess.run(test_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 默认不覆盖(-n)；显式开启覆盖时使用(-o)
            unzip_mode = "-o" if overwrite_existing else "-n"
            extract_cmd = ["unzip", unzip_mode, str(zip_path), "-d", str(root_dir)]
            result = subprocess.run(
                extract_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if result.stdout.strip():
                print(result.stdout.strip())
            print(f"完成解压: {zip_path.name}")
        except subprocess.CalledProcessError as exc:
            output = exc.stdout.strip() if exc.stdout else str(exc)
            print(f"跳过损坏或不可解压 zip 文件: {zip_path}, 错误: {output}")
            skipped_archives.append(zip_path)
            continue

        extracted_archives.append(zip_path)
        if not keep_zip:
            remove_file_if_exists(zip_path, dry_run=dry_run)

    if skipped_archives:
        print(f"跳过了 {len(skipped_archives)} 个非法或损坏的 zip 文件")

    return extracted_archives


def collect_device_folders(root_dir: Path, devices: Iterable[str]) -> dict[str, list[DeviceFolder]]:
    device_set = set(devices)
    collected: dict[str, list[DeviceFolder]] = {device: [] for device in device_set}

    for child in root_dir.iterdir():
        if not child.is_dir():
            continue

        folder = parse_device_folder(child)
        if folder is None or folder.device not in device_set:
            continue
        collected[folder.device].append(folder)

    for device in collected:
        collected[device].sort(key=lambda item: item.timestamp)

    return collected


def collect_root_databases(root_dir: Path) -> list[RootDatabase]:
    databases: list[RootDatabase] = []
    for child in root_dir.iterdir():
        if not child.is_file() or child.name.endswith(".db-journal"):
            continue

        db_info = parse_root_database(child)
        if db_info is not None:
            databases.append(db_info)

    databases.sort(key=lambda item: item.timestamp)
    return databases


def print_db_tables(db_path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] 检查表结构: {db_path}")
        return

    if not db_path.exists():
        print(f"数据库不存在，无法检查表结构: {db_path}")
        return

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        print(f"读取数据库表结构失败: {db_path}, 错误: {exc}")
        return
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if tables:
        print(f"数据库表({db_path.name}): {', '.join(tables)}")
    else:
        print(f"数据库表({db_path.name}): 无用户表")


def build_mapping(
    root_databases: list[RootDatabase],
    device_folders: dict[str, list[DeviceFolder]],
) -> list[MappingGroup]:
    if not root_databases:
        raise ValueError("root 目录下没有找到形如 YYMMDD-HHMMSS.db 的数据库文件")

    required_count = len(root_databases)
    participating_devices = sorted(
        device for device, folders in device_folders.items() if len(folders) >= required_count
    )
    skipped_devices = sorted(
        device for device, folders in device_folders.items() if len(folders) < required_count
    )

    if skipped_devices:
        detail = ", ".join(f"{device}={len(device_folders[device])}" for device in skipped_devices)
        print(f"跳过目录数量不足的设备: {detail}; 需要 {required_count} 个目录")

    if not participating_devices:
        detail = ", ".join(f"{device}={len(folders)}" for device, folders in sorted(device_folders.items()))
        raise ValueError(f"没有可参与匹配的完整设备: {detail}, db={required_count}")

    mapping: list[MappingGroup] = []
    for index, root_db in enumerate(root_databases):
        folders = [
            device_folders[device][index]
            for device in participating_devices
        ]

        mapping.append(MappingGroup(source_db=root_db, device_folders=folders))

    return mapping


def summarize_mapping(mapping: list[MappingGroup]) -> None:
    print(f"共匹配 {len(mapping)} 组")
    participating_devices = {folder.device for group in mapping for folder in group.device_folders}
    for index, group in enumerate(mapping, start=1):
        present_devices = {folder.device for folder in group.device_folders}
        missing_devices = sorted(participating_devices - present_devices)
        folder_names = ", ".join(folder.path.name for folder in group.device_folders)
        if missing_devices:
            print(
                f"[{index:02d}] {group.source_db.path.name} -> {folder_names} "
                f"(缺少: {', '.join(missing_devices)})"
            )
        else:
            print(f"[{index:02d}] {group.source_db.path.name} -> {folder_names}")


def copy_db_to_folder(source_db: Path, target_folder: Path, dry_run: bool) -> Path:
    target_db = target_folder / source_db.name
    if dry_run:
        print(f"[dry-run] 复制: {source_db} -> {target_db}")
        return target_db

    shutil.copy2(source_db, target_db)
    print(f"复制: {source_db} -> {target_db}")
    return target_db


def process_read_phase(
    root_dir: Path,
    devices: Iterable[str],
    dry_run: bool,
    cleanup_read_db: bool,
) -> None:
    device_set = set(devices)
    unit_dirs: list[Path] = []
    for child in root_dir.iterdir():
        if not child.is_dir():
            continue
        folder = parse_device_folder(child)
        if folder is None or folder.device not in device_set:
            continue
        unit_dirs.append(child)

    unit_dirs.sort(key=lambda p: p.name)
    print(f"read-only: 候选目录 {len(unit_dirs)} 个")

    found_db = 0
    for unit_dir in unit_dirs:
        db_path = unit_dir / f"{unit_dir.name}.db"
        if not db_path.exists():
            continue

        found_db += 1
        print(f"\nread-only 导出: {db_path}")
        print_db_tables(db_path, dry_run=dry_run)
        if dry_run:
            print(f"[dry-run] 导出 CSV: {db_path}")
        else:
            process_single_database(str(db_path))

        if cleanup_read_db:
            remove_file_if_exists(db_path, dry_run=dry_run)
            remove_file_if_exists(journal_path(db_path), dry_run=dry_run)

    print(f"read-only: 发现并处理同名 db {found_db} 个")


def process_mapping_group(
    group: MappingGroup,
    dry_run: bool,
) -> None:
    print(f"\n处理源数据库: {group.source_db.path.name}")
    print_db_tables(group.source_db.path, dry_run=dry_run)

    for folder in group.device_folders:
        copy_db_to_folder(group.source_db.path, folder.path, dry_run=dry_run)


def main() -> int:
    args = parse_args()
    d = args.d.expanduser().resolve()

    if not d.exists():
        raise FileNotFoundError(f"目录不存在: {d}")
    if not d.is_dir():
        raise NotADirectoryError(f"不是目录: {d}")

    print(f"root 目录: {d}")
    print(f"目标设备: {', '.join(args.devices)}")

    # 兼容旧参数映射
    if args.overwrite_extracted:
        args.overwrite = True
    if args.cleanup_read_db:
        args.cleanup_read = True
    if args.cleanup_db:
        args.cleanup_root_db = True

    if not args.skip_extract:
        extract_archives(
            d,
            keep_zip=args.keep_zip,
            dry_run=args.dry_run,
            overwrite_existing=args.overwrite,
        )

    if args.extract_only:
        print("仅解压模式完成，跳过后续 db 匹配、复制与导出")
        return 0

    # 第二阶段: 读取已解压子目录中的同名 db 并导出 csv
    process_read_phase(
        root_dir=d,
        devices=args.devices,
        dry_run=args.dry_run,
        cleanup_read_db=args.cleanup_read,
    )

    if args.read_only:
        # read-only 模式到读取阶段结束
        print("\nread-only 处理完成")
        return 0

    device_folders = collect_device_folders(d, args.devices)
    for device in sorted(device_folders):
        print(f"设备 {device}: {len(device_folders[device])} 个目录")

    root_databases = collect_root_databases(d)
    print(f"root 顶层 db: {len(root_databases)} 个")

    mapping = build_mapping(root_databases, device_folders)
    summarize_mapping(mapping)

    for group in mapping:
        process_mapping_group(
            group,
            dry_run=args.dry_run,
        )

    if args.cleanup_root_db:
        print("\n复制阶段完成，开始删除 root 顶层 db / db-journal")
        for root_db in root_databases:
            remove_file_if_exists(root_db.path, dry_run=args.dry_run)
            remove_file_if_exists(journal_path(root_db.path), dry_run=args.dry_run)

    print("\n全部处理完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())