"""
按时间顺序将场景下 GT/HW/SM/RM 子文件夹的 db 文件一一对应，生成 CSV，
并通过命名规则校验匹配正确性。

命名规则:
  - GT 文件名 (YYMMDD-HHMMSS.db): 录制结束时间
  - 手机子文件夹 (YYYYMMDD_HHMMSS_Device): 录制开始时间
  校验: 手机开始时间 < GT 结束时间，差值 = 录制时长

用法:
    uv run python converters/AlignDbByTime.py /Users/qi/Resources/论文数据/S1
    uv run python converters/AlignDbByTime.py /Users/qi/Resources/论文数据/S1 --no-validate
    uv run python converters/AlignDbByTime.py /Users/qi/Resources/论文数据/S1 -o output.csv
"""

import argparse
import csv
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

# 阈值常量
MAX_DURATION_SEC = 1800   # 录制时长超过 30 分钟警告
MIN_DURATION_SEC = 10     # 录制时长不足 10 秒警告
MAX_DEVICE_SPREAD_SEC = 300  # 同组设备间开始时间偏差超过 5 分钟警告


def extract_gt_timestamp(filename: str) -> datetime | None:
    """从 GT db 文件名提取录制结束时间, 格式: YYMMDD-HHMMSS.db"""
    m = re.match(r"(\d{6})-(\d{6})\.db$", filename)
    if not m:
        return None
    date_str, time_str = m.groups()
    return datetime.strptime(f"{date_str}{time_str}", "%y%m%d%H%M%S")


def extract_phone_timestamp(dirname: str) -> datetime | None:
    """从手机子文件夹名提取录制开始时间, 格式: YYYYMMDD_HHMMSS_DeviceName"""
    m = re.match(r"(\d{8})_(\d{6})_", dirname)
    if not m:
        return None
    date_str, time_str = m.groups()
    return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")


def collect_gt_files(gt_dir: Path) -> list[tuple[datetime, Path]]:
    """收集 GT 目录下所有 .db 文件, 返回 (录制结束时间, 路径) 列表"""
    entries = []
    for f in gt_dir.iterdir():
        if not f.is_file():
            continue
        ts = extract_gt_timestamp(f.name)
        if ts is not None:
            entries.append((ts, f))
    entries.sort(key=lambda x: x[0])
    return entries


def collect_phone_files(phone_dir: Path) -> list[tuple[datetime, Path]]:
    """收集手机目录下子文件夹中的 .db 文件, 返回 (录制开始时间, 路径) 列表"""
    entries = []
    for subdir in sorted(phone_dir.iterdir()):
        if not subdir.is_dir():
            continue
        ts = extract_phone_timestamp(subdir.name)
        if ts is None:
            continue
        db_files = [f for f in subdir.iterdir()
                    if f.is_file() and f.suffix == ".db"]
        if db_files:
            entries.append((ts, db_files[0]))
    entries.sort(key=lambda x: x[0])
    return entries


def display_width(text: str) -> int:
    """计算字符串的终端显示宽度（CJK字符宽度=2, ASCII=1）"""
    w = 0
    for ch in text:
        ea = unicodedata.east_asian_width(ch)
        w += 2 if ea in ("W", "F") else 1
    return w


def pad_center(text: str, width: int, fill: str = " ") -> str:
    """按显示宽度居中对齐"""
    dw = display_width(text)
    if dw >= width:
        return text
    left = (width - dw) // 2
    right = width - dw - left
    return fill * left + text + fill * right


def pad_left(text: str, width: int, fill: str = " ") -> str:
    """按显示宽度左对齐（右补空格）"""
    dw = display_width(text)
    return text + fill * (width - dw) if dw < width else text


def fmt_duration(seconds: float) -> str:
    """格式化秒数为可读字符串"""
    if seconds < 0:
        return f"{-seconds:.0f}s(倒挂!)"
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"


def validate_and_report(
    sources: dict[str, list[tuple[datetime, Path]]],
    max_rows: int,
) -> tuple[list[dict], bool]:
    """
    校验匹配结果并生成报告。
    返回 (行数据列表, 是否有错误)
    """
    phone_sources = ["HW", "SM", "RM"]
    rows = []
    has_error = False

    # 列宽定义（按终端显示宽度）
    COL_W = [3, 16, 20, 20, 20, 8]  # 行, GT, HW, SM, RM, 偏差
    COLS = ["行", "GT结束时间", "HW开始(时长)", "SM开始(时长)", "RM开始(时长)", "设备偏差"]
    table_header = " | ".join(
        pad_center(COLS[k], COL_W[k]) for k in range(len(COLS))
    ) + " | 状态"
    banner_width = max(display_width(table_header), 70)

    print("\n" + "=" * banner_width)
    print("校验报告: GT 文件名=结束时间, 手机文件夹=开始时间")
    print("=" * banner_width)

    for i in range(max_rows):
        gt_entries = sources.get("GT", [])
        gt_end = gt_entries[i][0] if i < len(gt_entries) else None

        row_data: dict = {"index": i + 1}
        status_parts = []
        durations: dict[str, float] = {}

        for src in phone_sources:
            entries = sources.get(src, [])
            if i >= len(entries):
                row_data[f"{src}_start"] = None
                row_data[f"{src}_duration"] = None
                continue

            phone_start = entries[i][0]
            row_data[f"{src}_start"] = phone_start

            if gt_end is not None:
                dur = (gt_end - phone_start).total_seconds()
                durations[src] = dur
                row_data[f"{src}_duration"] = dur

                if dur < 0:
                    status_parts.append(f"{src}:倒挂({fmt_duration(dur)})")
                    has_error = True
                elif dur > MAX_DURATION_SEC:
                    status_parts.append(f"{src}:过长({fmt_duration(dur)})")
                elif dur < MIN_DURATION_SEC:
                    status_parts.append(f"{src}:过短({fmt_duration(dur)})")
            else:
                row_data[f"{src}_duration"] = None

        # 同组设备间开始时间偏差检查
        phone_starts = [
            (src, row_data[f"{src}_start"])
            for src in phone_sources
            if row_data.get(f"{src}_start") is not None
        ]
        if len(phone_starts) >= 2:
            earliest_src, earliest_ts = min(phone_starts, key=lambda x: x[1])
            latest_src, latest_ts = max(phone_starts, key=lambda x: x[1])
            spread = (latest_ts - earliest_ts).total_seconds()
            row_data["device_spread"] = spread
            if spread > MAX_DEVICE_SPREAD_SEC:
                status_parts.append(f"设备偏差大({fmt_duration(spread)})")

        row_data["status"] = " | ".join(status_parts) if status_parts else "✓"
        rows.append(row_data)

    # 打印表头
    sep_line = "-" * max(display_width(table_header), 70)
    print(table_header)
    print(sep_line)

    for row in rows:
        i = row["index"]
        gt_entries = sources.get("GT", [])
        gt_end = gt_entries[i - 1][0] if i - 1 < len(gt_entries) else None
        gt_str = gt_end.strftime("%m-%d %H:%M:%S") if gt_end else "-"

        values = [
            str(i),
            gt_str,
        ]
        for src in phone_sources:
            start = row.get(f"{src}_start")
            dur = row.get(f"{src}_duration")
            if start is not None and dur is not None:
                values.append(f"{start.strftime('%H:%M:%S')}({fmt_duration(dur)})")
            elif start is not None:
                values.append(f"{start.strftime('%H:%M:%S')}(无GT)")
            else:
                values.append("-")

        spread = row.get("device_spread")
        values.append(fmt_duration(spread) if spread is not None else "-")

        cells = [pad_center(values[k], COL_W[k]) for k in range(len(COLS))]
        print(" | ".join(cells) + " | " + row["status"])

    print(sep_line)

    # 统计
    total = len(rows)
    ok_count = sum(1 for r in rows if r["status"] == "✓")
    warn_count = sum(1 for r in rows if r["status"] != "✓" and "倒挂" not in r["status"])
    err_count = sum(1 for r in rows if "倒挂" in r["status"])

    print(f"总计 {total} 行: ✓={ok_count}  ⚠️={warn_count}  ❌={err_count}")
    if has_error:
        print("❌ 存在时间倒挂行，匹配可能不正确!")
    else:
        print("✓ 所有行通过基本校验")
    print("=" * max(display_width(table_header), 70) + "\n")

    return rows, has_error


def align_and_export(
    scene_dir: Path,
    output_path: Path | None = None,
    validate: bool = True,
) -> None:
    """按时间顺序对齐并导出 CSV"""
    scene_name = scene_dir.name

    sources: dict[str, list[tuple[datetime, Path]]] = {}
    for src_type in ("GT", "HW", "SM", "RM"):
        src_dir = scene_dir / f"{src_type}_{scene_name}"
        if not src_dir.is_dir():
            print(f"警告: 目录不存在 {src_dir}, 跳过")
            continue
        if src_type == "GT":
            sources[src_type] = collect_gt_files(src_dir)
        else:
            sources[src_type] = collect_phone_files(src_dir)
        print(f"{src_type}: 找到 {len(sources[src_type])} 个 db 文件")

    max_rows = max((len(v) for v in sources.values()), default=0)
    if max_rows == 0:
        print("未找到任何 db 文件")
        return

    # 校验
    if validate:
        validate_and_report(sources, max_rows)

    # 确定输出路径
    if output_path is None:
        output_path = scene_dir / f"{scene_name}_aligned.csv"

    # 写入 CSV (包含校验信息)
    fieldnames = ["index"]
    for src in ("GT", "HW", "SM", "RM"):
        fieldnames.append(f"{src}_file")
    for src in ("HW", "SM", "RM"):
        fieldnames.append(f"{src}_duration_seconds")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i in range(max_rows):
            row: dict = {"index": i + 1}
            for src in ("GT", "HW", "SM", "RM"):
                entries = sources.get(src, [])
                if i < len(entries):
                    _, path = entries[i]
                    row[f"{src}_file"] = str(path)
                else:
                    row[f"{src}_file"] = ""

            # 计算录制时长 (GT结束时间 - 手机开始时间)
            gt_entries = sources.get("GT", [])
            gt_end = gt_entries[i][0] if i < len(gt_entries) else None
            for src in ("HW", "SM", "RM"):
                entries = sources.get(src, [])
                if i < len(entries) and gt_end is not None:
                    phone_start = entries[i][0]
                    row[f"{src}_duration_seconds"] = (gt_end - phone_start).total_seconds()
                else:
                    row[f"{src}_duration_seconds"] = ""

            writer.writerow(row)

    print(f"已导出 {max_rows} 行到 {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="按时间顺序对齐场景下 GT/HW/SM/RM 的 db 文件并生成 CSV"
    )
    parser.add_argument("scene_dir", type=Path, help="场景目录路径, 如 /path/to/S1")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="输出 CSV 路径, 默认为场景目录下的 <场景名>_aligned.csv")
    parser.add_argument("--no-validate", action="store_true",
                        help="跳过校验")
    args = parser.parse_args()

    if not args.scene_dir.is_dir():
        print(f"错误: 目录不存在 {args.scene_dir}", file=sys.stderr)
        sys.exit(1)

    align_and_export(args.scene_dir, args.output, validate=not args.no_validate)


if __name__ == "__main__":
    main()
