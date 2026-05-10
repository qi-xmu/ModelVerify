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
from datetime import datetime
from pathlib import Path

from base.tab_show import display_width, pad_center

# 阈值常量
MAX_DURATION_SEC = 600    # 录制时长超过 10 分钟警告
MIN_DURATION_SEC = 10     # 录制时长不足 10 秒警告
MAX_DEVICE_SPREAD_SEC = 300  # 同组设备间开始时间偏差超过 5 分钟警告
MATCH_MAX_DURATION_SEC = 360  # 匹配时允许的最大录制时长（超过此值认为不匹配）


def match_by_time(
    gt_entries: list[tuple[datetime, Path]],
    phone_entries: list[tuple[datetime, Path]],
) -> list[tuple[int | None, int | None]]:
    """
    基于时间戳的贪心匹配，保持时序一致性。
    每个 GT 最多匹配一个手机，某个手机开始时间后最近的 GT 结束时间对应。
    返回 [(gt_index, phone_index), ...] 列表，未匹配项为 None。
    """
    pairs: list[tuple[int | None, int | None]] = []
    j = 0  # GT 游标
    for i in range(len(phone_entries)):
        phone_ts = phone_entries[i][0]
        # 跳过在手机开始时间之前结束的 GT（不可能匹配）
        while j < len(gt_entries) and gt_entries[j][0] <= phone_ts:
            j += 1
        if j < len(gt_entries):
            dur = (gt_entries[j][0] - phone_ts).total_seconds()
            if 0 < dur <= MATCH_MAX_DURATION_SEC:
                pairs.append((j, i))
                j += 1
                continue
        # 无匹配 GT（时长超限或无后续 GT）
        pairs.append((None, i))

    # 补充未被匹配的 GT（时序在最后一个手机之后的 GT）
    matched_gt = {p[0] for p in pairs if p[0] is not None}
    for k in range(len(gt_entries)):
        if k not in matched_gt:
            pairs.append((k, None))

    # 重新按 GT 时间排序（保持行输出以 GT 为基准）
    pairs.sort(key=lambda p: (
        gt_entries[p[0]][0] if p[0] is not None else datetime.max,
        phone_entries[p[1]][0] if p[1] is not None else datetime.max,
    ))
    return pairs


def extract_gt_timestamp(filename: str) -> datetime | None:
    """从 GT db 文件名提取时间戳, 格式: YYMMDD-HHMMSS[no].db"""
    m = re.match(r"(\d{6})-(\d{6})(?:no)?\.db$", filename)
    if not m:
        return None
    date_str, time_str = m.groups()
    return datetime.strptime(f"{date_str}{time_str}", "%y%m%d%H%M%S")


def extract_phone_timestamp(dirname: str) -> datetime | None:
    """从手机子文件夹名提取时间戳, 格式: YYYYMMDD_HHMMSS_DeviceName"""
    m = re.match(r"(\d{8})_(\d{6})_", dirname)
    if not m:
        return None
    date_str, time_str = m.groups()
    return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")


def collect_gt_files(gt_dir: Path) -> list[tuple[datetime, Path]]:
    """收集 GT 目录下所有 .db 文件（含无效文件，匹配后再清理）, 返回 (时间戳, 路径) 列表"""
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
    """收集手机目录下子文件夹中的所有 .db 文件（含无效文件）, 返回 (时间戳, 路径) 列表"""
    entries = []
    for subdir in sorted(phone_dir.iterdir()):
        if not subdir.is_dir():
            continue
        ts = extract_phone_timestamp(subdir.name)
        if ts is None:
            continue
        db_files = [f for f in subdir.iterdir() if f.is_file() and f.suffix == ".db"]
        if db_files:
            entries.append((ts, db_files[0]))
    entries.sort(key=lambda x: x[0])
    return entries


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
) -> tuple[str, list[dict], bool]:
    """
    校验匹配结果并生成报告。
    返回 (报告文本, 行数据列表, 是否有错误)
    """
    phone_sources = ["HW", "SM", "RM"]
    rows = []
    has_error = False

    # 列宽定义（按终端显示宽度）
    COL_W = [3, 16, 20, 20, 20, 8]  # 行, GT, HW, SM, RM, 偏差
    COLS = [
        "行",
        "GT结束时间",
        "HW开始(时长)",
        "SM开始(时长)",
        "RM开始(时长)",
        "设备偏差",
    ]
    table_header = (
        " | ".join(pad_center(COLS[k], COL_W[k]) for k in range(len(COLS))) + " | 状态"
    )
    banner_width = max(display_width(table_header), 70)

    report_lines = [
        "",
        "=" * banner_width,
        "校验报告: GT 文件名=结束时间, 手机文件夹=开始时间",
        "=" * banner_width,
    ]

    for i in range(max_rows):
        gt_entries = sources.get("GT", [])
        gt_entry = gt_entries[i] if i < len(gt_entries) else None
        gt_end = gt_entry[0] if gt_entry is not None else None

        row_data: dict = {"index": i + 1}
        status_parts = []
        durations: dict[str, float] = {}

        for src in phone_sources:
            entries = sources.get(src, [])
            phone_entry = entries[i] if i < len(entries) else None
            if phone_entry is None:
                row_data[f"{src}_start"] = None
                row_data[f"{src}_duration"] = None
                continue

            phone_start = phone_entry[0]
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

        # 检测无效数据标记
        if gt_entry is not None and "no" in str(gt_entry[1]):
            status_parts.append("GT无效")
        for src in phone_sources:
            entry = sources.get(src, [])
            phone_entry = entry[i] if i < len(entry) else None
            if phone_entry is not None and "no" in str(phone_entry[1]):
                status_parts.append(f"{src}无效")

        row_data["status"] = " | ".join(status_parts) if status_parts else "✓"
        rows.append(row_data)

    # 构建表头
    sep_line = "-" * max(display_width(table_header), 70)
    report_lines.append(table_header)
    report_lines.append(sep_line)

    for row in rows:
        i = row["index"]
        gt_entries = sources.get("GT", [])
        gt_entry = gt_entries[i - 1] if i - 1 < len(gt_entries) else None
        gt_end = gt_entry[0] if gt_entry is not None else None
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
        report_lines.append(" | ".join(cells) + " | " + row["status"])

    report_lines.append(sep_line)

    # 统计
    total = len(rows)
    ok_count = sum(1 for r in rows if r["status"] == "✓")
    warn_count = sum(
        1 for r in rows if r["status"] != "✓" and "倒挂" not in r["status"]
    )
    err_count = sum(1 for r in rows if "倒挂" in r["status"])

    report_lines.append(f"总计 {total} 行: ✓={ok_count}  !={warn_count}  X={err_count}")
    if has_error:
        report_lines.append("X 存在时间倒挂行，匹配可能不正确!")
    else:
        report_lines.append("✓ 所有行通过基本校验")
    report_lines.append("=" * max(display_width(table_header), 70))

    report = "\n".join(report_lines)
    print(report + "\n")
    return report, rows, has_error


def align_and_export(
    scene_dir: Path,
    output_path: Path | None = None,
    validate: bool = True,
) -> None:
    """按时间顺序对齐并导出 CSV"""
    scene_name = scene_dir.name

    sources: dict[str, list[tuple[datetime, Path] | None]] = {}
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

    if not sources:
        print("未找到任何 db 文件")
        return

    # 基于时间戳匹配 GT 与手机数据（以 HW 为参考）
    gt_entries = sources.get("GT", [])
    hw_entries = sources.get("HW", [])
    if not gt_entries or not hw_entries:
        max_rows = max((len(v) for v in sources.values()), default=0)
    else:
        pairs = match_by_time(gt_entries, hw_entries)

        # 根据匹配结果重组各数据源，保证同一行的 GT 和手机对应
        aligned_gt: list[tuple[datetime, Path] | None] = []
        aligned_hw: list[tuple[datetime, Path] | None] = []
        aligned_sm: list[tuple[datetime, Path] | None] = []
        aligned_rm: list[tuple[datetime, Path] | None] = []
        sm_entries = sources.get("SM", [])
        rm_entries = sources.get("RM", [])

        for gt_i, hw_i in pairs:
            aligned_gt.append(gt_entries[gt_i] if gt_i is not None else None)
            aligned_hw.append(hw_entries[hw_i] if hw_i is not None else None)
            aligned_sm.append(sm_entries[hw_i] if hw_i is not None and hw_i < len(sm_entries) else None)
            aligned_rm.append(rm_entries[hw_i] if hw_i is not None and hw_i < len(rm_entries) else None)

        sources["GT"] = aligned_gt
        sources["HW"] = aligned_hw
        sources["SM"] = aligned_sm
        sources["RM"] = aligned_rm
        max_rows = len(pairs)

    # 校验报告（包含全部行，含 no 标记的无效数据）
    if validate and "GT" in sources and "HW" in sources:
        report, _, _ = validate_and_report(sources, max_rows)
        txt_path = scene_dir / "ALIGN.txt"
        txt_path.write_text(report.strip() + "\n", encoding="utf-8")
        print(f"报告已保存至 {txt_path}")

    # 剔除含 "no" 的无效行（仅影响 CSV 输出）
    valid_indices = []
    for i in range(max_rows):
        has_no = False
        for entries in sources.values():
            entry = entries[i] if i < len(entries) else None
            if entry is not None and "no" in str(entry[1]):
                has_no = True
                break
        if not has_no:
            valid_indices.append(i)

    # 确定输出路径
    if output_path is None:
        output_path = scene_dir / f"{scene_name}_aligned.csv"

    # 写入 CSV（仅有效行）
    fieldnames = ["index"]
    for src in ("GT", "HW", "SM", "RM"):
        fieldnames.append(f"{src}_file")
    for src in ("HW", "SM", "RM"):
        fieldnames.append(f"{src}_duration_seconds")

    csv_index = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for vi in valid_indices:
            csv_index += 1
            row: dict = {"index": csv_index}
            gt_end: datetime | None = None

            for src in ("GT", "HW", "SM", "RM"):
                entries = sources.get(src, [])
                entry = entries[vi] if vi < len(entries) else None
                if entry is not None:
                    ts, path = entry
                    row[f"{src}_file"] = str(path)
                    if src == "GT":
                        gt_end = ts
                    elif gt_end is not None:
                        row[f"{src}_duration_seconds"] = (gt_end - ts).total_seconds()
                else:
                    row[f"{src}_file"] = ""
                    if src != "GT":
                        row[f"{src}_duration_seconds"] = ""

            for src in ("HW", "SM", "RM"):
                if f"{src}_duration_seconds" not in row:
                    row[f"{src}_duration_seconds"] = ""

            writer.writerow(row)

    print(f"已导出 {csv_index} 行到 {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="按时间顺序对齐场景下 GT/HW/SM/RM 的 db 文件并生成 CSV"
    )
    parser.add_argument("scene_dir", type=Path, help="场景目录路径, 如 /path/to/S1")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径, 默认为场景目录下的 <场景名>_aligned.csv",
    )
    parser.add_argument("--no-validate", action="store_true", help="跳过校验")
    args = parser.parse_args()

    if not args.scene_dir.is_dir():
        print(f"错误: 目录不存在 {args.scene_dir}", file=sys.stderr)
        sys.exit(1)

    align_and_export(args.scene_dir, args.output, validate=not args.no_validate)


if __name__ == "__main__":
    main()
