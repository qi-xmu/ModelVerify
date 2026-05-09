"""RTAB-Map 数据库可视化工具

读取 RTAB-Map .db 文件，绘制轨迹（原始节点 + 优化后位姿）。
批量处理目录时输出汇总统计表格。
"""

import unicodedata
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from base.rtab import RTABData

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["SimSun", "Songti SC", "STSong"]
plt.rcParams["axes.unicode_minus"] = False


def _compute_length(ps: np.ndarray) -> float:
    if len(ps) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(ps, axis=0), axis=1)))


def _display_width(text: str) -> int:
    """计算字符串的终端显示宽度（CJK字符宽度=2, ASCII=1）"""
    w = 0
    for ch in text:
        ea = unicodedata.east_asian_width(ch)
        w += 2 if ea in ("W", "F") else 1
    return w


def _pad_center(text: str, width: int, fill: str = " ") -> str:
    """按显示宽度居中对齐"""
    dw = _display_width(text)
    if dw >= width:
        return text
    left = (width - dw) // 2
    right = width - dw - left
    return fill * left + text + fill * right


def _pad_right(text: str, width: int, fill: str = " ") -> str:
    """按显示宽度右对齐"""
    dw = _display_width(text)
    return text + fill * (width - dw) if dw < width else text


def plot_trajectory(rtab: RTABData, save_path: Path):
    node_ps = rtab.node_ps
    opt_ps = rtab.opt_ps if rtab.opt_ids else None

    fig = plt.figure(figsize=(14, 6))

    # 2D 俯视图
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(node_ps[:, 0], node_ps[:, 1], linewidth=0.6, color="lightgray", alpha=0.7, label="Node")
    if opt_ps is not None and len(opt_ps) > 0:
        ax1.plot(opt_ps[:, 0], opt_ps[:, 1], linewidth=0.8, color="steelblue", label="Optimized")
        ax1.scatter(opt_ps[0, 0], opt_ps[0, 1], c="green", marker="o", s=60, zorder=3)
        ax1.scatter(opt_ps[-1, 0], opt_ps[-1, 1], c="red", marker="X", s=60, zorder=3)
    else:
        ax1.scatter(node_ps[0, 0], node_ps[0, 1], c="green", marker="o", s=60, zorder=3)
        ax1.scatter(node_ps[-1, 0], node_ps[-1, 1], c="red", marker="X", s=60, zorder=3)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("轨迹俯视图")
    ax1.legend()
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)

    # 3D 视图
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(node_ps[:, 0], node_ps[:, 1], node_ps[:, 2], linewidth=0.5, color="lightgray", alpha=0.5, label="Node")
    if opt_ps is not None and len(opt_ps) > 0:
        ax2.plot(opt_ps[:, 0], opt_ps[:, 1], opt_ps[:, 2], linewidth=0.8, color="steelblue", label="Optimized")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_title("轨迹 3D 视图")
    ax2.legend()
    # 3D 等比例尺
    pts = node_ps if opt_ps is None else np.vstack([node_ps, opt_ps])
    ranges = pts.max(axis=0) - pts.min(axis=0)
    mid = pts.min(axis=0) + ranges / 2
    max_range = ranges.max() / 2
    ax2.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax2.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax2.set_zlim(mid[2] - max_range, mid[2] + max_range)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"轨迹图已保存至 {save_path}")


def plot_link_graph(rtab: RTABData, save_path: Path):
    links = rtab.get_links()
    if not links:
        return

    node_ps = rtab.node_ps
    id_to_idx = {nid: i for i, nid in enumerate(rtab.node_ids)}

    fig, ax = plt.subplots(figsize=(10, 8))

    for from_id, to_id in links:
        if from_id in id_to_idx and to_id in id_to_idx:
            fi, ti = id_to_idx[from_id], id_to_idx[to_id]
            ax.plot(
                [node_ps[fi, 0], node_ps[ti, 0]],
                [node_ps[fi, 1], node_ps[ti, 1]],
                linewidth=0.3,
                color="gray",
                alpha=0.4,
            )

    ax.scatter(node_ps[:, 0], node_ps[:, 1], s=4, c="steelblue", alpha=0.6)
    ax.scatter(node_ps[0, 0], node_ps[0, 1], c="green", marker="o", s=60, zorder=3, label="起点")
    ax.scatter(node_ps[-1, 0], node_ps[-1, 1], c="red", marker="X", s=60, zorder=3, label="终点")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"节点连接图（{len(links)} 条边）")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"连接图已保存至 {save_path}")


def visualize(db_path: Path, save_csv: bool = False) -> dict[str, Any]:
    """处理单个 db 文件，返回统计信息字典"""
    name = db_path.stem
    output_dir = db_path.parent

    rtab = RTABData(db_path)
    rtab.transform_local()
    rtab.transform_global()

    if save_csv:
        rtab.save_csv(output_dir / f"{name}.csv", using_opt=True)

    node_len = _compute_length(rtab.node_ps)
    opt_len = _compute_length(rtab.opt_ps) if rtab.opt_ids else None

    stats: dict[str, Any] = {
        "name": name,
        "node_count": len(rtab),
        "duration_s": rtab.t_len_s,
        "rate_hz": rtab.rate,
        "node_len_m": node_len,
        "opt_count": len(rtab.opt_ids) if rtab.opt_ids else 0,
        "opt_len_m": opt_len,
    }

    plot_trajectory(rtab, output_dir / f"{name}_trajectory.png")
    plot_link_graph(rtab, output_dir / f"{name}_link_graph.png")

    rtab.conn.close()
    return stats


def print_summary_table(all_stats: list[dict[str, Any]]) -> None:
    """打印所有 db 文件的汇总统计表格"""
    COL_W = [25, 6, 7, 6, 8, 7, 8]  # 文件名, 节点, 时长, 帧率, 轨迹, Opt数, Opt轨迹
    COLS = ["文件名", "节点", "时长(s)", "帧率", "轨迹(m)", "Opt数", "Opt轨(m)"]
    EMPTY = "-"

    table_header = " | ".join(
        _pad_center(COLS[k], COL_W[k]) for k in range(len(COLS))
    )
    banner_w = max(_display_width(table_header), 60)
    sep = "-" * banner_w

    print("\n" + sep)
    print("RTAB-Map 轨迹统计汇总")
    print(sep)
    print(table_header)
    print(sep)

    def fmt_val(v: Any, precision: int = 1) -> str:
        if v is None or v == 0:
            return EMPTY
        if isinstance(v, float):
            return f"{v:.{precision}f}"
        return str(v)

    for s in all_stats:
        values = [
            s["name"],
            str(s["node_count"]),
            fmt_val(s["duration_s"]),
            fmt_val(s["rate_hz"]),
            fmt_val(s["node_len_m"], 2),
            str(s["opt_count"]) if s["opt_count"] > 0 else EMPTY,
            fmt_val(s["opt_len_m"], 2),
        ]
        cells = [_pad_center(values[k], COL_W[k]) for k in range(len(COLS))]
        print(" | ".join(cells))

    print(sep)

    # 汇总
    total_nodes = sum(s["node_count"] for s in all_stats)
    total_dur = sum(s["duration_s"] for s in all_stats)
    total_node_len = sum(s["node_len_m"] for s in all_stats)
    total_opt = sum(s["opt_count"] for s in all_stats)
    total_opt_len = sum(s["opt_len_m"] for s in all_stats if s["opt_len_m"] is not None)

    summary_values = [
        f"合计({len(all_stats)}文件)",
        str(total_nodes),
        f"{total_dur:.0f}",
        "-",
        f"{total_node_len:.1f}",
        str(total_opt),
        f"{total_opt_len:.1f}",
    ]
    row = [_pad_center(summary_values[k], COL_W[k]) for k in range(len(COLS))]
    print(" | ".join(row))
    print(sep + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: uv run python validators/VisualRtabDb.py <db_path_or_dir> [--csv]")
        sys.exit(1)

    save_csv = "--csv" in sys.argv
    path = Path(sys.argv[1])
    if path.is_dir():
        db_files = sorted(path.rglob("*.db"))
        if not db_files:
            print(f"目录 {path} 中未找到 .db 文件")
            sys.exit(1)
        print(f"找到 {len(db_files)} 个数据库文件\n")
        all_stats = []
        for db_file in db_files:
            stats = visualize(db_file, save_csv=save_csv)
            all_stats.append(stats)
        print_summary_table(all_stats)
    else:
        visualize(path, save_csv=save_csv)
