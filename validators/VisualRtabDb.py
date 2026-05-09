"""RTAB-Map 数据库可视化工具

读取 RTAB-Map .db 文件，绘制轨迹（原始节点 + 优化后位姿）。
"""

from pathlib import Path

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


def visualize(db_path: Path, save_csv: bool = False):
    name = db_path.stem
    output_dir = db_path.parent

    rtab = RTABData(db_path)
    rtab.transform_local()
    rtab.transform_global()

    if save_csv:
        rtab.save_csv(output_dir / f"{name}.csv", using_opt=True)

    node_len = _compute_length(rtab.node_ps)
    opt_len = _compute_length(rtab.opt_ps) if rtab.opt_ids else None

    print(f"[{name}]")
    print(f"  Node 数量: {len(rtab)}")
    print(f"  时长: {rtab.t_len_s:.1f} s")
    print(f"  帧率: {rtab.rate:.1f} Hz")
    print(f"  Node 轨迹长度: {node_len:.2f} m")
    if opt_len is not None:
        print(f"  Optimized 数量: {len(rtab.opt_ids)}")
        print(f"  Optimized 轨迹长度: {opt_len:.2f} m")

    plot_trajectory(rtab, output_dir / f"{name}_trajectory.png")
    plot_link_graph(rtab, output_dir / f"{name}_link_graph.png")

    rtab.conn.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: uv run python validators/VisualRtabDb.py <db_path_or_dir> [--csv]")
        sys.exit(1)

    save_csv = "--csv" in sys.argv
    path = Path(sys.argv[1])
    if path.is_dir():
        db_files = sorted(path.glob("*.db"))
        if not db_files:
            print(f"目录 {path} 中未找到 .db 文件")
            sys.exit(1)
        print(f"找到 {len(db_files)} 个数据库文件\n")
        for db_file in db_files:
            visualize(db_file, save_csv=save_csv)
    else:
        visualize(path, save_csv=save_csv)
