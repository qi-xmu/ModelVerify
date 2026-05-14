#!/usr/bin/env python3
"""
轨迹绘制对比 — GT / Fusion / Camera 轨迹 2D 地图 + Rerun 可视化

用法:
    uv run python visualizers/DrawTrajCompare.py -u <db_or_csv_or_results.csv>
    uv run python visualizers/DrawTrajCompare.py -u <results.csv> --session 1 --device SM
    uv run python visualizers/DrawTrajCompare.py -d <scene_dir> --plot
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import base.rerun_ext as bre
from base.args_parser import DatasetArgsParser
from base.obj import Obj
from base.scene import FusionIndex, Session, SessionObj
from base.scene.cal import TrajEvaluator, export_csv, export_yaw

# 中文字体设置
matplotlib.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei",
    "PingFang SC",
    "Heiti SC",
    "Microsoft YaHei",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def _plot_trajectory(obj: SessionObj, save_path: Path) -> None:
    """绘制二维轨迹对比图，保存到 save_path"""
    fig, ax = plt.subplots(figsize=(8, 8))

    configs = [
        (obj.gt_pose.ps[:, :2], "真值", "red", "^-"),
        (obj.fusion_pose.ps[:, :2], "FUS", "green", "s-"),
        (obj.cam_pose.ps[:, :2], "VIO", "gold", ".-"),
    ]
    if obj.extra_pose is not None:
        map_name = {"network": "INO", "fix": "FIX"}
        configs.append(
            (obj.extra_pose.ps[:, :2], map_name[obj.session.extra_kind], "blue", "x-")
        )

    for xy, label, color, marker in configs:
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            marker,
            color=color,
            label=label,
            linewidth=1,
            markersize=3,
            markerfacecolor=color,
            markeredgewidth=0.5,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"> 轨迹图已保存: {save_path}")


def main():
    dap = DatasetArgsParser()
    dap.parser.add_argument("--session", type=int, default="1", help="指定场次 ID")
    dap.parser.add_argument(
        "--device", type=str, default="SM", help="指定设备 (HW/SM/RM)"
    )
    dap.parser.add_argument(
        "--yaw",
        type=float,
        nargs=2,
        default=None,
        metavar=("GT_YAW", "EXTRA_YAW"),
        help="手动指定 yaw 角度（度）",
    )
    dap.parser.add_argument(
        "--plot",
        action="store_true",
        help="绘制二维轨迹对比图并保存（PNG, dpi=300）",
    )
    dap.parse()

    def action(session: Session):
        te_temp_file = Path(session.parent_dir) / f"TE_ALG_{session.label}.pkl"

        if te_temp_file.exists() and not dap.regen:
            te = Obj.load(te_temp_file)
            obj: SessionObj = te.obj
        else:
            obj = SessionObj(session)
            obj.align_time(rate=1)
            obj.align_space()
            te = TrajEvaluator(obj)
            Obj.save(te, te_temp_file)

        te.report()

        if dap.visual:
            bre.rerun_init(f"{session.device}_s{session.session_id}")
            bre.send_pose_data(obj.gt_pose, "Groundtruth", color=[192, 72, 72])
            bre.send_pose_data(obj.fusion_pose, "Fusion", color=[72, 192, 72])
            bre.send_pose_data(obj.cam_pose, "Camera", color=[72, 72, 192])
            if obj.extra_pose is not None:
                bre.send_pose_data(obj.extra_pose, "Extra", color=[192, 192, 72])

            print(
                f"[{session.label}] GT={len(obj.gt_pose)}, Fusion={len(obj.fusion_pose)}"
            )

        if dap.args.plot:
            plot_path = session.parent_dir / f"trajectory_{session.label}.png"
            _plot_trajectory(obj, plot_path)

        return te

    if dap.unit:
        path = Path(dap.unit)
        s_id = dap.args.session
        device = dap.args.device
        index = FusionIndex.from_scene(path) if path.is_dir() else FusionIndex(path)
        session = index[(s_id, device)]
        if dap.args.yaw:
            session.gt_yaw, session.extra_yaw = dap.args.yaw
            session.has_yaw = True

        print(
            f"> 角度 [{session.label}] gt_yaw={session.gt_yaw}, extra_yaw={session.extra_yaw}"
        )
        action(session)

    elif dap.dataset:
        path = Path(dap.dataset)
        index = FusionIndex.from_scene(path) if path.is_dir() else FusionIndex(path)
        index.print_summary()
        te_list = [action(s) for s in index]
        output_dir = path if path.is_dir() else path.parent
        export_csv(te_list, output_dir)
        if not index.has_yaw:
            export_yaw(te_list, output_dir)

    else:
        dap.parser.print_help()


if __name__ == "__main__":
    main()
