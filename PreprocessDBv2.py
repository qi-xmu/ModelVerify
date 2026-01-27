#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from base.calibration.time import match21
from base.datatype import GroundTruthData, ImuData, PosesData
from base.interpolate import get_time_series
from base.rtab import RTABData
from base.serialize import PosesDataSerializer

# def draw_trajectory(raw_gt: GroundTruthData, output_path: Path, opt_gt: PosesData = None, gap_info: dict = None):
#     """绘制 2D 轨迹图，参考 DataCheck 风格"""
#     print(f"正在生成轨迹图：{output_path.name}")
#     plt.figure(figsize=(10, 8))
    
#     # 绘制主要轨迹 (gt.csv 数据)
#     plt.plot(raw_gt.ps[:, 0], raw_gt.ps[:, 1], 'b-', label='GT Trajectory (Node)', alpha=0.7, linewidth=1.5)
    
#     # 如果指定了对比数据库优化位姿 (-dbp)
#     if opt_gt is not None:
#         plt.plot(opt_gt.ps[:, 0], opt_gt.ps[:, 1], 'g--', label='Optimized Trajectory (DB Opt)', alpha=0.6)

#     # 标记时间空洞 (Gaps) - 红点
#     if gap_info and gap_info["gap_idxs"]:
#         gap_ps = raw_gt.ps[gap_info["gap_idxs"]]
#         plt.scatter(gap_ps[:, 0], gap_ps[:, 1], c='red', s=30, label='Time Gaps > 1s', zorder=5)

#     plt.xlabel('X (m)')
#     plt.ylabel('Y (m)')
#     plt.title('Trajectory 2D View')
#     plt.legend()
#     plt.grid(True, linestyle=':', alpha=0.5)
#     plt.axis('equal')
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.close()
def draw_trajectory(raw_gt: GroundTruthData, output_path: Path, opt_gt: PosesData = None, gap_info: dict = None):
    """绘制 2D 轨迹图，复用 base.draw.Poses 的绘图风格"""
    print(f"正在生成轨迹图：{output_path.name}")
    
    # 创建一个新的图形
    plt.figure(figsize=(10, 8))
    
    # 提取位置数据
    positions = raw_gt.ps
    x = positions[:, 0]
    y = positions[:, 1]
    
    # 绘制主要轨迹 (gt.csv 数据) - 参考 Poses.py 的绘图风格
    plt.plot(x, y, 'b-', label='GT Trajectory (Node)', alpha=0.7, linewidth=1.5)
    
    # 绘制起点（用绿色圆点）- 参考 Poses.py 的标记风格
    plt.plot(
        x[0], y[0], 
        marker='o', color='green', markersize=10, 
        label='Start', linestyle='None'
    )
    
    # 绘制终点（用红色叉）- 参考 Poses.py 的标记风格
    plt.plot(
        x[-1], y[-1], 
        marker='x', color='red', markersize=10, 
        label='End', linestyle='None'
    )
    
    # 如果指定了对比数据库优化位姿 (-dbp)
    if opt_gt is not None:
        opt_positions = opt_gt.ps
        opt_x = opt_positions[:, 0]
        opt_y = opt_positions[:, 1]
        
        # 绘制优化轨迹 - 使用虚线样式
        plt.plot(opt_x, opt_y, 'g--', label='Optimized Trajectory (DB Opt)', 
                 alpha=0.6, linewidth=1.5)
        
        # 绘制优化轨迹的起点和终点
        plt.plot(
            opt_x[0], opt_y[0], 
            marker='o', color='green', markersize=8, 
            linestyle='None', alpha=0.6
        )
        plt.plot(
            opt_x[-1], opt_y[-1], 
            marker='x', color='red', markersize=8, 
            linestyle='None', alpha=0.6
        )

    # 标记时间空洞 (Gaps) - 红点
    if gap_info and gap_info["gap_idxs"]:
        gap_ps = raw_gt.ps[gap_info["gap_idxs"]]
        plt.scatter(gap_ps[:, 0], gap_ps[:, 1], c='red', s=30, 
                   label='Time Gaps > 1s', zorder=5, alpha=0.8)

    # 设置图表属性 - 参考 Poses.py 的样式
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory 2D View')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def check_groundtruth_gap(t_us, max_gap_s=1.0):
    """真值空洞检测逻辑"""
    ts_diff = np.diff(t_us) * 1e-6
    idxs = np.where(ts_diff > max_gap_s)[0].tolist()
    gap_diffs = ts_diff[ts_diff > max_gap_s].tolist()
    return {
        "max_gap": float(np.max(ts_diff)) if len(ts_diff)>0 else 0,
        "mean_gap": float(np.mean(ts_diff)) if len(ts_diff)>0 else 0,
        "gap_idxs": idxs,
        "gap_diff": gap_diffs,
        "note": "检测时间间隔"
    }

def time_sync(rtab_data: RTABData, imu_data: ImuData, time_range=(0, 50)) -> tuple[GroundTruthData, int]:
    """时间同步：仅平移，不插值"""
    print(f"正在进行时间同步 (参考范围: {time_range}s)...")
    gt_data = GroundTruthData(rtab_data.node_t_us, rtab_data.node_rots, rtab_data.node_ps)
    time_gc = match21(imu_data.to_poses(), gt_data, time_range=time_range)
    gt_data.t_us += time_gc
    gt_data.reset_start()
    return gt_data, int(time_gc)

def plot_time_diff(imu_data: ImuData, gt_data: GroundTruthData, output_dir: Path, time_range=(0, 50)):
    """绘制角速度对比图"""
    def get_angvels(t_us, rots):
        dt_s = np.diff(t_us) * 1e-6
        dt_s[dt_s <= 0] = 1e-6
        angs = [np.linalg.norm((rots[i].inv() * rots[i+1]).as_rotvec()) / dt_s[i] for i in range(len(rots)-1)]
        return np.array(angs), t_us[:-1]

    v_imu, t_imu = get_angvels(imu_data.t_us, imu_data.ahrs)
    v_gt, t_gt = get_angvels(gt_data.t_us, gt_data.rots)
    
    plt.figure(figsize=(12, 6))
    t0 = t_imu[0]
    t_imu_rel = (t_imu - t0) / 1e6
    t_gt_rel = (t_gt - t0) / 1e6

    plt.plot(t_imu_rel, v_imu, label='IMU', alpha=0.5, linewidth=1)
    plt.plot(t_gt_rel, v_gt, label='GT (Raw)', alpha=0.8, linewidth=1)
    
    # --- 修改处：动态调整 X 轴显示范围 ---
    # 如果数据长度小于 time_range，则以数据长度为准；否则以 time_range 为准
    max_view_time = min(max(t_imu_rel.max(), t_gt_rel.max()), time_range[1] + 5)
    plt.xlim(0, max_view_time) 
    
    plt.xlabel('Time (s)')
    plt.ylabel('Rad/s')
    plt.title('Angular Velocity Time Alignment (Sharp Peaks)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.savefig(output_dir / "Timediff.png", dpi=300, bbox_inches="tight")
    plt.close()

def plot_coordinate_transform(before_gt: GroundTruthData, after_gt: GroundTruthData, output_dir: Path):
    """坐标系转换验证 (初始5s)"""
    show_range = (0, 5.0)
    v_bef = before_gt.get_time_range(show_range)
    v_aft = after_gt.get_time_range(show_range)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    g_vec = [0, 0, -9.81]
    for ax, data, title in zip([ax1, ax2], [v_bef, v_aft], ['BEFORE (Y-UP)', 'AFTER (Z-UP)']):
        grav = data.rots.apply(g_vec)
        t_rel = (data.t_us - data.t_us[0]) / 1e6 if len(data.t_us)>0 else []
        ax.plot(t_rel, grav[:, 0], label='X'); ax.plot(t_rel, grav[:, 1], label='Y'); ax.plot(t_rel, grav[:, 2], label='Z')
        ax.set_title(title); ax.set_ylim([-12, 12]); ax.legend(); ax.grid(True)
    plt.savefig(output_dir / "Transform.png", dpi=300); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="预处理RTAB数据库文件为gt.csv")
    parser.add_argument("--db", required=True)
    parser.add_argument("--imu", required=True)
    parser.add_argument("--output")
    parser.add_argument("-t", "--time-range", type=float, nargs=2, default=[0, 50])
    parser.add_argument("-dbp", "--draw-db", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path(args.db).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载
    rtab_data = RTABData(args.db)
    imu_data = ImuData.from_csv(Path(args.imu))

    # 同步 (不插值)
    raw_gt, t_offset_us = time_sync(rtab_data, imu_data, time_range=tuple(args.time_range))
    
    # 记录 Gap 和 Rate 信息
    gap_info = check_groundtruth_gap(raw_gt.t_us)
    imu_rate = float(1e6 / np.mean(np.diff(imu_data.t_us)))

    # 输出 TimeCheck.json
    check_report = {
        "data_path": str(Path(args.db).parent.absolute()),
        "device_name": Path(args.db).parent.name,
        "check_data_rate": {"imu": imu_rate, "rtab": float(raw_gt.rate)},
        "check_groundtruth_gap": gap_info,
        "check_time_diff": {"time_diff_21_us": t_offset_us, "note": "检测两个序列的时间偏移"}
    }
    with open(output_dir / "TimeCheck.json", "w", encoding="utf-8") as f:
        json.dump(check_report, f, indent=4, ensure_ascii=False)

    # 转换坐标系前备份
    before_gt = GroundTruthData(raw_gt.t_us.copy(), Rotation.from_quat(raw_gt.rots.as_quat()), raw_gt.ps.copy())
    
    # 绘图：对齐验证
    plot_time_diff(imu_data, before_gt, output_dir, time_range=args.time_range)

    # 执行转换
    y_to_z = Rotation.from_rotvec([np.pi / 2, 0, 0])
    raw_gt.rots = y_to_z * raw_gt.rots
    
    # 绘图：转换验证
    plot_coordinate_transform(before_gt, raw_gt, output_dir)

    # 绘图：轨迹图 (解决 AttributeError)
    opt_poses = None
    if args.draw_db:
        # 如果需要对比 DB 优化轨迹，也应用相同的偏移和转换
        opt_poses = PosesData(rtab_data.opt_t_us + t_offset_us, y_to_z * rtab_data.opt_rots, rtab_data.opt_ps)
        opt_poses.ps -= opt_poses.ps[0] # reset start
        
    draw_trajectory(raw_gt, output_dir / "Trajectory.png", opt_gt=opt_poses, gap_info=gap_info)

    # 保存 CSV
    PosesDataSerializer(raw_gt).save(output_dir / "gt.csv")
    print("预处理完成！")

if __name__ == "__main__":
    main()