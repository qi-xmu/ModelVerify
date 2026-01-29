#!/usr/bin/env python3
import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import rerun as rr

from base.calibration.time import match21
from base.datatype import GroundTruthData, ImuData, PosesData
from base.interpolate import get_time_series
from base.rtab import RTABData
from base.serialize import PosesDataSerializer
from base.rerun_ext import RerunView, send_imu_data, send_pose_data

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
    plt.title('Angular Velocity Time Alignment')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.savefig(output_dir / "TimeDiff.png", dpi=300, bbox_inches="tight")
    plt.close()

# def plot_coordinate_transform(before_gt: GroundTruthData, after_gt: GroundTruthData, output_dir: Path):
#     """坐标系转换验证 (初始5s)"""
#     show_range = (0, 5.0)
#     v_bef = before_gt.get_time_range(show_range)
#     v_aft = after_gt.get_time_range(show_range)
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#     g_vec = [0, 0, -9.81]
#     for ax, data, title in zip([ax1, ax2], [v_bef, v_aft], ['BEFORE (Y-UP)', 'AFTER (Z-UP)']):
#         grav = data.rots.apply(g_vec)
#         t_rel = (data.t_us - data.t_us[0]) / 1e6 if len(data.t_us)>0 else []
#         ax.plot(t_rel, grav[:, 0], label='X'); ax.plot(t_rel, grav[:, 1], label='Y'); ax.plot(t_rel, grav[:, 2], label='Z')
#         ax.set_title(title); ax.set_ylim([-12, 12]); ax.legend(); ax.grid(True)
#     plt.savefig(output_dir / "Transform.png", dpi=300); plt.close(fig)


def rerun_verify_coordinate_transform(imu_data: ImuData, gt_data: GroundTruthData):
    """
    使用 Rerun 验证坐标系转换是否正确
    
    验证原理：
    1. 将统一时间IMU 数据（body frame）发送到 Rerun，标记为 "body"
    2. 使用 gt_data 的旋转（已经经过时间同步和坐标系转换）将 IMU 数据转换到 global frame
    3. 将转换后的 IMU 数据发送到 Rerun，标记为 "global"
    4. 验证标准：在正确的 Z-UP 坐标系中，重力应该只出现在 z 轴上（约 9.8 m/s²）
    """
    print("正在使用 Rerun 验证坐标系转换...\n")
    
    # ImuData.transform 方法会将 acce, gyro, magn, ahrs 从 local frame 转换到 global frame
    imu_global = imu_data.transform(rots=gt_data.rots)
    
    # 创建 Rerun 视图
    # 使用 RerunView 创建可视化，只有一个视图包含两个标签
    view = RerunView()
    view.add_imu_view(visible=True, tags=["body", "global"])
    
    # 初始化 Rerun
    rr.init("Coordinate Transform Verification", spawn=True)
    
    # 发送视图蓝图
    view.send("Coordinate Transform Verification")
    
    # 发送原始 IMU 数据（body frame）
    send_imu_data(imu_data, tag="body")
    
    # 发送使用 Z-UP Global Frame 旋转转换后的 IMU 数据
    send_imu_data(imu_global, tag="global")
    

def main():
    parser = argparse.ArgumentParser(
        description="预处理RTAB数据库文件为gt.csv",
        epilog="""
        坐标转换流程：
        1. RTAB-Map数据：存储在RTAB Body Frame<-> Y-UP Global Frame
        2. transform_local()：从 RTAB Body Frame 转换到 IMU Body Frame
        3. 时间同步：使用 match21 算法计算 IMU 和 GT 的时间偏移
        4. 同一时间轴插值：将 IMU、GT 数据插值统一频率
        5. 坐标转换：Y-UP Global Frame → Z-UP Global Frame (绕 X 轴旋转 90°)
        6. rerun验证，统一频率IMU记为body、利用GT 数据的旋转到全局坐标系后记为global

        使用模式：
        1. 首次处理（推荐）: -u <数据目录> [--output <输出目录>]
        - 数据目录必须包含：imu.csv 和一个 .db 文件
        - 生成文件：gt.csv, TimeCheck.json, Trajectory.png, TimeDiff.png
        - 默认输出到数据目录本身
        
        2. 兼容旧模式: --db <db文件> --imu <imu.csv> [--output <输出目录>]
        - 生成文件：gt.csv, TimeCheck.json, Trajectory.png, TimeDiff.png
        - 默认输出到 imu.csv 同文件夹
        """
    )
    parser.add_argument("-u", "--unit-dir", type=str, default=None,
                       help="指定单元数据目录（包含imu.csv和.db文件）")
    parser.add_argument("-db", type=str, default=None,
                       help="RTAB数据库文件路径(.db)，用于兼容旧模式")
    parser.add_argument("-imu", type=str, default=None,
                       help="IMU数据文件路径(imu.csv)，用于兼容旧模式")
    parser.add_argument("-output", type=str, default=None,
                       help="输出目录路径（可选）")
    parser.add_argument("-t", "--time-range", type=float, nargs=2, default=[0, 50],
                       help="时间同步范围（秒），默认 [0, 50]")
    parser.add_argument("-dbp", "--draw-db", action="store_true",
                       help="绘制DB优化轨迹对比图")
    parser.add_argument("-rv", "--rerun-verify", action="store_true", 
                       help="使用 Rerun 验证坐标系转换是否正确")
    args = parser.parse_args()

    # 检查参数
    if args.unit_dir:
        # 使用单元目录模式（推荐）：目录包含 imu.csv 和 .db 文件
        unit_dir = Path(args.unit_dir)
        print(f"使用单元目录模式: {unit_dir}")
        
        # 检查必需文件
        imu_file = unit_dir / "imu.csv"
        if not imu_file.exists():
            raise FileNotFoundError(f"未找到 IMU 数据文件: {imu_file}")
        
        # 检查是否存在 RTAB 数据库文件
        db_file = RTABData.get_db_file(unit_dir)
        if db_file is None:
            raise FileNotFoundError(f"未找到 RTAB 数据库文件(.db)在目录: {unit_dir}")
        
        # 从 RTAB-Map 加载原始数据（不使用 ExtUnitData，避免生成 opt.csv）
        print(f"正在从数据库加载数据: {db_file}")
        rtab_data = RTABData(db_file)
        rtab_data.transform_local()  # 应用 RTAB Body Frame → IMU Body Frame 转换
        
        # 从 CSV 加载 IMU 数据
        imu_data = ImuData.from_csv(imu_file)
        
        # 阶段1：时间对齐
        raw_gt, t_offset_us = time_sync(rtab_data, imu_data, time_range=tuple(args.time_range))
        
        # 阶段1输出：保存 TimeCheck.json 和绘制 TimeDiff.png
        data_path = str(unit_dir.absolute())
        device_name = unit_dir.name
        
    elif args.db and args.imu:
        # 兼容旧模式：直接指定 db 和 imu 文件
        print(f"使用旧模式: --db {args.db} --imu {args.imu}")
        
        rtab_data = RTABData(args.db)
        rtab_data.transform_local()
        
        imu_data = ImuData.from_csv(Path(args.imu))
        
        # 阶段1：时间对齐（不插值）
        raw_gt, t_offset_us = time_sync(rtab_data, imu_data, time_range=tuple(args.time_range))
        
        data_path = str(Path(args.imu).parent.absolute())
        device_name = Path(args.imu).parent.name
    else:
        parser.error("必须指定 -u/--unit-dir 或 (--db 和 --imu)")
    
    # 确定输出目录
    output_dir = Path(args.output) if args.output else (unit_dir if args.unit_dir else Path(args.imu).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === 阶段1：时间对齐（不插值）===
    print("\n=== 阶段1：时间对齐（不插值）===")
    
    # 记录 Gap 和 Rate 信息（使用未插值的数据）
    gap_info = check_groundtruth_gap(raw_gt.t_us)
    imu_rate = float(1e6 / np.mean(np.diff(imu_data.t_us)))

    # 输出 TimeCheck.json
    check_report = {
        "data_path": data_path,
        "device_name": device_name,
        "check_data_rate": {"imu": imu_rate, "rtab": float(raw_gt.rate)},
        "check_groundtruth_gap": gap_info,
        "check_time_diff": {"time_diff_21_us": t_offset_us, "note": "检测两个序列的时间偏移"}
    }
    with open(output_dir / "TimeCheck.json", "w", encoding="utf-8") as f:
        json.dump(check_report, f, indent=4, ensure_ascii=False)
    print(f"已保存时间检查信息：{output_dir / 'TimeCheck.json'}")
    
    # 绘制对齐验证图（使用未插值的数据）
    plot_time_diff(imu_data, raw_gt, output_dir, time_range=args.time_range)
    print(f"已绘制时间对齐图：{output_dir / 'TimeDiff.png'}")
    

    
    # === 阶段2：时间同步插值 ===
    print("\n=== 阶段2：时间同步插值 ===")
    
    # 使用 IMU 和 GT 时间戳的并集作为新时间序列
    t_new_us = get_time_series([imu_data.t_us, raw_gt.t_us])
    print(f"统一时间序列：{len(t_new_us)} 个时间点")
    
    # 插值到统一时间序列
    raw_gt = raw_gt.interpolate(t_new_us)
    imu_data = imu_data.interpolate(t_new_us)
    print(f"插值完成：GT 和 IMU 都有 {len(raw_gt)} 个点")

        # 绘制轨迹图（使用未插值但对齐后的数据）
    opt_poses = None
    if args.draw_db:
        # 如果需要对比 DB 优化轨迹
        if args.unit_dir:
            # 单元目录模式：从 rtab_data 加载
            opt_poses = PosesData(rtab_data.opt_t_us, rtab_data.opt_rots, rtab_data.opt_ps)
        else:
            # 旧模式：从 rtab_data 加载
            opt_poses = PosesData(rtab_data.opt_t_us, rtab_data.opt_rots, rtab_data.opt_ps)
        
        # 应用与 raw_gt 相同的时间偏移（不应用坐标转换，因为raw_gt还未转换）
        opt_poses.t_us += t_offset_us
        opt_poses.ps -= opt_poses.ps[0]  # reset start
        
    draw_trajectory(raw_gt, output_dir / "Trajectory.png", opt_gt=opt_poses, gap_info=gap_info)
    print(f"已绘制轨迹图：{output_dir / 'Trajectory.png'}")
    
    # Rerun 验证坐标系转换
    if args.rerun_verify:
        rerun_verify_coordinate_transform(imu_data, raw_gt)
    
    # === 阶段4：保存 CSV ===
    print("\n=== 阶段4：保存 CSV ===")
    
    # 保存插值且坐标转换后的 gt.csv
    # 注意：如果文件已存在（如ExtUnitData生成的），需要先删除才能覆盖
    gt_csv_path = output_dir / "gt.csv"
    if gt_csv_path.exists():
        gt_csv_path.unlink()  # 删除旧文件
    PosesDataSerializer(raw_gt).save(gt_csv_path)
    print(f"已保存 gt.csv：{gt_csv_path}") 
    print("\n预处理完成！")

if __name__ == "__main__":
    main()