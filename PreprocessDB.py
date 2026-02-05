#!/usr/bin/env python3
"""
PreprocessDB - RTAB-Map 数据预处理脚本

功能说明：
    1. 从 RTAB-Map 数据库加载轨迹数据（.db）
    2. 从 CSV 文件加载 IMU 数据（imu.csv）
    3. 应用 Body Frame 转换（RTAB Body → IMU Body）
    4. 时间对齐 IMU 和 GT 数据
    5. 将 GT 和 IMU 插值到统一时间轴
    6. 生成真值轨迹文件（gt.csv）
    7. 提供可视化验证（轨迹图、角速度对比图、Rerun）

坐标系说明：
    - RTAB Body Frame: RTAB-Map 采集设备（相机）的载体坐标系
    - IMU Body Frame: IMU 传感器的载体坐标系
    - Global Frame: 世界坐标系（移动设备使用 Z-UP）
    - DefaultBodyRotation: 绕 Y 轴旋转 -90 度，用于 RTAB Body → IMU Body 转换
"""

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
    """
    绘制 2D 轨迹图
    
    参数说明：
        raw_gt: 真值轨迹数据（GroundTruthData）
            - ps: 位置数据（N x 3 数组）
        output_path: 输出图片路径
        opt_gt: 可选，优化轨迹数据（PosesData）用于对比
        gap_info: 可选，时间空洞信息字典
            - gap_idxs: 空洞索引列表
            - gap_diff: 空洞时长列表
    
    图表元素：
        - 蓝色实线：原始轨迹
        - 绿色圆点：起点
        - 红色叉：终点
        - 绿色虚线（可选）：优化轨迹对比
        - 红点：时间空洞位置（间隔 > 1.0 秒）
    """
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
    """
    真值时间空洞检测
    
    参数说明：
        t_us: 时间戳数组（微秒）
        max_gap_s: 最大允许的时间间隔（秒），默认 1.0 秒
    
    返回值：
        字典包含：
        - max_gap: 最大时间间隔（秒）
        - mean_gap: 平均时间间隔（秒）
        - gap_idxs: 超过阈值的空洞索引列表
        - gap_diff: 超过阈值的时间间隔列表
        - note: 说明文本
    
    用途：
        检测轨迹数据中的时间空洞（如 RTAB-Map 暂停、跟踪丢失等情况）
        在轨迹图中用红点标记空洞位置
    """
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
    """
    时间同步：计算 IMU 和 GT 的时间偏移
    
    参数说明：
        rtab_data: RTAB 数据对象（已应用 transform_local）
            - node_t_us: 关键帧时间戳
            - node_rots: 关键帧旋转
            - node_ps: 关键帧位置
        imu_data: IMU 数据对象
            - t_us: IMU 时间戳
            - gyr, acce, ahrs: IMU 传感器数据
        time_range: 时间同步范围（秒），默认 (0, 50)
            - 在此范围内搜索最佳时间对齐
    
    返回值：
        (GroundTruthData, int): 
            - GroundTruthData: 已对齐的真值数据（包含时间偏移）
            - int: 时间偏移量（微秒）
    
    处理流程：
        1. 使用 RTAB 数据创建 GroundTruthData 对象
        2. 调用 match21 算法计算时间偏移
           - 在 time_range 范围内搜索
           - 使用角速度序列进行匹配
        3. 将时间偏移应用到 GroundTruthData
        4. 重置起始时间（t_us[0] = 0）
    
    注意：
        - 此函数不进行插值，只计算时间偏移
        - 插值在后续步骤中完成
    """
    print(f"正在进行时间同步 (范围: {time_range}s)...")
    gt_data = GroundTruthData(rtab_data.node_t_us, rtab_data.node_rots, rtab_data.node_ps)
    time_gc = match21(imu_data.to_poses(), gt_data, time_range=time_range)
    gt_data.t_us += time_gc
    gt_data.reset_start()
    return gt_data, int(time_gc)

def plot_time_diff(imu_data: ImuData, gt_data: GroundTruthData, output_dir: Path, time_range=(0, 50)):
    """
    绘制角速度对比图，用于验证时间对齐效果
    
    参数说明：
        imu_data: IMU 数据对象
            - t_us: IMU 时间戳
            - ahrs: IMU 姿态（四元数）
        gt_data: GroundTruthData 对象（已对齐）
            - t_us: GT 时间戳
            - rots: GT 旋转
        output_dir: 输出目录
        time_range: X 轴显示范围（秒），默认 (0, 50)
    
    处理流程：
        1. 计算角速度序列：
           - 角速度 = |ω₂ - ω₁| / Δt
           - 使用相邻旋转的四元数差分
        2. 绘制对比曲线：
           - 蓝色曲线：IMU 角速度
           - 橙色曲线：GT 角速度
        3. 动态调整 X 轴范围：
           - 取数据长度和 time_range 的较小值
    
    验证标准：
        - 如果时间对齐正确，两条曲线应该重合
        - 如果存在时间偏移，两条曲线会有相位差
    """
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

def rerun_verify_coordinate_transform(imu_data: ImuData, gt_data: GroundTruthData):
    """
    使用 Rerun 验证坐标系转换是否正确
    
    参数说明：
        imu_data: IMU 数据对象（body frame）
            - acce: 加速度（包含重力）
            - gyro: 角速度
            - magn: 磁场
        gt_data: GroundTruthData 对象（已对齐）
            - rots: 旋转矩阵（body → global）
    
    验证原理：
        1. 将 IMU 数据发送到 Rerun（body frame），标记为 "body"
        2. 使用 gt_data.rots 将 IMU 数据转换到 global frame
        3. 将转换后的 IMU 数据发送到 Rerun（global frame），标记为 "global"
        4. 验证重力只出现在 Z 轴上（约 9.8 m/s²）
    
    坐标系说明：
        - Body Frame: IMU 传感器坐标系
            - 静态时，重力可能出现在任意轴
        - Global Frame: 世界坐标系（Z-UP）
            - Z 轴向上（重力方向）
            - 静态时，重力应该只出现在 Z 轴
    
    验证标准：
        - 静态时，global.acce[2] ≈ 9.8 m/s²（重力）
        - global.acce[0] ≈ 0，global.acce[1] ≈ 0
        - 如果重力出现在其他轴，说明坐标系转换有问题
    
    Rerun 可视化：
        - 打开 Rerun 窗口
        - 显示 "body" 和 "global" 两个标签
        - 每个标签显示对应帧的 IMU 数据
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
        # ========== 模式 1：单元目录模式（推荐） ==========
        # 目录必须包含：imu.csv 和一个 .db 文件
        unit_dir = Path(args.unit_dir)
        print(f"使用单元目录模式: {unit_dir}")
        
        # 检查必需文件
        imu_file = unit_dir / "imu.csv"
        if not imu_file.exists():
            raise FileNotFoundError(f"未找到 IMU 数据文件: {imu_file}")
        
        # 检查是否存在 RTAB 数据库文件（自动查找 .db 文件）
        db_file = RTABData.get_db_file(unit_dir)
        if db_file is None:
            raise FileNotFoundError(f"未找到 RTAB 数据库文件(.db)在目录: {unit_dir}")
        
        # ========== 步骤 1：加载数据 ==========
        # 从 RTAB-Map 加载原始数据（不使用 ExtUnitData，避免生成 opt.csv）
        print(f"正在从数据库加载数据: {db_file}")
        rtab_data = RTABData(db_file)
        # 应用 RTAB Body Frame → IMU Body Frame 转换
        rtab_data.transform_local()
        
        # 从 CSV 加载 IMU 数据
        imu_data = ImuData.from_csv(imu_file)
        
        # ========== 步骤 3：时间对齐（不插值）==========
        raw_gt, t_offset_us = time_sync(rtab_data, imu_data, time_range=tuple(args.time_range))
        
        # 记录输出路径信息
        data_path = str(unit_dir.absolute())
        device_name = unit_dir.name
        
    elif args.db and args.imu:
        # ========== 模式 2：兼容旧模式 ==========
        # 直接指定 db 和 imu 文件
        print(f"使用旧模式: --db {args.db} --imu {args.imu}")
        
        # 步骤 1：加载数据
        rtab_data = RTABData(args.db)
        rtab_data.transform_local()
        
        imu_data = ImuData.from_csv(Path(args.imu))
        
        # 步骤 3：时间对齐（不插值）
        raw_gt, t_offset_us = time_sync(rtab_data, imu_data, time_range=tuple(args.time_range))
        
        # 记录输出路径信息
        data_path = str(Path(args.imu).parent.absolute())
        device_name = Path(args.imu).parent.name
    else:
        parser.error("必须指定 -u/--unit-dir 或 (--db 和 --imu)")
    
    # ========== 确定输出目录 ==========
    output_dir = Path(args.output) if args.output else (unit_dir if args.unit_dir else Path(args.imu).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 阶段 1：时间对齐（不插值）==========
    print("\n=== 阶段1：时间对齐（不插值）===")
    
    # ========== 步骤 3.2：记录 Gap 和 Rate 信息 ==========
    # 使用未插值的数据进行检测
    gap_info = check_groundtruth_gap(raw_gt.t_us)
    imu_rate = float(1e6 / np.mean(np.diff(imu_data.t_us)))

    # ========== 步骤 3.3：输出 TimeCheck.json ==========
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
    

    
    # 时间同步插值 
    print("\n=== 时间同步插值 ===")
    
    # 使用 IMU 和 GT 时间戳的并集作为新时间序列
    t_new_us = get_time_series([imu_data.t_us, raw_gt.t_us])
    print(f"统一时间序列：{len(t_new_us)} 个时间点")
    
    # 插值到统一时间序列
    raw_gt = raw_gt.interpolate(t_new_us)
    imu_data = imu_data.interpolate(t_new_us)
    print(f"插值完成：GT 和 IMU 都有 {len(raw_gt)} 个点")

    # 绘制轨迹图（使用插值对齐后的数据）
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
    print("\n=== 保存 gt.csv ===")
    
    # 保存插值且坐标转换后的 gt.csv
    # 注意：如果文件已存在，需要先删除才能覆盖
    gt_csv_path = output_dir / "gt.csv"
    if gt_csv_path.exists():
        gt_csv_path.unlink()  # 删除旧文件
    PosesDataSerializer(raw_gt).save(gt_csv_path)
    print(f"已保存 gt.csv：{gt_csv_path}") 
    
    # === 保存插值对齐后的 IMU 数据 ===
    print("\n=== 保存 imu_in.csv ===")
    
    # 保存插值对齐后的 IMU 数据
    imu_in_csv_path = output_dir / "imu.csv"
    if imu_in_csv_path.exists():
        imu_in_csv_path.unlink()  # 删除旧文件
    from base.serialize import ImuDataSerializer
    ImuDataSerializer(imu_data).save(imu_in_csv_path)
    print(f"已保存 imu.csv：{imu_in_csv_path}")
    
    print("\n预处理完成！")

if __name__ == "__main__":
    main()