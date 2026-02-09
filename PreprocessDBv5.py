#!/usr/bin/env python3
"""
PreprocessDB - RTAB-Map 数据预处理脚本

功能说明：
    1. 从 RTAB-Map 数据库/CSV 加载轨迹数据（.db/.csv）
    2. 从 CSV 文件加载 IMU 数据（imu.csv）
    3. 加载优化轨迹（opt）并与 GT 同步对齐
    4. 应用 Body Frame 转换（RTAB Body → IMU Body）
    5. 时间对齐 IMU 和 GT 数据
    6. 将 IMU/GT/OPT 插值到统一时间轴
    7. 生成真值轨迹与可视化（gt.csv、gt.png）
    8. 输出 IMU/OPT 标准化结果（imu-c.csv/opt.csv）

坐标系说明：
    - RTAB Body Frame: RTAB-Map 采集设备（相机）的载体坐标系
    - IMU Body Frame: IMU 传感器的载体坐标系
    - Global Frame: 世界坐标系（移动设备使用 Z-UP）
    - DefaultBodyRotation: 绕 Y 轴旋转 -90 度，用于 RTAB Body → IMU Body 转换
"""

import argparse
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation

import rerun as rr

from base.calibration.time import match21
from base.datatype import GroundTruthData, ImuData, PosesData
from base.draw.Poses import draw_trajectory_2d, draw_trajectory_2d_compare
from base.interpolate import get_time_series
from base.rtab import RTABData
from base.serialize import PosesDataSerializer
from base.rerun_ext import RerunView, send_imu_data

def find_unit_directories(base_dir: Path, depth: int) -> list[Path]:
    """
    查找符合条件的单元目录
    
    参数说明：
        base_dir: 基础目录
        depth: 查找深度
            - depth=1: 查找 base_dir 的直接子目录
            - depth=2: 查找 base_dir 的子目录的子目录
            - depth=3: 查找 base_dir 的子目录的子目录的子目录
    
    返回：
        符合条件的单元目录列表（包含 imu.csv 和 .db 文件的目录）
    """
    unit_dirs = []
    
    # 递归查找指定深度的目录
    def _find_dirs(current_dir: Path, current_depth: int):
        if current_depth == 0:
            # 检查当前目录是否包含必需文件
            imu_file = current_dir / "imu.csv"
            db_files = list(current_dir.glob("*.db"))
            if imu_file.exists() and len(db_files) > 0:
                unit_dirs.append(current_dir)
            return
        
        # 继续向下查找
        for child in current_dir.iterdir():
            if child.is_dir():
                _find_dirs(child, current_depth - 1)
    
    _find_dirs(base_dir, depth)
    return unit_dirs

def draw_trajectory(raw_gt: GroundTruthData, output_path: Path, opt_gt: PosesData = None):
    """
    绘制 2D 轨迹图
    
    参数说明：
        raw_gt: 真值轨迹数据（GroundTruthData）
            - ps: 位置数据（N x 3 数组）
        output_path: 输出图片路径
        opt_gt: 可选，优化轨迹数据（PosesData）用于对比
    
    图表元素：
        - GT/OPT 轨迹对比（参考 GenerateValid 的绘图风格）
    """
    print(f"正在生成轨迹图：{output_path.name}")
    if opt_gt is None:
        draw_trajectory_2d(raw_gt, show=False, save_path=output_path)
        return

    draw_trajectory_2d_compare(
        [raw_gt, opt_gt],
        ["GT", "OPT"],
        show=False,
        save_path=output_path,
    )

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

def time_sync(
    rtab_data: RTABData,
    imu_data: ImuData,
    coarse_range: tuple[float, float] | None = None,
    fine_range: tuple[float, float] | None = None,
    output_dir: Path | None = None,
) -> tuple[GroundTruthData, int]:
    """
    时间同步：计算 IMU 和 GT 的时间偏移（两步法）
    
    参数说明：
        rtab_data: RTAB 数据对象（已应用 transform_local）
            - node_t_us: 关键帧时间戳
            - node_rots: 关键帧旋转
            - node_ps: 关键帧位置
        imu_data: IMU 数据对象
            - t_us: IMU 时间戳
            - gyr, acce, ahrs: IMU 传感器数据
        coarse_range: 粗对齐时间范围（秒），例如 (0, 50)
            - 指定后用于粗对齐的大范围搜索
        fine_range: 精对齐时间范围（秒），例如 (0, 10)
            - 指定后用于精对齐的小范围搜索
            - 不指定则在粗对齐范围内自动选择尖峰最密集的约 10 秒窗口
        output_dir: 输出目录（可选）
            - 如果提供，match21 将生成时间对齐图 TimeDiff1.png 和 TimeDiff2.png
    
    返回值：
        (GroundTruthData, int): 
            - GroundTruthData: 已对齐的真值数据（包含时间偏移）
            - int: 时间偏移量（微秒）
    
    处理流程：
        1. 使用 RTAB 数据创建 GroundTruthData 对象
        2. 第一步：粗对齐（大范围搜索）
           - 生成 TimeDiff1.png
        3. 第二步：精对齐（小范围微调，支持指定或自动选择窗口）
           - 生成 TimeDiff2.png
        4. 将时间偏移应用到 GroundTruthData
        5. 重置起始时间（t_us[0] = 0）
    
    注意：
        - 此函数不进行插值，只计算时间偏移
        - 插值在后续步骤中完成
    """
    print(f"=== 时间同步检查（两步法）===")

    coarse_range = coarse_range or (0.0, 50.0)

    def _get_angvels_local(t_us: np.ndarray, rots: Rotation, step: int = 1):
        """获取角速度列表（本地实现，避免改动 time.py）"""
        n = len(rots)
        step = max(int(step), 1)
        assert n >= 2, "At least two rotations are required"

        As: list[float] = []
        Ts: list[int] = []
        for i in range(0, n - step, step):
            drot = rots[i].inv() * rots[i + step]
            angle = float(np.linalg.norm(drot.as_rotvec()))
            dt_s = (t_us[i + step] - t_us[i]) * 1e-6
            if dt_s <= 0:
                continue
            ang_vel = angle / dt_s
            As.append(ang_vel)
            Ts.append(int(t_us[i]))
        return np.array(As, dtype=float), np.array(Ts, dtype=np.int64)

    def _auto_find_fine_range(
        imu_poses: PosesData,
        gt_poses: PosesData,
        window_s: float = 10.0,
        coarse_range_local: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """自动选择尖峰最密集的时间窗口（相对起点，单位秒）。"""
        t_start_us = max(imu_poses.t_us[0], gt_poses.t_us[0])
        t_end_us = min(imu_poses.t_us[-1], gt_poses.t_us[-1])
        duration_s = max((t_end_us - t_start_us) * 1e-6, 0.0)
        if duration_s <= 0:
            return (0.0, window_s)

        window_s = min(window_s, duration_s)
        step = max(int(imu_poses.rate / 200), 1)
        ang_vels, ang_ts = _get_angvels_local(imu_poses.t_us, imu_poses.rots, step=step)
        if len(ang_ts) == 0:
            return (0.0, window_s)

        # 仅保留重叠区间内的角速度
        mask = (ang_ts >= t_start_us) & (ang_ts <= t_end_us)
        ang_ts = ang_ts[mask]
        ang_vels = ang_vels[mask]
        if len(ang_ts) == 0:
            return (0.0, window_s)

        # 若指定了粗对齐范围，则限制自动窗口在粗对齐范围内
        if coarse_range_local is not None:
            coarse_start_s, coarse_end_s = coarse_range_local
            coarse_start_us = t_start_us + int(coarse_start_s * 1e6)
            coarse_end_us = t_start_us + int(coarse_end_s * 1e6)
            coarse_mask = (ang_ts >= coarse_start_us) & (ang_ts <= coarse_end_us)
            ang_ts = ang_ts[coarse_mask]
            ang_vels = ang_vels[coarse_mask]
            if len(ang_ts) == 0:
                return (coarse_start_s, min(coarse_start_s + window_s, coarse_end_s))

        # 双指针滑动窗口，最大化窗口内“尖峰数量”
        if len(ang_vels) >= 10:
            peak_threshold = float(np.percentile(ang_vels, 80))
        else:
            peak_threshold = float(np.max(ang_vels) * 0.9)

        peak_flags = (ang_vels >= peak_threshold).astype(np.int32)
        peak_prefix = np.concatenate([[0], np.cumsum(peak_flags)])
        sum_prefix = np.concatenate([[0.0], np.cumsum(ang_vels)])

        best_peak_cnt = -1
        best_sum = -1.0
        best_i = 0
        best_j = 1
        j = 0
        window_us = window_s * 1e6
        for i in range(len(ang_ts)):
            if j < i:
                j = i
            while j < len(ang_ts) and (ang_ts[j] - ang_ts[i]) <= window_us:
                j += 1
            window_peak_cnt = int(peak_prefix[j] - peak_prefix[i])
            window_sum = float(sum_prefix[j] - sum_prefix[i])
            if window_peak_cnt > best_peak_cnt or (
                window_peak_cnt == best_peak_cnt and window_sum > best_sum
            ):
                best_peak_cnt = window_peak_cnt
                best_sum = window_sum
                best_i = i
                best_j = j

        start_us = int(ang_ts[best_i])
        end_us = int(min(start_us + window_us, t_end_us))
        start_s = (start_us - t_start_us) * 1e-6
        end_s = (end_us - t_start_us) * 1e-6
        return (max(start_s, 0.0), max(end_s, start_s + 1e-3))
    
    gt_data = GroundTruthData(rtab_data.node_t_us, rtab_data.node_rots, rtab_data.node_ps)
    imu_poses = imu_data.to_poses()
    
    # 参考 DataCheck.py 中的 check_time_diff 方法
    print(f"IMU数据点数: {len(imu_data)}")
    print(f"GT数据点数: {len(gt_data)}")
    print(f"IMU采样率: {1e6 / np.mean(np.diff(imu_data.t_us)):.1f} Hz")
    print(f"GT采样率: {gt_data.rate:.1f} Hz")
    
    # 第一步：粗对齐（大范围搜索）
    print(f"\n=== 第一步：粗对齐（大范围搜索）===")
    print(f"搜索范围: {coarse_range[0]}s 到 {coarse_range[1]}s")
    
    coarse_save_path = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        coarse_save_path = output_dir / "TimeDiff1.png"
    
    # 使用较低的分辨率进行粗对齐
    coarse_time_gc = match21(imu_poses, gt_data, 
                           time_range=coarse_range, 
                           save_path=coarse_save_path,
                           resolution=100)  
    
    print(f"粗对齐结果: {coarse_time_gc / 1e6:.6f} 秒")
    
    # 第二步：精对齐（小范围微调）
    print(f"\n=== 第二步：精对齐（小范围微调）===")
    
    # 创建一个临时的GT数据副本，应用粗对齐结果
    gt_data_fine = GroundTruthData(rtab_data.node_t_us, rtab_data.node_rots, rtab_data.node_ps)
    gt_data_fine.t_us += coarse_time_gc
    
    # 在小范围内搜索相对偏移（支持指定或自动）
    if fine_range is None:
        fine_range = _auto_find_fine_range(
            imu_poses,
            gt_data_fine,
            window_s=15.0,
            coarse_range_local=coarse_range,
        )
        print(
            "精对齐搜索范围: 自动选择尖峰最密集窗口 "
            f"{fine_range[0]:.2f}s 到 {fine_range[1]:.2f}s"
        )
    else:
        fine_range = (float(fine_range[0]), float(fine_range[1]))
        print(f"精对齐搜索范围: 手动指定 {fine_range[0]:.2f}s 到 {fine_range[1]:.2f}s")
    
    fine_save_path = None
    if output_dir:
        fine_save_path = output_dir / "TimeDiff2.png"
    
    # 使用较高的分辨率搜索相对偏移
    # 注意：这里搜索的是相对偏移，不是绝对偏移
    relative_offset = match21(imu_poses, gt_data_fine,
                            time_range=fine_range,
                            save_path=fine_save_path,
                            resolution=200)  # 较高分辨率
    
    # 最终时间偏移 = 粗对齐结果 + 相对偏移
    fine_time_gc = coarse_time_gc + relative_offset
    
    print("=" * 60)
    print(f"粗对齐结果: {coarse_time_gc / 1e6:.6f} 秒")
    print(f"精对齐相对偏移: {relative_offset / 1e6:.6f} 秒")
    print(f"最终时间偏移: {fine_time_gc / 1e6:.6f} 秒 ({fine_time_gc} μs)")
    
    # 应用最终的时间偏移
    gt_data.t_us += fine_time_gc
    gt_data.reset_start()
    
    print(f"\n时间同步完成（两步法）")
    print("=" * 80)
    
    return gt_data, int(fine_time_gc)


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


def process_unit_directory(
    unit_dir: Path,
    args,
) -> tuple[GroundTruthData, ImuData, PosesData | None, int, dict]:
    """
    处理单个单元目录
    
    参数说明：
        unit_dir: 单元目录路径
        args: 命令行参数
    
    返回：
        (raw_gt, imu_data, opt_data, t_offset_us, gap_info): 
            - raw_gt: 真值数据
            - imu_data: IMU数据
            - opt_data: 优化轨迹（可选）
            - t_offset_us: 时间偏移量
            - gap_info: 时间空洞信息
    """
    print(f"\n处理单元目录: {unit_dir}")
    
    # 检查必需文件
    imu_file = unit_dir / "imu.csv"
    if not imu_file.exists():
        raise FileNotFoundError(f"未找到 IMU 数据文件: {imu_file}")
    
    # 检查是否存在 RTAB 数据库文件（自动查找 .db 文件）
    db_file = RTABData.get_db_file(unit_dir)
    if db_file is None:
        raise FileNotFoundError(f"未找到 RTAB 数据库文件(.db)在目录: {unit_dir}")
    
    # 从 RTAB-Map 加载原始数据
    print(f"正在从数据库加载数据: {db_file}")
    rtab_data = RTABData(db_file)
    # 应用 RTAB Body Frame → IMU Body Frame 转换
    rtab_data.transform_local()
    
    # 从 CSV 加载 IMU 数据
    imu_data = ImuData.from_csv(imu_file)

    # 从 RTAB-Map 加载优化轨迹（可选）
    opt_data = None
    if hasattr(rtab_data, "opt_ps"):
        opt_t_us = getattr(rtab_data, "opt_t_us", rtab_data.node_t_us)
        opt_data = PosesData(opt_t_us, rtab_data.opt_rots, rtab_data.opt_ps)
    
    # 时间对齐
    # 传入单元目录作为输出目录，让 match21 直接生成时间对齐图
    raw_gt, t_offset_us = time_sync(
        rtab_data,
        imu_data,
        coarse_range=args.time_range_coarse,
        fine_range=args.time_range_fine,
        output_dir=unit_dir,
    )

    # 检测时间空洞
    gap_info = check_groundtruth_gap(raw_gt.t_us)

    return raw_gt, imu_data, opt_data, t_offset_us, gap_info

def save_results(
    raw_gt: GroundTruthData,
    imu_data: ImuData,
    opt_data: PosesData | None,
    t_offset_us: int,
    gap_info: dict,
    output_dir: Path,
    args,
    data_path: str,
    device_name: str,
):
    """
    保存处理结果
    
    参数说明：
        raw_gt: 真值数据
        imu_data: IMU数据
        t_offset_us: 时间偏移量
        gap_info: 时间空洞信息
        output_dir: 输出目录
        args: 命令行参数
        data_path: 数据路径
        device_name: 设备名称
    """
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算 IMU 频率
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
    
    # 时间对齐图已经在 time_sync 中由 match21 生成（两步法）
    # plot_time_diff(imu_data, raw_gt, output_dir, time_range=args.time_range, time_offset_us=t_offset_us)
    
    print(f"时间对齐图已由 match21 生成（两步法）：")
    print(f"  - 粗对齐图：{output_dir / 'TimeDiff1.png'}")
    print(f"  - 精对齐图：{output_dir / 'TimeDiff2.png'}")
    
    # 时间同步插值
    print("\n=== 时间同步插值 ===")
    ts_us = [imu_data.t_us, raw_gt.t_us]
    if opt_data is not None:
        opt_data.t_us = opt_data.t_us + t_offset_us
        ts_us.append(opt_data.t_us)

    t_new_us = get_time_series(ts_us)
    print(f"统一时间序列：{len(t_new_us)} 个时间点")
    
    # 插值到统一时间序列
    raw_gt = raw_gt.interpolate(t_new_us)
    imu_data = imu_data.interpolate(t_new_us)
    if opt_data is not None:
        opt_data = opt_data.interpolate(t_new_us)
    print(f"插值完成：GT 和 IMU 都有 {len(raw_gt)} 个点")

    # 重置时间起点，使 t=0 对齐
    t_start_us = int(t_new_us[0])
    raw_gt.t_us = raw_gt.t_us - t_start_us
    imu_data.t_us = imu_data.t_us - t_start_us
    if opt_data is not None:
        opt_data.t_us = opt_data.t_us - t_start_us
    
    # 绘制轨迹图
    opt_poses = opt_data
    
    draw_trajectory(raw_gt, output_dir / "gt.png", opt_gt=opt_poses)
    print(f"已绘制轨迹图：{output_dir / 'gt.png'}")
    
    # Rerun 验证坐标系转换
    if args.rerun_verify:
        rerun_verify_coordinate_transform(imu_data, raw_gt)
    
    # 保存 gt.csv
    gt_csv_path = output_dir / "gt.csv"
    if gt_csv_path.exists():
        gt_csv_path.unlink()  # 删除旧文件
    PosesDataSerializer(raw_gt).save(gt_csv_path)
    print(f"已保存 gt.csv：{gt_csv_path}")
    
    # 保存 imu.csv
    imu_csv_path = output_dir / "imu-c.csv"
    if imu_csv_path.exists():
        imu_csv_path.unlink()  # 删除旧文件
    from base.serialize import ImuDataSerializer
    ImuDataSerializer(imu_data).save(imu_csv_path)
    print(f"已保存 imu-c.csv：{imu_csv_path}")

    # 保存 opt.csv
    if opt_data is not None:
        opt_csv_path = output_dir / "opt.csv"
        if opt_csv_path.exists():
            opt_csv_path.unlink()
        PosesDataSerializer(opt_data).save(opt_csv_path)
        print(f"已保存 opt.csv：{opt_csv_path}")


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
        1. 单元目录模式: -u <数据目录> [--output <输出目录>]
        - 数据目录必须包含：imu.csv 和一个 .db 文件
        - 生成文件：gt.csv, imu-c.csv, opt.csv(可选), gt.png, TimeCheck.json, TimeDiff1/2.png
        - 默认输出到数据目录本身
        
        2. 组目录模式: -g <组目录> 
        - 处理组目录下的所有子目录（每个子目录应包含imu.csv和.db文件）
        - 输出到各自的子目录
        
        3. 项目目录模式: -p <项目目录>
        - 处理项目目录下的所有子目录的子目录（两层深度）
        - 输出到各自的单元目录
        
        4. 数据集目录模式: -d <数据集目录>
        - 处理数据集目录下的所有子目录的子目录的子目录（三层深度）
        - 输出到各自的单元目录
        
        5. 兼容旧模式: --db <db文件> --imu <imu.csv> [--output <输出目录>]
        - 生成文件：gt.csv, imu-c.csv, opt.csv(可选), gt.png, TimeCheck.json, TimeDiff1/2.png
        - 默认输出到 imu.csv 同文件夹
        """
    )
    parser.add_argument("-u", "--unit-dir", type=str, default=None,
                       help="指定单元数据目录（包含imu.csv和.db文件）")
    parser.add_argument("-g", "--group-dir", type=str, default=None,
                       help="指定组数据目录，处理该目录下的所有子目录（每个子目录应包含imu.csv和.db文件）")
    parser.add_argument("-p", "--project-dir", type=str, default=None,
                       help="指定项目数据目录，处理该目录下的所有子目录的子目录（两层深度）")
    parser.add_argument("-d", "--dataset-dir", type=str, default=None,
                       help="指定数据集目录，处理该目录下的所有子目录的子目录的子目录（三层深度）")
    parser.add_argument("-db", type=str, default=None,
                       help="RTAB数据库文件路径(.db)，用于兼容旧模式")
    parser.add_argument("-imu", type=str, default=None,
                       help="IMU数据文件路径(imu.csv)，用于兼容旧模式")
    parser.add_argument("-output", type=str, default=None,
                       help="输出目录路径（可选），批量处理时忽略此参数")
    parser.add_argument(
        "-t1",
        "--time-range-coarse",
        type=float,
        nargs=2,
        default=[0, 50],
        help="粗对齐时间范围（秒），如: -t1 0 50",
    )
    parser.add_argument(
        "-t2",
        "--time-range-fine",
        type=float,
        nargs=2,
        default=None,
        help="精对齐时间范围（秒），如: -t2 0 10；不指定则自动选择尖峰最多的约10s窗口",
    )
    parser.add_argument("-rv", "--rerun-verify", action="store_true", 
                       help="使用 Rerun 验证坐标系转换是否正确")
    args = parser.parse_args()

    # 检查参数组合
    mode_count = sum([
        args.unit_dir is not None,
        args.group_dir is not None,
        args.project_dir is not None,
        args.dataset_dir is not None,
        args.db is not None and args.imu is not None
    ])
    
    if mode_count == 0:
        parser.error("必须指定以下模式之一：\n"
                    "  -u/--unit-dir <目录> (单元目录模式)\n"
                    "  -g/--group-dir <目录> (组目录模式)\n"
                    "  -p/--project-dir <目录> (项目目录模式)\n"
                    "  -d/--dataset-dir <目录> (数据集目录模式)\n"
                    "  --db <文件> --imu <文件> (兼容旧模式)")
    elif mode_count > 1:
        parser.error("只能指定一种处理模式")
    
    # ========== 模式 1：单元目录模式 ==========
    if args.unit_dir:
        print(f"使用单元目录模式: {args.unit_dir}")
        unit_dir = Path(args.unit_dir)
        
        try:
            # 处理单个单元目录
            raw_gt, imu_data, opt_data, t_offset_us, gap_info = process_unit_directory(
                unit_dir,
                args,
            )
            
            # 确定输出目录
            output_dir = Path(args.output) if args.output else unit_dir
            data_path = str(unit_dir.absolute())
            device_name = unit_dir.name
            
            # 保存结果
            save_results(
                raw_gt,
                imu_data,
                opt_data,
                t_offset_us,
                gap_info,
                output_dir,
                args,
                data_path,
                device_name,
            )
            
            print(f"\n单元目录处理完成: {unit_dir}")
            
        except Exception as e:
            print(f"处理单元目录失败: {unit_dir}")
            print(f"错误信息: {e}")
            raise
    
    # ========== 模式 2：组目录模式 (-g) ==========
    elif args.group_dir:
        group_dir = Path(args.group_dir)
        print(f"使用组目录模式: {group_dir}")
        print(f"查找深度: 1 (直接子目录)")
        
        # 查找符合条件的单元目录
        unit_dirs = find_unit_directories(group_dir, depth=1)
        
        if not unit_dirs:
            print(f"警告：在组目录 {group_dir} 中未找到符合条件的单元目录")
            return
        
        print(f"找到 {len(unit_dirs)} 个单元目录需要处理")
        
        # 批量处理每个单元目录
        success_count = 0
        for i, unit_dir in enumerate(unit_dirs, 1):
            print(f"\n{'='*60}")
            print(f"处理第 {i}/{len(unit_dirs)} 个单元目录: {unit_dir}")
            print(f"{'='*60}")
            
            try:
                # 处理单个单元目录
                raw_gt, imu_data, opt_data, t_offset_us, gap_info = process_unit_directory(
                    unit_dir,
                    args,
                )
                
                # 输出目录为单元目录本身
                output_dir = unit_dir
                data_path = str(unit_dir.absolute())
                device_name = unit_dir.name
                
                # 保存结果
                save_results(
                    raw_gt,
                    imu_data,
                    opt_data,
                    t_offset_us,
                    gap_info,
                    output_dir,
                    args,
                    data_path,
                    device_name,
                )
                
                success_count += 1
                print(f"✓ 单元目录处理成功: {unit_dir}")
                
            except Exception as e:
                print(f"✗ 单元目录处理失败: {unit_dir}")
                print(f"错误信息: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"组目录处理完成")
        print(f"成功: {success_count}/{len(unit_dirs)}")
        print(f"失败: {len(unit_dirs) - success_count}/{len(unit_dirs)}")
        print(f"{'='*60}")
    
    # ========== 模式 3：项目目录模式 (-p) ==========
    elif args.project_dir:
        project_dir = Path(args.project_dir)
        print(f"使用项目目录模式: {project_dir}")
        print(f"查找深度: 2 (子目录的子目录)")
        
        # 查找符合条件的单元目录
        unit_dirs = find_unit_directories(project_dir, depth=2)
        
        if not unit_dirs:
            print(f"警告：在项目目录 {project_dir} 中未找到符合条件的单元目录")
            return
        
        print(f"找到 {len(unit_dirs)} 个单元目录需要处理")
        
        # 批量处理每个单元目录
        success_count = 0
        for i, unit_dir in enumerate(unit_dirs, 1):
            print(f"\n{'='*60}")
            print(f"处理第 {i}/{len(unit_dirs)} 个单元目录: {unit_dir}")
            print(f"{'='*60}")
            
            try:
                # 处理单个单元目录
                raw_gt, imu_data, opt_data, t_offset_us, gap_info = process_unit_directory(
                    unit_dir,
                    args,
                )
                
                # 输出目录为单元目录本身
                output_dir = unit_dir
                data_path = str(unit_dir.absolute())
                device_name = unit_dir.name
                
                # 保存结果
                save_results(
                    raw_gt,
                    imu_data,
                    opt_data,
                    t_offset_us,
                    gap_info,
                    output_dir,
                    args,
                    data_path,
                    device_name,
                )
                
                success_count += 1
                print(f"✓ 单元目录处理成功: {unit_dir}")
                
            except Exception as e:
                print(f"✗ 单元目录处理失败: {unit_dir}")
                print(f"错误信息: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"项目目录处理完成")
        print(f"成功: {success_count}/{len(unit_dirs)}")
        print(f"失败: {len(unit_dirs) - success_count}/{len(unit_dirs)}")
        print(f"{'='*60}")
    
    # ========== 模式 4：数据集目录模式 (-d) ==========
    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        print(f"使用数据集目录模式: {dataset_dir}")
        print(f"查找深度: 3 (子目录的子目录的子目录)")
        
        # 查找符合条件的单元目录
        unit_dirs = find_unit_directories(dataset_dir, depth=3)
        
        if not unit_dirs:
            print(f"警告：在数据集目录 {dataset_dir} 中未找到符合条件的单元目录")
            return
        
        print(f"找到 {len(unit_dirs)} 个单元目录需要处理")
        
        # 批量处理每个单元目录
        success_count = 0
        for i, unit_dir in enumerate(unit_dirs, 1):
            print(f"\n{'='*60}")
            print(f"处理第 {i}/{len(unit_dirs)} 个单元目录: {unit_dir}")
            print(f"{'='*60}")
            
            try:
                # 处理单个单元目录
                raw_gt, imu_data, opt_data, t_offset_us, gap_info = process_unit_directory(
                    unit_dir,
                    args,
                )
                
                # 输出目录为单元目录本身
                output_dir = unit_dir
                data_path = str(unit_dir.absolute())
                device_name = unit_dir.name
                
                # 保存结果
                save_results(
                    raw_gt,
                    imu_data,
                    opt_data,
                    t_offset_us,
                    gap_info,
                    output_dir,
                    args,
                    data_path,
                    device_name,
                )
                
                success_count += 1
                print(f"✓ 单元目录处理成功: {unit_dir}")
                
            except Exception as e:
                print(f"✗ 单元目录处理失败: {unit_dir}")
                print(f"错误信息: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"数据集目录处理完成")
        print(f"成功: {success_count}/{len(unit_dirs)}")
        print(f"失败: {len(unit_dirs) - success_count}/{len(unit_dirs)}")
        print(f"{'='*60}")
    
    # ========== 模式 5：兼容旧模式 ==========
    elif args.db and args.imu:
        print(f"使用兼容旧模式: --db {args.db} --imu {args.imu}")
        
        # 直接指定 db 和 imu 文件
        db_file = Path(args.db)
        imu_file = Path(args.imu)
        
        if not db_file.exists():
            raise FileNotFoundError(f"未找到数据库文件: {db_file}")
        if not imu_file.exists():
            raise FileNotFoundError(f"未找到 IMU 数据文件: {imu_file}")
        
        # 从 RTAB-Map 加载原始数据
        print(f"正在从数据库加载数据: {db_file}")
        rtab_data = RTABData(str(db_file))
        rtab_data.transform_local()
        
        # 从 CSV 加载 IMU 数据
        imu_data = ImuData.from_csv(imu_file)

        # 从 RTAB-Map 加载优化轨迹（可选）
        opt_data = None
        if hasattr(rtab_data, "opt_ps"):
            opt_t_us = getattr(rtab_data, "opt_t_us", rtab_data.node_t_us)
            opt_data = PosesData(opt_t_us, rtab_data.opt_rots, rtab_data.opt_ps)
        
        # 确定输出目录
        output_dir = Path(args.output) if args.output else imu_file.parent

        # 时间对齐（不插值）
        # 传入输出目录，让 match21 直接生成时间对齐图
        raw_gt, t_offset_us = time_sync(
            rtab_data,
            imu_data,
            coarse_range=args.time_range_coarse,
            fine_range=args.time_range_fine,
            output_dir=output_dir,
        )
        
        # 检测时间空洞
        gap_info = check_groundtruth_gap(raw_gt.t_us)
        
        # 输出目录已确定
        data_path = str(imu_file.parent.absolute())
        device_name = imu_file.parent.name
        
        # 保存结果
        save_results(
            raw_gt,
            imu_data,
            opt_data,
            t_offset_us,
            gap_info,
            output_dir,
            args,
            data_path,
            device_name,
        )
        
        print(f"\n兼容旧模式处理完成")
    
    print("\n所有处理完成！")

if __name__ == "__main__":
    main()