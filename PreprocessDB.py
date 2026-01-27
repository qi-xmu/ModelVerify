#!/usr/bin/env python3
"""
预处理脚本：将RTAB数据库(.db)文件转换为gt.csv文件

功能说明:
    该脚本用于从原始RTAB数据库文件中生成经过预处理的gt.csv真值数据。主要功能包括:

    1. 数据加载与同步:
       - 从.db文件加载RTAB真值数据
       - 从imu.csv文件加载IMU数据
       - 计算共同时间窗口，并进行插值对齐
       - 通过时间匹配算法校准真值数据与IMU数据的时间戳
       - 生成时间对齐效果图(Timediff.png)

    2. 坐标系转换:
       - 将RTAB的Y-UP坐标系转换为Z-UP坐标系
       - 生成坐标系转换效果图(Ttansfom.png)，比较前后的三轴加速度

    3. 数据保存:
       - 保存时间同步和坐标系转换后的gt.csv文件

使用示例:
    python PreprocessDB.py --db <db_file> --imu <imu_file> --output <output_dir>

输入数据格式:
    - db_file: RTAB数据库文件(.db)
    - imu_file: IMU数据文件(imu.csv)，包含时间戳和IMU测量信息

输出结果:
    output/
    ├── gt.csv          # 处理后的真值数据
    ├── Timediff.png    # 时间对齐效果图
    └── Ttansfom.png    # 坐标系转换效果图
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation

from base.calibration.time import match21
from base.datatype import GroundTruthData, ImuData
from base.draw.Poses import draw_trajectory_2d_compare
from base.interpolate import get_time_series
from base.rtab import RTABData
from base.serialize import PosesDataSerializer


def load_rtab_data(db_path: Path) -> RTABData:
    """加载RTAB数据"""
    print("正在加载RTAB数据库数据...")
    rtab_data = RTABData(db_path)
    print(f"加载完成，共 {len(rtab_data)} 个位姿")
    return rtab_data


def load_imu_data(imu_path: Path) -> ImuData:
    """加载IMU数据"""
    print("正在加载IMU数据...")
    if not imu_path.exists():
        raise FileNotFoundError(f"IMU文件不存在: {imu_path}")
    
    try:
        imu_data = ImuData.from_csv(imu_path)
        print(f"加载完成，共 {len(imu_data)} 个IMU测量")
        return imu_data
    except UnicodeDecodeError as e:
        print(f"文件编码错误，尝试使用 latin-1 编码读取: {e}")
        # 尝试其他编码
        try:
            import pandas as pd
            raw = pd.read_csv(imu_path, encoding='latin-1').dropna().to_numpy()
            imu_data = ImuData.from_raw(raw)
            print(f"使用 latin-1 编码加载完成，共 {len(imu_data)} 个IMU测量")
            return imu_data
        except Exception as e2:
            raise RuntimeError(f"无法读取IMU文件: {e2}")
    except Exception as e:
        raise RuntimeError(f"加载IMU数据失败: {e}")


def time_sync(rtab_data: RTABData, imu_data: ImuData) -> tuple[GroundTruthData, ImuData, int]:
    """时间同步RTAB和IMU数据"""
    print("正在进行时间同步...")

    # 将RTAB数据转换为PosesData格式
    gt_data = GroundTruthData(rtab_data.node_t_us, rtab_data.node_rots, rtab_data.node_ps)

    # 时间匹配
    time_gc = match21(imu_data.to_poses(), gt_data)
    gt_data.t_us += time_gc
    gt_data.reset_start()

    print(f"时间差：{time_gc} 微秒")

    # 获取共同时间窗口
    ts_us = [gt_data.t_us, imu_data.t_us]
    t_new_us = get_time_series(ts_us)

    # 插值对齐
    gt_data = gt_data.interpolate(t_new_us)
    imu_data = imu_data.interpolate(t_new_us)

    print(f"插值对齐完成，共同时间序列长度：{len(t_new_us)}")

    return gt_data, imu_data, time_gc


def coordinate_transform(gt_data: GroundTruthData) -> GroundTruthData:
    """坐标系转换：Y-UP 到 Z-UP"""
    print("正在进行坐标系转换：Y-UP -> Z-UP")

    # Y-UP 到 Z-UP 的旋转：绕X轴90度
    y_to_z_rotation = Rotation.from_rotvec([np.pi / 2, 0, 0])  # 90度

    # 变换旋转
    gt_data.rots = y_to_z_rotation * gt_data.rots

    # 变换位置（如果需要，根据重力方向调整）
    # 注意：位置变换取决于坐标系原点，这里假设原点不变

    print("坐标系转换完成")
    return gt_data


def plot_time_diff(imu_data: ImuData, gt_data: GroundTruthData, output_dir: Path):
    """绘制角速度对比图"""
    print("正在生成角速度对比图...")

    # 1. 定义计算角速度的内部辅助函数
    def get_angvels(t_us, rots, step=1):
        n = len(rots)
        step = max(int(step), 1)
        As = []
        Ts = []
        for i in range(0, n - step, step):
            # 计算相邻帧之间的旋转差异
            drot = rots[i].inv() * rots[i + step]
            angle = float(np.linalg.norm(drot.as_rotvec()))
            dt_s = (t_us[i + step] - t_us[i]) * 1e-6
            if dt_s > 0:
                ang_vel = angle / dt_s
                As.append(ang_vel)
                Ts.append(t_us[i])
        return np.array(As), np.array(Ts)

    # 2. 准备数据
    imu_poses = imu_data.to_poses()
    rate = min(imu_poses.rate, gt_data.rate)
    # 这里的 step 逻辑保持与原代码一致
    step = max(int(rate / 100), 1) 

    seq_imu, t_imu = get_angvels(imu_poses.t_us, imu_poses.rots, step=step)
    seq_gt, t_gt = get_angvels(gt_data.t_us, gt_data.rots, step=step)

    # 3. 创建绘图
    plt.figure(figsize=(12, 6))
    
    plt.plot(t_imu / 1e6, seq_imu, label='IMU Angular Velocity', alpha=0.7, linewidth=1)
    plt.plot(t_gt / 1e6, seq_gt, label='GT Angular Velocity', alpha=0.7, linewidth=1)
    
    # 4. 设置图表属性
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity Peaks Comparison (Time Alignment)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 5. 保存结果
    plt.tight_layout()
    png_path = output_dir / "Timediff.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"角速度对比图已保存：{png_path}")
    
def plot_coordinate_transform(before_gt: GroundTruthData, after_gt: GroundTruthData, output_dir: Path):
    """绘制坐标系转换效果图
    
    通过比较转换前后的重力向量来验证坐标系变换的正确性。
    """
    print("正在生成坐标系转换效果图...")

    # 调整画布大小：宽度16，高度6
    fig, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 计算重力向量在传感器坐标系下的投影
    # 将世界坐标系中的重力向量 [0, 0, -9.81] m/s² 变换到各时刻的传感器坐标系
    gravity_before = before_gt.rots.apply([0, 0, -9.81])
    gravity_after = after_gt.rots.apply([0, 0, -9.81])

    # 2. 绘制转换前的重力分量
    ax5.plot(before_gt.t_us / 1e6, gravity_before[:, 0], label='X', alpha=0.8)
    ax5.plot(before_gt.t_us / 1e6, gravity_before[:, 1], label='Y', alpha=0.8)
    ax5.plot(before_gt.t_us / 1e6, gravity_before[:, 2], label='Z', alpha=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Gravity (m/s²)')
    ax5.set_title('Gravity Vector BEFORE Transform (e.g. Y-UP)')
    ax5.legend(loc='upper right')
    ax5.grid(True, linestyle='--', alpha=0.6)

    # 3. 绘制转换后的重力分量
    ax6.plot(after_gt.t_us / 1e6, gravity_after[:, 0], label='X', alpha=0.8)
    ax6.plot(after_gt.t_us / 1e6, gravity_after[:, 1], label='Y', alpha=0.8)
    ax6.plot(after_gt.t_us / 1e6, gravity_after[:, 2], label='Z', alpha=0.8)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Gravity (m/s²)')
    ax6.set_title('Gravity Vector AFTER Transform (e.g. Z-UP)')
    ax6.legend(loc='upper right')
    ax6.grid(True, linestyle='--', alpha=0.6)

    # 4. 优化布局并保存
    plt.tight_layout()
    png_path = output_dir / "Transform.png"  # 修正了文件名拼写 Ttansfom -> Transform
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"坐标系转换效果图已保存：{png_path}")

def save_gt_csv(gt_data: GroundTruthData, output_dir: Path):
    """保存gt.csv文件"""
    csv_path = output_dir / "gt.csv"
    PosesDataSerializer(gt_data).save(csv_path)
    print(f"gt.csv文件已保存：{csv_path}")


def main():
    parser = argparse.ArgumentParser(description="预处理RTAB数据库文件为gt.csv")
    parser.add_argument("--db", required=True, help="RTAB数据库文件路径(.db)")
    parser.add_argument("--imu", required=True, help="IMU数据文件路径(imu.csv)")
    parser.add_argument("--output", help="输出目录路径（可选，默认与.db文件同目录）")

    args = parser.parse_args()

    db_path = Path(args.db)
    imu_path = Path(args.imu)
    output_dir = Path(args.output) if args.output else db_path.parent

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    rtab_data = load_rtab_data(db_path)
    imu_data = load_imu_data(imu_path)

    # 时间同步
    gt_data, synced_imu_data, time_offset = time_sync(rtab_data, imu_data)

    # 保存转换前的副本用于绘图
    before_transform_gt = GroundTruthData(gt_data.t_us.copy(), Rotation.from_quat(gt_data.rots.as_quat()), gt_data.ps.copy())

    # 坐标系转换
    gt_data = coordinate_transform(gt_data)

    # 生成图表
    plot_time_diff(synced_imu_data, before_transform_gt, output_dir)
    plot_coordinate_transform(before_transform_gt, gt_data, output_dir)

    # 保存结果
    save_gt_csv(gt_data, output_dir)

    print("预处理完成！")


if __name__ == "__main__":
    main()