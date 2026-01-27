#!/usr/bin/env python3
"""
预处理脚本：将RTAB数据库(.db)文件转换为gt.csv文件

功能说明:
    该脚本用于从原始RTAB数据库文件中生成经过预处理的gt.csv真值数据。主要功能包括:

    1. 数据加载与同步:
       - 从.db文件加载RTAB真值数据
       - 从imu.csv文件加载IMU数据
       - 通过时间匹配算法校准真值数据与IMU数据的时间戳
       - 生成时间对齐效果图(Timediff.png)

    2. 坐标系转换:
       - 将RTAB的Y-UP坐标系转换为Z-UP坐标系
       - 生成坐标系转换效果图(Transform.png)，比较前后的初始5s三轴加速度

    3. 数据保存:
       - 保存时间同步和坐标系转换后的gt.csv文件

使用示例:
    uv run PreprocessDB.py --db <db_file> --imu <imu_file>

输入数据格式:
    - db_file: RTAB数据库文件(.db)
    - imu_file: IMU数据文件(imu.csv)，包含时间戳和IMU测量信息

输出结果:
    output/
    ├── gt.csv          # 处理后的真值数据
    ├── Timediff.png    # 时间对齐效果图
    └── Transform.png   # 坐标系转换效果图
"""

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation

from base.calibration.time import match21
from base.datatype import GroundTruthData, ImuData, PosesData
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


def time_sync(rtab_data: RTABData, imu_data: ImuData, time_range=(0, 50)) -> tuple[GroundTruthData, int]:
    """时间同步RTAB和IMU数据：仅平移时间戳，不插值"""
    print(f"正在进行时间同步 (参考范围: {time_range}s)...")
    
    # 使用原始节点构造
    gt_data = GroundTruthData(rtab_data.node_t_us, rtab_data.node_rots, rtab_data.node_ps)

    # 1. 使用 match21 算出偏移量
    time_gc = match21(imu_data.to_poses(), gt_data, time_range=time_range)
    
    # 2. 仅平移原始节点的时间戳
    gt_data.t_us += time_gc
    gt_data.reset_start()

    print(f"计算得出最佳时间偏移量：{time_gc} 微秒")
    return gt_data, int(time_gc)


def coordinate_transform(gt_data: GroundTruthData) -> GroundTruthData:
    """坐标系转换：Y-UP 到 Z-UP"""
    print("正在进行坐标系转换：Y-UP -> Z-UP")

    # Y-UP 到 Z-UP 的旋转：绕X轴90度
    y_to_z_rotation = Rotation.from_rotvec([np.pi / 2, 0, 0])  # 90度

    # 变换旋转 (对原始节点进行旋转)
    gt_data.rots = y_to_z_rotation * gt_data.rots

    # 变换位置（如果需要，根据重力方向调整）
    # 注意：位置变换取决于坐标系原点，这里假设原点不变

    print("坐标系转换完成")
    return gt_data


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
    """绘制坐标系转换效果图 - 仅展示初始静止段验证重力方向"""
    print("正在生成坐标系转换验证图 (初始静止段)...")

    # 仅截取前 5 秒的数据进行验证（静止段最能体现重力轴向）
    # get_time_range 内部会处理相对于起始时间的偏移
    show_range = (0, 5.0) 
    view_before = before_gt.get_time_range(show_range)
    view_after = after_gt.get_time_range(show_range)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    g_vec = [0, 0, -9.81] # 世界坐标系下的重力

    for ax, data, title in zip([ax1, ax2], [view_before, view_after], ['BEFORE (Y-UP?)', 'AFTER (Z-UP?)']):
        # 计算传感器坐标系下的重力分量
        gravity = data.rots.apply(g_vec)
        
        # 时间轴从 0 开始
        t_rel = (data.t_us - data.t_us[0]) / 1e6 if len(data.t_us) > 0 else []
        
        ax.plot(t_rel, gravity[:, 0], label='Gravity-X', alpha=0.8)
        ax.plot(t_rel, gravity[:, 1], label='Gravity-Y', alpha=0.8)
        ax.plot(t_rel, gravity[:, 2], label='Gravity-Z', alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('m/s²')
        ax.set_ylim([-12, 12]) # 固定量程方便对比
        ax.set_title(f'Static Gravity Vector {title}')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    png_path = output_dir / "Transform.png"
    fig.savefig(png_path, dpi=300)
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
    parser.add_argument("-t", "--time-range", type=float, nargs=2, default=[0, 50])
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
    raw_gt, t_offset_us = time_sync(rtab_data, imu_data, time_range=tuple(args.time_range))

    # 保存转换前的副本用于绘图 (原始同步节点)
    before_transform_gt = GroundTruthData(raw_gt.t_us.copy(), Rotation.from_quat(raw_gt.rots.as_quat()), raw_gt.ps.copy())

    # 坐标系转换 (对原始节点进行旋转变换)
    raw_gt = coordinate_transform(raw_gt)

    # 生成图表 
    plot_time_diff(imu_data, before_transform_gt, output_dir)
    plot_coordinate_transform(before_transform_gt, raw_gt, output_dir)

    # 保存结果 - 保存的是对齐了时间、转换了坐标系的【原始节点】，无插值数据
    save_gt_csv(raw_gt, output_dir)

    print("预处理完成！")


if __name__ == "__main__":
    main()