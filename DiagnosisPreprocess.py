#!/usr/bin/env python3
"""
预处理诊断脚本 - 评估预处理后的数据质量

功能说明：
    1. 评估时间同步质量
       - 检查时间戳对齐程度
       - 计算时间同步的相关性
       - 检测时间异常
    
    2. 验证 IMU 坐标系转换
       - 检查重力加速度是否正确指向 Z 轴
       - 验证静止状态下的数据一致性
       - 计算坐标系转换的置信度
    
    3. 生成诊断报告
       - 总体数据质量评分
       - 详细的分析结果
       - 可视化诊断图表
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.transform import Rotation

from src.base.datatype import PosesData, ImuData, GroundTruthData


class DiagnosisReport:
    """诊断报告类"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report = {
            "time_sync_diagnosis": {},
            "coordinate_transform_diagnosis": {},
            "overall_quality_score": 0.0,
            "warnings": [],
            "recommendations": [],
        }
    
    def add_time_sync_result(self, key: str, value):
        """添加时间同步诊断结果"""
        self.report["time_sync_diagnosis"][key] = value
    
    def add_transform_result(self, key: str, value):
        """添加坐标系转换诊断结果"""
        self.report["coordinate_transform_diagnosis"][key] = value
    
    def add_warning(self, warning: str):
        """添加警告信息"""
        self.report["warnings"].append(warning)
    
    def add_recommendation(self, recommendation: str):
        """添加建议信息"""
        self.report["recommendations"].append(recommendation)
    
    def set_quality_score(self, score: float):
        """设置总体质量评分 (0-100)"""
        self.report["overall_quality_score"] = float(score)
    
    def save(self, filename: str = "diagnosis_report.json"):
        """保存诊断报告"""
        report_path = self.output_dir / filename
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        print(f"诊断报告已保存: {report_path}")
        return report_path
    
    def print_summary(self):
        """打印诊断摘要"""
        print("\n" + "="*80)
        print("预处理数据诊断报告")
        print("="*80)
        print(f"\n总体质量评分: {self.report['overall_quality_score']:.1f}/100\n")
        
        if self.report['warnings']:
            print("[!] 警告:")
            for i, warning in enumerate(self.report['warnings'], 1):
                print(f"  {i}. {warning}")
        
        print("\n[*] 时间同步诊断:")
        for key, value in self.report['time_sync_diagnosis'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n[*] 坐标系转换诊断:")
        for key, value in self.report['coordinate_transform_diagnosis'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        if self.report['recommendations']:
            print("\n[+] 建议:")
            for i, rec in enumerate(self.report['recommendations'], 1):
                print(f"  {i}. {rec}")
        print("\n" + "="*80)


class TimeSyncDiagnosis:
    """时间同步诊断"""
    
    @staticmethod
    def analyze_time_alignment(gt_poses: PosesData, imu_data: ImuData, 
                              window_size: int = 100) -> Dict:
        """
        分析时间对齐质量
        
        检查内容：
            1. 时间范围重叠情况
            2. 数据率一致性
            3. 时间戳异常检测
            4. IMU 运动与 GT 运动的相关性（关键指标）
        
        参数说明：
            gt_poses: GT 位姿数据
            imu_data: IMU 数据
            window_size: 滑动窗口大小
        
        返回值：
            包含对齐质量指标的字典
        """
        results = {}
        
        # 1. 检查时间范围重叠
        gt_t_start = gt_poses.t_us[0]
        gt_t_end = gt_poses.t_us[-1]
        imu_t_start = imu_data.t_us[0]
        imu_t_end = imu_data.t_us[-1]
        
        overlap_start = max(gt_t_start, imu_t_start)
        overlap_end = min(gt_t_end, imu_t_end)
        overlap_duration_s = (overlap_end - overlap_start) * 1e-6
        
        results["overlap_duration_s"] = overlap_duration_s
        results["gt_duration_s"] = (gt_t_end - gt_t_start) * 1e-6
        results["imu_duration_s"] = (imu_t_end - imu_t_start) * 1e-6
        
        # 2. 检查数据率一致性
        gt_rate = float(1e6 / np.mean(np.diff(gt_poses.t_us)))
        imu_rate = float(1e6 / np.mean(np.diff(imu_data.t_us)))
        
        results["gt_rate_hz"] = gt_rate
        results["imu_rate_hz"] = imu_rate
        results["rate_ratio"] = imu_rate / gt_rate
        
        # 3. 检查时间戳的平滑性（检测时间跳跃或丢包）
        gt_dt = np.diff(gt_poses.t_us) * 1e-6  # 秒
        imu_dt = np.diff(imu_data.t_us) * 1e-6  # 秒
        
        gt_dt_mean = np.mean(gt_dt)
        imu_dt_mean = np.mean(imu_dt)
        
        # 检测异常的时间间隔（> 3 倍标准差）
        gt_anomalies = np.where(np.abs(gt_dt - gt_dt_mean) > 3 * np.std(gt_dt))[0]
        imu_anomalies = np.where(np.abs(imu_dt - imu_dt_mean) > 3 * np.std(imu_dt))[0]
        
        results["gt_time_anomalies"] = len(gt_anomalies)
        results["imu_time_anomalies"] = len(imu_anomalies)
        
        # 4. Key: Check motion correspondence between IMU and GT
        # Principle: If time sync is correct, IMU angular velocity should correspond to GT rotation changes
        print("    > Computing motion correlation between IMU and GT...")
        
        # 计算 GT 的角速度（从旋转矩阵的变化率）
        gt_angvel = TimeSyncDiagnosis._compute_angular_velocity(
            gt_poses.rots, gt_poses.t_us
        )
        
        # 计算 IMU 的角速度（陀螺仪数据，需要积分验证）
        imu_angvel = np.linalg.norm(imu_data.gyro, axis=1)
        
        # 对齐时间戳，计算相关性
        # 重采样到相同的时间轴
        common_t_start = max(gt_poses.t_us[0], imu_data.t_us[0])
        common_t_end = min(gt_poses.t_us[-1], imu_data.t_us[-1])
        
        # 创建公共时间轴（使用较低的采样率以避免噪声）
        n_samples = min(len(gt_poses), len(imu_data)) // 2
        common_times = np.linspace(common_t_start, common_t_end, n_samples)
        
        # 在公共时间轴上重采样
        gt_angvel_resampled = np.interp(common_times, gt_poses.t_us, gt_angvel)
        imu_angvel_resampled = np.interp(common_times, imu_data.t_us, imu_angvel)
        
        # 标准化（使相关性计算不受幅度影响）
        gt_angvel_norm = (gt_angvel_resampled - np.mean(gt_angvel_resampled)) / (np.std(gt_angvel_resampled) + 1e-6)
        imu_angvel_norm = (imu_angvel_resampled - np.mean(imu_angvel_resampled)) / (np.std(imu_angvel_resampled) + 1e-6)
        
        # 计算相关系数
        correlation = float(np.corrcoef(gt_angvel_norm, imu_angvel_norm)[0, 1])
        results["motion_correlation"] = correlation
        
        # 如果相关性很低，说明时间同步可能有问题
        if not np.isnan(correlation):
            results["sync_quality"] = "Good" if correlation > 0.7 else ("Fair" if correlation > 0.5 else "Poor")
        else:
            results["sync_quality"] = "Unknown"
        
        return results
    
    @staticmethod
    def _compute_angular_velocity(rots, t_us, step: int = 1) -> np.ndarray:
        """
        从旋转序列计算角速度
        
        参数说明：
            rots: Rotation 对象序列
            t_us: 时间戳数组
            step: 计算步长
        
        返回值：
            角速度大小数组
        """
        n = len(rots)
        step = max(int(step), 1)
        
        angvels = []
        for i in range(n - step):
            drot = rots[i].inv() * rots[i + step]
            angle = float(np.linalg.norm(drot.as_rotvec()))
            dt_s = (t_us[i + step] - t_us[i]) * 1e-6
            if dt_s > 0:
                angvel = angle / dt_s
                angvels.append(angvel)
            else:
                angvels.append(0.0)
        
        # 补齐长度
        angvels.append(angvels[-1] if angvels else 0.0)
        return np.array(angvels)
    
    @staticmethod
    def plot_time_alignment(gt_poses: PosesData, imu_data: ImuData, 
                           output_path: Path):
        """Plot time alignment diagnosis charts"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Calculate angular velocities for comparison
        gt_angvel = TimeSyncDiagnosis._compute_angular_velocity(gt_poses.rots, gt_poses.t_us)
        imu_angvel = np.linalg.norm(imu_data.gyro, axis=1)
        
        # 1. Angular velocity comparison (key: should be similar)
        ax = axes[0, 0]
        t_gt_s = (gt_poses.t_us - gt_poses.t_us[0]) * 1e-6
        t_imu_s = (imu_data.t_us - imu_data.t_us[0]) * 1e-6
        
        ax.plot(t_gt_s, gt_angvel, label='GT Angular Velocity', alpha=0.7, linewidth=0.8)
        ax.plot(t_imu_s, imu_angvel, label='IMU Angular Velocity', alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('IMU vs GT Angular Velocity\n(Should match well if time sync is correct)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 加速度大小对比
        ax = axes[0, 1]
        gt_acce_mag = np.linalg.norm(imu_data.acce, axis=1)  # IMU 加速度
        ax.plot(t_imu_s, gt_acce_mag, label='IMU Acceleration', alpha=0.7, linewidth=0.8)
        ax.axhline(y=np.mean(gt_acce_mag), color='r', linestyle='--', 
                  label=f'Mean: {np.mean(gt_acce_mag):.2f} m/s²', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_title('IMU Acceleration Magnitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 时间间隔分析
        ax = axes[1, 0]
        gt_dt = np.diff(gt_poses.t_us) * 1e-3  # 毫秒
        imu_dt = np.diff(imu_data.t_us) * 1e-3  # 毫秒
        ax.hist(gt_dt, bins=50, alpha=0.6, label='GT', density=True)
        ax.hist(imu_dt, bins=50, alpha=0.6, label='IMU', density=True)
        ax.set_xlabel('Time Interval (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Time Interval Distribution\n(Should be identical)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 相关性热力图
        ax = axes[1, 1]
        # 计算滑动窗口相关性
        window = 500
        correlations = []
        window_centers = []
        
        for i in range(0, len(imu_data) - window, window // 2):
            imu_seg = imu_angvel[i:i+window]
            
            # 在 GT 上找对应的时间窗口
            t_imu_seg_start = imu_data.t_us[i]
            t_imu_seg_end = imu_data.t_us[min(i+window, len(imu_data)-1)]
            
            gt_mask = (gt_poses.t_us >= t_imu_seg_start) & (gt_poses.t_us <= t_imu_seg_end)
            if np.sum(gt_mask) > 10:
                gt_seg = gt_angvel[gt_mask]
                
                # 重采样到相同长度
                if len(gt_seg) > 1 and len(imu_seg) > 1:
                    gt_seg_resampled = np.interp(
                        np.linspace(0, 1, len(imu_seg)),
                        np.linspace(0, 1, len(gt_seg)),
                        gt_seg
                    )
                    
                    # 标准化
                    gt_norm = (gt_seg_resampled - np.mean(gt_seg_resampled)) / (np.std(gt_seg_resampled) + 1e-6)
                    imu_norm = (imu_seg - np.mean(imu_seg)) / (np.std(imu_seg) + 1e-6)
                    
                    corr = float(np.corrcoef(gt_norm, imu_norm)[0, 1])
                    if not np.isnan(corr):
                        correlations.append(corr)
                        window_centers.append(t_imu_seg_start * 1e-6)
        
        if correlations:
            window_centers = np.array(window_centers) - window_centers[0]
            ax.plot(window_centers, correlations, 'o-', linewidth=2, markersize=4)
            ax.axhline(y=0.7, color='g', linestyle='--', label='Good (>0.7)', alpha=0.5)
            ax.axhline(y=0.5, color='orange', linestyle='--', label='Fair (>0.5)', alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Correlation Coefficient')
            ax.set_title('Motion Correlation Over Time\n(Angular velocity correlation)')
            ax.set_ylim([-1, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"时间对齐图已保存: {output_path}")


class CoordinateTransformDiagnosis:
    """坐标系转换诊断"""
    
    @staticmethod
    def analyze_gravity_alignment(imu_data: ImuData, gt_poses: PosesData,
                                 window_size: int = 200) -> Dict:
        """
        分析重力加速度的对齐情况
        
        原理：
            - 先将 IMU 数据从 body frame 转换到 global frame（使用 GT 的旋转）
            - 在静止或缓慢运动时，重力加速度应该只在 Z 轴上（全局坐标系）
            - 检查加速度的 Z 分量是否接近 9.8 m/s²
            - 检查 X, Y 分量是否接近 0
        
        参数说明：
            imu_data: IMU 数据（body frame）
            gt_poses: GT 位姿数据（包含 body → global 的旋转）
            window_size: 滑动窗口大小
        
        返回值：
            包含重力对齐质量指标的字典
        """
        results = {}
        
        # Important: Transform IMU from body frame to global frame first
        print("    > Transforming coordinates (Body Frame -> Global Frame)...")
        imu_global = imu_data.transform(rots=gt_poses.rots)
        
        acce = imu_global.acce  # 转换后的加速度（全局坐标系）
        t_us = imu_global.t_us
        
        # 计算加速度大小
        acce_mag = np.linalg.norm(acce, axis=1)
        results["acce_magnitude_mean"] = float(np.mean(acce_mag))
        results["acce_magnitude_std"] = float(np.std(acce_mag))
        
        # 计算重力分量（假设转换后数据在全局坐标系）
        # Z 轴应该是上方向（重力方向）
        gz = acce[:, 2]  # Z 分量
        gxy = np.linalg.norm(acce[:, :2], axis=1)  # XY 平面大小
        
        results["gz_mean"] = float(np.mean(gz))
        results["gz_std"] = float(np.std(gz))
        results["gxy_mean"] = float(np.mean(gxy))
        results["gxy_std"] = float(np.std(gxy))
        
        # 理想的重力值应该是 9.8 m/s²（或因为世界坐标系可能是 Z-UP）
        # 检查重力是否主要集中在 Z 轴
        gravity_alignment_ratio = np.mean(np.abs(gz)) / (np.mean(acce_mag) + 1e-6)
        results["gravity_alignment_ratio"] = float(gravity_alignment_ratio)
        
        # 检查 Z 轴对齐度（与理想 9.8 的差异）
        ideal_gravity = 9.81
        gz_deviation = np.abs(np.mean(gz)) - ideal_gravity
        results["gravity_deviation_from_ideal"] = float(gz_deviation)
        
        # 分析动态和静态状态
        # 使用加速度幅度的变化来检测运动状态
        acce_mag_filtered = signal.savgol_filter(acce_mag, window_length=min(101, len(acce_mag) if len(acce_mag) % 2 == 1 else len(acce_mag)-1), polyorder=3)
        acce_mag_derivative = np.abs(np.gradient(acce_mag_filtered))
        
        # 找到静态段（导数小）
        static_threshold = np.percentile(acce_mag_derivative, 25)
        static_mask = acce_mag_derivative < static_threshold
        
        if np.sum(static_mask) > 10:
            static_acce = acce[static_mask]
            results["static_gz_mean"] = float(np.mean(static_acce[:, 2]))
            results["static_gz_std"] = float(np.std(static_acce[:, 2]))
            results["static_gxy_mean"] = float(np.mean(np.linalg.norm(static_acce[:, :2], axis=1)))
        
        # 计算转换质量评分
        # 评分标准：
        # 1. gravity_alignment_ratio 应该 > 0.9（重力集中在 Z 轴）
        # 2. gz 应该接近 9.8（误差 < 1.0）
        # 3. gxy 应该很小（< 1.0）
        
        score = 0.0
        if gravity_alignment_ratio > 0.95:
            score += 40
        elif gravity_alignment_ratio > 0.85:
            score += 30
        elif gravity_alignment_ratio > 0.70:
            score += 15
        
        if abs(gz_deviation) < 0.5:
            score += 30
        elif abs(gz_deviation) < 1.0:
            score += 20
        elif abs(gz_deviation) < 2.0:
            score += 10
        
        if results["gxy_mean"] < 0.5:
            score += 30
        elif results["gxy_mean"] < 1.0:
            score += 15
        
        results["coordinate_transform_score"] = score
        
        return results
    
    @staticmethod
    def plot_gravity_analysis(imu_data: ImuData, gt_poses: PosesData, output_path: Path):
        """
        绘制重力加速度分析图
        
        参数说明：
            imu_data: IMU 数据（body frame，需要转换）
            gt_poses: GT 位姿数据（用于坐标系转换）
            output_path: 输出路径
        """
        # Important: Transform to global frame first
        print("    > Transforming IMU data to global frame for visualization...")
        imu_global = imu_data.transform(rots=gt_poses.rots)
        acce = imu_global.acce  # 使用转换后的加速度
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Acceleration components time series
        ax = axes[0, 0]
        ax.plot(acce[:, 0], label='X (Global)', alpha=0.7, linewidth=0.8, color='red')
        ax.plot(acce[:, 1], label='Y (Global)', alpha=0.7, linewidth=0.8, color='green')
        ax.plot(acce[:, 2], label='Z (Global)', alpha=0.7, linewidth=0.8, color='blue')
        ax.axhline(y=9.8, color='purple', linestyle='--', label='Expected Gravity (9.8 m/s²)', alpha=0.5, linewidth=2)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_title('Acceleration Components Over Time (Global Frame)\nZ-axis should be ~9.8 (gravity), X/Y should be ~0')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 加速度大小分布
        ax = axes[0, 1]
        acce_mag = np.linalg.norm(acce, axis=1)
        ax.hist(acce_mag, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(acce_mag), color='r', linestyle='--', label=f'Mean: {np.mean(acce_mag):.2f}')
        ax.set_xlabel('Acceleration Magnitude (m/s²)')
        ax.set_ylabel('Frequency')
        ax.set_title('Acceleration Magnitude Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Gravity component analysis
        ax = axes[1, 0]
        gz = acce[:, 2]
        gxy = np.linalg.norm(acce[:, :2], axis=1)
        ax.scatter(gxy, gz, alpha=0.3, s=10)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Ideal: X/Y = 0')
        ax.axhline(y=9.8, color='g', linestyle='--', alpha=0.5, label='Ideal: Z = 9.8')
        # Mark ideal point
        ax.scatter([0], [9.8], color='purple', s=200, marker='*', 
                  label='Ideal Point (0, 9.8)', zorder=5, edgecolors='black', linewidths=2)
        ax.set_xlabel('Horizontal Acceleration (m/s²)')
        ax.set_ylabel('Vertical Acceleration (m/s²)')
        ax.set_title('Gravity Alignment\nPoints should cluster near the purple star')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 加速度主要方向分析
        ax = axes[1, 1]
        # 计算每个轴的方差占比
        var_per_axis = np.var(acce, axis=0)
        total_var = np.sum(var_per_axis)
        var_ratio = var_per_axis / total_var * 100
        
        axes_labels = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue']
        ax.bar(axes_labels, var_ratio, color=colors, alpha=0.7)
        ax.set_ylabel('Variance Ratio (%)')
        ax.set_title('Acceleration Variance Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"重力分析图已保存: {output_path}")


def diagnose_preprocessed_data(data_dir: Path, output_dir: Path = None) -> DiagnosisReport:
    """
    诊断预处理后的数据
    
    参数说明：
        data_dir: 包含 gt.csv 和 imu*.csv 的目录
        output_dir: 输出目录（默认为 data_dir）
    
    返回值：
        DiagnosisReport: 诊断报告对象
    """
    if output_dir is None:
        output_dir = data_dir
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # 查找 gt.csv 和 imu 文件
    gt_csv = None
    imu_csv = None
    
    for f in data_dir.glob("*.csv"):
        if f.name == "gt.csv":
            gt_csv = f
        elif "imu" in f.name.lower():
            imu_csv = f
    
    if gt_csv is None:
        raise FileNotFoundError(f"gt.csv not found in {data_dir}")
    if imu_csv is None:
        raise FileNotFoundError(f"imu*.csv not found in {data_dir}")
    
    print(f"正在诊断预处理数据...")
    print(f"  GT 数据: {gt_csv}")
    print(f"  IMU 数据: {imu_csv}\n")
    
    # Load data
    print("加载数据...")
    # GT data loaded via GroundTruthData
    gt_data_raw = GroundTruthData.from_csv(gt_csv)
    gt_poses = gt_data_raw  # GroundTruthData inherits from PosesData
    imu_data = ImuData.from_csv(imu_csv)
    print(f"  GT 数据点数: {len(gt_poses)}")
    print(f"  IMU 数据点数: {len(imu_data)}\n")
    
    # 创建诊断报告
    report = DiagnosisReport(output_dir)
    
    # 1. Time synchronization diagnosis
    print("="*80)
    print("1. 时间同步诊断")
    print("   验证 PreprocessDBv5 两步法时间同步的质量")
    print("   （粗对齐 + 精对齐）")
    print("   方法: 检查 IMU 与 GT 的运动相关性")
    print("="*80)
    
    time_sync_results = TimeSyncDiagnosis.analyze_time_alignment(gt_poses, imu_data)
    
    for key, value in time_sync_results.items():
        report.add_time_sync_result(key, value)
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Check time sync quality
    motion_corr = time_sync_results.get("motion_correlation", 0)
    sync_quality = time_sync_results.get("sync_quality", "Unknown")
    
    if sync_quality == "Poor" or motion_corr < 0.5:
        report.add_warning(f"时间同步质量较差: 运动相关性仅 {motion_corr:.4f}（应大于 0.7）")
    elif sync_quality == "Fair":
        report.add_warning(f"时间同步质量一般: 运动相关性 {motion_corr:.4f}（建议 > 0.7）")
    
    # 绘制时间对齐诊断图
    TimeSyncDiagnosis.plot_time_alignment(
        gt_poses, imu_data, 
        output_dir / "diagnosis_time_alignment.png"
    )
    
    # 2. Coordinate transformation diagnosis
    print("\n" + "="*80)
    print("2. 坐标系转换诊断")
    print("   验证 IMU 载体坐标系 -> 全局坐标系 转换")
    print("   方法: 检查重力对齐（Z 轴应为 ~9.8 m/s²）")
    print("="*80)
    
    transform_results = CoordinateTransformDiagnosis.analyze_gravity_alignment(
        imu_data, gt_poses
    )
    
    for key, value in transform_results.items():
        report.add_transform_result(key, value)
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Check coordinate transformation quality
    if transform_results["gravity_alignment_ratio"] < 0.85:
        report.add_warning(f"重力对齐比率偏低: {transform_results['gravity_alignment_ratio']:.4f} < 0.85")
    
    if abs(transform_results["gravity_deviation_from_ideal"]) > 1.0:
        report.add_warning(f"重力偏差较大: {transform_results['gravity_deviation_from_ideal']:.4f} m/s²")
    
    if transform_results["gxy_mean"] > 1.0:
        report.add_warning(f"水平重力分量较大: {transform_results['gxy_mean']:.4f} m/s²")
    
    # 绘制重力加速度分析图
    CoordinateTransformDiagnosis.plot_gravity_analysis(
        imu_data, gt_poses,  # 传入 gt_poses 用于坐标系转换
        output_dir / "diagnosis_gravity_analysis.png"
    )
    
    # 3. Overall quality score
    print("\n" + "="*80)
    print("3. 总体质量评分")
    print("="*80)
    
    time_sync_score = 100.0
    
    # 时间同步得分（基于运动相关性）
    motion_corr = time_sync_results.get("motion_correlation", 0)
    if not np.isnan(motion_corr):
        # 相关性 > 0.7 为优秀，> 0.5 为良好，> 0.3 为一般，否则为差
        if motion_corr > 0.7:
            time_sync_score = 95
        elif motion_corr > 0.5:
            time_sync_score = 75
        elif motion_corr > 0.3:
            time_sync_score = 50
        else:
            time_sync_score = 20
    
    # 时间异常得分
    anomaly_count = time_sync_results["gt_time_anomalies"] + time_sync_results["imu_time_anomalies"]
    if anomaly_count > 0:
        time_sync_score -= min(anomaly_count * 5, 25)
    
    transform_score = transform_results.get("coordinate_transform_score", 0)
    
    overall_score = (time_sync_score * 0.4 + transform_score * 0.6)
    overall_score = max(0.0, min(100.0, overall_score))
    
    report.set_quality_score(overall_score)
    
    print(f"\n  时间同步评分: {time_sync_score:.1f}/100 (运动相关性: {motion_corr:.4f})")
    print(f"  坐标系转换评分: {transform_score:.1f}/100")
    print(f"  总体质量评分: {overall_score:.1f}/100")
    
    # Generate recommendations
    if overall_score >= 85:
        report.add_recommendation("数据质量良好，适合用于模型训练。")
    elif overall_score >= 70:
        report.add_recommendation("数据质量可接受。请查看警告了解详情。")
    else:
        report.add_recommendation("数据质量较差。建议重新预处理。")
    
    if transform_results["gravity_alignment_ratio"] < 0.99:
        report.add_recommendation("检查 IMU 坐标系转换旋转矩阵（重力对齐度偏低）。")
    
    if motion_corr < 0.7:
        report.add_recommendation(f"改进时间同步（运动相关性 {motion_corr:.4f}，建议 > 0.7）。考虑调整 PreprocessDBv5.py 中的时间范围。")
    
    # 保存诊断报告
    report.save("diagnosis_report.json")
    
    # 打印摘要
    report.print_summary()
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="预处理数据诊断工具",
        epilog="""
        诊断项目：
        1. 时间同步诊断
           - 检查 GT 和 IMU 的时间对齐情况
           - 分析数据率一致性
           - 检测时间戳异常
        
        2. 坐标系转换诊断
           - 验证重力加速度是否正确指向 Z 轴
           - 检查 IMU 转换后的数据一致性
           - 分析加速度分布
        
        3. 生成诊断报告和可视化图表
        """
    )
    parser.add_argument(
        "-d", "--data-dir", 
        type=str, 
        required=True,
        help="包含 gt.csv 和 imu*.csv 的数据目录"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认为数据目录）"
    )
    
    args = parser.parse_args()
    
    try:
        report = diagnose_preprocessed_data(
            Path(args.data_dir),
            Path(args.output_dir) if args.output_dir else None
        )
    except Exception as e:
        print(f"诊断失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
