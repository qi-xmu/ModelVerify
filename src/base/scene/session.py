"""Session 数据类"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from base.calibration.time import match21
from base.datatype import FusionData, GroundTruthData, Pose, PosesData
from base.interpolate import get_time_series
from base.rtab import RTABData
from base.types.navio_db import NaVIODB


@dataclass
class Session:
    """单次采集场次，持有真值、原始数据和融合结果的路径引用"""

    session_id: int
    device: str
    duration_s: float
    gt_db: Path
    navio_db: Path
    fusion_csv: Path
    status: str
    fusion_csv_extra: Path | None = None
    gt_yaw: float = 0.0
    extra_yaw: float = 0.0
    has_yaw: bool = False

    @property
    def is_ok(self) -> bool:
        return self.status == "OK"

    @property
    def key(self) -> tuple[int, str]:
        return (self.session_id, self.device)

    @property
    def gt_stem(self) -> str:
        return self.gt_db.stem

    @property
    def label(self) -> str:
        """显示标签：fusion_csv 的父文件夹名"""
        return self.fusion_csv.parent.name

    @property
    def parent_dir(self) -> Path:
        return self.fusion_csv.parent


def _yaw_from_velocity(
    a: PosesData, b: PosesData, duration_s: float = 40, skip_s: float = 5
) -> float:
    """根据前 duration_s 秒（跳过前 skip_s 秒）的速度方向计算 a 到 b 的 yaw 角（度）"""
    t0 = a.t_us[0] + skip_s * 1e6
    t1 = t0 + duration_s * 1e6
    mask_a = (a.t_us >= t0) & (a.t_us < t1)
    mask_b = (b.t_us >= t0) & (b.t_us < t1)
    vel_a = a.ps[mask_a][-1] - a.ps[mask_a][0]
    vel_b = b.ps[mask_b][-1] - b.ps[mask_b][0]
    yaw_rad = np.arctan2(vel_b[1], vel_b[0]) - np.arctan2(vel_a[1], vel_a[0])
    return float(np.degrees(yaw_rad))


def _yaw_from_rotation_avg(a: PosesData, b: PosesData, duration_s: float = 20) -> float:
    """通过前 duration_s 秒旋转矩阵 SVD 平均计算 a 到 b 的 yaw 角（度）"""
    t0 = a.t_us[0] + 5e6
    start_idx = int(np.searchsorted(a.t_us, t0))
    n = int(np.searchsorted(a.t_us, t0 + duration_s * 1e6)) - start_idx
    n = min(n, len(a) - start_idx, len(b) - start_idx)
    rel_rots = []
    for i in range(start_idx, start_idx + n):
        rel_rots.append(a.get_pose(i).between(b.get_pose(i)).rot.as_matrix())
    R_sum = np.sum(rel_rots, axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    yaw = Rotation.from_matrix(R_mean).as_euler("ZXY")[0]
    return float(np.degrees(yaw))


def _yaw_from_both(a: PosesData, b: PosesData) -> float:
    """综合速度方向和旋转平均，取绝对值较小者"""
    yaw_vel = _yaw_from_velocity(a, b)
    return yaw_vel
    yaw_rot = _yaw_from_rotation_avg(a, b)
    if abs(yaw_vel) <= abs(yaw_rot):
        return yaw_vel
    return yaw_rot


class SessionObj:
    """解析 Session，加载 GT / Fusion / Cam 轨迹数据"""

    gt_pose: PosesData
    fusion_pose: PosesData
    cam_pose: PosesData
    extra_pose: PosesData | None

    space_match: Pose | None = None

    def __init__(self, session: Session):
        self.session = session
        # 真值
        if session.gt_db.suffix == ".db":
            rtab = RTABData(session.gt_db)
            rtab.transform_local()
            rtab.transform_global()
            self.gt_pose = GroundTruthData.from_rtab(rtab)
            rtab.conn.close()
        else:
            self.gt_pose = GroundTruthData.from_csv(session.gt_db)

        self.fusion_pose = FusionData.from_csv(self.session.fusion_csv)
        # 基于相机的传感器姿态
        navio = NaVIODB(self.session.navio_db)
        self.cam_pose = navio.get_pose_data()
        self.imu_pose = navio.get_imu_data().to_poses()
        navio.close()

        self.extra_pose = None
        if self.session.fusion_csv_extra and self.session.fusion_csv_extra.exists():
            self.extra_pose = FusionData.from_csv(self.session.fusion_csv_extra)

    def align_time(self, rate: int = 200) -> None:
        """通过角速度互相关计算时间偏移，对齐所有轨迹到共同时间窗口"""
        self.gt_pose.t_us = (
            self.gt_pose.t_us - self.gt_pose.t_us[0] + self.imu_pose.t_us[0]
        )

        # 计算 GT 与 fusion 之间的时间偏移，移动 GT
        check_path = self.session.fusion_csv.parent / "time_align_check.png"
        t_offset_us = match21(self.gt_pose, self.imu_pose, check_plot_path=check_path)
        self.gt_pose.t_us -= t_offset_us

        # 获取共同时间窗口并插值
        rel_ts = [self.gt_pose.t_us, self.fusion_pose.t_us, self.cam_pose.t_us]
        if self.extra_pose is not None:
            rel_ts.append(self.extra_pose.t_us)

        t_new_us = get_time_series(rel_ts, rate=rate)
        # 屏蔽前1s的数据
        # t_new_us = t_new_us[rate * 10 :]

        self.gt_pose = self.gt_pose.interpolate(t_new_us)
        self.fusion_pose = self.fusion_pose.interpolate(t_new_us)
        self.cam_pose = self.cam_pose.interpolate(t_new_us)
        if self.extra_pose is not None:
            self.extra_pose = self.extra_pose.interpolate(t_new_us)

    def align_space(self) -> Pose:
        """计算 GT/Network 到 fusion 的 yaw 变换，直接应用"""
        self.gt_pose.reset_start()
        self.fusion_pose.reset_start()
        self.cam_pose.reset_start()
        if self.extra_pose is not None:
            self.extra_pose.reset_start()

        if not self.session.has_yaw:
            self.session.gt_yaw = _yaw_from_both(self.gt_pose, self.fusion_pose)
        rot_yaw = Rotation.from_rotvec([0, 0, np.radians(self.session.gt_yaw)])
        print(f"> GT -> Fusion Yaw: {self.session.gt_yaw:.1f}°")

        space_match = Pose.from_rotation(rot_yaw)
        self.gt_pose.transform_global(space_match)

        if self.extra_pose is not None:
            if not self.session.has_yaw:
                self.session.extra_yaw = _yaw_from_both(
                    self.extra_pose, self.fusion_pose
                )
            net_rot = Rotation.from_rotvec([0, 0, np.radians(self.session.extra_yaw)])
            self.extra_pose.transform_global(Pose.from_rotation(net_rot))
            print(f"> Network -> Fusion Yaw: {self.session.extra_yaw:.1f}°")

        return space_match
