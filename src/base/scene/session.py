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
        navio.close()

        self.extra_pose = None
        if self.session.fusion_csv_extra and self.session.fusion_csv_extra.exists():
            self.extra_pose = FusionData.from_csv(self.session.fusion_csv_extra)

    def align_time(self, rate: int = 200) -> None:
        """通过角速度互相关计算时间偏移，对齐所有轨迹到共同时间窗口"""

        self.gt_pose.t_us = (
            self.gt_pose.t_us - self.gt_pose.t_us[0] + self.fusion_pose.t_us[0]
        )

        # 计算 GT 与 fusion 之间的时间偏移，移动 GT
        check_path = self.session.fusion_csv.parent / "time_align_check.png"
        t_offset_us = match21(
            self.gt_pose, self.fusion_pose, check_plot_path=check_path
        )
        self.gt_pose.t_us -= t_offset_us

        # 获取共同时间窗口并插值
        rel_ts = [self.gt_pose.t_us, self.fusion_pose.t_us, self.cam_pose.t_us]
        if self.extra_pose is not None:
            rel_ts.append(self.extra_pose.t_us)

        t_new_us = get_time_series(rel_ts, rate=rate)
        self.gt_pose = self.gt_pose.interpolate(t_new_us)
        self.fusion_pose = self.fusion_pose.interpolate(t_new_us)
        self.cam_pose = self.cam_pose.interpolate(t_new_us)
        if self.extra_pose is not None:
            self.extra_pose = self.extra_pose.interpolate(t_new_us)

    def align_space(self) -> Pose:
        """计算 GT 到 fusion 的变换（yaw + XY），直接应用到 gt_pose"""
        rel_rots = []
        for g_pose, f_pose in zip(
            self.gt_pose,
            self.fusion_pose,  # pyright: ignore[reportArgumentType]
        ):
            rel_rots.append(g_pose.between(f_pose).rot.as_matrix())

        R_sum = np.sum(rel_rots, axis=0)
        U, _, Vt = np.linalg.svd(R_sum)
        R_mean = U @ Vt
        if np.linalg.det(R_mean) < 0:
            U[:, -1] *= -1
            R_mean = U @ Vt

        yaw = Rotation.from_matrix(R_mean).as_euler("ZXY")[0]
        rot_yaw = Rotation.from_rotvec([0, 0, yaw])

        # 旋转 GT 后再算平移
        # gt_rotated = PosesData(
        #     self.gt_pose.t_us.copy(),
        #     rot_yaw * self.gt_pose.rots,
        #     rot_yaw.apply(self.gt_pose.ps),
        # )
        # rel_ps = []
        # for g_pose, f_pose in zip(gt_rotated, self.fusion_pose):
        #     rel_ps.append(f_pose.p - g_pose.p)

        # t_mean = np.mean(rel_ps, axis=0)
        # t_xy = np.array([t_mean[0], t_mean[1], 0])

        # print(
        #     f"> GT -> Fusion: Yaw={np.degrees(yaw):.1f}°, XY=[{t_mean[0]:.3f} {t_mean[1]:.3f}] m"
        # )

        space_match = Pose.from_rotation(rot_yaw)
        self.gt_pose.transform_global(space_match)
        return space_match
