import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from base.interpolate import get_time_series, interpolate_vector3, slerp_rotation
from base.rtab import RTABData


@dataclass
class Pose:
    rot: Rotation
    p: NDArray
    t_us: int = 0

    def copy(self):
        return Pose(self.rot, self.p.copy(), self.t_us)

    @staticmethod
    def identity():
        return Pose(Rotation.identity(), np.zeros(3))

    @staticmethod
    def from_rotation(rot: Rotation):
        return Pose(rot, np.zeros(3))

    @staticmethod
    def from_transform(trans: NDArray):
        return Pose(Rotation.identity(), trans)

    def compose(self, other: Self):
        return Pose(self.rot * other.rot, self.rot.apply(other.p) + self.p)

    def compose_self(self, other: Self):
        self.p = self.rot.apply(other.p) + self.p
        self.rot = self.rot * other.rot

    def compose_trans_self(self, trans: NDArray):
        self.p += self.rot.apply(trans)

    def inverse(self):
        return Pose(self.rot.inv(), -self.rot.inv().apply(self.p))

    def get_yaw_rot(self):
        yaw = self.rot.as_euler("ZXY")[0]
        return Pose(Rotation.from_rotvec([0, 0, yaw]), np.zeros(3))


Frame: TypeAlias = Literal["local", "global"]


@dataclass
class PosesData:
    t_us: NDArray
    rots: Rotation
    ps: NDArray

    def __len__(self):
        return len(self.t_us)

    def __getitem__(self, index: int | slice):
        return PosesData(self.t_us[index], self.rots[index], self.ps[index])

    def __iter__(self):
        for i in range(len(self.t_us)):
            yield Pose(self.rots[i], self.ps[i], self.t_us[i])

    @staticmethod
    def from_list(pose_list: list[Pose]):
        rotations = []
        translates = []
        t_us = []
        for pose in pose_list:
            rotations.append(pose.rot.as_matrix())
            translates.append(pose.p)
            t_us.append(pose.t_us)

        t_us = np.array(t_us)
        rots = Rotation.from_matrix(rotations)
        ps = np.array(translates)
        return PosesData(t_us, rots, ps)

    def get_pose(self, index: int = 0):
        return Pose(self.rots[index], self.ps[index], self.t_us[index])

    def get_time_range(self, time_range: tuple[float | None, float | None]):
        ts, te = self.t_us[0], self.t_us[-1]
        if time_range[0] is not None:
            ts = time_range[0] * 1e6 + self.t_us[0]
        if time_range[1] is not None:
            te = time_range[1] * 1e6 + self.t_us[0]
        return self.get_time_region((ts, te))

    def get_time_region(self, t_region: tuple[int, int]):
        ts, te = t_region
        mask = (self.t_us >= ts) & (self.t_us <= te)
        return PosesData(self.t_us[mask], self.rots[mask], self.ps[mask])

    def interpolate(self, t_new_us: NDArray):
        rots = slerp_rotation(self.rots, self.t_us, t_new_us)
        trans = interpolate_vector3(self.ps, self.t_us, t_new_us)
        return PosesData(t_new_us, rots, trans)

    def transform_local(self, tf: Pose):
        # R = R * R_loc
        # t = t + R * t_loc
        self.ps = self.ps + self.rots.apply(tf.p)
        self.rots = self.rots * tf.rot

    def transform_global(self, tf: Pose):
        # R = R_glo * R
        # t = t_glo + R_glo * t
        self.rots = tf.rot * self.rots
        self.ps = tf.p + tf.rot.apply(self.ps)

    @property
    def rate(self):
        return float(1e6 / np.mean(np.diff(self.t_us)))

    @property
    def length_meter(self):
        """
        获取轨迹长度 meter
        """
        disps = np.diff(self.ps, axis=0)
        disp_lens = np.linalg.norm(disps, axis=1)
        return np.sum(disp_lens)

    def length_meter_fix_rate(self, rate: int = 1):
        """
        设定固定频率 获取轨迹长度 meter
        """
        t_new_us = get_time_series([self.t_us], rate=rate)
        return self.interpolate(t_new_us).length_meter


@dataclass
class ImuData:
    t_us: NDArray
    gyro: NDArray
    acce: NDArray
    ahrs: Rotation
    magn: NDArray

    frame: Frame = "local"

    def __getitem__(self, idx):
        return ImuData(
            self.t_us[idx],
            self.gyro[idx],
            self.acce[idx],
            self.ahrs[idx],
            self.magn[idx],
            self.frame,
        )

    def __len__(self):
        return self.t_us.__len__()

    @staticmethod
    def from_raw(raw: NDArray):
        assert raw.shape[1] >= 12, f"Invalid raw data shape: {raw.shape}"
        gyro = raw[:, 1:4]
        acce = raw[:, 4:7]
        ahrs = Rotation.from_quat(raw[:, 7:11], scalar_first=True)
        t_us = raw[:, 0] + raw[:, 11][0] + -raw[:, 0][0]

        if raw.shape[1] >= 15:
            magn = raw[:, 12:15]
        else:
            magn = np.zeros_like(gyro)
            # raise ValueError("Invalid raw data shape: missing magnetometer data")
        return ImuData(t_us, gyro, acce, ahrs, magn)

    @staticmethod
    def from_csv(path: Path):
        raw = pd.read_csv(path).dropna().to_numpy()
        return ImuData.from_raw(raw)

    def interpolate(self, t_new_us: NDArray):
        acce = interpolate_vector3(self.acce, self.t_us, t_new_us)
        gyro = interpolate_vector3(self.gyro, self.t_us, t_new_us)
        ahrs = slerp_rotation(self.ahrs, self.t_us, t_new_us)
        # 磁场
        magn = interpolate_vector3(self.magn, self.t_us, t_new_us)
        return ImuData(t_new_us, gyro, acce, ahrs, magn)

    def transform(self, rots: Rotation | None = None):
        if rots is None:
            rots = self.ahrs
        acce = rots.apply(self.acce)
        gyro = rots.apply(self.gyro)
        magn = rots.apply(self.magn)
        return ImuData(self.t_us, gyro, acce, rots, magn, frame="global")

    def get_time_range(self, time_range: tuple[float | None, float | None]):
        ts, te = self.t_us[0], self.t_us[-1]
        if time_range[0]:
            ts = time_range[0] * 1e6 + self.t_us[0]
        if time_range[1]:
            te = time_range[1] * 1e6 + self.t_us[0]
        m = (self.t_us >= ts) & (self.t_us <= te)
        return ImuData(
            self.t_us[m],
            self.gyro[m],
            self.acce[m],
            self.ahrs[m],
            self.magn[m],
            self.frame,
        )


class GroundTruthData(PosesData):
    @staticmethod
    def from_raw(raw: NDArray):
        t_us = raw[:, 0]
        trans = raw[:, 1:4]
        quats = raw[:, 4:8]
        # qwxyz
        rots = Rotation.from_quat(quats, scalar_first=True)
        return GroundTruthData(t_us, rots, trans)

    @staticmethod
    def from_csv(path: Path):
        raw = pd.read_csv(path).to_numpy()
        return GroundTruthData.from_raw(raw)


class CameraColumn:
    """
    #timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],p_CS_C_x [m],p_CS_C_y [m],p_CS_C_z [m],q_CS_w [],q_CS_x [],q_CS_y [],q_CS_z [],t_system [us]
    """

    t = ["#timestamp [us]"]
    ps = ["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]
    qs = ["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]
    pc = ["p_CS_C_x [m]", "p_CS_C_y [m]", "p_CS_C_z [m]"]
    qc = ["q_CS_w []", "q_CS_x []", "q_CS_y []", "q_CS_z []"]
    t_sys = ["t_system [us]"]

    all = t + ps + qs + pc + qc + t_sys


class CameraData(PosesData):
    @staticmethod
    def from_csv(path: Path):
        df = pd.read_csv(path).drop_duplicates(CameraColumn.t)
        t_sensor_us = df[CameraColumn.t].to_numpy().flatten()
        t_us = df[CameraColumn.t_sys].to_numpy().flatten()
        trans = df[CameraColumn.ps].to_numpy()
        rots = df[CameraColumn.qs].to_numpy()
        rots = Rotation.from_quat(rots, scalar_first=True)

        t_us = t_sensor_us - t_sensor_us[0] + t_us[0]
        return CameraData(t_us, rots, trans)


class FusionData(PosesData):
    """
    t_us: 此时间为系统时间，非UTC时间戳
    """

    @staticmethod
    def from_raw(raw: NDArray):
        t_us = raw[:, 0]
        trans = raw[:, 1:4]
        quats = raw[:, 4:8]
        # cam_trans = raw[:, 8:11]
        # cam_quats = raw[:, 11:15]
        rots = Rotation.from_quat(quats, scalar_first=True)
        return FusionData(t_us, rots, trans)

    @staticmethod
    def from_csv(path: Path):
        raw = pd.read_csv(path).to_numpy()
        return FusionData.from_raw(raw)


@dataclass
class CalibrationData:
    tf_sg_local: Pose
    tf_sg_global: Pose

    @staticmethod
    def from_json(path: Path):
        with open(path, "r") as f:
            data = json.load(f)
            if not isinstance(data, list) or len(data) != 1:
                raise ValueError("Invalid JSON format")
            data = data[0]

            rot_local = np.array(data["rot_sensor_gt"])
            trans_local = np.array(data["trans_sensor_gt"]).flatten()
            tf_sg_local = Pose(Rotation.from_matrix(rot_local), trans_local)

            rot_global = np.array(data["rot_ref_sensor_gt"])
            trans_global = np.array(data["trans_ref_sensor_gt"]).flatten()
            tf_sg_global = Pose(Rotation.from_matrix(rot_global), trans_global)

            print(f"Load Calib File: {path}")

            return CalibrationData(tf_sg_local, tf_sg_global)


@dataclass
class DataCheck:
    t_gi_us: int

    @staticmethod
    def from_json(path: Path):
        with open(path, "r") as f:
            data = json.load(f)
            assert "check_time_diff" in data
            check_time_diff = data["check_time_diff"]
            assert "time_diff_21_us" in check_time_diff
            t_gi_us = check_time_diff["time_diff_21_us"]
            return DataCheck(t_gi_us)


class UnitData:
    imu_data: ImuData
    gt_data: PosesData
    fusion_data: PosesData
    check_data: DataCheck
    calib_data: CalibrationData

    def __init__(self, base_dir: Path | str, using_ext: int = True):
        base_dir = Path(base_dir)
        self.name = base_dir.name
        self.base_dir = base_dir

        # 设备名称
        spl = self.name.split("_")
        device_name = spl[2] if len(spl) > 2 else "default"
        self.device_name = device_name  # type: ignore

        self._imu_path = base_dir / "imu.csv"
        self._cam_path = base_dir / "cam.csv"
        self.__gt_path_1 = base_dir / "rtab.csv"
        self.__gt_path_2 = base_dir / "gt.csv"
        if not self.__gt_path_1.exists():
            self._gt_path = self.__gt_path_2
        # 两种文件均不存在
        if not self._gt_path.exists():
            # 生成gt文件
            db_file = RTABData.get_db_file(base_dir)
            rtab_data = RTABData(db_file)
            rtab_data.save_csv(self._gt_path)
            print(f"Generated {self._gt_path}")

        self._gt_path_opt = base_dir / "opt.csv"
        if not self._gt_path_opt.exists():
            db_file = RTABData.get_db_file(base_dir)
            rtab_data = RTABData(db_file)
            rtab_data.save_csv(self._gt_path_opt, using_opt=True)

        self._fusion_path = base_dir / "fusion.csv"
        self.has_fusion = self._fusion_path.exists()

        # 读取标定数据
        if using_ext:
            self._calib_file = base_dir / "Calibration.json"
            self._check_file = base_dir / "DataCheck.json"
            self.load_data()

    def load_data(self):
        imu_data = ImuData.from_csv(self._imu_path)
        gt_data = GroundTruthData.from_csv(self._gt_path)
        if self.has_fusion:
            fusion_data = FusionData.from_csv(self._fusion_path)
            self.fusion_data = fusion_data

        self.correct(gt_data, imu_data)

    def correct(self, gt_data: GroundTruthData, imu_data: ImuData):
        self.calib_data = CalibrationData.from_json(self._calib_file)
        self.check_data = DataCheck.from_json(self._check_file)
        # 时间修正
        gt_data.t_us += self.check_data.t_gi_us
        # 空间变换
        gt_data.transform_local(self.calib_data.tf_sg_local.inverse())
        # gt_data.transform_global(self.calib_data.tf_sg_global)
        gt_data.transform_global(gt_data.get_pose(0).get_yaw_rot().inverse())

        # 数据对齐
        t_new_us = get_time_series([imu_data.t_us, gt_data.t_us])
        self.imu_data = imu_data.interpolate(t_new_us)
        self.gt_data = gt_data.interpolate(t_new_us)


class DeviceDataset:
    def __init__(self, base_dir: Path | str, using_ext: bool = True):
        self.base_dir = Path(base_dir)
        self.device_name = self.base_dir.name
        self.units = [
            UnitData(path, using_ext)
            for path in self.base_dir.iterdir()
            if path.is_dir()
        ]

    def __getitem__(self, index):
        return self.units[index]

    def __len__(self):
        return len(self.units)
