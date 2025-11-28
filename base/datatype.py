import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from base.interpolate import get_time_series, interpolate_vector3, slerp_rotation


@dataclass
class Pose:
    rot: Rotation
    p: NDArray

    def copy(self):
        return Pose(self.rot, self.p.copy())

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
        return Pose(Rotation.from_euler("ZXY", [yaw, 0, 0]), np.zeros(3))


Frame: TypeAlias = Literal["local", "global"]


@dataclass
class ImuData:
    t_us: NDArray
    gyro: NDArray
    acce: NDArray
    ahrs: Rotation

    frame: Frame = "local"

    def __getitem__(self, idx):
        return ImuData(
            self.t_us[idx], self.gyro[idx], self.acce[idx], self.ahrs[idx], self.frame
        )

    def __len__(self):
        return self.t_us.__len__()

    @staticmethod
    def from_raw(raw: NDArray):
        assert raw.shape[1] == 12, f"Invalid raw data shape: {raw.shape}"
        gyro = raw[:, 1:4]
        acce = raw[:, 4:7]
        ahrs = Rotation.from_quat(raw[:, 7:11], scalar_first=True)
        t_us = raw[:, 0] + raw[:, 11][0] + -raw[:, 0][0]
        return ImuData(t_us, gyro, acce, ahrs)

    @staticmethod
    def from_csv(path: Path):
        raw = pd.read_csv(path).dropna().to_numpy()
        return ImuData.from_raw(raw)

    def interpolate(self, t_new_us: NDArray):
        acce = interpolate_vector3(self.acce, self.t_us, t_new_us)
        gyro = interpolate_vector3(self.gyro, self.t_us, t_new_us)
        ahrs = slerp_rotation(self.ahrs, self.t_us, t_new_us)
        return ImuData(t_new_us, gyro, acce, ahrs)

    def transform(self, rots: Rotation | None = None):
        if rots is None:
            rots = self.ahrs
        acce = rots.apply(self.acce)
        gyro = rots.apply(self.gyro)
        return ImuData(self.t_us, gyro, acce, rots, frame="global")

    def get_time_range(self, time_range: tuple[float | None, float | None]):
        ts, te = self.t_us[0], self.t_us[-1]
        if time_range[0]:
            ts = time_range[0] * 1e6 + self.t_us[0]
        if time_range[1]:
            te = time_range[1] * 1e6 + self.t_us[0]
        m = (self.t_us >= ts) & (self.t_us <= te)
        return ImuData(
            self.t_us[m], self.gyro[m], self.acce[m], self.ahrs[m], frame=self.frame
        )


@dataclass
class PosesData:
    t_us: NDArray
    rots: Rotation
    trans: NDArray

    def __len__(self):
        return len(self.t_us)

    def __getitem__(self, index):
        if isinstance(index, int):
            return Pose(self.rots[index], self.trans[index])
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

    def get_time_range(self, time_range: tuple[float | None, float | None]):
        ts, te = self.t_us[0], self.t_us[-1]
        if time_range[0] is not None:
            ts = time_range[0] * 1e6 + self.t_us[0]
        if time_range[1] is not None:
            te = time_range[1] * 1e6 + self.t_us[0]

        mask = (self.t_us >= ts) & (self.t_us <= te)
        return PosesData(self.t_us[mask], self.rots[mask], self.trans[mask])

    def interpolate(self, t_new_us: NDArray):
        rots = slerp_rotation(self.rots, self.t_us, t_new_us)
        trans = interpolate_vector3(self.trans, self.t_us, t_new_us)
        return PosesData(t_new_us, rots, trans)

    def transform_local(self, tf: Pose):
        # R = R * R_loc
        # t = t + R * t_loc
        self.trans = self.trans + self.rots.apply(tf.p)
        self.rots = self.rots * tf.rot

    def transform_global(self, tf: Pose):
        # R = R_glo * R
        # t = t_glo + R_glo * t
        self.rots = tf.rot * self.rots
        self.trans = tf.p + tf.rot.apply(self.trans)


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


def get_ang_vec(rot: Rotation):
    rot_vec = rot.as_rotvec()
    angle = np.linalg.norm(rot_vec)
    vec = rot_vec / angle
    return vec.tolist(), float(angle) * 180 / np.pi


class FusionData(PosesData):
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

    def __init__(self, base_dir: Path | str):
        base_dir = Path(base_dir)
        self.name = base_dir.name

        self._imu_path = base_dir / "imu.csv"
        self._gt_path = base_dir / "rtab.csv"
        if not self._gt_path.exists():
            self._gt_path = base_dir / "gt.csv"
        self._fusion_path = base_dir / "fusion.csv"
        self.has_fusion = self._fusion_path.exists()

        # 读取标定数据
        self._calib_file = base_dir / "Calibration.json"
        self._check_file = base_dir / "DataCheck.json"

        self.calib_data = CalibrationData.from_json(self._calib_file)
        self.check_data = DataCheck.from_json(self._check_file)

        self.load_data()

    def load_data(self):
        imu_data = ImuData.from_csv(self._imu_path)
        gt_data = GroundTruthData.from_csv(self._gt_path)
        if self.has_fusion:
            fusion_data = FusionData.from_csv(self._fusion_path)
            self.fusion_data = fusion_data

        # 时间修正
        gt_data.t_us += self.check_data.t_gi_us

        # 空间变换
        gt_data.transform_local(self.calib_data.tf_sg_local.inverse())
        # gt_data.transform_global(self.calib_data.tf_sg_global)
        gt_data.transform_global(gt_data[0].get_yaw_rot().inverse())

        # 数据对齐
        t_new_us = get_time_series([imu_data.t_us, gt_data.t_us])
        self.imu_data = imu_data.interpolate(t_new_us)
        self.gt_data = gt_data.interpolate(t_new_us)

    # def set_time_range(self, time_range:tuple[float, float]):
    # pass


class DeviceDataset:
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.device_name = self.base_dir.name
        self.units = [
            UnitData(path) for path in self.base_dir.iterdir() if path.is_dir()
        ]

    def __getitem__(self, index):
        return self.units[index]

    def __len__(self):
        return len(self.units)
