from pathlib import Path
from typing import Literal

from scipy.spatial.transform import Rotation

from base.dataset.H5Type import H5Dataset, ResampledData, Sequence
from base.datatype import GroundTruthData, ImuData, UnitData


class H5UnitData(UnitData):
    def __init__(self, sequence: Sequence):
        self.sequence = sequence

        self.name = sequence.name
        self._get_deivce_from_name()

    def load_data(self, using_opt=False):
        """从H5序列的resampled数据加载IMU和真值数据"""
        assert self.sequence.resampled is not None, (
            "Sequence has resampled but it is None"
        )
        resampled: ResampledData = self.sequence.resampled

        # 获取时间戳 [N]
        t_us = resampled.get_t_us()
        assert t_us is not None, "Resampled data has no time stamps"

        # 获取IMU数据 [M, N, 6] -> 取第一个传感器
        imu_raw = resampled.get_imu()
        assert imu_raw is not None, "Resampled data has no IMU data"
        imu_sensor = imu_raw[0] if imu_raw.ndim == 3 else imu_raw
        gyro = imu_sensor[:, 0:3]  # [N, 3]
        acce = imu_sensor[:, 3:6]  # [N, 3]

        # 获取磁场数据
        mag_raw = resampled.get_mag()
        assert mag_raw is not None, "Resampled data has no magnetometer data"
        mag_sensor = mag_raw[0] if mag_raw.ndim == 3 else mag_raw
        mag = mag_sensor[:, 0:3]  # [N, 3]
        assert len(mag) == len(gyro), (
            f"Length mismatch between gyro and mag data: {len(gyro)} != {len(mag)}"
        )

        # 获取真值数据 [N, 10] -> pos(3), qwxyz(4), vel(3)
        gt_raw = resampled.get_ground_truth()
        assert gt_raw is not None, "Resampled data has no ground truth data"
        pos = gt_raw[:, 0:3]  # [N, 3]
        quats = gt_raw[:, 3:7]  # [N, 4] qwxyz
        rots = Rotation.from_quat(quats, scalar_first=True)

        # 创建ImuData
        self.imu_data = ImuData(
            t_us=t_us,
            gyro=gyro,
            acce=acce,
            ahrs=rots,
            magn=mag,
        )

        # 创建gt_data
        self.gt_data = GroundTruthData(t_us, rots, pos)


class H5SplitDataset(H5Dataset):
    def __init__(
        self,
        file_path: Path | str,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.file_path = Path(file_path)
        self.split = split
        self.name = self.file_path.stem

        super().__init__(file_path, mode="r")
        super().__enter__()

        # 读取相应划分的数据名称
        seq_lists = self.load_sequence_lists()
        self.sequence_names = getattr(seq_lists, self.split)
        assert isinstance(self.sequence_names, list)

    def __getitem__(self, idx):
        seq_name = self.sequence_names[idx]
        sequence = self.get_sequence(seq_name)
        assert sequence is not None

        return H5UnitData(sequence)

    def __iter__(self):
        for seq_name in self.sequence_names:
            sequence = self.get_sequence(seq_name)
            assert sequence is not None

            yield H5UnitData(sequence)

    def __len__(self):
        return len(self.sequence_names)
