from pathlib import Path

import h5py
import numpy as np
from h5py import Group
from scipy.spatial.transform import Rotation

from base.datatype import GroundTruthData, ImuData
from base.serialize import TLIOSerializer, UnitSerializer

sample = "/Users/qi/Resources/SimpleDemo.h5"
main = "/Users/qi/Resources/YT_server_200Hz.h5"

# 文件结构
# 'para', 'test', 'train', 'valid'
# "mid360_2025-10-27_221831_00"
# 'acc', 'acc_bias', 'gyr', 'gyr_bias', 'mag', 'motion_qua', 'pos', 'qua', 'ts'


class H5Loader:
    def __init__(self, filepath: str) -> None:
        self.filepath = Path(filepath)
        self.h5_data = h5py.File(filepath, "r")

        self.savepath = self.filepath.parent / self.filepath.name.split(".")[0]

    def _parse_unit(self, unit: Group, remove_bias=False):
        """
        <KeysViewHDF5 ['acc', 'acc_bias', 'gyr', 'gyr_bias', 'mag', 'motion_qua', 'pos', 'qua', 'ts']>
        """
        acc = np.array(unit["acc"])
        _acc_bias = np.array(unit["acc_bias"])
        gyr = np.array(unit["gyr"])
        _gyr_bias = np.array(unit["gyr_bias"])
        mag = np.array(unit["mag"])
        _motion_qua = np.array(unit["motion_qua"])
        pos = np.array(unit["pos"])
        qua = np.array(unit["qua"])
        ts = np.array(unit["ts"])

        if remove_bias:
            acc = acc - _acc_bias
            gyr = gyr - _gyr_bias

        t_us = (ts * 1e6).astype(np.uint64)
        rots = Rotation.from_quat(qua, scalar_first=True)

        imu_data = ImuData(t_us, gyr, acc, rots, mag, "local")
        gt_data = GroundTruthData(t_us, rots, pos)
        return imu_data, gt_data

    def convert(self, group: Group):
        for key in group:
            assert isinstance(key, str)
            print(key, end=": ")
            # 访问数据
            unit = group[key]
            assert isinstance(unit, Group)
            # 解析数据
            data = self._parse_unit(unit, remove_bias=True)
            # 保存数据
            base_unit_dir = self.base_group_dir / key
            tlio_unit_dir = self.tlio_group_dir / key
            UnitSerializer(*data).save(base_unit_dir)
            TLIOSerializer(*data).save(tlio_unit_dir)

    def convert_all(self):
        self.base_dir = self.savepath
        self.tlio_dir = self.savepath.parent / f"TLIO_{self.savepath.name}"
        keys = ["train", "test", "valid"]
        for key in keys:
            self.base_group_dir = self.base_dir / key
            self.tlio_group_dir = self.tlio_dir / key

            group = self.h5_data[key]
            assert isinstance(group, Group)
            self.convert(group)


if __name__ == "__main__":
    # 探索文件结构
    h5_loader = H5Loader(main)
    # h5_loader.convert_all()
