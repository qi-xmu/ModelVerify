"""

TODO:
    - [x] 计算 ATE 和 RTE

"""

import json
from pathlib import Path

import numpy as np
import sophuspy as sp
from numpy.typing import NDArray

from base.interpolate import get_time_series

from .datatype import Pose, PosesData


class Evaluation:
    ref_poses: PosesData
    rte_duration: int = 60  # sec

    resdict: dict

    def __init__(self, ref_poses: PosesData):
        self.ref_poses = ref_poses
        self.resdict = {}

        # 计算其他信息
        self.length = ref_poses.length_meter

        # map
        self.resdict["ref_length"] = self.length

    def __str__(self):
        return json.dumps(self.resdict, indent=4)

    def print(self):
        print(self)

    def save(self, file: Path | str):
        with open(file, "w") as f:
            json.dump(self.resdict, f, indent=4)

    @staticmethod
    def __get_error(pose1: Pose, pose2: Pose):
        ref_se3 = sp.SE3(pose1.rot.as_matrix(), pose1.p)
        eva_se3 = sp.SE3(pose2.rot.as_matrix(), pose2.p)

        err_se3 = ref_se3.inverse() * eva_se3
        err = err_se3.log()
        return err

    def get_eval(self, eva_poses: PosesData, tag: str):
        ref_poses = self.ref_poses

        # t_region = self.__get_time_region(ref_poses.t_us, eva_poses.t_us)
        rate = int(ref_poses.rate)
        t_new_us = get_time_series([ref_poses.t_us, eva_poses.t_us], rate=rate)
        ref_poses = ref_poses.interpolate(t_new_us)
        eva_poses = eva_poses.interpolate(t_new_us)

        # 长度
        size = len(ref_poses)
        assert size != 0, f"size = {size}, {ref_poses}"
        assert size == len(eva_poses), f" {size} != {len(eva_poses)}"

        cnt = 0
        ape_square_sum = 0
        ate_square_sum = 0
        for ref_pose, eva_pose in zip(ref_poses, eva_poses):
            # 计算 APE
            err = self.__get_error(ref_pose, eva_pose)
            ape_square_sum += np.square(err)
            ate_square_sum += np.square(eva_pose.p - ref_pose.p)
            cnt += 1

        ape_vec: NDArray = np.sqrt(ape_square_sum / cnt)
        ate_vec: NDArray = np.sqrt(ate_square_sum / cnt)
        self.resdict[f"{tag}_ape_vec"] = ape_vec.flatten().tolist()
        self.resdict[f"{tag}_ate_vec"] = ate_vec.flatten().tolist()

        # 计算 RPE
        gap = int(rate * self.rte_duration)
        cnt = 0
        rpe_square_sum = 0
        for i in range(size - gap):
            ref_pose_pre = ref_poses.get_pose(i)
            ref_pose_nex = ref_poses.get_pose(i + gap)
            ref_pose = ref_pose_pre.inverse().compose(ref_pose_nex)

            eva_pose_pre = eva_poses.get_pose(i)
            eva_pose_nex = eva_poses.get_pose(i + gap)
            eva_pose = eva_pose_pre.inverse().compose(eva_pose_nex)

            err = self.__get_error(ref_pose, eva_pose)
            rpe_square_sum += np.square(err)

            cnt += 1

        rpe_vec: NDArray = np.sqrt(ape_square_sum / cnt)
        self.resdict[f"{tag}_rpe_vec"] = rpe_vec.flatten().tolist()

        return self.resdict
