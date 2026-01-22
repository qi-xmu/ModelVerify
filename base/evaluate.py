"""

TODO:
    - [x] 计算 ATE 和 RTE
    - [ ] 计算 CDF

"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from base.interpolate import get_time_series

from .datatype import PosesData


class Evaluation:
    ref_poses: PosesData
    resdict: dict

    def __init__(
        self,
        ref_poses: PosesData,
        rel_duration: int = 60,
        rate: int = 200,
        name: str = "",
    ):
        self.ref_poses = ref_poses
        self.rel_duration = rel_duration
        self.rate = rate
        self.name = name

        self.resdict = {}
        self.inner = {}

        # 计算其他信息
        self.length = ref_poses.length_meter
        self.time_length = (ref_poses.t_us[-1] - ref_poses.t_us[0]) / 1e6
        self.mean_velocity = self.length / self.time_length

        # map
        self.resdict["length(m)"] = self.length
        self.resdict["time_length(s)"] = self.time_length
        self.resdict["mean_velocity(m/s)"] = self.mean_velocity

    def __str__(self):
        return json.dumps(self.resdict, indent=4)

    def print(self):
        print(self)

    def save(self, file: Path | str):
        with open(file, "w") as f:
            json.dump(self.resdict, f, indent=4)

    def __get_APE(self, eva_poses: PosesData):
        errs = []
        for ref_pose, eva_pose in zip(self.ref_poses, eva_poses):
            err_vec = ref_pose.between(eva_pose).log()
            err = np.linalg.norm(err_vec)
            errs.append(err)
        return np.array(errs)

    def __get_ATE(self, eva_poses: PosesData):
        errs = []
        for ref_pose, eva_pose in zip(self.ref_poses, eva_poses):
            err = np.linalg.norm(eva_pose.p - ref_pose.p)
            errs.append(err)
        return np.array(errs)

    def __get_RPE(self, eva_poses: PosesData):
        gap = int(self.rate * self.rel_duration)
        size = len(eva_poses)
        assert size >= gap, f"eva_poses size {size} is less than gap {gap}"

        errs = []
        for i in range(size - gap):
            ref_rel_pose = self.ref_poses.get_between(i, i + gap)
            eva_rel_pose = eva_poses.get_between(i, i + gap)
            err_vec = ref_rel_pose.between(eva_rel_pose).log()
            err = np.linalg.norm(err_vec)
            errs.append(err)

        return np.array(errs)

    def __get_RTE(self, eva_poses: PosesData):
        gap = int(self.rate * self.rel_duration)

        size = len(eva_poses)
        errs = []
        for i in range(size - gap):
            # NOTE：这里没有使用全局坐标系下的误差。
            ref_rel_pose = self.ref_poses.get_between(i, i + gap)
            eva_rel_pose = eva_poses.get_between(i, i + gap)
            err = np.linalg.norm(eva_rel_pose.p - ref_rel_pose.p)
            errs.append(err)

        return np.array(errs)

    def get_eval(self, eva_poses: PosesData, tag: str):
        """
        对比PosesData, 返回包含各误差指标的字典:
        {
            "rate": int,          # 采样率 (Hz)
            "size": int,          # 姿态数据点数
            "APE(_)": float,      # 绝对姿态误差均值 (rad)
            "APE_CDF": dict,      # APE的CDF数据
            "ATE(m)": float,      # 绝对轨迹误差均值 (m)
            "ATE_CDF": dict,      # ATE的CDF数据
            "RPE(_)": float,      # 相对姿态误差均值 (rad), 当time_length > rel_duration时
            "RPE_CDF": dict,      # RPE的CDF数据, 当time_length > rel_duration时
            "RTE(m)": float,      # 相对轨迹误差均值 (m), 当time_length > rel_duration时
            "RTE_CDF": dict,      # RTE的CDF数据, 当time_length > rel_duration时
        }

        Returns:
            tuple: (resdict[tag], inner[tag]), 分别为结果字典和CDF详细数据
        """
        if tag not in self.inner:
            self.inner[tag] = {}
            self.resdict[tag] = {}

        res = {}
        ref_poses = self.ref_poses
        res["rate"] = self.rate

        t_new_us = get_time_series([ref_poses.t_us, eva_poses.t_us], rate=self.rate)
        ref_poses = ref_poses.interpolate(t_new_us)
        eva_poses = eva_poses.interpolate(t_new_us)
        # 长度
        size = len(ref_poses)
        res["size"] = size
        assert size != 0, f"size = {size}, {ref_poses}"
        assert size == len(eva_poses), f" {size} != {len(eva_poses)}"

        # 计算误差
        ape_errs = self.__get_APE(eva_poses)
        ate_errs = self.__get_ATE(eva_poses)
        ape = np.mean(ape_errs)
        ate = np.mean(ate_errs)
        ape_cdf = get_cdf_from_err(ape_errs)
        ate_cdf = get_cdf_from_err(ate_errs)
        # 记录值
        res["APE(_)"] = ape
        res["ATE(m)"] = ate
        res["APE_CDF"] = ape_cdf["percentiles"]
        res["ATE_CDF"] = ate_cdf["percentiles"]
        self.inner[tag]["APE_CDF"] = ape_cdf
        self.inner[tag]["ATE_CDF"] = ate_cdf

        # 计算 RPE
        if self.time_length > self.rel_duration:
            rpe_errs = self.__get_RPE(eva_poses)
            rte_errs = self.__get_RTE(eva_poses)
            rpe_cdf = get_cdf_from_err(rpe_errs)
            rte_cdf = get_cdf_from_err(rte_errs)
            rpe = np.mean(rpe_errs)
            rte = np.mean(rte_errs)
            # 记录值
            res["RPE(_)"] = rpe
            res["RTE(m)"] = rte
            res["RPE_CDF"] = rpe_cdf["percentiles"]
            res["RTE_CDF"] = rte_cdf["percentiles"]
            self.inner[tag]["RPE_CDF"] = rpe_cdf
            self.inner[tag]["RTE_CDF"] = rte_cdf

        self.resdict[tag].update(res)
        return res, self.inner[tag]

    def get_cdf(self, tag: str, err_type: Literal["APE", "ATE", "RPE", "RTE"] = "ATE"):
        return self.inner[tag][f"{err_type}_CDF"]


def get_cdf_from_err(errors: NDArray | list, tag: str = "") -> dict:
    """
    计算误差的累积分布函数 (CDF)

    Args:
        errors: 误差值数组

    Returns:
        包含CDF数据的字典:
        {
            "errors": 误差值数组,
            "cdf": CDF值数组,
            "percentiles": 百分位数 {50%, 90%, 95%, 99%}
        }
    """
    # 计算CDF
    errors = np.array(errors)
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)

    # 计算百分位数
    percentiles = {
        "50%": np.percentile(errors, 50),
        "90%": np.percentile(errors, 90),
        "95%": np.percentile(errors, 95),
        "99%": np.percentile(errors, 99),
    }

    result = {
        "tag": tag,
        "errors": sorted_errors,
        "cdf": cdf,
        "percentiles": percentiles,
    }

    return result
