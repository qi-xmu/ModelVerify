import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, TypeAlias

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from sophuspy import SE3

from base.interpolate import get_time_series, interpolate_vector3, slerp_rotation
from base.rtab import RTABData


@dataclass
class Pose:
    """
    位姿数据类，表示刚体的旋转和平移

    Attributes:
        rot (Rotation): 旋转四元数，表示姿态
        p (NDArray): 3D平移向量 [x, y, z]，单位：米
        t_us (int): 时间戳，单位：微秒

    数学表示:
        T = [R, t]
            [0, 1]
        其中 R 为旋转矩阵，t 为平移向量
    """
    rot: Rotation
    p: NDArray
    t_us: int = 0

    def copy(self):
        """创建位姿的深拷贝"""
        return Pose(self.rot, self.p.copy(), self.t_us)

    @staticmethod
    def identity():
        """
        创建单位位姿（无旋转、无平移）

        Returns:
            Pose: 单位位姿
        """
        return Pose(Rotation.identity(), np.zeros(3))

    @staticmethod
    def from_rotation(rot: Rotation):
        """
        从旋转创建位姿（无平移）

        Args:
            rot: 旋转四元数

        Returns:
            Pose: 只有旋转的位姿
        """
        return Pose(rot, np.zeros(3))

    @staticmethod
    def from_transform(trans: NDArray):
        """
        从平移创建位姿（无旋转）

        Args:
            trans: 3D平移向量

        Returns:
            Pose: 只有平移的位姿
        """
        return Pose(Rotation.identity(), trans)

    def compose(self, other: Self):
        """
        位姿复合（变换组合）

        数学公式:
            T_total = T_self * T_other
            R_total = R_self * R_other
            p_total = R_self * p_other + p_self

        应用场景:
            - 将局部坐标系下的点转换到全局坐标系
            - 组合多个连续变换

        Args:
            other: 另一个位姿

        Returns:
            Pose: 复合后的新位姿
        """
        return Pose(self.rot * other.rot, self.rot.apply(other.p) + self.p)

    def compose_self(self, other: Self):
        """
        将另一个位姿复合到当前位姿上（原地修改）

        数学公式:
            R_new = R_self * R_other
            p_new = R_self * p_other + p_self

        Args:
            other: 要复合的另一个位姿
        """
        self.p = self.rot.apply(other.p) + self.p
        self.rot = self.rot * other.rot

    def compose_trans_self(self, trans: NDArray):
        """
        将平移向量复合到当前位姿上（原地修改）

        数学公式:
            p_new = p_self + R_self * trans

        应用场景:
            - 在当前坐标系下添加平移偏移

        Args:
            trans: 平移向量，在局部坐标系下
        """
        self.p += self.rot.apply(trans)

    def inverse(self):
        """
        计算位姿的逆

        数学公式:
            T_inv = T^{-1}
            R_inv = R^T (旋转矩阵的转置)
            p_inv = -R^T * p

        应用场景:
            - 从全局坐标系转换到局部坐标系
            - 计算相对变换

        Returns:
            Pose: 逆位姿
        """
        return Pose(self.rot.inv(), -self.rot.inv().apply(self.p))

    def between(self, to: Self):
        """
        计算两个位姿之间的相对变换

        数学公式:
            T_rel = T_self^{-1} * T_to

        应用场景:
            - 计算从位置A到位置B的相对运动
            - 计算相机帧之间的相对位姿

        Args:
            to: 目标位姿

        Returns:
            Pose: 相对位姿
        """
        return self.inverse().compose(to)

    def log(self):
        """
        计算位姿的李代数表示

        数学公式:
            se(3) = log(SE(3))

        应用场景:
            - 位姿优化
            - 位姿插值

        Returns:
            SE3的李代数向量（6维）
        """
        return SE3(self.rot.as_matrix(), self.p).log()

    def get_yaw_pose(self):
        """
        提取偏航角（绕Z轴旋转）并创建只有偏航的位姿

        应用场景:
            - 只关心水平方向朝向的场景
            - 2D导航和平面运动分析

        Returns:
            Pose: 只包含偏航角旋转的位姿
        """
        yaw = self.rot.as_euler("ZXY")[0]
        return Pose(Rotation.from_rotvec([0, 0, yaw]), np.zeros(3))


Frame: TypeAlias = Literal["local", "global"]


@dataclass
class PosesData:
    """
    位姿数据序列类，存储一系列位姿

    Attributes:
        t_us (NDArray): 时间戳数组 [t1, t2, ..., tn]，单位：微秒
        rots (Rotation): 旋转四元数数组，包含N个旋转
        ps (NDArray): 位置数组，形状为[N, 3]，单位：米

    应用场景:
        - 存储轨迹数据
        - 存储相机/IMU位姿序列
        - 存储Ground Truth轨迹
    """
    t_us: NDArray
    rots: Rotation
    ps: NDArray

    def __len__(self):
        """返回位姿数量"""
        return len(self.t_us)

    def __getitem__(self, index: int | slice):
        """
        索引访问单个或多个位姿

        Args:
            index: 整数索引或切片

        Returns:
            PosesData: 子集位姿数据
        """
        return PosesData(self.t_us[index], self.rots[index], self.ps[index])

    def __iter__(self):
        """
        迭代位姿序列

        Yields:
            Pose: 每个时间步的位姿
        """
        for i in range(len(self.t_us)):
            yield Pose(self.rots[i], self.ps[i], self.t_us[i])

    @staticmethod
    def from_list(pose_list: list[Pose]):
        """
        从位姿列表创建PosesData对象

        Args:
            pose_list: Pose对象列表

        Returns:
            PosesData: 新的位姿数据对象
        """
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
        """
        获取指定索引的位姿

        Args:
            index: 位姿索引，默认为0

        Returns:
            Pose: 指定位置的位姿对象
        """
        return Pose(self.rots[index], self.ps[index], self.t_us[index])

    def get_time_range(self, time_range: tuple[float | None, float | None]):
        """
        根据相对时间范围获取位姿子集

        Args:
            time_range: (start_sec, end_sec) 相对起始时间的秒数
                       None表示不限制

        Returns:
            PosesData: 指定时间范围内的位姿数据
        """
        ts, te = self.t_us[0], self.t_us[-1]
        if time_range[0] is not None:
            ts = time_range[0] * 1e6 + self.t_us[0]
        if time_range[1] is not None:
            te = time_range[1] * 1e6 + self.t_us[0]
        return self.get_time_region((ts, te))

    def get_time_region(self, t_region: tuple[int, int]) -> "PosesData":
        """
        根据绝对时间范围获取位姿子集

        Args:
            t_region: (start_us, end_us) 绝对时间戳范围（微秒）

        Returns:
            PosesData: 指定时间范围内的位姿数据
        """
        ts, te = t_region
        mask = (self.t_us >= ts) & (self.t_us <= te)
        return PosesData(self.t_us[mask], self.rots[mask], self.ps[mask])

    def interpolate(self, t_new_us: NDArray, bounds_error: bool = False) -> "PosesData":
        """
        在新的时间点上插值位姿

        应用场景:
            - 不同传感器的数据对齐
            - 固定频率重采样
            - 填补缺失数据

        Args:
            t_new_us: 新的时间戳数组
            bounds_error: 超出时间范围时是否报错

        Returns:
            PosesData: 插值后的位姿数据
        """
        # 使用球面线性插值（SLERP）插值旋转
        rots = slerp_rotation(self.rots, self.t_us, t_new_us)
        # 使用线性插值插值位置
        trans = interpolate_vector3(self.ps, self.t_us, t_new_us, bounds_error)
        return PosesData(t_new_us, rots, trans)

    def transform_local(self, tf: Pose):
        """
        应用局部坐标变换（右乘变换）

        数学公式:
            R_new = R * R_tf
            p_new = p + R * p_tf

        应用场景:
            - 传感器坐标系转换（如IMU到车身坐标系）
            - 应用标定参数

        Args:
            tf: 局部变换位姿
        """
        # R = R * R_loc
        # t = t + R * t_loc
        self.ps = self.ps + self.rots.apply(tf.p)
        self.rots = self.rots * tf.rot

    def transform_global(self, tf: Pose):
        """
        应用全局坐标变换（左乘变换）

        数学公式:
            R_new = R_tf * R
            p_new = p_tf + R_tf * p

        应用场景:
            - 世界坐标系转换
            - 位姿平移和旋转

        Args:
            tf: 全局变换位姿
        """
        # R = R_glo * R
        # t = t_glo + R_glo * t
        self.rots = tf.rot * self.rots
        self.ps = tf.p + tf.rot.apply(self.ps)

    def get_between(self, s: int, e: int) -> Pose:
        """
        计算两个时间点之间的相对位姿

        应用场景:
            - 计算相邻帧之间的运动
            - 计算时间段内的累积运动

        Args:
            s: 起始索引
            e: 结束索引

        Returns:
            Pose: 相对位姿
        """
        s_pose = self.get_pose(s)
        e_pose = self.get_pose(e)
        rel_pose = s_pose.between(e_pose)
        return rel_pose

    def reset_start(self):
        """
        重置出发点到原点

        应用场景:
            - 将轨迹起点设为原点
            - 计算相对位移而非绝对位置
        """
        self.ps -= self.ps[0]

    @property
    def rate(self) -> float:
        """
        计算平均采样频率

        Returns:
            float: 频率，单位：Hz
        """
        return float(1e6 / np.mean(np.diff(self.t_us)))

    @property
    def length_meter(self) -> float:
        """
        计算轨迹总长度（相邻点距离之和）

        Returns:
            float: 轨迹长度，单位：米
        """
        disps = np.diff(self.ps, axis=0)
        disp_lens = np.linalg.norm(disps, axis=1)
        return np.sum(disp_lens)

    def length_meter_fix_rate(self, rate: int = 1) -> float:
        """
        按固定频率重采样后计算轨迹长度

        应用场景:
            - 不同采样率的轨迹比较
            - 统一标准评估轨迹长度

        Args:
            rate: 重采样频率，单位：Hz

        Returns:
            float: 轨迹长度，单位：米
        """
        t_new_us = get_time_series([self.t_us], rate=rate)
        return self.interpolate(t_new_us).length_meter


@dataclass
class ImuData:
    """
    IMU数据类，存储惯性测量单元的传感器数据

    Attributes:
        t_us (NDArray): 时间戳数组 [t1, t2, ..., tn]，单位：微秒
        gyro (NDArray): 陀螺仪数据，形状[N, 3]，单位：rad/s
        acce (NDArray): 加速度计数据，形状[N, 3]，单位：m/s²
        ahrs (Rotation): AHRS姿态四元数数组，表示设备方向
        magn (NDArray): 磁力计数据，形状[N, 3]，单位：μT（或其他磁场单位）
        frame (Frame): 坐标系类型，"local"为局部坐标系，"global"为全局坐标系

    应用场景:
        - 存储IMU原始数据
        - 存储融合后的姿态估计
        - 数据预处理和坐标变换

    注意:
        - frame属性仅用于标记数据所在的坐标系
        - acce_bias和gyro_bias为预留字段，可用于存储零偏
    """
    t_us: NDArray
    gyro: NDArray
    acce: NDArray
    ahrs: Rotation
    magn: NDArray

    frame: Frame = "local"
    # 补充内容
    # acce_bias: NDArray = np.zeros(0)
    # gyro_bias: NDArray = np.zeros(0)

    def __getitem__(self, idx):
        """
        索引访问IMU数据子集

        Args:
            idx: 整数索引或切片

        Returns:
            ImuData: 子集IMU数据
        """
        return ImuData(
            self.t_us[idx],
            self.gyro[idx],
            self.acce[idx],
            self.ahrs[idx],
            self.magn[idx],
            self.frame,
        )

    def __len__(self):
        """返回IMU数据数量（时间步数）"""
        return self.t_us.__len__()

    @staticmethod
    def from_raw(raw: NDArray) -> "ImuData":
        """
        从原始数组创建ImuData

        Args:
            raw: 原始数据数组，形状为 [N, M]，其中:
                - N: 时间步数
                - M >= 12: 数据列数
                  - col 0: 时间戳偏移量 [us]
                  - col 1-3: 陀螺仪数据 [rad/s] (x, y, z)
                  - col 4-6: 加速度计数据 [m/s^2] (x, y, z)
                  - col 7-10: AHRS四元数 [qw, qx, qy, qz]
                  - col 11: 时间戳基准值 [us]
                  - col 12-14: 磁力计数据 [可选] (x, y, z)

        Returns:
            ImuData对象

        Raises:
            AssertionError: 当数据列数少于12列时
        """
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
    def from_csv(path: Path) -> "ImuData":
        raw = pd.read_csv(path).dropna().to_numpy()
        return ImuData.from_raw(raw)

    def interpolate(
        self,
        t_new_us: NDArray,
        bounds_error: bool = False,
    ) -> "ImuData":
        """
        在新的时间点上插值IMU数据

        应用场景:
            - 不同传感器的数据对齐
            - 固定频率重采样
            - 填补缺失数据点

        Args:
            t_new_us: 新的时间戳数组
            bounds_error: 超出时间范围时是否报错

        Returns:
            ImuData: 插值后的IMU数据
        """
        # 使用线性插值插值加速度、陀螺仪和磁场数据
        acce = interpolate_vector3(self.acce, self.t_us, t_new_us, bounds_error)
        gyro = interpolate_vector3(self.gyro, self.t_us, t_new_us, bounds_error)
        # 使用球面线性插值（SLERP）插值姿态四元数
        ahrs = slerp_rotation(self.ahrs, self.t_us, t_new_us)
        # 磁场
        magn = interpolate_vector3(self.magn, self.t_us, t_new_us, bounds_error)
        return ImuData(t_new_us, gyro, acce, ahrs, magn)

    # 传入gt位姿去转换
    def transform(self, rots: Rotation | None = None):
        """
        将IMU数据从局部坐标系转换到全局坐标系

        Args:
            rots: 旋转变换矩阵，用于将局部坐标系数据转换到全局坐标系。
                  如果为None，则使用自身AHRS姿态四元数作为旋转基准。
                  适用于：
                  - None: 使用设备自身的AHRS姿态进行变换
                  - 外部传入: 使用指定的旋转矩阵（如GT位姿）进行变换

        Returns:
            ImuData: 转换到全局坐标系的新IMU数据对象，frame标记为"global"

        坐标变换原理:
            - 加速度计: a_global = R * a_local
            - 陀螺仪:     ω_global = R * ω_local
            - 磁力计:      m_global = R * m_local
            - AHRS姿态:   q_global = R * q_local

        应用场景:
            1. 使用自身AHRS进行姿态估计和重力补偿
            2. 使用GT位姿将数据投影到世界坐标系进行可视化
            3. 数据融合和轨迹可视化时统一坐标系
        """
        # 如果没有提供旋转矩阵，使用自身的AHRS姿态四元数作为旋转基准
        if rots is None:
            rots = self.ahrs

        # 将加速度计测量值从局部坐标系旋转变换到全局坐标系
        # a_global = R * a_local (消除了设备方向的影响)
        acce = rots.apply(self.acce)

        # 将陀螺仪测量值从局部坐标系旋转变换到全局坐标系
        # ω_global = R * ω_local
        gyro = rots.apply(self.gyro)

        # 将磁力计测量值从局部坐标系旋转变换到全局坐标系
        # m_global = R * m_local
        magn = rots.apply(self.magn)

        # 更新AHRS姿态四元数
        # q_global = R * q_local
        # 使用四元数乘法组合旋转变换，表示设备在世界坐标系中的新方向
        ahrs = rots * self.ahrs

        # 返回新的ImuData对象，标记坐标系为"global"
        # 注意：这是一个新对象，不修改原始数据
        return ImuData(self.t_us, gyro, acce, ahrs, magn, frame="global")

    def get_time_range(self, time_range: tuple[float | None, float | None]):
        """
        根据相对时间范围获取IMU数据子集

        Args:
            time_range: (start_sec, end_sec) 相对起始时间的秒数
                       None表示不限制

        Returns:
            ImuData: 指定时间范围内的IMU数据
        """
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

    def to_poses(self) -> PosesData:
        """
        将IMU姿态数据转换为PosesData对象

        注意: 位置设为零向量，因为IMU只提供姿态信息

        Returns:
            PosesData: 包含旋转但位置为零的位姿数据
        """
        ps = np.zeros((len(self.t_us), 3))
        return PosesData(self.t_us, self.ahrs, ps)

    def calibrate_with(self, config_path: Path):
        """
        使用标定文件校准IMU数据（原地修改）

        校准公式:
            acce_calibrated = Ra * acce - Ba
            gyro_calibrated = Rg * gyro - Bg

        其中:
            - Ra, Rg: 加速度计和陀螺仪的标定旋转矩阵
            - Ba, Bg: 加速度计和陀螺仪的零偏

        应用场景:
            - 消除传感器误差
            - 统一不同设备的数据

        Args:
            config_path: 标定配置文件路径（YAML格式）
        """
        imu_calib = ImuCalibration(config_path)

        # 应用加速度计标定：旋转并减去零偏
        self.acce = imu_calib.ra.apply(self.acce) - imu_calib.ba
        # 应用陀螺仪标定：旋转并减去零偏
        self.gyro = imu_calib.rg.apply(self.gyro) - imu_calib.bg


class ImuCalibration:
    """
    IMU标定数据类，存储IMU传感器的标定参数

    Attributes:
        ra (Rotation): 加速度计标定旋转矩阵（刻度因子和轴对齐）
        rg (Rotation): 陀螺仪标定旋转矩阵（刻度因子和轴对齐）
        ba (NDArray): 加速度计零偏，单位：m/s²
        bg (NDArray): 陀螺仪零偏，单位：rad/s

    应用场景:
        - 消除传感器系统性误差
        - 补偿刻度因子不一致
        - 补偿轴间不正交性
        - 补偿静态零偏

    标定原理:
        加速度计: a_calibrated = Ra * a_raw - Ba
        陀螺仪:     ω_calibrated = Rg * ω_raw - Bg

    配置文件格式 (YAML):
        Calibration:
            ra: [r11, r12, r13, r21, r22, r23, r31, r32, r33]  # 3x3矩阵，行优先
            rg: [r11, r12, r13, r21, r22, r23, r31, r32, r33]
            ba: [bx, by, bz]
            bg: [bx, by, bz]
    """

    def __init__(self, config_path: Path):
        """
        从YAML配置文件加载IMU标定参数

        Args:
            config_path: 标定配置文件路径

        Raises:
            KeyError: 当配置文件缺少必要字段时
        """
        # 初始化为单位旋转和零零偏（默认值）
        self.ra = Rotation.identity()
        self.rg = Rotation.identity()
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)

        # 读取YAML配置文件
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if "Calibration" in config:
                calib = config["Calibration"]
                # 加载加速度计标定旋转矩阵
                ra_arr = np.array(calib["ra"]).reshape((3, 3))
                # 加载陀螺仪标定旋转矩阵
                rg_arr = np.array(calib["rg"]).reshape((3, 3))
                self.ra = Rotation.from_matrix(ra_arr)
                self.rg = Rotation.from_matrix(rg_arr)
                # 加载加速度计零偏
                self.ba = np.array(calib["ba"])
                # 加载陀螺仪零偏
                self.bg = np.array(calib["bg"])

    def calibrate(self, imu_data: ImuData) -> ImuData:
        """
        应用标定参数校准IMU数据

        校准公式:
            acce_calibrated = Ra * acce_raw - Ba
            gyro_calibrated = Rg * gyro_raw - Bg

        Args:
            imu_data: 待校准的IMU数据对象

        Returns:
            ImuData: 校准后的IMU数据（原地修改）
        """
        imu_data.acce = self.ra.apply(imu_data.acce) - self.ba
        imu_data.gyro = self.rg.apply(imu_data.gyro) - self.bg
        return imu_data

    def __repr__(self):
        """返回标定参数的字符串表示"""
        return f"ImuCalibration(Ra={self.ra}, Rg={self.rg}, Ba={self.ba}, Bg={self.bg})"


class GroundTruthData(PosesData):
    """
    地面真值位姿数据类，继承自PosesData

    数据格式:
        CSV文件包含以下列:
        - col 0: 时间戳 [us]
        - col 1-3: 位置 [m] (x, y, z)
        - col 4-7: 四元数 [qw, qx, qy, qz]

    应用场景:
        - 存储运动捕捉系统测量的真值位姿
        - 存储SLAM/视觉里程计的真值轨迹
        - 用于算法评估和误差分析
    """

    @staticmethod
    def from_raw(raw: NDArray) -> "GroundTruthData":
        """
        从原始数组创建GroundTruthData对象

        Args:
            raw: 原始数据数组，形状为 [N, 8]
                - col 0: 时间戳 [us]
                - col 1-3: 位置 [m]
                - col 4-7: 四元数 [qw, qx, qy, qz]

        Returns:
            GroundTruthData: 地面真值数据对象
        """
        t_us = raw[:, 0]
        trans = raw[:, 1:4]
        quats = raw[:, 4:8]
        # qwxyz 格式
        rots = Rotation.from_quat(quats, scalar_first=True)
        return GroundTruthData(t_us, rots, trans)

    @staticmethod
    def from_csv(path: Path) -> "GroundTruthData":
        """
        从CSV文件加载地面真值数据

        Args:
            path: CSV文件路径

        Returns:
            GroundTruthData: 地面真值数据对象
        """
        raw = pd.read_csv(path).to_numpy()
        return GroundTruthData.from_raw(raw)


class CameraColumn:
    """
    相机数据CSV列名定义

    CSV格式:
        #timestamp [us],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],p_CS_C_x [m],p_CS_C_y [m],p_CS_C_z [m],q_CS_w [],q_CS_x [],q_CS_y [],q_CS_z [],t_system [us]

    列说明:
        - #timestamp [us]: 相机传感器时间戳
        - p_RS_R_x/y/z [m]: 参考坐标系下传感器位置
        - q_RS_w/x/y/z []: 参考坐标系下传感器姿态（四元数）
        - p_CS_C_x/y/z [m]: 相机坐标系下位置
        - q_CS_w/x/y/z []: 相机坐标系下姿态（四元数）
        - t_system [us]: 系统时间戳

    命名规则:
        - R: Reference frame (参考坐标系，如世界坐标系)
        - S: Sensor frame (传感器坐标系)
        - C: Camera frame (相机坐标系)
        - p_X_Y: X到Y的平移向量
        - q_X_Y: X到Y的旋转四元数
    """

    t = ["#timestamp [us]"]  # 相机传感器时间戳
    ps = ["p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]"]  # 传感器位置
    qs = ["q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []"]  # 传感器姿态
    pc = ["p_CS_C_x [m]", "p_CS_C_y [m]", "p_CS_C_z [m]"]  # 相机位置
    qc = ["q_CS_w []", "q_CS_x []", "q_CS_y []", "q_CS_z []"]  # 相机姿态
    t_sys = ["t_system [us]"]  # 系统时间戳

    all = t + ps + qs + pc + qc + t_sys  # 所有列名


class CameraData(PosesData):
    """
    相机位姿数据类，继承自PosesData

    存储相机的位姿序列，包含传感器时间戳和系统时间戳

    应用场景:
        - 存储SLAM算法估计的相机位姿
        - 存储视觉里程计输出
        - 相机轨迹可视化
    """

    @staticmethod
    def from_csv(path: Path) -> "CameraData":
        """
        从CSV文件加载相机位姿数据

        处理流程:
            1. 读取CSV文件
            2. 按时间戳排序
            3. 去除重复时间戳
            4. 将传感器时间戳转换为系统时间戳

        Args:
            path: 相机数据CSV文件路径

        Returns:
            CameraData: 相机位姿数据对象
        """
        # 读取CSV文件并按时间戳排序，去除重复的时间戳
        df = (
            pd.read_csv(path)
            .sort_values(by=CameraColumn.t)
            .drop_duplicates(CameraColumn.t)
        )
        # 提取传感器时间戳和系统时间戳
        t_sensor_us = df[CameraColumn.t].to_numpy().flatten()
        t_us = df[CameraColumn.t_sys].to_numpy().flatten()
        # 提取位置和旋转
        trans = df[CameraColumn.ps].to_numpy()
        rots = df[CameraColumn.qs].to_numpy()
        rots = Rotation.from_quat(rots, scalar_first=True)

        # 将传感器时间戳转换为相对时间戳
        t_us = t_sensor_us - t_sensor_us[0] + t_us[0]
        return CameraData(t_us, rots, trans)


class FusionData(PosesData):
    """
    融合位姿数据类，继承自PosesData

    存储多传感器融合后的位姿估计结果

    数据格式:
        CSV文件包含以下列:
        - col 0: 时间戳 [us] (系统时间，非UTC时间戳)
        - col 1-3: 位置 [m] (x, y, z)
        - col 4-7: 四元数 [qw, qx, qy, qz]
        - col 8-11: 相机位置 [m] (可选)
        - col 12-15: 相机四元数 (可选)

    注意:
        t_us: 此时间为系统时间，非UTC时间戳

    应用场景:
        - 存储IMU-视觉融合位姿
        - 存储多传感器SLAM输出
        - 存储滤波器融合结果
    """

    @staticmethod
    def from_raw(raw: NDArray) -> "FusionData":
        """
        从原始数组创建FusionData对象

        Args:
            raw: 原始数据数组，形状为 [N, 8+]
                - col 0: 时间戳 [us]
                - col 1-3: 位置 [m]
                - col 4-7: 四元数 [qw, qx, qy, qz]
                - col 8-11: 相机位置 [m] (可选，不使用)
                - col 12-15: 相机四元数 (可选，不使用)

        Returns:
            FusionData: 融合位姿数据对象
        """
        t_us = raw[:, 0]
        trans = raw[:, 1:4]
        quats = raw[:, 4:8]
        # cam_trans = raw[:, 8:11]  # 相机位置（未使用）
        # cam_quats = raw[:, 11:15]  # 相机四元数（未使用）
        rots = Rotation.from_quat(quats, scalar_first=True)
        return FusionData(t_us, rots, trans)

    @staticmethod
    def from_csv(path: Path) -> "FusionData":
        """
        从CSV文件加载融合位姿数据

        处理流程:
            1. 读取CSV文件
            2. 去除重复的时间戳

        Args:
            path: 融合数据CSV文件路径

        Returns:
            FusionData: 融合位姿数据对象
        """
        raw = pd.read_csv(path).drop_duplicates(CameraColumn.t).to_numpy()
        return FusionData.from_raw(raw)


@dataclass
class CalibrationData:
    """
    标定数据类，存储传感器到真值坐标系的变换

    Attributes:
        tf_sg_local (Pose): 传感器到真值的局部变换
                           tf_sg_local = T_sensor_gt (传感器坐标系到真值坐标系)
        tf_sg_global (Pose): 传感器到真值的全局变换
                            tf_sg_global = T_ref_sensor_gt (参考传感器到真值坐标系)

    应用场景:
        - 传感器坐标系转换
        - 多设备标定数据存储
        - 坐标系对齐

    JSON格式:
        [
            {
                "rot_sensor_gt": [3x3 matrix],
                "trans_sensor_gt": [3x1 vector],
                "rot_ref_sensor_gt": [3x3 matrix],
                "trans_ref_sensor_gt": [3x1 vector]
            }
        ]
    """

    tf_sg_local: Pose
    tf_sg_global: Pose

    @staticmethod
    def from_json(path: Path) -> "CalibrationData":
        """
        从JSON文件加载标定数据

        Args:
            path: JSON文件路径

        Returns:
            CalibrationData: 标定数据对象

        Raises:
            ValueError: 当JSON格式无效时
        """
        with open(path, "r") as f:
            data = json.load(f)
            # 检查JSON格式
            if not isinstance(data, list) or len(data) != 1:
                raise ValueError("Invalid JSON format")
            data = data[0]

            # 加载局部变换：传感器到真值坐标系
            rot_local = np.array(data["rot_sensor_gt"])
            trans_local = np.array(data["trans_sensor_gt"]).flatten()
            tf_sg_local = Pose(Rotation.from_matrix(rot_local), trans_local)

            # 加载全局变换：参考传感器到真值坐标系
            rot_global = np.array(data["rot_ref_sensor_gt"])
            trans_global = np.array(data["trans_ref_sensor_gt"]).flatten()
            tf_sg_global = Pose(Rotation.from_matrix(rot_global), trans_global)

            print(f"Load Calib File: {path}")

            return CalibrationData(tf_sg_local, tf_sg_global)


@dataclass
class DataCheck:
    """
    数据检查类，存储时间同步信息

    Attributes:
        t_gi_us (int): GT（Ground Truth）与IMU之间的时间差，单位：微秒
                     t_gt = t_imu + t_gi_us

    应用场景:
        - 多传感器时间同步
        - 时间戳对齐
        - 消除系统时间偏差

    JSON格式:
        {
            "check_time_diff": {
                "time_diff_21_us": 时间差值 (us)
            }
        }
    """

    t_gi_us: int

    @staticmethod
    def from_json(path: Path) -> "DataCheck":
        """
        从JSON文件加载数据检查信息

        Args:
            path: JSON文件路径

        Returns:
            DataCheck: 数据检查对象

        Raises:
            AssertionError: 当JSON格式缺少必要字段时
        """
        with open(path, "r") as f:
            data = json.load(f)
            # 检查时间差字段
            assert "check_time_diff" in data
            check_time_diff = data["check_time_diff"]
            assert "time_diff_21_us" in check_time_diff
            t_gi_us = check_time_diff["time_diff_21_us"]
            return DataCheck(t_gi_us)


class UnitData:
    """
    单元数据类，管理单个数据集单元的所有数据

    该类负责管理和加载单个实验单元的IMU数据、地面真值数据和可选数据

    Attributes:
        imu_data (ImuData): IMU传感器数据
        gt_data (PosesData): 地面真值位姿数据
        opt_data (PosesData): 可选数据（如优化后的轨迹）
        base_dir (Path): 数据集单元的基础目录
        name (str): 数据集单元名称
        device_name (str): 设备名称（从目录名解析）
        has_opt (bool): 是否存在可选数据
        using_ext (bool): 是否使用扩展数据

    目录结构:
        base_dir/
            ├── imu.csv       # IMU数据
            ├── cam.csv       # 相机数据
            ├── gt.csv        # 地面真值数据
            └── opt.csv       # 可选数据（可选）

    命名规则:
        目录名格式: YYYYMMDD_HHMMSS_DeviceName
        例如: 20260126_174027_Redmi-K30-Pro
    """

    imu_data: ImuData
    gt_data: PosesData
    opt_data: PosesData

    def __init__(self, base_dir: Path | str):
        """
        初始化单元数据对象

        Args:
            base_dir: 数据集单元的基础目录路径
        """
        self.base_dir = Path(base_dir)
        self.name = self.base_dir.name
        self._get_deivce_from_name()

        # 设置数据文件路径
        self._imu_path = self.base_dir / "imu.csv"
        self._cam_path = self.base_dir / "cam.csv"
        self._gt_path = self.base_dir / "gt.csv"
        self._opt_path = self.base_dir / "opt.csv"

        # 检查是否存在可选数据
        self.has_opt = self._opt_path.exists()
        self.using_ext = False

    def _get_deivce_from_name(self):
        """
        从目录名解析设备名称

        目录名格式: YYYYMMDD_HHMMSS_DeviceName
        提取第3部分作为设备名称

        示例:
            "20260126_174027_Redmi-K30-Pro" -> "Redmi-K30-Pro"
            "20260126_174027" -> "default"
        """
        # 设备名称
        spl = self.name.split("_")
        device_name = spl[2] if len(spl) > 2 else "default"
        self.device_name = device_name

    def load_data(self, using_opt=False):
        """
        加载单元数据

        加载流程:
            1. 加载IMU数据
            2. 加载地面真值数据
            3. 如果存在可选数据，加载可选数据
            4. 如果指定using_opt，使用可选数据作为地面真值
            5. 检查IMU和GT数据长度是否匹配

        Args:
            using_opt: 是否使用可选数据作为地面真值
        """
        # 加载IMU数据
        self.imu_data = ImuData.from_csv(self._imu_path)
        # 加载地面真值数据
        self.gt_data = GroundTruthData.from_csv(self._gt_path)
        # 如果存在可选数据，加载可选数据
        if self.has_opt:
            self.opt_data = GroundTruthData.from_csv(self._opt_path)
            # 如果指定使用可选数据，将可选数据作为地面真值
            if using_opt:
                self.gt_data = self.opt_data

        # 检查IMU和GT数据长度是否匹配
        if len(self.imu_data) != len(self.gt_data):
            print("! imu and gt data length not match")


class ExtUnitData(UnitData):
    """
    扩展单元数据类，继承自UnitData

    相比UnitData，增加了以下功能:
        - 从RTAB数据库文件生成CSV
        - 加载融合数据
        - 加载标定数据和时间检查数据
        - 自动进行数据修正（时间对齐和空间变换）

    Attributes:
        imu_data (ImuData): IMU传感器数据
        gt_data (PosesData): 地面真值位姿数据
        opt_data (PosesData): 可选数据
        fusion_data (PosesData): 融合位姿数据
        check_data (DataCheck): 数据检查信息（时间同步）
        calib_data (CalibrationData): 标定数据
        has_fusion (bool): 是否存在融合数据
        using_ext (bool): 是否使用扩展数据

    目录结构:
        base_dir/
            ├── imu.csv              # IMU数据
            ├── cam.csv              # 相机数据
            ├── gt.csv               # 地面真值数据
            ├── opt.csv              # 可选数据（可选）
            ├── fusion.csv           # 融合数据（可选）
            ├── Calibration.json     # 标定数据
            ├── DataCheck.json       # 数据检查信息
            └── *.db                 # RTAB数据库文件（可选）
    """

    imu_data: ImuData
    gt_data: PosesData
    opt_data: PosesData
    fusion_data: PosesData
    check_data: DataCheck
    calib_data: CalibrationData

    def __init__(self, base_dir: Path | str, using_ext: bool = True):
        """
        初始化扩展单元数据对象

        处理流程:
            1. 调用父类初始化
            2. 检查是否存在RTAB数据库文件
            3. 如果存在数据库，生成CSV文件
            4. 检查是否存在融合数据
            5. 如果使用扩展数据，设置标定文件路径

        Args:
            base_dir: 数据集单元的基础目录路径
            using_ext: 是否使用扩展数据（标定和时间检查），默认为True
        """
        super().__init__(base_dir)
        self.using_ext = using_ext

        # 读取真值：检查是否存在RTAB数据库文件
        db_file = RTABData.get_db_file(self.base_dir)
        if db_file is not None:
            # 如果数据库文件存在但CSV文件不存在，从数据库生成CSV
            rtab_data = RTABData(db_file)

            # 应用局部变换并保存为CSV
            rtab_data.transform_local()
            rtab_data.save_csv(self._gt_path)
            rtab_data.save_csv(self._opt_path, using_opt=True)

        # 检查融合数据文件
        self._fusion_path = self.base_dir / "fusion.csv"
        self.has_fusion = self._fusion_path.exists()

        # 如果使用扩展数据，设置标定文件路径
        if using_ext:
            self._calib_file = self.base_dir / "Calibration.json"
            self._check_file = self.base_dir / "DataCheck.json"

    def load_data(self, using_opt=False):
        """
        加载扩展单元数据

        加载流程:
            1. 调用父类加载基础数据（IMU, GT, Opt）
            2. 如果存在融合数据，加载融合数据
            3. 应用数据修正（时间对齐和空间变换）

        Args:
            using_opt: 是否使用可选数据作为地面真值
        """
        # 加载基础数据
        super().load_data(using_opt)

        # 如果存在融合数据，加载融合数据
        if self.has_fusion:
            fusion_data = FusionData.from_csv(self._fusion_path)
            self.fusion_data = fusion_data

        # 应用数据修正
        self.correct(self.gt_data, self.imu_data)

    def correct(self, gt_data: PosesData, imu_data: ImuData):
        """
        修正GT和IMU数据，进行时间对齐和空间变换

        修正流程:
            1. 加载标定数据和数据检查信息
            2. 应用时间修正（补偿时间戳偏差）
            3. 应用空间变换（坐标系转换）
            4. 数据对齐（重采样到统一时间序列）

        Args:
            gt_data: 地面真值位姿数据（将被原地修改）
            imu_data: IMU数据（将被原地修改）
        """
        # 加载标定数据和数据检查信息
        self.calib_data = CalibrationData.from_json(self._calib_file)
        self.check_data = DataCheck.from_json(self._check_file)

        # 时间修正：补偿GT和IMU之间的时间差
        gt_data.t_us += self.check_data.t_gi_us

        # 空间变换：将GT数据从传感器坐标系转换到IMU坐标系
        gt_data.transform_local(self.calib_data.tf_sg_local.inverse())

        # 以下为可选的全局变换（注释掉）
        # gt_data.transform_global(self.calib_data.tf_sg_global)
        # gt_data.transform_global(gt_data.get_pose(0).get_yaw_pose().inverse())

        # 数据对齐：重采样到统一时间序列
        # 使用IMU和GT时间戳的并集作为新时间序列
        t_new_us = get_time_series([imu_data.t_us, gt_data.t_us])
        self.imu_data = imu_data.interpolate(t_new_us)
        self.gt_data = gt_data.interpolate(t_new_us)


class DeviceDataset:
    """
    设备数据集类，管理单个设备的所有数据单元

    该类用于组织和管理属于同一设备的多个数据单元集合

    Attributes:
        base_dir (Path): 设备数据集的基础目录
        device_name (str): 设备名称（从目录名解析）
        units (list[UnitData]): 数据单元列表

    目录结构:
        device_name/              # 设备目录
            ├── unit_1/           # 数据单元1
            │   ├── imu.csv
            │   ├── gt.csv
            │   └── ...
            ├── unit_2/           # 数据单元2
            │   ├── imu.csv
            │   ├── gt.csv
            │   └── ...
            └── ...

    应用场景:
        - 管理同一设备的多次实验数据
        - 批量处理和迭代访问数据单元
        - 设备级别的数据分析

    使用示例:
        >>> dataset = DeviceDataset("/path/to/device")
        >>> for unit in dataset:
        ...     unit.load_data()
        ...     # 处理数据
        >>> first_unit = dataset[0]
        >>> print(f"Dataset has {len(dataset)} units")
    """

    def __init__(self, base_dir: Path | str):
        """
        初始化设备数据集对象

        处理流程:
            1. 设置基础目录
            2. 从目录名解析设备名称
            3. 扫描所有子目录，创建数据单元对象

        Args:
            base_dir: 设备数据集的基础目录路径
        """
        self.base_dir = Path(base_dir)
        self.device_name = self.base_dir.name
        # 扫描所有子目录，为每个子目录创建UnitData对象
        self.units = [
            UnitData(path) for path in self.base_dir.iterdir() if path.is_dir()
        ]

    def __getitem__(self, index: int) -> UnitData:
        """
        通过索引访问数据单元

        Args:
            index: 数据单元索引

        Returns:
            UnitData: 指定索引的数据单元对象
        """
        return self.units[index]

    def __iter__(self):
        """
        迭代所有数据单元

        Yields:
            UnitData: 每个数据单元对象
        """
        return iter(self.units)

    def __len__(self) -> int:
        """
        返回数据单元数量

        Returns:
            int: 数据单元总数
        """
        return len(self.units)
