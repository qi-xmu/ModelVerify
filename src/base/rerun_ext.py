"""
Rerun 扩展模块

提供用于可视化 IMU 数据、位姿数据和误差统计的 Rerun 工具类和函数。
主要功能：
- 创建和管理多种可视化视图（IMU 时序图、3D 空间视图、CDF 误差分布图）
- 发送 IMU 数据（陀螺仪、加速度计、磁力计）
- 发送位姿数据（旋转、平移、轨迹）
- 记录坐标系和变换
"""

from itertools import accumulate

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray

from .datatype import ImuData, Pose, PosesData


class RerunView:
    """
    Rerun 视图管理器

    用于创建和管理多个 Rerun 可视化视图，支持：
    - IMU 时序视图（陀螺仪、加速度计、磁力计）
    - 3D 空间视图（轨迹可视化）
    - CDF 误差分布视图（误差累积分布函数）
    """

    def __init__(self):
        """初始化视图列表"""
        self.views = []

    def add_imu_view(
        self,
        visible: bool = False,
        tags: list = [],
        range: tuple[float, float] = (-15.0, 15.0),
    ):
        """
        添加 IMU 时序视图

        创建用于显示陀螺仪、加速度计、磁力计数据的时间序列视图。

        Args:
            visible: 视图是否默认可见
            tags: 数据标签列表，为每个 tag 创建独立的 IMU 视图
            range: 时间轴范围（秒），相对于光标位置，默认显示前后 15 秒

        Returns:
            self: 支持链式调用
        """
        # 定义 XYZ 三个轴的名称和对应颜色
        # x 轴：红色，y 轴：绿色，z 轴：蓝色
        XYZ_AXIS_NAMES = ["x", "y", "z"]
        XYZ_AXIS_COLORS = [[(231, 76, 60), (39, 174, 96), (52, 120, 219)]]

        # 设置时间轴的显示范围（相对于光标的前后时间）
        time_range_start = rrb.TimeRangeBoundary.cursor_relative(seconds=range[0])
        time_range_end = rrb.TimeRangeBoundary.cursor_relative(seconds=range[1])
        # 为每个 tag 创建陀螺仪视图
        gyro_views = [
            rrb.TimeSeriesView(
                origin=f"gyroscope_{tag}",  # 数据源路径
                name=f"Gyroscope_{tag}",  # 视图显示名称
                overrides={
                    # type: ignore[arg-type]
                    f"/gyroscope_{tag}": rr.SeriesLines.from_fields(
                        names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS
                    ),
                },
                axis_x=rrb.archetypes.TimeAxis(
                    view_range=rrb.TimeRange(start=time_range_start, end=time_range_end)
                ),
            )
            for tag in tags
        ]

        # 为每个 tag 创建加速度计视图
        acce_views = [
            rrb.TimeSeriesView(
                origin=f"accelerometer_{tag}",
                name=f"Accelerometer_{tag}",
                overrides={
                    # type: ignore[arg-type]
                    f"/accelerometer_{tag}": rr.SeriesLines.from_fields(
                        names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS
                    ),
                },
                axis_x=rrb.archetypes.TimeAxis(
                    view_range=rrb.TimeRange(start=time_range_start, end=time_range_end)
                ),
            )
            for tag in tags
        ]

        # 为每个 tag 创建磁力计视图
        magn_views = [
            rrb.TimeSeriesView(
                origin=f"magnetometer_{tag}",
                name=f"Magnetometer_{tag}",
                overrides={
                    # type: ignore[arg-type]
                    f"/magnetometer_{tag}": rr.SeriesLines.from_fields(
                        names=XYZ_AXIS_NAMES, colors=XYZ_AXIS_COLORS
                    ),
                },
                axis_x=rrb.archetypes.TimeAxis(
                    view_range=rrb.TimeRange(start=time_range_start, end=time_range_end)
                ),
            )
            for tag in tags
        ]

        # 将所有 IMU 视图垂直排列组合成一个视图
        imu_view = rrb.Vertical(*gyro_views, *acce_views, *magn_views, visible=visible)

        self.views.append(imu_view)
        return self

    def add_spatial_view(
        self,
        eye_at_entity: str = "/world/Groundtruth",
    ):
        """
        添加 3D 空间视图

        创建用于可视化 3D 轨迹和位姿的空间视图。

        Args:
            eye_at_entity: 相机跟踪的实体路径，默认跟踪真值轨迹

        Returns:
            self: 支持链式调用
        """
        spatial_view = rrb.Spatial3DView(
            origin="/",  # 从根路径开始
            name="World position",  # 视图名称
            spatial_information=rrb.SpatialInformation(show_axes=True),  # 显示坐标轴
            eye_controls=rrb.EyeControls3D(
                kind="Orbital",  # 轨道式相机控制
                tracking_entity=eye_at_entity  # 跟踪指定的实体
            ),
        )
        self.views.append(spatial_view)
        return self

    def add_cdf_view(
        self,
        visible: bool = True,
        tags: list = [],
        error_types: list = ["ATE", "RTE"],
        range: tuple[float, float] = (-15.0, 15.0),
    ):
        """
        添加 CDF（累积分布函数）视图，用于显示误差的累积分布函数

        用于分析不同误差类型（绝对位姿误差、相对位姿误差等）的统计分布特性。

        Args:
            visible: 视图是否默认可见
            tags: 数据标签列表，为每个 tag 创建独立的视图
            error_types: 误差类型列表，如 ['ATE', 'RTE', 'APE', 'RPE']
                         ATE: Absolute Trajectory Error（绝对轨迹误差）
                         RTE: Relative Trajectory Error（相对轨迹误差）
                         APE: Absolute Pose Error（绝对位姿误差）
                         RPE: Relative Pose Error（相对位姿误差）
            range: 时间轴范围（相对于光标），单位：秒

        Returns:
            self: 支持链式调用
        """
        # 为每个误差类型创建颜色映射
        ERROR_TYPE_COLORS = {
            "APE": [(231, 76, 60)],  # 红色：绝对位姿误差
            "ATE": [(39, 174, 96)],  # 绿色：绝对轨迹误差
            "RPE": [(52, 120, 219)],  # 蓝色：相对位姿误差
            "RTE": [(255, 152, 0)],  # 橙色：相对轨迹误差
        }

        # 设置时间轴的显示范围
        time_range_start = rrb.TimeRangeBoundary.cursor_relative(seconds=range[0])
        time_range_end = rrb.TimeRangeBoundary.cursor_relative(seconds=range[1])

        # 为每个 tag 和 error_type 组合创建视图
        cdf_views = []
        for tag in tags:
            for error_type in error_types:
                view_name = f"CDF_{error_type}_{tag}"
                origin = f"cdf_{error_type}_{tag}"

                # 创建时间序列视图来显示 CDF 曲线
                view = rrb.TimeSeriesView(
                    origin=origin,  # 数据源路径
                    name=view_name,  # 视图显示名称
                    overrides={
                        origin: rr.SeriesLines.from_fields(
                            names=["CDF"],  # CDF 值序列
                            colors=[
                                ERROR_TYPE_COLORS.get(error_type, [(128, 128, 128)])
                            ],
                        ),
                    },
                    axis_x=rrb.archetypes.TimeAxis(
                        view_range=rrb.TimeRange(
                            start=time_range_start, end=time_range_end
                        )
                    ),
                    visible=visible,
                )
                cdf_views.append(view)

        # 将所有 CDF 视图垂直排列组合成一个视图
        if cdf_views:
            cdf_view = rrb.Vertical(*cdf_views, visible=visible)
            self.views.append(cdf_view)

        return self

    def send(self, name: str, shares: list[float] | None = []):
        """
        发送视图蓝图到 Rerun

        初始化 Rerun 并发送所有已配置的视图布局。

        Args:
            name: Rerun 应用的名称
            shares: 各列的宽度比例，例如 [0.3, 0.7] 表示第一列占 30%，第二列占 70%

        Returns:
            self: 支持链式调用
        """
        # 初始化 Rerun 应用并启动查看器
        rr.init(name, spawn=True)
        # 将所有视图水平排列，创建布局蓝图
        blueprint = rrb.Horizontal(*self.views, column_shares=shares)
        # 发送布局蓝图
        rr.send_blueprint(blueprint)
        return self


def rerun_init(name: str):
    """
    快速初始化 Rerun，创建一个包含 3D 空间视图的默认配置

    Args:
        name: Rerun 应用的名称
    """
    RerunView().add_spatial_view().send(name)


def log_coordinate(
    entity_path: str,
    view_coordinate: rr.AsComponents = rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    length: float = 1.0,
    radii=None,
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    labels=["X", "Y", "Z"],
    show_labels=None,
):
    """
    在指定路径记录坐标系（坐标轴箭头）

    用于可视化局部坐标系的朝向，默认显示右手坐标系，Z 轴向上。

    Args:
        entity_path: 实体路径，如 "/world/Pose"
        view_coordinate: 坐标系类型，默认右手坐标系 Z 轴向上
        length: 坐标轴箭头的长度
        radii: 箭头的粗细
        colors: XYZ 轴的颜色数组，默认 [红, 绿, 蓝]
        labels: XYZ 轴的标签
        show_labels: 是否显示轴标签
    """
    rr.log(
        entity_path,
        view_coordinate,
        rr.Arrows3D(
            vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * length,  # XYZ 方向的向量
            radii=radii,
            colors=colors,
            labels=labels,
            show_labels=show_labels,
        ),
        static=True,  # 静态实体，不随时间变化
    )


def set_world_tf(tf: Pose):
    """
    设置世界坐标系的变换

    在 /world 路径记录一个静态的 3D 变换，用于定义世界坐标系。

    Args:
        tf: 位姿对象，包含旋转矩阵和平移向量
    """
    rr.log(
        "/world",
        rr.Transform3D(mat3x3=tf.rot.as_matrix(), translation=tf.p),
        static=True,
    )


def send_imu_data(imu_data: ImuData, tag: str = ""):
    """
    发送 IMU 数据到 Rerun

    将陀螺仪、加速度计、磁力计数据发送到 Rerun 进行可视化。

    Args:
        imu_data: IMU 数据对象，包含时间戳和传感器数据
        tag: 数据标签，用于区分不同来源的 IMU 数据
    """
    # 创建时间列，将微秒转换为秒
    ts_imu = rr.TimeColumn("timestamp", timestamp=imu_data.t_us * 1e-6)

    # 发送陀螺仪数据
    rr.send_columns(
        f"/gyroscope_{tag}",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.gyro),
    )

    # 发送加速度计数据
    rr.send_columns(
        f"/accelerometer_{tag}",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.acce),
    )

    # 发送磁力计数据
    rr.send_columns(
        f"/magnetometer_{tag}",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.magn),
    )


def send_pose_data(poses_data: PosesData, tag: str = "Pose", color=None):
    """
    发送位姿数据到 Rerun（静态轨迹）

    将位姿序列发送到 Rerun 进行可视化，包括坐标系和完整轨迹。

    Args:
        poses_data: 位姿数据对象，包含时间戳、位置和旋转
        tag: 位姿标签，用于区分不同来源的位姿数据
        color: 轨迹颜色，格式为 [R, G, B]，范围 0-255
    """
    # 创建时间列，将微秒转换为秒
    times = rr.TimeColumn("timestamp", timestamp=poses_data.t_us * 1e-6)
    ps = poses_data.ps  # 位置数据
    qs = poses_data.rots.as_quat(canonical=True)  # 旋转四元数（标准形式）

    # 在位姿位置记录坐标系
    log_coordinate(f"/world/{tag}", length=1, labels=[tag], show_labels=False)

    # 发送位姿变换数据（随时间变化的位姿）
    rr.send_columns(
        f"/world/{tag}",
        indexes=[times],
        columns=rr.Transform3D.columns(quaternion=qs, translation=ps),
    )

    # 绘制完整轨迹（静态）
    colors = [color] if color is not None else None
    rr.log(
        f"/world/{tag}_path",
        rr.LineStrips3D(strips=[ps], labels=[tag], colors=colors, show_labels=True),
        static=True,
    )


def send_pose_data_dyn(poses_data: PosesData, tag: str = "Pose", color=None):
    """
    发送位姿数据到 Rerun（动态轨迹）

    将位姿序列发送到 Rerun 进行可视化，包括坐标系和随时间增长的轨迹。
    与 send_pose_data 的区别在于轨迹是动态绘制的，会随着时间推移逐步显示。

    Args:
        poses_data: 位姿数据对象，包含时间戳、位置和旋转
        tag: 位姿标签，用于区分不同来源的位姿数据
        color: 轨迹颜色，格式为 [R, G, B]，范围 0-255
    """
    # 创建时间列，将微秒转换为秒
    times = rr.TimeColumn("timestamp", timestamp=poses_data.t_us * 1e-6)
    ps = poses_data.ps  # 位置数据
    qs = poses_data.rots.as_quat(canonical=True)  # 旋转四元数（标准形式）

    # 在位姿位置记录坐标系
    log_coordinate(f"/world/{tag}", length=1, labels=[tag], show_labels=False)

    # 发送位姿变换数据（随时间变化的位姿）
    rr.send_columns(
        f"/world/{tag}",
        indexes=[times],
        columns=rr.Transform3D.columns(quaternion=qs, translation=ps),
    )

    def _gen_path(ps: np.ndarray | list[np.ndarray]):
        """
        生成逐步增长的轨迹序列

        例如：如果有 5 个位置点，会生成：
        [p0], [p0, p1], [p0, p1, p2], [p0, p1, p2, p3], [p0, p1, p2, p3, p4]

        Args:
            ps: 位置点数组

        Returns:
            逐步增长的轨迹列表
        """
        paths = list(accumulate(ps, lambda acc, x: acc + [x], initial=[]))
        return paths[1:]

    # 生成动态轨迹（每个时间点显示从起点到当前点的轨迹）
    paths = _gen_path(ps)
    tags = [tag] * len(paths) if tag is not None else None
    colors = [np.array([color])] * len(paths) if color is not None else None

    # 发送动态轨迹数据
    rr.send_columns(
        f"/world/{tag}_path",
        indexes=[times],
        columns=rr.LineStrips3D.columns(strips=paths, labels=tags, colors=colors),
    )


def log_network_pose(
    t_us: int, pose: Pose, path: list[NDArray] | None = None, tag: str = ""
):
    """
    记录单个网络输出的位姿（适用于实时推理场景）

    在指定时间戳记录一个位姿，通常用于神经网络实时输出的位姿可视化。

    Args:
        t_us: 时间戳（微秒）
        pose: 位姿对象，包含旋转和平移
        path: 历史路径点列表，可选。如果提供，会绘制完整轨迹
        tag: 位姿标签，用于区分不同来源的位姿数据
    """
    # 设置当前时间戳（将微秒转换为秒）
    rr.set_time("timestamp", timestamp=t_us / 1e6)

    # 记录当前位姿的 3D 变换
    rr.log(
        f"/world/{tag}",
        rr.Transform3D(mat3x3=pose.rot.as_matrix(), translation=pose.p),
    )

    # 如果提供了路径，绘制轨迹
    if path is not None:
        rr.log(
            f"/world/{tag}_path",
            rr.LineStrips3D(
                strips=[path],  # 轨迹线段
                labels=[f"{tag}"],
                colors=[[192, 72, 192]],  # 紫色轨迹
                show_labels=True,
            ),
        )
