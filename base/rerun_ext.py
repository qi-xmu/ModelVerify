import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from numpy.typing import NDArray

from .datatype import ImuData, Pose, PosesData


def log_coordinate(
    entity_path: str,
    view_coordinate: rr.AsComponents = rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    length: float = 1.0,
    radii=None,
    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    labels=["X", "Y", "Z"],
    show_labels=None,
):
    rr.log(
        entity_path,
        view_coordinate,
        rr.Arrows3D(
            vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * length,
            radii=radii,
            colors=colors,
            labels=labels,
            show_labels=show_labels,
        ),
        static=True,
    )


def rerun_init(
    name: str,
    imu_view_visible: bool = False,
    imu_view_tags: list = [""],
    imu_view_range: tuple[float, float] = (-15.0, 15.0),
):
    XYZ_AXIS_NAMES = ["x", "y", "z"]
    XYZ_AXIS_COLORS = [[(231, 76, 60), (39, 174, 96), (52, 120, 219)]]

    assert len(imu_view_tags) > 0

    time_range_start = rrb.TimeRangeBoundary.cursor_relative(seconds=imu_view_range[0])
    time_range_end = rrb.TimeRangeBoundary.cursor_relative(seconds=imu_view_range[1])

    gyro_views = [
        rrb.TimeSeriesView(
            origin=f"gyroscope_{tag}",
            name=f"Gyroscope_{tag}",
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
        for tag in imu_view_tags
    ]
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
        for tag in imu_view_tags
    ]

    imu_views = gyro_views + acce_views

    rr.init(name, spawn=True)
    blueprint = rrb.Horizontal(
        rrb.Vertical(*imu_views, visible=imu_view_visible),
        rrb.Spatial3DView(
            origin="/",
            name="World position",
            spatial_information=rrb.SpatialInformation(show_axes=True),
            eye_controls=rrb.EyeControls3D(
                kind="Orbital", tracking_entity="/world/groundtruth"
            ),
        ),
        column_shares=[0.40, 0.60],
    )
    rr.send_blueprint(blueprint)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def set_world_tf(tf: Pose):
    rr.log(
        "/world",
        rr.Transform3D(mat3x3=tf.rot.as_matrix(), translation=tf.p),
        static=True,
    )


def send_imu_data(imu_data: ImuData, tag: str = ""):
    ts_imu = rr.TimeColumn("timestamp", timestamp=imu_data.t_us * 1e-6)
    rr.send_columns(
        f"/gyroscope_{tag}",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.gyro),
    )
    rr.send_columns(
        f"/accelerometer_{tag}",
        indexes=[ts_imu],
        columns=rr.Scalars.columns(scalars=imu_data.acce),
    )


def send_pose_data(poses_data: PosesData, tag: str = "Pose", color=None):
    times = rr.TimeColumn("timestamp", timestamp=poses_data.t_us * 1e-6)
    ps = poses_data.trans
    qs = poses_data.rots.as_quat(canonical=True)

    log_coordinate(f"/world/{tag}", length=1, labels=[tag], show_labels=False)
    rr.send_columns(
        f"/world/{tag}",
        indexes=[times],
        columns=rr.Transform3D.columns(quaternion=qs, translation=ps),
    )

    colors = [color] if color is not None else None
    rr.log(
        f"/world/{tag}_path",
        rr.LineStrips3D(strips=[ps], labels=[tag], colors=colors),
        static=True,
    )


def log_network_pose(
    t_us: int, pose: Pose, path: list[NDArray] | None = None, tag: str = ""
):
    rr.set_time("timestamp", timestamp=t_us / 1e6)
    rr.log(
        f"/world/{tag}",
        rr.Transform3D(mat3x3=pose.rot.as_matrix(), translation=pose.p),
    )
    if path is not None:
        rr.log(
            f"/world/{tag}_path",
            rr.LineStrips3D(strips=[path], labels=[f"{tag}"]),
        )
