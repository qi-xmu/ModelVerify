import sqlite3
from pathlib import Path

import numpy as np

from base.datatype import CameraData, ImuData
from base.types.type_pointcloud import Point, PointCloudFrame


class NaVIODB:
    def __init__(self, db_path: Path):  # noqa: F821
        self.name = db_path.stem
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ─── IMU ───────────────────────────────────────────────────────

    def get_imu_data(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT timestamp_ns, w_RS_s_x, w_RS_s_y, w_RS_s_z,
                   a_RS_s_x, a_RS_s_y, a_RS_s_z,
                   q_RS_w, q_RS_x, q_RS_y, q_RS_z,
                   t_system_ns, m_RS_s_x, m_RS_s_y, m_RS_s_z
            FROM imu_data
            ORDER BY timestamp_ns
            """
        )
        rows = cursor.fetchall()

        if not rows:
            print("Warning: No IMU data found in database")

        data = np.array(rows)
        # ns to us
        data[:, 0] = data[:, 0] / 1000
        data[:, 11] = data[:, 11] / 1000

        return ImuData.from_raw(data)

    def read_imu_data_in_range(self, start_time_ns: int, end_time_ns: int):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT timestamp_ns, w_RS_s_x, w_RS_s_y, w_RS_s_z,
                   a_RS_s_x, a_RS_s_y, a_RS_s_z,
                   q_RS_w, q_RS_x, q_RS_y, q_RS_z,
                   t_system_ns, m_RS_s_x, m_RS_s_y, m_RS_s_z
            FROM imu_data
            WHERE timestamp_ns >= ? AND timestamp_ns <= ?
            ORDER BY timestamp_ns
            """,
            (start_time_ns, end_time_ns),
        )
        rows = cursor.fetchall()

        if not rows:
            return None

        data = np.array(rows)
        data[:, 0] = data[:, 0] / 1000
        data[:, 11] = data[:, 11] / 1000

        return ImuData.from_raw(data)

    def read_imu_at(self, timestamp_ns: int):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT timestamp_ns, w_RS_s_x, w_RS_s_y, w_RS_s_z,
                   a_RS_s_x, a_RS_s_y, a_RS_s_z,
                   q_RS_w, q_RS_x, q_RS_y, q_RS_z,
                   t_system_ns, m_RS_s_x, m_RS_s_y, m_RS_s_z
            FROM imu_data
            WHERE timestamp_ns = ?
            """,
            (timestamp_ns,),
        )
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"IMU data not found at timestamp: {timestamp_ns}")

        data = np.array([row])
        data[:, 0] = data[:, 0] / 1000
        data[:, 11] = data[:, 11] / 1000

        return ImuData.from_raw(data)

    def get_imu_timestamps(self) -> list[int]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DISTINCT timestamp_ns FROM imu_data ORDER BY timestamp_ns"
        )
        return [row[0] for row in cursor.fetchall()]

    # ─── Pose / Camera ────────────────────────────────────────────

    def get_pose_data(self):
        """获取传感器姿态、相机姿态、系统时间戳和跟踪状态"""

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT timestamp_ns, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z,
                   p_cs_c_x, p_cs_c_y, p_cs_c_z, q_cs_w, q_cs_x, q_cs_y, q_cs_z,
                   t_system_ns, tracking_state
            FROM poses
            WHERE p_RS_R_x IS NOT NULL
            ORDER BY timestamp_ns
            """
        )
        rows = cursor.fetchall()

        if not rows:
            print("Warning: No pose data found in database (poses table is empty)")
            return None

        data = np.array(rows)
        # ns to us
        data[:, 0] = data[:, 0] / 1000
        data[:, 15] = data[:, 15] / 1000

        # 检查0列是否包含重复行
        _, unique_indices = np.unique(data[:, 0], return_index=True)
        data = data[np.sort(unique_indices)]

        _track_state = data[:16]

        return CameraData.from_raw(data)

    def read_cam_data_in_range(self, start_time_ns: int, end_time_ns: int):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT timestamp_ns, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z,
                   p_cs_c_x, p_cs_c_y, p_cs_c_z, q_cs_w, q_cs_x, q_cs_y, q_cs_z,
                   t_system_ns, tracking_state
            FROM poses
            WHERE timestamp_ns >= ? AND timestamp_ns <= ?
            AND p_RS_R_x IS NOT NULL
            ORDER BY timestamp_ns
            """,
            (start_time_ns, end_time_ns),
        )
        rows = cursor.fetchall()

        if not rows:
            return None

        data = np.array(rows)
        data[:, 0] = data[:, 0] / 1000
        data[:, 15] = data[:, 15] / 1000

        return CameraData.from_raw(data)

    def read_cam_at(self, timestamp_ns: int):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT timestamp_ns, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z,
                   p_cs_c_x, p_cs_c_y, p_cs_c_z, q_cs_w, q_cs_x, q_cs_y, q_cs_z,
                   t_system_ns, tracking_state
            FROM poses
            WHERE timestamp_ns = ?
            AND p_RS_R_x IS NOT NULL
            """,
            (timestamp_ns,),
        )
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"Camera data not found at timestamp: {timestamp_ns}")

        data = np.array([row])
        data[:, 0] = data[:, 0] / 1000
        data[:, 15] = data[:, 15] / 1000

        return CameraData.from_raw(data)

    def get_cam_timestamps(self) -> list[int]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT timestamp_ns FROM poses
            WHERE p_RS_R_x IS NOT NULL
            ORDER BY timestamp_ns
            """
        )
        return [row[0] for row in cursor.fetchall()]

    # ─── Point Cloud ──────────────────────────────────────────────

    def read_all_pointcloud_frames(self) -> list[PointCloudFrame]:
        timestamps = self.get_pointcloud_timestamps()
        frames = []
        for ts_ns in timestamps:
            try:
                frame = self.read_pointcloud_frame_at(ts_ns)
                if frame.points:
                    frames.append(frame)
            except Exception as e:
                print(f"Error reading frame at {ts_ns}: {e}")
        return frames

    def read_pointcloud_frames_in_range(
        self, start_time_ns: int, end_time_ns: int
    ) -> list[PointCloudFrame]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT timestamp_ns
            FROM pointcloud_observations
            WHERE timestamp_ns >= ? AND timestamp_ns <= ?
            ORDER BY timestamp_ns
            """,
            (int(start_time_ns), int(end_time_ns)),
        )
        timestamps = [row[0] for row in cursor.fetchall()]

        frames = []
        for ts_ns in timestamps:
            try:
                frame = self.read_pointcloud_frame_at(ts_ns)
                if frame.points:
                    frames.append(frame)
            except Exception as e:
                print(f"Error reading frame at {ts_ns}: {e}")
        return frames

    def read_pointcloud_frame_at(self, timestamp_ns: int) -> PointCloudFrame:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                p.id, p.x, p.y, p.z, p.confidence,
                p.first_seen, p.last_seen, p.observation_count
            FROM pointcloud_points p
            INNER JOIN pointcloud_observations o ON p.id = o.point_id
            WHERE o.timestamp_ns = ?
            ORDER BY p.id
            """,
            (int(timestamp_ns),),
        )

        frame = PointCloudFrame(t_us=int(timestamp_ns / 1e3))
        for row in cursor.fetchall():
            point = Point(
                id=row[0],
                position=np.array([row[1], row[2], row[3]]),
                confidence=row[4],
                first_seen=row[5] / 1e9,
                last_seen=row[6] / 1e9,
                observation_count=row[7],
            )
            frame.points.append(point)

        return frame

    def get_pointcloud_timestamps(self) -> list[int]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DISTINCT timestamp_ns FROM pointcloud_observations ORDER BY timestamp_ns"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_pointcloud_time_range(self) -> tuple[int, int] | None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT MIN(timestamp_ns), MAX(timestamp_ns)
            FROM pointcloud_observations
            """
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return row[0], row[1]

    def get_pointcloud_statistics(
        self,
    ) -> tuple[int, int, int] | None:
        """Returns (total_points, total_observations, unique_point_ids)."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_points,
                SUM(observation_count) as total_observations,
                COUNT(DISTINCT id) as unique_point_ids
            FROM pointcloud_points
            """
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return row[0], row[1] or 0, row[2]

    # ─── Track Result ─────────────────────────────────────────────

    def get_track_result(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT timestamp_ns, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z,
                   p_cs_c_x, p_cs_c_y, p_cs_c_z, q_cs_w, q_cs_x, q_cs_y, q_cs_z, t_system_ns
            FROM track_result
            ORDER BY timestamp_ns
            """
        )
        rows = cursor.fetchall()

        if not rows:
            print("Warning: No track result data found in database")
            return None

        data = np.array(rows)
        # ns to us
        data[:, 0] = data[:, 0] / 1000
        data[:, 15] = data[:, 15] / 1000

        # 时间戳已经是 us
        return CameraData.from_raw(data)
