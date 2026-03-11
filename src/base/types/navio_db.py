import sqlite3

import numpy as np

from base.datatype import CameraData, ImuData
from base.rtab import Path


class NaVIODB:
    def __init__(self, db_path: Path):  # noqa: F821
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

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

    def get_pose_data(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT timestamp_ns, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z,
                   p_cs_c_x, p_cs_c_y, p_cs_c_z, q_cs_w, q_cs_x, q_cs_y, q_cs_z,
                   t_system_ns, tracking_state
            FROM poses
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

        _track_state = data[:16]

        return CameraData.from_raw(data)

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
