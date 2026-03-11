"""
从 SQLite 数据库导出 IMU 和 GT 数据为 CSV 格式

用法:
    uv run python converters/GetCSVFromDB.py <db_path> [--output_dir <output_dir>]

示例:
    uv run python converters/GetCSVFromDB.py /path/to/xxx.db
    uv run python converters/GetCSVFromDB.py /path/to/xxx.db --output_dir /path/to/output
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd



def export_imu_from_db(db_path: Path, output_path: Path):
    """从数据库导出 IMU 数据为 CSV

    来源: imu_data 表
    输出: imu.csv
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

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
    conn.close()

    if not rows:
        print("Warning: No IMU data found in database")
        return

    data = np.array(rows)

    # ns 转 us
    t_ns = data[:, 0].astype(np.uint64)
    t_us = t_ns // 1000

    # 创建 DataFrame (兼容原有 CSV 格式)
    df = pd.DataFrame(
        {
            "timestamp_us_offset": t_us - t_us[0],
            "gyro_x": data[:, 1],
            "gyro_y": data[:, 2],
            "gyro_z": data[:, 3],
            "acce_x": data[:, 4],
            "acce_y": data[:, 5],
            "acce_z": data[:, 6],
            "qw": data[:, 7],
            "qx": data[:, 8],
            "qy": data[:, 9],
            "qz": data[:, 10],
            "timestamp_us": t_us,
            "magn_x": data[:, 12],
            "magn_y": data[:, 13],
            "magn_z": data[:, 14],
        }
    )

    df.to_csv(output_path, index=False)
    print(f"IMU data saved to: {output_path}")
    print(f"  - Records: {len(df)}")


def export_cam_from_db(db_path: Path, output_path: Path):
    """从数据库导出 Camera Pose 数据为 CSV

    来源: poses 表 (IMU Pose)
    输出: cam.csv
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT timestamp_ns, p_RS_R_x, p_RS_R_y, p_RS_R_z,
               q_RS_w, q_RS_x, q_RS_y, q_RS_z
        FROM poses
        ORDER BY timestamp_ns
        """
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("Warning: No pose data found in database (poses table is empty)")
        return

    data = np.array(rows)

    # ns 转 us
    t_ns = data[:, 0].astype(np.uint64)
    t_us = t_ns // 1000

    # 创建 DataFrame
    df = pd.DataFrame(
        {
            "timestamp_us": t_us,
            "tx": data[:, 1],
            "ty": data[:, 2],
            "tz": data[:, 3],
            "qw": data[:, 4],
            "qx": data[:, 5],
            "qy": data[:, 6],
            "qz": data[:, 7],
        }
    )

    df.to_csv(output_path, index=False)
    print(f"Camera pose data saved to: {output_path}")
    print(f"  - Records: {len(df)}")


def export_fusion_from_db(db_path: Path, output_path: Path):
    """从数据库导出 Fusion 轨迹数据为 CSV

    来源: track_result 表 (用户轨迹)
    输出: fusion.csv
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT timestamp_us, p_RS_R_x, p_RS_R_y, p_RS_R_z,
               q_RS_w, q_RS_x, q_RS_y, q_RS_z
        FROM track_result
        ORDER BY timestamp_us
        """
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("Warning: No track result data found in database")
        return

    data = np.array(rows)

    # 时间戳已经是 us
    t_us = data[:, 0].astype(np.uint64)

    # 创建 DataFrame
    df = pd.DataFrame(
        {
            "timestamp_us": t_us,
            "tx": data[:, 1],
            "ty": data[:, 2],
            "tz": data[:, 3],
            "qw": data[:, 4],
            "qx": data[:, 5],
            "qy": data[:, 6],
            "qz": data[:, 7],
        }
    )

    df.to_csv(output_path, index=False)
    print(f"Fusion track data saved to: {output_path}")
    print(f"  - Records: {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export IMU and GT data from SQLite database to CSV"
    )
    parser.add_argument("db_path", type=Path, help="Path to the database file (.db)")
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory (default: same as db file)",
    )
    parser.add_argument(
        "--imu_only", action="store_true", help="Only export IMU data (imu.csv)"
    )
    parser.add_argument(
        "--cam_only", action="store_true", help="Only export camera pose (cam.csv)"
    )
    parser.add_argument(
        "--fusion_only",
        action="store_true",
        help="Only export fusion track (fusion.csv)",
    )

    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: Database file not found: {args.db_path}")
        return 1

    output_dir = args.output_dir or args.db_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 文件命名规则
    if args.output_dir:
        db_name = args.db_path.stem
        imu_filename = f"{db_name}_imu.csv"
        cam_filename = f"{db_name}_cam.csv"
        fusion_filename = f"{db_name}_fusion.csv"
    else:
        imu_filename = "imu.csv"
        cam_filename = "cam.csv"
        fusion_filename = "fusion.csv"

    # 导出各数据表
    if not args.cam_only and not args.fusion_only:
        imu_path = output_dir / imu_filename
        export_imu_from_db(args.db_path, imu_path)

    if not args.imu_only and not args.fusion_only:
        cam_path = output_dir / cam_filename
        export_cam_from_db(args.db_path, cam_path)

    if not args.imu_only and not args.cam_only:
        fusion_path = output_dir / fusion_filename
        export_fusion_from_db(args.db_path, fusion_path)

    return 0


if __name__ == "__main__":
    exit(main())
