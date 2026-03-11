"""
从 SQLite 数据库导出 IMU 和 GT 数据为 CSV 格式

用法:
    uv run python converters/GetCSVFromDB.py <db_path> [--output_dir <output_dir>]

示例:
    uv run python converters/GetCSVFromDB.py /path/to/xxx.db
    uv run python converters/GetCSVFromDB.py /path/to/xxx.db --output_dir /path/to/output
"""

import argparse
from pathlib import Path

from base.serialize import ImuDataSerializer, PosesDataSerializer
from base.types.navio_db import NaVIODB


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

    args = parser.parse_args()

    if not args.db_path.exists():
        print(f"Error: Database file not found: {args.db_path}")
        return 1

    output_dir = args.output_dir or args.db_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    ndb = NaVIODB(args.db_path)
    imu_data = ndb.get_imu_data()
    ImuDataSerializer(imu_data).save(output_dir / "imu.csv")

    cam_data = ndb.get_pose_data()
    if cam_data:
        PosesDataSerializer(cam_data).save(output_dir / "cam.csv")

    try:
        tracking_result = ndb.get_track_result()
        if tracking_result:
            PosesDataSerializer(tracking_result).save(output_dir / "result.csv")
    except Exception as e:
        print(f"Warning: Failed to export track result: {e}")

    return 0


if __name__ == "__main__":
    exit(main())
