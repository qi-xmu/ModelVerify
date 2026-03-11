# Dataset Database Schema

---

## Tables

### `pointcloud_points`

| Column              | Type    | Constraint  | Description                      |
| ------------------- | ------- | ----------- | -------------------------------- |
| `id`                | INTEGER | PRIMARY KEY | ARCore point ID                  |
| `x`                 | REAL    | NOT NULL    | X coordinate (meters)            |
| `y`                 | REAL    | NOT NULL    | Y coordinate (meters)            |
| `z`                 | REAL    | NOT NULL    | Z coordinate (meters)            |
| `confidence`        | REAL    | NOT NULL    | Confidence (0~1)                 |
| `first_seen`        | INTEGER | NOT NULL    | First observation timestamp (ns) |
| `last_seen`         | INTEGER | NOT NULL    | Last observation timestamp (ns)  |
| `observation_count` | INTEGER | DEFAULT 0   | Number of observations           |

### `pointcloud_observations`

| Column         | Type    | Constraint                | Description                       |
| -------------- | ------- | ------------------------- | --------------------------------- |
| `id`           | INTEGER | PRIMARY KEY AUTOINCREMENT | Row ID                            |
| `timestamp_ns` | INTEGER | NOT NULL                  | Observation timestamp (ns)        |
| `point_id`     | INTEGER | NOT NULL, FOREIGN KEY     | References `pointcloud_points.id` |

**Indexes:**

- `idx_obs_timestamp` on `timestamp_ns`
- `idx_obs_point_id` on `point_id`

### `poses`

Combined IMU and Camera pose data table.

| Column           | Type    | Constraint                | Description                                         |
| ---------------- | ------- | ------------------------- | --------------------------------------------------- |
| `id`             | INTEGER | PRIMARY KEY AUTOINCREMENT | Row ID                                              |
| `timestamp_ns`   | INTEGER | NOT NULL                  | Timestamp (ns)                                      |
| `p_RS_R_x`       | REAL    | NOT NULL                  | IMU position X (meters)                             |
| `p_RS_R_y`       | REAL    | NOT NULL                  | IMU position Y (meters)                             |
| `p_RS_R_z`       | REAL    | NOT NULL                  | IMU position Z (meters)                             |
| `q_RS_w`         | REAL    | NOT NULL                  | IMU quaternion W                                    |
| `q_RS_x`         | REAL    | NOT NULL                  | IMU quaternion X                                    |
| `q_RS_y`         | REAL    | NOT NULL                  | IMU quaternion Y                                    |
| `q_RS_z`         | REAL    | NOT NULL                  | IMU quaternion Z                                    |
| `p_cs_c_x`       | REAL    | NULL                      | Camera position X (meters)                          |
| `p_cs_c_y`       | REAL    | NULL                      | Camera position Y (meters)                          |
| `p_cs_c_z`       | REAL    | NULL                      | Camera position Z (meters)                          |
| `q_cs_w`         | REAL    | NULL                      | Camera quaternion W                                 |
| `q_cs_x`         | REAL    | NULL                      | Camera quaternion X                                 |
| `q_cs_y`         | REAL    | NULL                      | Camera quaternion Y                                 |
| `q_cs_z`         | REAL    | NULL                      | Camera quaternion Z                                 |
| `t_system_ns`    | INTEGER | NOT NULL                  | UTC timestamp (ns)                                  |
| `tracking_state` | INTEGER | NOT NULL DEFAULT 0        | AR tracking state (0=paused, 1=tracking, 2=stopped) |

**Index:**

- `idx_poses_timestamp` on `timestamp_ns`

### `imu_data`

IMU sensor data table matching CSV format.

| Column         | Type    | Constraint                | Description                |
| -------------- | ------- | ------------------------- | -------------------------- |
| `id`           | INTEGER | PRIMARY KEY AUTOINCREMENT | Row ID                     |
| `timestamp_ns` | INTEGER | NOT NULL                  | Timestamp (ns)             |
| `w_RS_s_x`     | REAL    | NOT NULL                  | Angular velocity X (rad/s) |
| `w_RS_s_y`     | REAL    | NOT NULL                  | Angular velocity Y (rad/s) |
| `w_RS_s_z`     | REAL    | NOT NULL                  | Angular velocity Z (rad/s) |
| `a_RS_s_x`     | REAL    | NOT NULL                  | Acceleration X (m/s²)      |
| `a_RS_s_y`     | REAL    | NOT NULL                  | Acceleration Y (m/s²)      |
| `a_RS_s_z`     | REAL    | NOT NULL                  | Acceleration Z (m/s²)      |
| `q_RS_w`       | REAL    | NOT NULL                  | Rotation quaternion W      |
| `q_RS_x`       | REAL    | NOT NULL                  | Rotation quaternion X      |
| `q_RS_y`       | REAL    | NOT NULL                  | Rotation quaternion Y      |
| `q_RS_z`       | REAL    | NOT NULL                  | Rotation quaternion Z      |
| `t_system_ns`  | INTEGER | NOT NULL                  | UTC timestamp (ns)         |
| `m_RS_s_x`     | REAL    | NULL                      | Magnetic field X (µT)      |
| `m_RS_s_y`     | REAL    | NULL                      | Magnetic field Y (µT)      |
| `m_RS_s_z`     | REAL    | NULL                      | Magnetic field Z (µT)      |

**Index:**

- `idx_imu_data_timestamp` on `timestamp_ns`

### `track_result`

Tracking result data table (microsecond timestamps).

| Column         | Type    | Constraint                | Description                  |
| -------------- | ------- | ------------------------- | ---------------------------- |
| `id`           | INTEGER | PRIMARY KEY AUTOINCREMENT | Row ID                       |
| `timestamp_us` | INTEGER | NOT NULL                  | Timestamp (microseconds)     |
| `p_RS_R_x`     | REAL    | NOT NULL                  | IMU position X (meters)      |
| `p_RS_R_y`     | REAL    | NOT NULL                  | IMU position Y (meters)      |
| `p_RS_R_z`     | REAL    | NOT NULL                  | IMU position Z (meters)      |
| `q_RS_w`       | REAL    | NOT NULL                  | IMU quaternion W             |
| `q_RS_x`       | REAL    | NOT NULL                  | IMU quaternion X             |
| `q_RS_y`       | REAL    | NOT NULL                  | IMU quaternion Y             |
| `q_RS_z`       | REAL    | NOT NULL                  | IMU quaternion Z             |
| `p_cs_c_x`     | REAL    | NULL                      | Camera position X (meters)   |
| `p_cs_c_y`     | REAL    | NULL                      | Camera position Y (meters)   |
| `p_cs_c_z`     | REAL    | NULL                      | Camera position Z (meters)   |
| `q_cs_w`       | REAL    | NULL                      | Camera quaternion W          |
| `q_cs_x`       | REAL    | NULL                      | Camera quaternion X          |
| `q_cs_y`       | REAL    | NULL                      | Camera quaternion Y          |
| `q_cs_z`       | REAL    | NULL                      | Camera quaternion Z          |
| `t_system_us`  | INTEGER | NOT NULL                  | UTC timestamp (microseconds) |

**Index:**

- `idx_track_result_timestamp` on `timestamp_us`

### `statistics`

Dataset statistics table recording metadata for each recording session.

| Column              | Type    | Constraint                | Description               |
| ------------------- | ------- | ------------------------- | ------------------------- |
| `id`                | INTEGER | PRIMARY KEY AUTOINCREMENT | Row ID                    |
| `imu_count`         | INTEGER | NOT NULL DEFAULT 0        | Total IMU records         |
| `pose_count`        | INTEGER | NOT NULL DEFAULT 0        | Total pose records        |
| `point_cloud_count` | INTEGER | NOT NULL DEFAULT 0        | Total point cloud frames  |
| `observation_count` | INTEGER | NOT NULL DEFAULT 0        | Total point observations  |
| `start_time_ns`     | INTEGER | NOT NULL                  | Recording start time (ns) |
| `end_time_ns`       | INTEGER | NOT NULL                  | Recording end time (ns)   |
| `duration_ms`       | INTEGER | NOT NULL                  | Recording duration (ms)   |

**Index:**

- `idx_stats_start_time` on `start_time_ns`

---

## Database Version

**Current Version:** 6

---

## Batch Write Configuration

| Data Type        | Batch Size  | Flush Interval |
| ---------------- | ----------- | -------------- |
| IMU data         | 100 records | 500 ms         |
| Pose data        | 5 records   | 100 ms         |
| Point cloud data | 5 records   | 100 ms         |
| Track result     | 5 records   | 100 ms         |

---

## ER Diagram

```
                    ┌─────────────────────┐
                    │     statistics      │
                    │  (session metadata) │
                    └─────────────────────┘

┌──────────────────┐
│ pointcloud_points│◄──────┐
│   (PK: id)       │       │
└──────────────────┘       │
        │                  │
        │ pointcloud_observations
        │ (FK: point_id)  │
        │                  │
        ▼                  │
┌──────────────────┐       │
│      poses       │       │
│  (IMU + Camera)  │       │
│  + tracking_state│       │
└──────────────────┘       │
        │                  │
        │ timestamp_ns     │
        │                  │
        ▼                  │
┌──────────────────┐       │
│    imu_data      │       │
│  (200Hz sensors) │       │
└──────────────────┘       │
                           │
┌──────────────────┐       │
│   track_result   │       │
│  (µs timestamps) │       │
└──────────────────┘       │
                           │
```
