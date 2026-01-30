# PreprocessDB.py 完整说明文档

## 概述

`PreprocessDB.py` 是一个数据预处理脚本，专门用于处理 RTAB-Map 数据库文件（.db）和 IMU 数据文件（imu.csv），将其转换为时间对齐、插值统一后的真值轨迹文件（gt.csv）和 IMU 数据文件（imu_in.csv）。

### 核心功能

1. **数据加载**：从 RTAB-Map 数据库和 IMU CSV 文件加载数据
2. **坐标系转换**：应用 RTAB Body Frame → IMU Body Frame 转换
3. **时间对齐**：使用 match21 算法计算 IMU 和 GT 的时间偏移
4. **数据插值**：将 GT 和 IMU 数据插值到统一时间轴
5. **可视化验证**：提供轨迹图、角速度对比图、Rerun 3D 可视化
6. **数据输出**：保存时间对齐、插值后的 gt.csv 和 imu_in.csv

### 支持的输入模式

- **单元目录模式**（推荐）：指定包含 imu.csv 和 .db 文件的目录
- **兼容旧模式**：直接指定 .db 文件和 imu.csv 文件路径

## 核心函数详解

### 1. 数据加载阶段

#### 1.1 RTABData 类加载
```python
rtab_data = RTABData(db_file)
```

**功能**：从 RTAB-Map 数据库加载轨迹数据

**加载的数据**：
- `node_t_us`: 关键帧时间戳（微秒）
- `node_rots`: 关键帧旋转（四元数）
- `node_ps`: 关键帧位置（米）
- `opt_t_us`: 优化轨迹时间戳（可选）
- `opt_rots`: 优化轨迹旋转（可选）
- `opt_ps`: 优化轨迹位置（可选）

#### 1.2 ImuData 类加载
```python
imu_data = ImuData.from_csv(imu_file)
```

**功能**：从 CSV 文件加载 IMU 数据

**加载的数据**：
- `t_us`: 时间戳（微秒）
- `gyr`: 角速度（rad/s），形状 (N, 3)
- `acce`: 加速度（m/s²），形状 (N, 3)
- `ahrs`: 姿态（四元数），形状 (N, 4)
- `magn`: 磁场强度（可选），形状 (N, 3)

### 2. 坐标系转换

#### 2.1 transform_local() 函数
```python
rtab_data.transform_local()
```

**功能**：应用 DefaultBodyRotation 转换

**转换公式**：
```python
DefaultBodyRotation = Rotation.from_rotvec([0, -90, 0], degrees=True)
node_rots = node_rots * DefaultBodyRotation
```

**坐标系说明**：
- **RTAB Body Frame**: RTAB-Map 采集设备（通常是相机）的载体坐标系
- **IMU Body Frame**: IMU 传感器的载体坐标系
- **转换目的**: 将采集设备的姿态转换为 IMU 传感器的姿态

### 3. 时间对齐

#### 3.1 time_sync() 函数
```python
raw_gt, t_offset_us = time_sync(rtab_data, imu_data, time_range=(0, 50))
```

**功能**：计算 IMU 和 GT 数据的时间偏移

**处理流程**：
1. 使用 RTAB 数据创建 GroundTruthData 对象
2. 调用 `match21()` 算法计算时间偏移
   - 在指定时间范围内搜索
   - 使用角速度序列进行匹配
   - 返回最优时间偏移（微秒）
3. 将时间偏移应用到 GroundTruthData
4. 重置起始时间（t_us[0] = 0）

**match21 算法原理**：
- 比较两个角速度序列的相关性
- 在时间范围内滑动搜索最佳匹配
- 返回使误差最小的时间偏移

### 4. 可视化函数

#### 4.1 draw_trajectory() - 轨迹图绘制
```python
draw_trajectory(raw_gt, output_dir / "Trajectory.png", opt_gt=opt_poses, gap_info=gap_info)
```

**功能**：绘制 2D 轨迹图

**图表元素**：
- **蓝色实线**：原始轨迹（gt.csv 数据）
- **绿色圆点**：轨迹起点
- **红色叉**：轨迹终点
- **绿色虚线**（可选）：优化轨迹对比（-dbp 参数）
- **红点**：时间空洞位置（间隔 > 1.0 秒）

#### 4.2 plot_time_diff() - 角速度对比图
```python
plot_time_diff(imu_data, raw_gt, output_dir, time_range=args.time_range)
```

**功能**：绘制角速度对比图验证时间对齐

**角速度计算**：
```python
# 角速度 = |ω₂ - ω₁| / Δt
def get_angvels(t_us, rots):
    dt_s = np.diff(t_us) * 1e-6  # 微秒转秒
    dt_s[dt_s <= 0] = 1e-6  # 避免除零
    
    # 相邻旋转的四元数差分
    angs = [
        np.linalg.norm((rots[i].inv() * rots[i+1]).as_rotvec()) / dt_s[i]
        for i in range(len(rots)-1)
    ]
    return np.array(angs), t_us[:-1]
```

**验证标准**：
- 如果时间对齐正确，IMU 和 GT 角速度曲线应该重合
- 如果存在时间偏移，两条曲线会有相位差

#### 4.3 rerun_verify_coordinate_transform() - Rerun 3D 验证
```python
rerun_verify_coordinate_transform(imu_data, raw_gt)
```

**功能**：使用 Rerun 验证坐标系转换正确性

**验证原理**：
1. 将原始 IMU 数据发送到 Rerun（标记为 "body"）
2. 使用 gt_data.rots 将 IMU 数据转换到 global frame
3. 将转换后的数据发送到 Rerun（标记为 "global"）
4. 验证重力只出现在 Z 轴上（约 9.8 m/s²）

**坐标系验证标准**：
- **静态时**：
  - `global.acce[2]` ≈ 9.8 m/s²（重力）
  - `global.acce[0]` ≈ 0，`global.acce[1]` ≈ 0
- **如果重力出现在其他轴**：说明坐标系转换有问题

### 5. 数据插值

#### 5.1 get_time_series() - 生成统一时间序列
```python
t_new_us = get_time_series([imu_data.t_us, raw_gt.t_us])
```

**功能**：取 IMU 和 GT 时间戳的并集作为统一时间序列

#### 5.2 interpolate() - 插值到统一时间轴
```python
raw_gt = raw_gt.interpolate(t_new_us)
imu_data = imu_data.interpolate(t_new_us)
```

**插值方法**：
- **时间戳**：线性插值
- **旋转**：Slerp（球面线性插值）
- **位置**：线性插值
- **IMU 数据**：线性插值

**结果**：IMU 和 GT 都有相同数量的时间点，便于后续对齐分析

### 6. 数据输出

#### 6.1 gt.csv 输出
```python
PosesDataSerializer(raw_gt).save(gt_csv_path)
```

**格式**：`t_us,q0,q1,q2,q3,x,y,z`
- `t_us`: 时间戳（微秒）
- `q0-q3`: 四元数（旋转）
- `x,y,z`: 位置（米）

#### 6.2 imu_in.csv 输出（新增功能）
```python
ImuDataSerializer(imu_data).save(imu_in_csv_path)
```

**格式**：`t_us,gyr_x,gyr_y,gyr_z,acce_x,acce_y,acce_z,ahrs_q0,ahrs_q1,ahrs_q2,ahrs_q3`
- `t_us`: 时间戳（微秒）
- `gyr_x,gyr_y,gyr_z`: 角速度（rad/s）
- `acce_x,acce_y,acce_z`: 加速度（m/s²）
- `ahrs_q0,ahrs_q1,ahrs_q2,ahrs_q3`: 姿态四元数

## 命令行参数详解

### 必需参数（二选一）

#### 单元目录模式（推荐）
```bash
-u, --unit-dir <目录路径>
```
- 指定包含 `imu.csv` 和 `.db` 文件的目录
- 自动检测 `.db` 文件：`RTABData.get_db_file(unit_dir)`
- 默认输出到数据目录本身

#### 兼容旧模式
```bash
--db <db文件路径> --imu <imu.csv文件路径>
```
- 直接指定数据库和 IMU 文件路径
- 默认输出到 imu.csv 的父目录

### 可选参数

```bash
--output <目录路径>
    指定输出目录（可选）
    默认：单元目录或 imu.csv 的父目录

-t, --time-range <开始秒> <结束秒>
    时间同步范围（秒），默认 [0, 50]
    在此范围内搜索最佳时间对齐

-dbp, --draw-db
    绘制 DB 优化轨迹对比图
    在轨迹图中叠加显示优化轨迹（绿色虚线）

-rv, --rerun-verify
    使用 Rerun 验证坐标系转换是否正确
    打开 3D 可视化窗口显示 body 和 global 帧
```

## 使用示例

### 示例 1：单元目录模式（推荐）

```bash
# 基本使用
uv run PreprocessDB.py -u data/20260126_174407_SM-G9900/

# 指定输出目录
uv run PreprocessDB.py -u data/20260126_174407_SM-G9900/ --output /tmp/output

# 调整时间同步范围
uv run PreprocessDB.py -u data/20260126_174407_SM-G9900/ -t 0 30

# 绘制优化轨迹对比
uv run PreprocessDB.py -u data/20260126_174407_SM-G9900/ -dbp

# Rerun 验证坐标系
uv run PreprocessDB.py -u data/20260126_174407_SM-G9900/ -rv

# 完整功能
uv run PreprocessDB.py -u data/20260126_174407_SM-G9900/ -t 0 30 -dbp -rv
```

### 示例 2：兼容旧模式

```bash
# 基本使用
uv run PreprocessDB.py --db data/20260126_174407_SM-G9900/dataset.db --imu data/20260126_174407_SM-G9900/imu.csv

# 指定输出目录
uv run PreprocessDB.py --db data/20260126_174407_SM-G9900/dataset.db --imu data/20260126_174407_SM-G9900/imu.csv --output /tmp/output

# 完整功能
uv run PreprocessDB.py --db data/20260126_174407_SM-G9900/dataset.db --imu data/20260126_174407_SM-G9900/imu.csv --output /tmp/test_preprocess -t 0 30 -dbp -rv
```

## 输出文件说明

### 必需输出文件

| 文件名 | 说明 | 格式 |
|--------|------|------|
| `gt.csv` | 真值轨迹数据 | `t_us,q0,q1,q2,q3,x,y,z` |
| `imu_in.csv` | 插值对齐后的 IMU 数据 | `t_us,gyr_x,gyr_y,gyr_z,acce_x,acce_y,acce_z,ahrs_q0,ahrs_q1,ahrs_q2,ahrs_q3` |
| `TimeCheck.json` | 时间检查报告 | JSON 格式 |
| `Trajectory.png` | 轨迹图 | PNG 图片 |
| `TimeDiff.png` | 角速度对比图 | PNG 图片 |

### TimeCheck.json 格式说明

```json
{
  "data_path": "/path/to/data",
  "device_name": "device_name",
  "check_data_rate": {
    "imu": 200.0,
    "rtab": 30.0
  },
  "check_groundtruth_gap": {
    "max_gap": 0.5,
    "mean_gap": 0.2,
    "gap_idxs": [10, 25],
    "gap_diff": [1.2, 1.5],
    "note": "检测时间间隔"
  },
  "check_time_diff": {
    "time_diff_21_us": 50000,
    "note": "检测两个序列的时间偏移"
  }
}
```

**字段说明**：
- `data_path`: 数据文件路径
- `device_name`: 设备名称
- `check_data_rate`: IMU 和 RTAB 的采样率（Hz）
- `check_groundtruth_gap`: 时间空洞检测结果
  - `max_gap`: 最大时间间隔（秒）
  - `mean_gap`: 平均时间间隔（秒）
  - `gap_idxs`: 超过阈值的空洞索引列表
  - `gap_diff`: 超过阈值的时间间隔列表
- `check_time_diff`: 时间偏移检测结果
  - `time_diff_21_us`: IMU 相对于 GT 的时间偏移（微秒）

## 处理流程详解

### 阶段 1：数据加载和 Body Frame 转换
```
1. 检查输入参数（单元目录模式或旧模式）
2. 加载 RTAB 数据库
3. 应用 transform_local()：RTAB Body → IMU Body
4. 加载 IMU CSV 数据
```

### 阶段 2：时间对齐（不插值）
```
1. 创建 GroundTruthData 对象
2. 调用 match21() 计算时间偏移
3. 应用时间偏移到 GT 数据
4. 重置起始时间（t_us[0] = 0）
5. 检测时间空洞
6. 保存 TimeCheck.json
7. 绘制角速度对比图
```

### 阶段 3：数据插值
```
1. 生成统一时间序列（IMU 和 GT 时间戳的并集）
2. 将 GT 和 IMU 数据插值到统一时间轴
3. 所有数据现在都有相同的时间点
```

### 阶段 4：可视化和验证
```
1. 绘制轨迹图（可选包含优化轨迹对比）
2. Rerun 3D 可视化验证（可选）
3. 验证坐标系转换正确性
```

### 阶段 5：数据输出
```
1. 保存 gt.csv（时间对齐、插值后的轨迹）
2. 保存 imu_in.csv（时间对齐、插值后的 IMU 数据）
3. 完成预处理
```

## 关键技术点

### 1. 时间对齐算法（match21）
- 使用角速度序列进行匹配
- 在指定时间范围内搜索最优偏移
- 基于最小二乘法计算误差
- 返回使误差最小的时间偏移

### 2. 坐标系转换
- DefaultBodyRotation：绕 Y 轴旋转 -90 度
- 只影响 Body Frame，不影响 Global Frame
- 适配不同传感器的安装方向

### 3. 数据插值
- 时间戳：线性插值
- 旋转：Slerp（球面线性插值）
- 保证插值后的四元数仍然是单位四元数
- 位置和 IMU 数据：线性插值

### 4. 可视化验证
- 2D 轨迹图：显示路径形状
- 角速度对比图：验证时间对齐
- Rerun 3D：验证坐标系转换

## 常见问题和解决方案

### Q1: 时间对齐失败
**原因**：时间范围设置不当，IMU 和 GT 数据没有重叠
**解决**：调整 `-t` 参数，扩大时间同步范围

### Q2: 坐标系验证失败
**原因**：DefaultBodyRotation 不正确
**解决**：检查传感器安装方向，可能需要调整旋转角度

### Q3: 插值结果异常
**原因**：时间戳跳跃过大或数据质量差
**解决**：检查原始数据，过滤异常点

### Q4: 内存不足
**原因**：数据量过大
**解决**：分段处理，或使用更高性能的机器

## 性能优化建议

1. **大文件处理**：使用 `-t` 参数限制处理范围
2. **内存优化**：及时释放不需要的数据
3. **并行处理**：可以考虑将不同阶段并行化
4. **缓存机制**：对重复计算的结果进行缓存

## 扩展功能

脚本支持以下扩展：
1. 添加更多的可视化方式
2. 支持其他时间对齐算法
3. 添加数据质量评估
4. 支持批处理多个数据集