# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ModelVerify is a Python toolkit for validating and testing inertial navigation models. It supports PyTorch model loading, IMU data processing, batch validation, visualization via Rerun SDK, and model conversion.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run Python scripts (use this instead of python directly)
uv run python <script_path>

# Run a single test
uv run python -m pytest tests/<test_file>::<test_name> -v

# Run all tests
uv run python -m pytest tests/ -v
```

## Architecture

### Core Module (`src/base/`)

**Data Types** (`datatype.py`):
- `Pose`: Rotation + position combination with `compose()`, `inverse()`, `between()` methods
- `ImuData`: IMU sensor data (gyro, acc, mag) with timestamps
- `GroundTruthData`: Ground truth poses with `PosesData` structure
- `PosesData`: Time-indexed array of poses with interpolation support
- `UnitData`: Container for a single data unit (IMU + GT + optional CAM/OPT/FUSION)

**Model Loading** (`model.py`):
- `InertialNetwork`: Loads TorchScript models (.pt), expects input shape (1, 6, 200)
- `ModelLoader`: Batch loads models from a directory
- `DataRunner`: Executes model predictions on UnitData

**Interpolation** (`interpolate.py`):
- `slerp_rotation()`: Spherical linear interpolation for rotations
- `interpolate_vector3()`: Linear interpolation for 3D vectors
- `get_time_series()`: Finds common time window across multiple data sources

**Evaluation** (`evaluate.py`):
- `Evaluation`: Computes ATE (Absolute Trajectory Error), APE, RPE metrics

### Script Organization

- `validators/`: Model validation scripts (VaildModel, VaildModelFromH5, DatasetAnalysis)
- `visualizers/`: Visualization scripts (DrawModel, DrawCompare, TLIOView)
- `converters/`: Data conversion scripts (GenerateH5Dataset, TLIO2H5, TorchScript2Android)

### Key Conventions

- Data uses microsecond timestamps (`t_us` as `np.uint64`)
- Quaternions use scalar-first format (w, x, y, z)
- Time alignment uses `DatasetArgsParser` with `-u` (unit) or `-d` (dataset) flags
- Results output to `results/<model_name>/` or specified output directory

## Project Context

- `.agent_history/`: Stores conversation history between user and Claude for project continuity
  - **Format**: Markdown files in directory root
  - **Naming**: `YYYY-MM-DD-标题.md`
  - **Content**: Session date, completed tasks, key decisions, modified files, and notes

### Agent Tools

- `.agent/agent_history.py`: Claude Code 会话历史记录工具
  - 用法: `uv run python .agent/agent_history.py "会话摘要"`
  - 自动生成文件名: `.agent_history/YYYY-MM-DD-标题.md`
