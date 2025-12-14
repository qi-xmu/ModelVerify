import importlib.util
from pathlib import Path
import importlib
import sys

from base.datatype import UnitData

PROJECT_PATH = Path(
    "/Users/qi/Codespace/Android/NAVIO/app/src/main/cpp/SensorFusionAndroid/"
)


class SensorFusion:
    TARGET = "SensorFusionAndroid"

    def __init__(self, project_path: str | Path = PROJECT_PATH):
        self.project_path = Path(project_path)
        self.configs_path = self.project_path / "configs"
        self.build_path = self.project_path / "build"

        self.run_py = self.project_path / "Run.py"
        spec = importlib.util.spec_from_file_location("Run", self.run_py.as_posix())
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot find module at {self.run_py.as_posix()}")

        # 加载py文件为模块
        module = importlib.util.module_from_spec(spec)
        sys.modules["Run"] = module
        spec.loader.exec_module(module)
        self.runner = module.main

    def run(
        self,
        unit: Path,
        *,
        model: Path | None = None,
        device: str,
        use_build: bool = False,
    ):
        config_path = self.configs_path / f"{device}.yaml"
        return self.runner(
            self.TARGET,
            unit,
            config_path,
            model,
            build_dir=self.build_path.as_posix(),
            use_build=use_build,
        )

    def unit_run(
        self,
        ud: UnitData,
        model: Path | None = None,
        use_build: bool = False,
    ):
        return self.run(
            ud.base_dir, model=model, device=ud.device_name, use_build=use_build
        )
