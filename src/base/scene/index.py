"""FusionIndex 索引类"""

import re
from pathlib import Path

import pandas as pd

from base.tab_show import display_width, pad_center

from .session import Session


class FusionIndex:
    """S2_results_fusion.csv 解析结果，(session_id, device) → Session"""

    @staticmethod
    def _parse_label(filename: str) -> str:
        """S2_results_fusion.csv → fusion"""
        m = re.search(r"results_(\w+)\.csv$", filename)
        return m.group(1) if m else filename

    def __init__(self, csv_path: str | Path):
        self.path = Path(csv_path)
        self._label = self._parse_label(self.path.name)
        self._extra_label: str | None = None
        df = pd.read_csv(self.path)
        self._sessions: dict[tuple[int, str], Session] = {}

        for _, row in df.iterrows():
            s = Session(
                session_id=int(row["session"]),
                device=row["device"],  # type: ignore
                duration_s=float(row["duration_s"]),  # type: ignore
                gt_db=Path(row["gt_file"]),  # type: ignore
                navio_db=Path(row["db_path"]),  # type: ignore
                fusion_csv=Path(row["output_csv"]),  # type: ignore
                status=row["status"],  # type: ignore
            )
            self._sessions[s.key] = s

    @property
    def label(self) -> str:
        return self._label

    @property
    def extra_label(self) -> str | None:
        return self._extra_label

    # ---- 索引规则 ----

    def all(self) -> list[Session]:
        """1. 返回所有 Session，按 (场次, 设备) 排序"""
        return sorted(self._sessions.values(), key=lambda s: (s.session_id, s.device))

    def by_device(self, device: str) -> list[Session]:
        """2. 按设备类型返回，按场次排序"""
        return [s for s in self.all() if s.device == device]

    def by_session(self, session_id: int) -> list[Session]:
        """3. 按场次 ID 返回，按设备排序"""
        return sorted(
            [s for s in self._sessions.values() if s.session_id == session_id],
            key=lambda s: s.device,
        )

    # ---- 附加 ----

    def attach(self, other: "FusionIndex") -> None:
        """附加另一个 FusionIndex，校验 GT 一致性后注入 fusion_csv_extra"""
        self._extra_label = other.label
        for key, s in self._sessions.items():
            if key not in other._sessions:
                raise KeyError(f"附加索引缺少 {key}")
            other_s = other._sessions[key]
            if s.gt_stem != other_s.gt_stem:
                raise ValueError(f"GT 不一致: [{key}] {s.gt_stem} != {other_s.gt_stem}")
            s.fusion_csv_extra = other_s.fusion_csv

    # ---- 便利接口 ----

    def __getitem__(self, key: tuple[int, str]) -> Session:
        return self._sessions[key]

    def __iter__(self):
        return iter(self.all())

    def __len__(self) -> int:
        return len(self._sessions)

    @property
    def session_ids(self) -> list[int]:
        return sorted({s.session_id for s in self._sessions.values()})

    @property
    def devices(self) -> list[str]:
        return sorted({s.device for s in self._sessions.values()})

    # ---- 自动发现 ----

    @classmethod
    def from_scene(cls, scene_dir: str | Path) -> "FusionIndex":
        """扫描场景目录，自动发现 *_fusion / *_network 子文件夹，将 network 附加到 fusion"""
        scene = Path(scene_dir)
        scene_name = scene.name
        fusion_idx: FusionIndex | None = None
        network_idx: FusionIndex | None = None

        for subdir in sorted(scene.iterdir()):
            if not subdir.is_dir():
                continue
            m = re.match(rf".*{re.escape(scene_name)}_(fusion|network)$", subdir.name)
            if not m:
                continue
            kind = m.group(1)
            pattern = f"{scene_name}_results_{kind}.csv"
            candidates = sorted(subdir.glob(pattern))
            if not candidates:
                candidates = sorted(subdir.glob(f"{scene_name}_results_*.csv"))
            if not candidates:
                continue
            csv_path = candidates[0]
            idx = cls(csv_path)
            if kind == "fusion":
                fusion_idx = idx
            elif kind == "network":
                network_idx = idx

        if fusion_idx is None:
            raise FileNotFoundError(
                f"场景 {scene_name} 中未找到 fusion 结果文件夹 "
                f"(期望子目录名称包含 '{scene_name}_fusion')"
            )
        if network_idx is not None:
            fusion_idx.attach(network_idx)
        return fusion_idx

    # ---- 表格 ----

    def print_summary(self) -> None:
        """打印索引摘要表格"""
        has_extra = any(s.fusion_csv_extra for s in self)

        COL_W = [5, 6, 8, 14, 30]
        COLS = ["场次", "设备", "时长(s)", "GT", f"标签({self.label})"]
        if has_extra:
            COL_W.append(30)
            extra = self.extra_label or "附加"
            COLS.append(f"附加({extra})")

        header = " | ".join(pad_center(COLS[k], COL_W[k]) for k in range(len(COLS)))
        sep = "-" * max(display_width(header), 64)

        print(f"\n融合索引: {self.path.name}")
        print(
            f"共 {len(self.session_ids)} 场次 × {len(self.devices)} 设备 = {len(self)} 条"
        )

        for device in self.devices:
            print(sep)
            print(header)
            print(sep)
            for s in self.by_device(device):
                values = [
                    str(s.session_id),
                    s.device,
                    str(s.duration_s),
                    s.gt_stem,
                    s.label,
                ]
                if has_extra:
                    extra_label = (
                        s.fusion_csv_extra.parent.name if s.fusion_csv_extra else "-"
                    )
                    values.append(extra_label)
                cells = [pad_center(values[k], COL_W[k]) for k in range(len(COLS))]
                print(" | ".join(cells))
        print(sep + "\n")
