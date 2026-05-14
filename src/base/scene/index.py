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
        self.kind: str = self._parse_label(self.path.name)
        self._extra_label: str | None = None
        df = pd.read_csv(self.path)
        self._sessions: dict[tuple[int, str], Session] = {}
        self.has_yaw = False

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

        # 自动修复路径：当场景目录移动后，CSV 中记录的绝对路径可能失效
        self._rebase_paths()

    def _rebase_paths(self) -> None:
        """根据 CSV 自身位置推断场景目录，修复不存在的路径。全部成功只打一行，失败则报错。"""
        subdir_name = self.path.parent.name
        scene_dir = self.path.parent.parent
        fixed = 0
        old_base: Path | None = None
        for s in self._sessions.values():
            for attr in ("gt_db", "navio_db", "fusion_csv", "extra_csv"):
                p: Path | None = getattr(s, attr)
                if p is not None and not p.exists():
                    if subdir_name not in p.parts:
                        raise RuntimeError(
                            f"> {p.parent} 重定向失败: 路径中未找到 '{subdir_name}'"
                        )
                    idx = p.parts.index(subdir_name)
                    if old_base is None:
                        old_base = Path(*p.parts[:idx])
                    tail = Path(*p.parts[idx:])
                    new_p = scene_dir / tail
                    if new_p.exists():
                        setattr(s, attr, new_p)
                        fixed += 1
                    else:
                        raise RuntimeError(
                            f"> {old_base} 重定向到 {scene_dir}: 失败 → {new_p} 目标也不存在"
                        )
        if fixed and old_base:
            print(f"> {old_base} 重定向到 {scene_dir}: {fixed} 个")
        # errors 为空且 fixed 为空说明无需修复，静默

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

    def attach_yaw(self, yaw_csv: str | Path) -> None:
        """读取 TE_yaw.csv，按 label 匹配并设置 Session 的 gt_yaw / extra_yaw"""
        df = pd.read_csv(yaw_csv)
        label_map: dict[str, tuple[float, float]] = {}
        for _, row in df.iterrows():
            label = str(row["label"])
            gt = float(row["gt_yaw"]) if row["gt_yaw"] != "-" else 0.0
            ex = float(row["extra_yaw"]) if row["extra_yaw"] != "-" else 0.0
            label_map[label] = (gt, ex)

        for s in self._sessions.values():
            if s.label in label_map:
                s.gt_yaw, s.extra_yaw = label_map[s.label]
                s.has_yaw = True

    def attach(self, other: "FusionIndex") -> None:
        """附加另一个 FusionIndex，校验 GT 一致性后注入 fusion_csv_extra"""
        self._extra_label = other.kind
        for key, s in self._sessions.items():
            if key not in other._sessions:
                raise KeyError(f"附加索引缺少 {key}")
            other_s = other._sessions[key]
            if s.gt_stem != other_s.gt_stem:
                raise ValueError(f"GT 不一致: [{key}] {s.gt_stem} != {other_s.gt_stem}")
            s.extra_csv = other_s.fusion_csv
            s.extra_kind = other.kind

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
        """扫描场景目录，自动发现 *_fusion / *_network / *_fix 子文件夹，将 network/fix 附加到 fusion"""
        scene = Path(scene_dir)
        scene_name = scene.name
        fusion_idx: FusionIndex | None = None
        extra_idx: FusionIndex | None = None

        for subdir in sorted(scene.iterdir()):
            if not subdir.is_dir():
                continue
            m = re.match(
                rf".*{re.escape(scene_name)}_(fusion|network|fix)$", subdir.name
            )
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
            elif kind in ("network", "fix"):
                extra_idx = idx

        if fusion_idx is None and extra_idx is not None:
            fusion_idx = extra_idx
            extra_idx = None

        if fusion_idx is None:
            raise FileNotFoundError(
                f"场景 {scene_name} 中未找到 fusion 结果文件夹 "
                f"(期望子目录名称包含 '{scene_name}_fusion')"
            )
        if extra_idx is not None:
            print(f"> 附加Index {extra_idx.kind}: {extra_idx.path}")
            fusion_idx.attach(extra_idx)

        # 自动附加 TE_yaw.csv（场景目录级别）
        yaw_csv = scene / "TE_yaw.csv"
        if yaw_csv.exists():
            print(f"> 附加YAW TE_yaw.csv: {yaw_csv}")
            fusion_idx.has_yaw = True
            fusion_idx.attach_yaw(yaw_csv)
        else:
            print(f"未找到 TE_yaw.csv: {yaw_csv}")

        return fusion_idx

    # ---- 表格 ----

    def print_summary(self) -> None:
        """打印索引摘要表格"""
        has_extra = any(s.extra_csv for s in self)

        COL_W = [5, 6, 8, 14, 30]
        COLS = ["场次", "设备", "时长(s)", "GT", f"标签({self.kind})"]
        if has_extra:
            COL_W.append(30)
            extra = self.extra_label or "附加"
            COLS.append(f"附加({extra})")
        COL_W.extend([6, 8])
        COLS.extend(["GT Yaw", "Net Yaw"])

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
                    extra_label = s.extra_csv.parent.name if s.extra_csv else "-"
                    values.append(extra_label)
                values.append(f"{s.gt_yaw:.1f}°")
                values.append(f"{s.extra_yaw:.1f}°")
                cells = [pad_center(values[k], COL_W[k]) for k in range(len(COLS))]
                print(" | ".join(cells))
        print(sep + "\n")
