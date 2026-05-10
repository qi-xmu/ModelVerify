"""轨迹误差评估（仅 XY 平面位置误差，不含姿态误差）"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from base.datatype import PosesData

from .session import SessionObj


@dataclass
class EvalResult:
    fusion_ate_list: list[NDArray] = field(default_factory=list)
    fusion_rte_list: list[NDArray] = field(default_factory=list)
    cam_ate_list: list[NDArray] = field(default_factory=list)
    cam_rte_list: list[NDArray] = field(default_factory=list)
    network_ate_list: list[NDArray] = field(default_factory=list)
    network_rte_list: list[NDArray] = field(default_factory=list)

    @staticmethod
    def _agg(err_list: list[NDArray]) -> tuple[float, float, float]:
        if not err_list:
            return 0.0, 0.0, 0.0
        errs = np.concatenate(err_list)
        return (
            float(np.mean(errs)),
            float(np.sqrt(np.mean(errs**2))),
            float(np.std(errs)),
        )

    def fusion_ate(self) -> tuple[float, float, float]:
        return self._agg(self.fusion_ate_list)

    def fusion_rte(self) -> tuple[float, float, float]:
        return self._agg(self.fusion_rte_list)

    def cam_ate(self) -> tuple[float, float, float]:
        return self._agg(self.cam_ate_list)

    def cam_rte(self) -> tuple[float, float, float]:
        return self._agg(self.cam_rte_list)

    def network_ate(self) -> tuple[float, float, float]:
        return self._agg(self.network_ate_list)

    def network_rte(self) -> tuple[float, float, float]:
        return self._agg(self.network_rte_list)


class TrajEvaluator:
    """评估 SessionObj 中 fusion/cam/network 轨迹相对于 GT 的 ATE/RTE"""

    def __init__(self, obj: SessionObj):
        self.obj: SessionObj = obj
        self._result: EvalResult | None = None

    @staticmethod
    def _ate(ref: PosesData, est: PosesData) -> NDArray:
        return np.linalg.norm((est.ps - ref.ps)[:, :2], axis=1)

    @staticmethod
    def _rte(ref: PosesData, est: PosesData, gap_s: float = 1.0) -> NDArray:
        dt = np.diff(ref.t_us)
        idx_step = max(1, int(gap_s * 1e6 / np.median(dt)))
        errs = []
        for i in range(len(ref.t_us) - idx_step):
            j = i + idx_step
            # R_i^{-1} * (p_j - p_i)
            d_ref = ref.rots[i].inv().apply(ref.ps[j] - ref.ps[i])
            d_est = est.rots[i].inv().apply(est.ps[j] - est.ps[i])
            errs.append(np.linalg.norm((d_est - d_ref)[:2]))
        return np.array(errs)

    def evaluate(self, gap_s: float = 1.0) -> EvalResult:
        """计算所有可用轨迹的 ATE 和 RTE"""
        ref = self.obj.gt_pose
        result = EvalResult()

        result.fusion_ate_list.append(self._ate(ref, self.obj.fusion_pose))
        result.fusion_rte_list.append(self._rte(ref, self.obj.fusion_pose, gap_s))

        result.cam_ate_list.append(self._ate(ref, self.obj.cam_pose))
        result.cam_rte_list.append(self._rte(ref, self.obj.cam_pose, gap_s))

        if self.obj.extra_pose is not None:
            result.network_ate_list.append(self._ate(ref, self.obj.extra_pose))
            result.network_rte_list.append(self._rte(ref, self.obj.extra_pose, gap_s))

        self._result = result
        return result

    def report(self) -> None:
        if self._result is None:
            self.evaluate()
        result = self._result
        fmt = "{:<10} {:>10} {:>10} {:>8} {:>8} {:>8}"
        lines = [
            fmt.format("", "ATE rmse", "RTE rmse", "RTE 50%", "RTE 90%", "RTE 99%")
        ]
        for tag, ate_fn, rte_fn in [
            ("fusion", result.fusion_ate, result.fusion_rte),  # type: ignore
            ("cam", result.cam_ate, result.cam_rte),  # type: ignore
            ("network", result.network_ate, result.network_rte),  # type: ignore
        ]:
            ate_m, ate_r, ate_s = ate_fn()
            rte_m, rte_r, rte_s = rte_fn()
            rte_all = (
                np.concatenate(
                    [a for a in getattr(result, f"{tag}_rte_list") if len(a) > 0]
                )
                if getattr(result, f"{tag}_rte_list")
                else np.array([])
            )
            p50 = float(np.percentile(rte_all, 50)) if len(rte_all) > 0 else 0
            p90 = float(np.percentile(rte_all, 90)) if len(rte_all) > 0 else 0
            p99 = float(np.percentile(rte_all, 99)) if len(rte_all) > 0 else 0
            lines.append(
                fmt.format(
                    tag,
                    f"{ate_r:.3f}",
                    f"{rte_r:.3f}",
                    f"{p50:.3f}",
                    f"{p90:.3f}",
                    f"{p99:.3f}",
                )
            )
        print("\n".join(lines))


def export_csv(te_list: list[TrajEvaluator], output_dir: Path) -> None:
    """将评估结果导出为三个 CSV 文件，按设备分组、场次排列"""
    import csv

    metrics = [
        (
            "ATE_RMSE.csv",
            lambda r: r.cam_ate()[1],
            lambda r: r.network_ate()[1],
            lambda r: r.fusion_ate()[1],
        ),
        (
            "RTE_RMSE.csv",
            lambda r: r.cam_rte()[1],
            lambda r: r.network_rte()[1],
            lambda r: r.fusion_rte()[1],
        ),
        (
            "RTE_90.csv",
            lambda r: _pct(r, "cam_rte_list", 90),
            lambda r: _pct(r, "network_rte_list", 90),
            lambda r: _pct(r, "fusion_rte_list", 90),
        ),
    ]

    # 构建 {(device, session_id): EvalResult}
    results: dict[tuple[str, int], EvalResult] = {}
    for te in te_list:
        s = te.obj.session
        results[(s.device, s.session_id)] = te._result  # type: ignore

    devices = sorted({d for d, _ in results})
    sessions = sorted({sid for _, sid in results})

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, f_cam, f_network, f_fusion in metrics:
        path = output_dir / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["", "cam", "network", "fusion"])
            for dev in devices:
                dev_rows = []
                for sid in sessions:
                    key = (dev, sid)
                    if key not in results:
                        continue
                    r = results[key]
                    dev_rows.append([
                        f"{dev}_s{sid}",
                        f"{f_cam(r):.3f}",
                        f"{f_network(r):.3f}",
                        f"{f_fusion(r):.3f}",
                    ])
                for row in dev_rows:
                    writer.writerow(row)
                if dev_rows:
                    n = len(dev_rows)
                    cam_avg = sum(float(r[1]) for r in dev_rows) / n
                    net_avg = sum(float(r[2]) for r in dev_rows) / n
                    fus_avg = sum(float(r[3]) for r in dev_rows) / n
                    writer.writerow([
                        f"{dev}合计",
                        f"{cam_avg:.3f}",
                        f"{net_avg:.3f}",
                        f"{fus_avg:.3f}",
                    ])
                    # 提升率: (cam - fusion) / cam * 100%
                    cam_imp = (cam_avg - fus_avg) / cam_avg * 100 if cam_avg > 0 else 0
                    net_imp = (net_avg - fus_avg) / net_avg * 100 if net_avg > 0 else 0
                    writer.writerow([
                        f"{dev}提升%",
                        f"{cam_imp:.1f}%",
                        f"{net_imp:.1f}%",
                        "-",
                    ])
                writer.writerow([])
        print(f"> {filename} saved to {path}")


def _pct(result: EvalResult, attr: str, p: int) -> float:
    err_list = getattr(result, attr)
    if not err_list:
        return 0.0
    all_errs = np.concatenate([a for a in err_list if len(a) > 0])
    return float(np.percentile(all_errs, p)) if len(all_errs) > 0 else 0.0
