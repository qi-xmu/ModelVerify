"""轨迹误差评估（仅 XY 平面位置误差，不含姿态误差）"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from base.datatype import PosesData

from .session import SessionObj

_EXTRA_PREFIX = {
    "network": "net",
    "fix": "fix",
}


@dataclass
class EvalResult:
    fusion_ate_list: list[NDArray] = field(default_factory=list)
    fusion_rte_list: list[NDArray] = field(default_factory=list)
    cam_ate_list: list[NDArray] = field(default_factory=list)
    cam_rte_list: list[NDArray] = field(default_factory=list)
    extra_ate_list: list[NDArray] = field(default_factory=list)
    extra_rte_list: list[NDArray] = field(default_factory=list)
    extra_kind: str = ""

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

    def extra_ate(self) -> tuple[float, float, float]:
        return self._agg(self.extra_ate_list)

    def extra_rte(self) -> tuple[float, float, float]:
        return self._agg(self.extra_rte_list)


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
            result.extra_ate_list.append(self._ate(ref, self.obj.extra_pose))
            result.extra_rte_list.append(self._rte(ref, self.obj.extra_pose, gap_s))
            result.extra_kind = self.obj.session.extra_kind
        else:
            print("> 附加数据不存在")

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
        rows = [
            ("fusion", result.fusion_ate, result.fusion_rte, "fusion"),  # type: ignore
            ("cam", result.cam_ate, result.cam_rte, "cam"),  # type: ignore
            (result.extra_kind or "extra", result.extra_ate, result.extra_rte, "extra"),  # type: ignore
        ]
        for tag, ate_fn, rte_fn, attr_tag in rows:
            ate_m, ate_r, ate_s = ate_fn()
            rte_m, rte_r, rte_s = rte_fn()
            rte_all = (
                np.concatenate(
                    [a for a in getattr(result, f"{attr_tag}_rte_list") if len(a) > 0]
                )
                if getattr(result, f"{attr_tag}_rte_list")
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
    """将评估结果导出为单个 CSV，按设备分组、场次排列，含 ATE/RTE/RTE90 及提升率"""
    import csv

    # 构建 {(device, session_id): EvalResult}
    results: dict[tuple[str, int], tuple[EvalResult, str]] = {}
    for te in te_list:
        s = te.obj.session
        results[(s.device, s.session_id)] = (te._result, s.label)  # type: ignore

    devices = sorted({d for d, _ in results})
    sessions = sorted({sid for _, sid in results})

    def _imp(before: float, after: float) -> str:
        if before <= 0:
            return "-"
        return f"{(before - after) / before * 100:.1f}%"

    # 从第一个有 extra 的结果中取 kind 前缀
    extra_prefix = "extra"
    for r, _ in results.values():
        if r.extra_kind and r.extra_kind in _EXTRA_PREFIX:
            extra_prefix = _EXTRA_PREFIX[r.extra_kind]
            break

    header = [
        "label",
        "cam_ATE",
        f"{extra_prefix}_ATE",
        "fus_ATE",
        "cam_ATE提升%",
        f"{extra_prefix}_ATE提升%",
        "cam_RTE",
        f"{extra_prefix}_RTE",
        "fus_RTE",
        "cam_RTE提升%",
        f"{extra_prefix}_RTE提升%",
        "cam_RTE90",
        f"{extra_prefix}_RTE90",
        "fus_RTE90",
        "cam_R90提升%",
        f"{extra_prefix}_R90提升%",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "TE_metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for dev in devices:
            dev_rows = []
            for sid in sessions:
                key = (dev, sid)
                if key not in results:
                    continue
                r, label = results[key]
                cam_ate = r.cam_ate()[1]
                extra_ate = r.extra_ate()[1]
                fus_ate = r.fusion_ate()[1]
                cam_rte = r.cam_rte()[1]
                extra_rte = r.extra_rte()[1]
                fus_rte = r.fusion_rte()[1]
                cam_r90 = _pct(r, "cam_rte_list", 90)
                extra_r90 = _pct(r, "extra_rte_list", 90)
                fus_r90 = _pct(r, "fusion_rte_list", 90)

                dev_rows.append(
                    [
                        label,
                        f"{cam_ate:.3f}",
                        f"{extra_ate:.3f}",
                        f"{fus_ate:.3f}",
                        _imp(cam_ate, fus_ate),
                        _imp(extra_ate, fus_ate),
                        f"{cam_rte:.3f}",
                        f"{extra_rte:.3f}",
                        f"{fus_rte:.3f}",
                        _imp(cam_rte, fus_rte),
                        _imp(extra_rte, fus_rte),
                        f"{cam_r90:.3f}",
                        f"{extra_r90:.3f}",
                        f"{fus_r90:.3f}",
                        _imp(cam_r90, fus_r90),
                        _imp(extra_r90, fus_r90),
                    ]
                )
            for row in dev_rows:
                writer.writerow(row)
            if dev_rows:
                n = len(dev_rows)
                cols = list(zip(*[r[1:] for r in dev_rows]))
                # 只对数值列取平均（跳过提升率列 % 字符串）
                idx = [
                    0,
                    1,
                    2,
                    5,
                    6,
                    7,
                    10,
                    11,
                    12,
                ]  # cam/extra/fus for ATE, RTE, RTE90
                avgs = [sum(float(cols[i][j]) for j in range(n)) / n for i in idx]
                cam_a, extra_a, fus_a = avgs[0], avgs[1], avgs[2]
                cam_r, extra_r, fus_r = avgs[3], avgs[4], avgs[5]
                cam_90, extra_90, fus_90 = avgs[6], avgs[7], avgs[8]
                writer.writerow(
                    [
                        f"{dev}合计",
                        f"{cam_a:.3f}",
                        f"{extra_a:.3f}",
                        f"{fus_a:.3f}",
                        _imp(cam_a, fus_a),
                        _imp(extra_a, fus_a),
                        f"{cam_r:.3f}",
                        f"{extra_r:.3f}",
                        f"{fus_r:.3f}",
                        _imp(cam_r, fus_r),
                        _imp(extra_r, fus_r),
                        f"{cam_90:.3f}",
                        f"{extra_90:.3f}",
                        f"{fus_90:.3f}",
                        _imp(cam_90, fus_90),
                        _imp(extra_90, fus_90),
                    ]
                )
            writer.writerow([])

    print(f"> TE_metrics.csv saved to {path}")


def export_yaw(te_list: list[TrajEvaluator], output_dir: Path) -> None:
    """导出每个 session 的 gt_yaw / extra_yaw（度）"""
    import csv

    devices = sorted({te.obj.session.device for te in te_list})
    sessions = sorted({te.obj.session.session_id for te in te_list})

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "TE_yaw.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "gt_yaw", "extra_yaw"])
        for dev in devices:
            for sid in sessions:
                for te in te_list:
                    s = te.obj.session
                    if s.device == dev and s.session_id == sid:
                        writer.writerow(
                            [
                                s.label,
                                f"{te.obj.session.gt_yaw:.1f}",
                                f"{te.obj.session.extra_yaw:.1f}",
                            ]
                        )
                        break
            writer.writerow([])
    print(f"> TE_yaw.csv saved to {path}")


def _pct(result: EvalResult, attr: str, p: int) -> float:
    err_list = getattr(result, attr)
    if not err_list:
        return 0.0
    all_errs = np.concatenate([a for a in err_list if len(a) > 0])
    return float(np.percentile(all_errs, p)) if len(all_errs) > 0 else 0.0
