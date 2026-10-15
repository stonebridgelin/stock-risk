"""T06b 的汇总表：逐日覆盖四格、排除日、通知负担、专项清单与单开关配对。纯函数加注入的写出端口。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from stock_risk.application.compare.burden import burden_by_set
from stock_risk.application.compare.variants import VariantSet, pairs
from stock_risk.application.compare.worker import LEVEL_CODE_NA, SetOutcome


class TableSink(Protocol):
    def write_csv(
        self, path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
    ) -> None: ...


PAIR_METRICS = (
    "per_year_start",
    "per_year_start_first",
    "per_year_start_short_restart",
    "per_year_escalate",
    "per_year_downgrade",
    "per_year_release",
    "per_year_recovery_obs",
    "per_year_recovery_revoked",
    "per_year_recovery_fail",
    "occupancy_pooled",
    "occupancy_median_across_securities",
    "n_long_episodes",
    "long_episode_days",
    "below_l_ep_20",
    "below_l_ep_60",
    "n_unreleased",
)


def pair_coverage(a: Mapping[str, bytes], b: Mapping[str, bytes]) -> dict[str, int]:
    """两套之间的逐日覆盖四格（只用双方都可计算的统计期间日）。"""
    cells = {"both": 0, "only_a": 0, "only_b": 0, "neither": 0, "common_days": 0}
    for target, la in a.items():
        lb = b.get(target)
        if lb is None or len(lb) != len(la):
            continue
        for x, y in zip(la, lb, strict=True):
            if LEVEL_CODE_NA in (x, y):
                continue
            cells["common_days"] += 1
            cells["both" if x and y else "only_a" if x else "only_b" if y else "neither"] += 1
    return cells


def write_tables(
    sink: TableSink, run_dir: Path, sets: Sequence[VariantSet], outcomes: Mapping[str, SetOutcome]
) -> list[str]:
    s, written = sink, []

    def dump(name: str, rows: list[dict[str, Any]]) -> None:
        cols: list[str] = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        s.write_csv(run_dir / name, cols or ["set_id"], rows)
        written.append(name)

    ordered = [outcomes[x.set_id] for x in sets if x.set_id in outcomes]
    dump("coverage_summary.csv", [c for o in ordered for c in o.coverage_cells])
    s.write_csv(
        run_dir / "excluded_days.csv",
        ["set_id", "target", "day", "v0_status", "new_status"],
        [
            dict(zip(["set_id", "target", "day", "v0_status", "new_status"], r, strict=True))
            for o in ordered
            for r in o.excluded
        ],
    )
    written.append("excluded_days.csv")
    burden = [b for o in ordered for b in o.burden]
    dump("burden.csv", burden)
    by_set = {o.set_id: burden_by_set(o.set_id, o.burden) for o in ordered if o.burden}
    dump("burden_by_set.csv", list(by_set.values()))
    dump("release_followups.csv", [f for o in ordered for f in o.followups])
    dump("unreleased.csv", [u for o in ordered for u in o.unreleased])
    dump("special_r2_filtered.csv", [r for o in ordered for r in o.r2_filtered])
    dump(
        "special_r2_filtered_days.csv",
        [
            {"set_id": o.set_id, "filtered_r2_days": o.r2_filtered_days}
            for o in ordered
            if outcomes[o.set_id] and o.set_id in {x.set_id for x in sets if x.switches.r2_floor}
        ],
    )
    dump(
        "special_cancelled_escalations.csv",
        [r for o in ordered for r in o.cancelled_escalations],
    )
    dump("special_l_path_starts.csv", [r for o in ordered for r in o.l_starts])
    dump(
        "special_v1_l_release_without_rise.csv",
        [r for o in ordered for r in o.l_release_no_rise],
    )
    pair_rows = _pair_rows(sets, outcomes, by_set)
    dump("pairs_summary.csv", pair_rows)
    return written


def _pair_rows(
    sets: Sequence[VariantSet],
    outcomes: Mapping[str, SetOutcome],
    by_set: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """单开关配对：两套的负担指标、差值与逐日覆盖四格。"""
    rows = []
    for pr in pairs(tuple(sets)):
        if pr.a not in outcomes or pr.b not in outcomes:
            continue
        ba, bb = by_set.get(pr.a, {}), by_set.get(pr.b, {})
        row: dict[str, Any] = {"pair_kind": pr.kind, "set_a": pr.a, "set_b": pr.b}
        for m in PAIR_METRICS:
            va, vb = ba.get(m, ""), bb.get(m, "")
            row[f"{m}_a"], row[f"{m}_b"] = va, vb
            row[f"{m}_diff"] = round(vb - va, 6) if va != "" and vb != "" else ""
        row |= {
            f"cov_{k}": v
            for k, v in pair_coverage(outcomes[pr.a].levels, outcomes[pr.b].levels).items()
        }
        rows.append(row)
    return rows
