"""T06b 的工作单元：一套 × 全部回放对象。在子进程中运行，结果写入该套的目录，汇总量返回父进程。

每个对象依次运行新变体、对应的 V0 基准（新变体不是 V0 时）与 r2_floor OFF 配对（新变体为 ON 时），
分析完即释放，不在内存中保留整套逐日结果。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from stock_risk.application.compare import analysis as an
from stock_risk.application.compare.burden import burden_row, release_followups, unreleased
from stock_risk.application.compare.timelines import TIMELINE_STOCKS, timeline_rows
from stock_risk.application.compare.variants import VariantSet
from stock_risk.application.compare.views import PERIOD_END, RunView, build_view
from stock_risk.application.replay import PreparedSymbol, SymbolReplay, replay_prepared
from stock_risk.application.replay_chart_data import ChartData, build_chart_data
from stock_risk.application.replay_outputs import (
    EPISODE_COLUMNS,
    EVENT_COLUMNS,
    daily_rows,
    episode_csv_rows,
    episode_rows,
    event_rows,
)
from stock_risk.application.replay_run import discrete_digest
from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.lifecycle.invariants import check_invariants
from stock_risk.domain.lifecycle.params import EXPLORATION_ASSUMPTIONS, EngineParams
from stock_risk.domain.lifecycle.runner import RunResult

COMPACT_COLUMNS = ["target", "day", "data_status", "display", "level", "episode_id"]
COVERAGE_COLUMNS = [
    "target",
    "day",
    "v0_display",
    "v0_level",
    "v0_episode_id",
    "new_display",
    "new_level",
    "new_episode_id",
]
OVERLAP_COLUMNS = [
    "target",
    "v0_episode_id",
    "new_episode_id",
    "overlap_days",
    "first_day",
    "last_day",
]
LEVEL_CODE_NA = 9  # 不可计算或不在统计期间（单开关配对的覆盖四格只用可计算日）


class CompareSink(Protocol):
    def set_dir(self, run_dir: Path, set_id: str) -> Path: ...

    def write_csv(
        self, path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]
    ) -> None: ...

    def write_csv_gz(
        self, path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]
    ) -> None: ...

    def write_full(
        self,
        run_dir: Path,
        set_id: str,
        target: str,
        daily: Sequence[Mapping[str, Any]],
        chart: ChartData,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerContext:
    run_dir: Path
    run_id: str
    prepared: tuple[PreparedSymbol, ...]
    calendar: TradingCalendar
    sets: Mapping[str, VariantSet]
    timeline_ids: frozenset[str]
    digest_ids: frozenset[tuple[str, str]]  # (套, 对象) 需要返回离散结果（I17 ②）
    background: dict[str, dict[date, float]]
    sink: CompareSink


@dataclass(slots=True)
class SetOutcome:
    set_id: str
    burden: list[dict[str, Any]] = field(default_factory=list)
    coverage_cells: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[tuple] = field(default_factory=list)
    cancelled_escalations: list[dict[str, Any]] = field(default_factory=list)
    l_starts: list[dict[str, Any]] = field(default_factory=list)
    l_release_no_rise: list[dict[str, Any]] = field(default_factory=list)
    r2_filtered: list[dict[str, Any]] = field(default_factory=list)
    r2_filtered_days: int = 0
    followups: list[dict[str, Any]] = field(default_factory=list)
    unreleased: list[dict[str, Any]] = field(default_factory=list)
    levels: dict[str, bytes] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    digests: dict[str, tuple] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)  # 程序检查发现的问题
    max_day: date | None = None


_CTX: WorkerContext | None = None


def init_worker(ctx: WorkerContext) -> None:
    """进程池初始化：每个子进程接收一次上下文（回放对象的数据与日历）。"""
    global _CTX
    _CTX = ctx


def params_of(vs: VariantSet) -> EngineParams:
    return EngineParams(vs.candidate, EXPLORATION_ASSUMPTIONS, switches=vs.switches)


def run_one(
    ctx: WorkerContext, vs: VariantSet, p: PreparedSymbol
) -> tuple[SymbolReplay, RunResult]:
    params = params_of(vs)
    result = replay_prepared(p, params, ctx.calendar)
    return SymbolReplay(p.target, vs.candidate_id, params, p, result), result


def levels_code(view: RunView) -> bytes:
    return bytes(
        view.level[i] if view.computable[i] and view.in_period(i) else LEVEL_CODE_NA
        for i in range(len(view.days))
    )


def run_set(set_id: str) -> SetOutcome:
    assert _CTX is not None, "子进程未初始化"
    return run_set_with(_CTX, set_id)


def run_set_with(ctx: WorkerContext, set_id: str) -> SetOutcome:
    vs = ctx.sets[set_id]
    base_vs = ctx.sets[vs.baseline_id]
    off_vs = (
        next(
            (
                s
                for s in ctx.sets.values()
                if s.variant == vs.variant
                and s.n == vs.n
                and s.candidate_id == vs.candidate_id
                and s.switches == replace(vs.switches, r2_floor=False)
            ),
            None,
        )
        if vs.switches.r2_floor
        else None
    )
    out = SetOutcome(set_id)
    events_all: list[dict[str, Any]] = []
    episodes_all: list[dict[str, Any]] = []
    compact: list[tuple] = []
    coverage_rows: list[tuple] = []
    compare_rows: list[dict[str, Any]] = []
    overlap_all: list[tuple] = []
    for p in ctx.prepared:
        label = p.target.label
        try:
            rep, result = run_one(ctx, vs, p)
        except Exception as e:  # noqa: BLE001 —— 记录失败，写入运行清单并使运行失败
            out.failed.append(f"{set_id}/{label}: {type(e).__name__}: {e}")
            continue
        out.violations += [
            f"{set_id}/{label}: {v}" for v in check_invariants(result.results, rep.params)
        ]
        view = build_view(rep, set_id)
        last = view.days[-1]
        out.max_day = last if out.max_day is None or last > out.max_day else out.max_day
        if (set_id, label) in ctx.digest_ids:
            out.digests[label] = discrete_digest(result)
        for r in event_rows(rep, ctx.run_id):
            events_all.append({"set_id": set_id, **r})
        for r in episode_csv_rows(rep, ctx.run_id, episode_rows(rep)):
            episodes_all.append({"set_id": set_id, **r})
        compact += [
            (
                label,
                r.record.day.isoformat(),
                r.record.data_status.value,
                r.record.display.value,
                view.level[i],
                view.episode[i] or "",
            )
            for i, r in enumerate(result.results)
        ]
        fu = release_followups(view)
        out.followups += [{"set_id": set_id, **f} for f in fu]
        out.unreleased += [{"set_id": set_id, **u} for u in unreleased(view)]
        out.burden.append({"set_id": set_id, **burden_row(view, fu)})
        out.levels[label] = levels_code(view)
        if set_id in ctx.timeline_ids:
            ctx.sink.write_full(
                ctx.run_dir,
                set_id,
                label,
                daily_rows(rep, ctx.run_id),
                build_chart_data(rep, ctx.run_id, ctx.background),
            )
            if p.target.symbol in TIMELINE_STOCKS:
                out.timeline += timeline_rows(set_id, view, result)
        if vs.is_baseline:
            continue
        brep, bres = run_one(ctx, base_vs, p)
        base = build_view(brep, base_vs.set_id)
        rows, cells, excluded = an.coverage(base, view)
        coverage_rows += rows
        out.coverage_cells.append({"set_id": set_id, "target": label, **cells})
        if (
            sum(cells[k] for k in ("both", "only_v0", "only_new", "neither"))
            != cells["common_days"]
        ):
            out.checks.append(f"{set_id}/{label}：覆盖四格之和不等于共同可计算日数")
        out.excluded += excluded
        cmp_rows = an.event_compare(base, view)
        compare_rows += cmp_rows
        out.checks += _coverage_check(set_id, label, base, view, cmp_rows)
        overlap_all += an.overlap_rows(base, view, an.overlaps(base, view))
        out.cancelled_escalations += [
            {"set_id": set_id, **r} for r in an.cancelled_escalations(base, view)
        ]
        out.l_starts += [{"set_id": set_id, **r} for r in an.l_path_starts(base, view)]
        if vs.variant == "V1":
            out.l_release_no_rise += [
                {"set_id": set_id, **r} for r in an.l_release_without_rise(view)
            ]
        if off_vs is not None:
            orep, _ = run_one(ctx, off_vs, p)
            rows_f, n_days = an.r2_filtered(build_view(orep, off_vs.set_id), view)
            out.r2_filtered += [
                {"set_id": set_id, "off_set_id": off_vs.set_id, **r} for r in rows_f
            ]
            out.r2_filtered_days += n_days
    d = ctx.sink.set_dir(ctx.run_dir, set_id)
    ctx.sink.write_csv(d / "events.csv", ["set_id", *EVENT_COLUMNS], events_all)
    ctx.sink.write_csv(d / "episodes.csv", ["set_id", *EPISODE_COLUMNS], episodes_all)
    ctx.sink.write_csv_gz(d / "daily_compact.csv.gz", COMPACT_COLUMNS, compact)
    if not vs.is_baseline:
        ctx.sink.write_csv_gz(
            d / f"coverage_daily_{set_id}.csv.gz", COVERAGE_COLUMNS, coverage_rows
        )
        cols = sorted({k for r in compare_rows for k in r}, key=_compare_order)
        ctx.sink.write_csv(d / f"event_compare_{set_id}.csv", cols, compare_rows)
        ctx.sink.write_csv(
            d / f"overlap_{set_id}.csv",
            OVERLAP_COLUMNS,
            [dict(zip(OVERLAP_COLUMNS, r, strict=True)) for r in overlap_all],
        )
    if out.max_day is not None and out.max_day > PERIOD_END:
        out.checks.append(f"{set_id}：回放结果出现 {out.max_day}，晚于 2019-12-31")
    return out


_ORDER = [
    "target",
    "direction",
    "event_type",
    "date",
    "episode_id",
    "close",
    "other_status",
    "other_episode_id",
    "first_20",
    "wait_20",
    "min_close_rel_20",
    "end_close_rel_20",
    "first_60",
    "wait_60",
    "min_close_rel_60",
    "end_close_rel_60",
    "partner_episode_id",
    "partner_end_date",
    "partner_end_reason",
    "release_day_diff",
    "tags",
]


def _compare_order(k: str) -> int:
    return _ORDER.index(k) if k in _ORDER else len(_ORDER)


def _coverage_check(
    set_id: str, label: str, base: RunView, new: RunView, rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """事件对照表覆盖每个 V0 START 与每个新变体 START（统计期间内）。"""
    problems = []
    for view, direction in ((base, "V0→新"), (new, "新→V0")):
        starts = {
            e.day.isoformat()
            for e in view.events
            if e.type.value == "START" and view.in_period(view.index[e.day])
        }
        listed = {
            r["date"] for r in rows if r["direction"] == direction and r["event_type"] == "START"
        }
        if starts != listed:
            problems.append(f"{set_id}/{label}：事件对照表缺少 {direction} 的 START")
    return problems
