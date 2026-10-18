"""T06b 的工作单元：一组（结构组合, N）内的若干套 × 全部回放对象。在子进程中运行，结果写入
各套的目录，汇总量返回父进程。

T08a D-04：同组内 V0 基准与 r2_floor OFF 配对各只运行一次。组内先运行 V0，再按变体全集的顺序
运行其余各套（OFF 排在对应的 ON 之前）；配对所需的视图按对象缓存，最后一个使用者用完即释放。
只请求部分套时，所需的基准或配对作为辅助运行（不输出）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

from stock_risk.application.compare import analysis as an
from stock_risk.application.compare.burden import burden_row, release_followups, unreleased
from stock_risk.application.compare.set_outputs import CompareSink, SetRows, coverage_check
from stock_risk.application.compare.timelines import TIMELINE_STOCKS, timeline_rows
from stock_risk.application.compare.variants import VariantSet
from stock_risk.application.compare.views import PERIOD_END, RunView, build_view
from stock_risk.application.replay import PreparedSymbol, SymbolReplay, replay_prepared
from stock_risk.application.replay_chart_data import build_chart_data
from stock_risk.application.replay_outputs import daily_rows
from stock_risk.application.replay_run_types import discrete_digest
from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.lifecycle.invariants import check_invariants
from stock_risk.domain.lifecycle.params import EXPLORATION_ASSUMPTIONS, EngineParams
from stock_risk.domain.lifecycle.runner import RunResult
from stock_risk.domain.lifecycle.semantic_digest import semantic_digest

LEVEL_CODE_NA = 9  # 不可计算或不在统计期间（单开关配对的覆盖四格只用可计算日）


@dataclass(frozen=True, slots=True)
class WorkerContext:
    run_dir: Path
    run_id: str
    prepared: tuple[PreparedSymbol, ...]
    calendar: TradingCalendar
    sets: Mapping[str, VariantSet]
    timeline_ids: frozenset[str]
    digest_ids: frozenset[tuple[str, str]]  # (套, 对象) 需要返回离散摘要与全语义摘要（I17 ②）
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
    semantic: dict[str, str] = field(default_factory=dict)  # 全语义摘要（T10-M0b）
    timeline: list[dict[str, Any]] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)  # 程序检查发现的问题
    max_day: date | None = None


@dataclass(frozen=True, slots=True)
class GroupPlan:
    """一组的运行顺序。steps：(套, 是否输出)；keep：需缓存视图的套；release_after：用完即释放。"""

    steps: tuple[tuple[VariantSet, bool], ...]
    keep: frozenset[str]
    release_after: Mapping[str, tuple[str, ...]]


_CTX: WorkerContext | None = None


def init_worker(ctx: WorkerContext) -> None:
    """进程池初始化：每个子进程接收一次上下文（回放对象的数据与日历）。"""
    global _CTX
    _CTX = ctx


def params_of(vs: VariantSet) -> EngineParams:
    return EngineParams(vs.candidate, EXPLORATION_ASSUMPTIONS, switches=vs.switches)


def group_key(vs: VariantSet) -> tuple[int, int]:
    """（结构组合, N）。"""
    return vs.structure_id, vs.n


def off_partner(sets: Mapping[str, VariantSet], vs: VariantSet) -> VariantSet | None:
    """r2_floor = ON 的套对应的 OFF 套（同变体、同候选、其余开关相同）。"""
    if not vs.switches.r2_floor:
        return None
    off = replace(vs.switches, r2_floor=False)
    return next(
        (
            s
            for s in sets.values()
            if s.variant == vs.variant
            and s.n == vs.n
            and s.candidate_id == vs.candidate_id
            and s.switches == off
        ),
        None,
    )


def _partners(sets: Mapping[str, VariantSet], vs: VariantSet) -> list[str]:
    out = [] if vs.is_baseline else [vs.baseline_id]
    off = off_partner(sets, vs)
    return out + ([off.set_id] if off is not None else [])


def group_plan(sets: Mapping[str, VariantSet], set_ids: Sequence[str]) -> GroupPlan:
    keys = {group_key(sets[s]) for s in set_ids}
    if len(keys) != 1:
        raise ValueError(f"一组只能含同一（结构组合, N）的套：{sorted(keys)}")
    order = {sid: i for i, sid in enumerate(sets)}
    wanted = set(set_ids)
    every = wanted | {p for s in set_ids for p in _partners(sets, sets[s])}
    ordered = sorted(every, key=lambda s: (not sets[s].is_baseline, order[s]))
    last_use: dict[str, str] = {}
    for sid in ordered:
        for p in _partners(sets, sets[sid]):
            if ordered.index(p) > ordered.index(sid):
                raise ValueError(f"{sid} 的配对 {p} 排在其后")
            last_use[p] = sid
    release: dict[str, list[str]] = {}
    for p, consumer in last_use.items():
        release.setdefault(consumer, []).append(p)
    return GroupPlan(
        tuple((sets[s], s in wanted) for s in ordered),
        frozenset(last_use),
        {k: tuple(v) for k, v in release.items()},
    )


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


def run_group(set_ids: Sequence[str]) -> list[SetOutcome]:
    assert _CTX is not None, "子进程未初始化"
    return run_group_with(_CTX, set_ids)


def run_group_with(ctx: WorkerContext, set_ids: Sequence[str]) -> list[SetOutcome]:
    plan = group_plan(ctx.sets, set_ids)
    cache: dict[str, dict[str, RunView]] = {}
    outs: list[SetOutcome] = []
    for vs, emit in plan.steps:
        out, views = _run_set(ctx, vs, emit, cache)
        if vs.set_id in plan.keep:
            cache[vs.set_id] = views
        for done in plan.release_after.get(vs.set_id, ()):
            cache.pop(done, None)
        if emit:
            outs.append(out)
    return outs


def run_set_with(ctx: WorkerContext, set_id: str) -> SetOutcome:
    """单独运行一套（所需的基准与配对作为辅助运行）。"""
    (out,) = run_group_with(ctx, [set_id])
    return out


def _run_set(
    ctx: WorkerContext, vs: VariantSet, emit: bool, cache: Mapping[str, dict[str, RunView]]
) -> tuple[SetOutcome, dict[str, RunView]]:
    out, rows = SetOutcome(vs.set_id), SetRows(vs.set_id)
    off_vs = off_partner(ctx.sets, vs)
    base_views = None if vs.is_baseline else cache[vs.baseline_id]
    off_views = None if off_vs is None else cache[off_vs.set_id]
    views: dict[str, RunView] = {}
    for p in ctx.prepared:
        label = p.target.label
        try:
            rep, result = run_one(ctx, vs, p)
        except Exception as e:  # noqa: BLE001 —— 记录失败，写入运行清单并使运行失败
            out.failed.append(f"{vs.set_id}/{label}: {type(e).__name__}: {e}")
            continue
        view = views[label] = build_view(rep, vs.set_id)
        if not emit:
            continue
        _record_target(ctx, vs, p, rep, result, view, out, rows)
        if base_views is None:
            continue
        base = base_views.get(label)
        if base is None:
            out.failed.append(f"{vs.set_id}/{label}: 基准 {vs.baseline_id} 未得出结果")
            continue
        _compare_target(vs, label, base, view, out, rows)
        if off_vs is not None and off_views is not None:
            off = off_views.get(label)
            if off is None:
                out.failed.append(f"{vs.set_id}/{label}: 配对 {off_vs.set_id} 未得出结果")
                continue
            rows_f, n_days = an.r2_filtered(off, view)
            out.r2_filtered += [
                {"set_id": vs.set_id, "off_set_id": off_vs.set_id, **r} for r in rows_f
            ]
            out.r2_filtered_days += n_days
    if emit:
        rows.write(ctx.sink, ctx.run_dir, with_compare=not vs.is_baseline)
        if out.max_day is not None and out.max_day > PERIOD_END:
            out.checks.append(f"{vs.set_id}：回放结果出现 {out.max_day}，晚于 2019-12-31")
    return out, views


def _record_target(
    ctx: WorkerContext,
    vs: VariantSet,
    p: PreparedSymbol,
    rep: SymbolReplay,
    result: RunResult,
    view: RunView,
    out: SetOutcome,
    rows: SetRows,
) -> None:
    """一套 × 一个对象：不变量、明细行、负担、逐日等级与时间线。"""
    set_id, label = vs.set_id, p.target.label
    out.violations += [
        f"{set_id}/{label}: {v}" for v in check_invariants(result.results, rep.params)
    ]
    last = view.days[-1]
    out.max_day = last if out.max_day is None or last > out.max_day else out.max_day
    if (set_id, label) in ctx.digest_ids:
        out.digests[label] = discrete_digest(result)
        out.semantic[label] = semantic_digest(result)
    rows.add_target(rep, result, view, ctx.run_id)
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


def _compare_target(
    vs: VariantSet, label: str, base: RunView, view: RunView, out: SetOutcome, rows: SetRows
) -> None:
    """新变体与同组 V0 的逐日覆盖、事件对照、重叠与专项清单。"""
    set_id = vs.set_id
    cov_rows, cells, excluded = an.coverage(base, view)
    rows.coverage += cov_rows
    out.coverage_cells.append({"set_id": set_id, "target": label, **cells})
    if sum(cells[k] for k in ("both", "only_v0", "only_new", "neither")) != cells["common_days"]:
        out.checks.append(f"{set_id}/{label}：覆盖四格之和不等于共同可计算日数")
    out.excluded += excluded
    cmp_rows = an.event_compare(base, view)
    rows.compare += cmp_rows
    out.checks += coverage_check(set_id, label, base, view, cmp_rows)
    rows.overlap += an.overlap_rows(base, view, an.overlaps(base, view))
    out.cancelled_escalations += [
        {"set_id": set_id, **r} for r in an.cancelled_escalations(base, view)
    ]
    out.l_starts += [{"set_id": set_id, **r} for r in an.l_path_starts(base, view)]
    if vs.variant == "V1":
        out.l_release_no_rise += [{"set_id": set_id, **r} for r in an.l_release_without_rise(view)]
