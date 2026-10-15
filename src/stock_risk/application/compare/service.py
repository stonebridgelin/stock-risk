"""用例：T06b 对照回放与 T06c 时间线。输出到 output/compare/<run_id>/。

按（结构组合, N）分组多进程（每个子进程初始化时接收一次全部回放对象的数据；组内 V0 基准与
r2_floor OFF 配对只运行一次，T08a D-04）。父进程汇总：逐日覆盖四格、
单开关配对、专项清单、通知负担、运行清单（I16 字段加变体全集），以及 I17 ② 检查与时间线。
只描述，不选参、不设门槛、不排名、不下结论。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from stock_risk.application.compare.set_outputs import CompareSink
from stock_risk.application.compare.tables import pair_coverage, write_tables
from stock_risk.application.compare.timelines import build_timelines
from stock_risk.application.compare.variants import (
    VariantSet,
    pairs,
    timeline_sets,
    variant_sets,
)
from stock_risk.application.compare.worker import (
    SetOutcome,
    WorkerContext,
    group_key,
    init_worker,
    params_of,
    run_group,
    run_set_with,
)
from stock_risk.application.progress import ProgressFn, quiet
from stock_risk.application.replay import PreparedSymbol, ReplayService, replay_prepared
from stock_risk.application.replay_chart_data import drawdown_63
from stock_risk.application.replay_run import BACKGROUND_SYMBOLS
from stock_risk.application.replay_run_types import discrete_digest
from stock_risk.application.run_manifest import (
    RunEnvironment,
    RunStart,
    base_manifest,
    finish_manifest,
    start_run,
)
from stock_risk.domain.lifecycle.params import EXPLORATION_ASSUMPTIONS

I17_SETS = (
    "V0_x0_x0_OFF_r2off_N20_s385",
    "V3_x8_x15_X1_r2off_N20_s385",
    "V4_x10_x20_X2_r2on_N20_s396",
)
I17_TARGETS = ("AAPL", "NFLX", "VRX", "GE", "SHLD_corrected")


class CompareOutputSink(CompareSink, Protocol):
    def create_run_dir(self, run_id: str) -> Path: ...

    def run_dir_exists(self, run_id: str) -> bool: ...

    def write_manifest(self, run_dir: Path, manifest: Mapping[str, Any]) -> Path: ...

    def write_plotly_js(self, run_dir: Path) -> None: ...

    def write_timeline_index(
        self, run_dir: Path, run_id: str, files: Sequence[Mapping[str, Any]]
    ) -> Path: ...

    def read_csv(self, path: Path) -> list[dict[str, str]]: ...


@dataclass(frozen=True, slots=True)
class _Setup:
    run: RunStart
    sets: tuple[VariantSet, ...]
    todo: list[str]
    prepared: tuple[PreparedSymbol, ...]
    ctx: WorkerContext
    timeline: tuple[VariantSet, VariantSet]


class CompareService:
    def __init__(
        self,
        replay: ReplayService,
        sink: CompareOutputSink,
        env: RunEnvironment,
        progress: ProgressFn = quiet,
    ) -> None:
        self._replay = replay
        self._sink = sink
        self._env = env
        self._progress = progress

    def run(
        self,
        started: datetime,
        clock: Callable[[], datetime],
        workers: int | None = None,
        only_sets: Sequence[str] | None = None,
        symbols: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        # 只运行部分套或部分对象时不得作为正式运行
        run = start_run(self._sink, self._env, started, not only_sets and not symbols)
        st = self._setup(run, only_sets, symbols)
        self._sink.write_plotly_js(run.run_dir)
        n_sets, n_targets = len(st.todo), len(st.prepared)
        self._progress(f"run_id = {run.run_id}（{run.run_kind}），{n_sets} 套 × {n_targets} 个对象")
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        t0 = clock()
        outcomes = self._run_parallel(st, workers)
        parallel_seconds = round((clock() - t0).total_seconds(), 1)
        manifest = self._manifest(st, outcomes, started)
        manifest |= {"workers": workers, "parallel_seconds": parallel_seconds}
        manifest = dict(sorted(manifest.items(), key=lambda kv: _MANIFEST_ORDER.index(kv[0])))
        self._sink.write_manifest(run.run_dir, finish_manifest(manifest, clock()))
        manifest["run_dir"] = str(run.run_dir)
        return manifest

    def _setup(
        self, run: RunStart, only_sets: Sequence[str] | None, symbols: Sequence[str] | None
    ) -> _Setup:
        sets = variant_sets()
        by_id = {s.set_id: s for s in sets}
        todo = [
            s.set_id
            for s in sets
            if not only_sets
            or s.set_id in only_sets
            or any(x.baseline_id == s.set_id for x in sets if x.set_id in only_sets)
        ]
        targets = [t for t in self._replay.targets(symbols) if t.variant != "uncorrected"]
        g = EXPLORATION_ASSUMPTIONS.gap_break
        prepared = tuple(self._replay.prepare(t, g) for t in targets)
        background = {
            s: drawdown_63(self._replay.loader.load(s).history.bars) for s in BACKGROUND_SYMBOLS
        }
        v0_tl, v3_tl = timeline_sets(sets)
        ctx = WorkerContext(
            run.run_dir,
            run.run_id,
            prepared,
            self._replay.loader.calendar(),
            by_id,
            frozenset({v0_tl.set_id, v3_tl.set_id}),
            frozenset((s, t) for s in I17_SETS for t in I17_TARGETS),
            background,
            self._sink,
        )
        return _Setup(run, sets, todo, prepared, ctx, (v0_tl, v3_tl))

    def _run_parallel(self, st: _Setup, workers: int) -> dict[str, SetOutcome]:
        outcomes: dict[str, SetOutcome] = {}
        groups = _groups(st.ctx.sets, st.todo)
        with ProcessPoolExecutor(
            max_workers=workers, initializer=init_worker, initargs=(st.ctx,)
        ) as pool:
            for group_outs in pool.map(run_group, groups, chunksize=1):
                outcomes |= {o.set_id: o for o in group_outs}
                self._progress(f"  {len(outcomes)}/{len(st.todo)} 套完成")
        return outcomes

    def _manifest(
        self, st: _Setup, outcomes: Mapping[str, SetOutcome], started: datetime
    ) -> dict[str, Any]:
        run, sets = st.run, st.sets
        v0_tl, v3_tl = st.timeline
        i17 = self._i17(st.ctx, st.prepared, outcomes)
        tables = write_tables(self._sink, run.run_dir, sets, outcomes)
        timelines = build_timelines(self._sink, run.run_dir, run.run_id, outcomes, v0_tl, v3_tl)
        failed = [f for o in outcomes.values() for f in o.failed]
        violations = [v for o in outcomes.values() for v in o.violations]
        checks = [c for o in outcomes.values() for c in o.checks] + timelines["problems"]
        return base_manifest(run, self._env, "compare_t06", started) | {
            "variant_sets": [_set_record(s) for s in sets],
            "variant_set_count": len(sets),
            "completed_sets": sorted(outcomes),
            "pairs": len(pairs(sets)),
            "targets": [p.target.label for p in st.prepared],
            "summary_period": ["2010-01-01", "2019-12-31"],
            "timeline_sets": [v0_tl.set_id, v3_tl.set_id],
            "failed": failed,
            "invariant_violations": len(violations),
            "invariant_examples": violations[:20],
            "program_checks": checks[:50],
            "program_check_problem_count": len(checks),
            "i17_check": i17,
            "i17_ok": all(x["equal"] for x in i17),
            "outputs": tables,
            "timelines": timelines["files"],
        }

    def _i17(
        self, ctx: WorkerContext, prepared: Sequence[Any], outcomes: Mapping[str, SetOutcome]
    ) -> list[dict[str, Any]]:
        """I17 ②：单进程（不缓存）与多进程（缓存）的离散结果相同（3 套 × 5 个对象）。"""
        out = []
        g = EXPLORATION_ASSUMPTIONS.gap_break
        for set_id in I17_SETS:
            if set_id not in outcomes:
                continue
            vs = ctx.sets[set_id]
            for p in prepared:
                if p.target.label not in I17_TARGETS:
                    continue
                fresh = self._replay.prepare(p.target, g, cache=False)
                single = discrete_digest(replay_prepared(fresh, params_of(vs), ctx.calendar))
                multi = outcomes[set_id].digests.get(p.target.label)
                out.append(
                    {
                        "set_id": set_id,
                        "target": p.target.label,
                        "equal": single == multi,
                        "days": len(single),
                    }
                )
        return out


# 运行清单的字段顺序（与 T06 正式运行相同）
_MANIFEST_ORDER = [
    "run_id",
    "run_kind",
    "mode",
    "git",
    "started_at",
    "documents_and_config",
    "versions",
    "parameters",
    "variant_sets",
    "variant_set_count",
    "completed_sets",
    "pairs",
    "targets",
    "summary_period",
    "timeline_sets",
    "workers",
    "parallel_seconds",
    "failed",
    "invariant_violations",
    "invariant_examples",
    "program_checks",
    "program_check_problem_count",
    "i17_check",
    "i17_ok",
    "outputs",
    "timelines",
]


def _groups(by_id: Mapping[str, VariantSet], todo: Sequence[str]) -> list[list[str]]:
    """按（结构组合, N）分组，套数多的组先提交（负载均衡）。组内顺序由 worker.group_plan 决定。"""
    groups: dict[tuple[int, int], list[str]] = {}
    for sid in todo:
        groups.setdefault(group_key(by_id[sid]), []).append(sid)
    return sorted(groups.values(), key=len, reverse=True)


def _set_record(s: VariantSet) -> dict[str, Any]:
    sw = s.switches
    return {
        "set_id": s.set_id,
        "group": s.group,
        "variant": s.variant,
        "candidate_id": s.candidate_id,
        "n": s.n,
        "confirm": s.candidate.confirm.value,
        "release_path": s.candidate.release_path.value,
        "atr_basis": s.candidate.atr_basis.value,
        "baseline": s.baseline_id,
        "switches": {
            "x1": sw.x1,
            "x2": sw.x2,
            "t11_full": sw.t11_full,
            "long_path": sw.long_path.value,
            "r2_floor": sw.r2_floor,
            "release_hyst": sw.release_hyst.value,
            "rec_scale": sw.rec_scale.value,
        },
    }


__all__ = ["CompareService", "pair_coverage", "run_set_with"]
