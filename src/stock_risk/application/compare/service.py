"""用例：T06b 对照回放与 T06c 时间线。输出到 output/compare/<run_id>/。

按套多进程（每个子进程初始化时接收一次全部回放对象的数据）。父进程汇总：逐日覆盖四格、
单开关配对、专项清单、通知负担、运行清单（I16 字段加变体全集），以及 I17 ② 检查与时间线。
只描述，不选参、不设门槛、不排名、不下结论。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from stock_risk.application.compare.tables import pair_coverage, write_tables
from stock_risk.application.compare.timelines import build_timelines
from stock_risk.application.compare.variants import (
    VariantSet,
    pairs,
    timeline_sets,
    variant_sets,
)
from stock_risk.application.compare.worker import (
    CompareSink,
    SetOutcome,
    WorkerContext,
    init_worker,
    params_of,
    run_set,
    run_set_with,
)
from stock_risk.application.replay import ReplayService, replay_prepared
from stock_risk.application.replay_chart_data import drawdown_63
from stock_risk.application.replay_run import (
    BACKGROUND_SYMBOLS,
    RunEnvironment,
    discrete_digest,
    make_run_id,
)
from stock_risk.domain.lifecycle.params import EXPLORATION_ASSUMPTIONS, FIXED
from stock_risk.domain.numeric import DEFAULT_TOLERANCES

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


class CompareService:
    def __init__(
        self,
        replay: ReplayService,
        sink: CompareOutputSink,
        env: RunEnvironment,
        progress: Callable[[str], None] = print,
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
        git = self._env.git_state()
        run_kind = (
            "formal"
            if (git.commit and not git.dirty and not only_sets and not symbols)
            else "debug"
        )
        run_id = make_run_id(started, git.commit)
        n = 2
        while self._sink.run_dir_exists(run_id):
            run_id = f"{make_run_id(started, git.commit)}-{n}"
            n += 1
        run_dir = self._sink.create_run_dir(run_id)
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
        calendar = self._replay.loader.calendar()
        background = {
            s: drawdown_63(self._replay.loader.load(s).history.bars) for s in BACKGROUND_SYMBOLS
        }
        v0_tl, v3_tl = timeline_sets(sets)
        digest_ids = frozenset((s, t) for s in I17_SETS for t in I17_TARGETS)
        ctx = WorkerContext(
            run_dir,
            run_id,
            prepared,
            calendar,
            by_id,
            frozenset({v0_tl.set_id, v3_tl.set_id}),
            digest_ids,
            background,
            self._sink,
        )
        self._sink.write_plotly_js(run_dir)
        self._progress(f"run_id = {run_id}（{run_kind}），{len(todo)} 套 × {len(prepared)} 个对象")
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        t0 = clock()
        outcomes: dict[str, SetOutcome] = {}
        with ProcessPoolExecutor(
            max_workers=workers, initializer=init_worker, initargs=(ctx,)
        ) as pool:
            for k, out in enumerate(pool.map(run_set, todo, chunksize=1), start=1):
                outcomes[out.set_id] = out
                if k % 24 == 0 or k == len(todo):
                    self._progress(f"  {k}/{len(todo)} 套完成")
        parallel_seconds = round((clock() - t0).total_seconds(), 1)
        i17 = self._i17(ctx, prepared, outcomes)
        tables = write_tables(self._sink, run_dir, sets, outcomes)
        timelines = build_timelines(self._sink, run_dir, run_id, outcomes, v0_tl, v3_tl)
        failed = [f for o in outcomes.values() for f in o.failed]
        violations = [v for o in outcomes.values() for v in o.violations]
        checks = [c for o in outcomes.values() for c in o.checks] + timelines["problems"]
        manifest = {
            "run_id": run_id,
            "run_kind": run_kind,
            "mode": "compare_t06",
            "git": asdict(git),
            "started_at": started.isoformat(),
            "documents_and_config": self._env.file_hashes(),
            "versions": self._env.versions(),
            "parameters": {
                "fixed_values": asdict(FIXED),
                "exploration_assumptions_I6a": {
                    **asdict(EXPLORATION_ASSUMPTIONS),
                    "note": "探索假设，非发起人批准值",
                },
                "numeric_tolerances_I6b": {**asdict(DEFAULT_TOLERANCES), "note": "待登记"},
            },
            "variant_sets": [_set_record(s) for s in sets],
            "variant_set_count": len(sets),
            "completed_sets": sorted(outcomes),
            "pairs": len(pairs(sets)),
            "targets": [p.target.label for p in prepared],
            "summary_period": ["2010-01-01", "2019-12-31"],
            "timeline_sets": [v0_tl.set_id, v3_tl.set_id],
            "workers": workers,
            "parallel_seconds": parallel_seconds,
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
        finished = clock()
        manifest["finished_at"] = finished.isoformat()
        manifest["elapsed_seconds"] = round((finished - started).total_seconds(), 1)
        self._sink.write_manifest(run_dir, manifest)
        manifest["run_dir"] = str(run_dir)
        return manifest

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
