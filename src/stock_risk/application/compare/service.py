"""用例：T06b 对照回放与 T06c 时间线。输出到 output/compare/<run_id>/。

按（结构组合, N）分组多进程（每个子进程初始化时接收一次全部回放对象的数据；组内 V0 基准与
r2_floor OFF 配对只运行一次，T08a D-04）。父进程汇总：逐日覆盖四格、
单开关配对、专项清单、通知负担、运行清单（I16 字段加变体全集），以及 I17 ② 检查与时间线。
I17 ②（T10-M0b 第 2 节）：预期检查键在运行开始时由 I17_SETS × I17_TARGETS 生成；固定抽样上，
单进程不缓存与多进程缓存两种模式的离散摘要与全语义摘要都必须相同；
缺项、重复或检查为空时 i17_ok 为假。
四模式与倒序提交由专用合成测试检查（tests/unit/application/test_i17_modes.py）。
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
from stock_risk.application.replay_run import BACKGROUND_SYMBOLS, i17_ok
from stock_risk.application.replay_run_types import discrete_digest
from stock_risk.application.run_manifest import (
    RunEnvironment,
    RunStart,
    base_manifest,
    data_identity,
    finish_manifest,
    start_run,
)
from stock_risk.domain.lifecycle.params import EXPLORATION_ASSUMPTIONS
from stock_risk.domain.lifecycle.semantic_digest import semantic_digest

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
    expected: tuple[tuple[str, str], ...]  # I17 ② 预期检查键（套, 对象）


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
        *,
        i17_sets: Sequence[str] = I17_SETS,
        i17_targets: Sequence[str] = I17_TARGETS,
        reverse_submission: bool = False,
    ) -> dict[str, Any]:
        """i17_sets、i17_targets、reverse_submission 只供专用合成测试使用；正式运行取默认值。"""
        # 只运行部分套或部分对象时不得作为正式运行
        run = start_run(self._sink, self._env, started, not only_sets and not symbols)
        expected = i17_expected_keys(i17_sets, i17_targets)
        st = self._setup(run, only_sets, symbols, expected)
        self._sink.write_plotly_js(run.run_dir)
        n_sets, n_targets = len(st.todo), len(st.prepared)
        self._progress(f"run_id = {run.run_id}（{run.run_kind}），{n_sets} 套 × {n_targets} 个对象")
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        t0 = clock()
        outcomes = self._run_parallel(st, workers, reverse_submission)
        parallel_seconds = round((clock() - t0).total_seconds(), 1)
        manifest = self._manifest(st, outcomes, started)
        manifest |= {"workers": workers, "parallel_seconds": parallel_seconds}
        manifest = dict(sorted(manifest.items(), key=lambda kv: _MANIFEST_ORDER.index(kv[0])))
        self._sink.write_manifest(run.run_dir, finish_manifest(manifest, clock()))
        manifest["run_dir"] = str(run.run_dir)
        return manifest

    def _setup(
        self,
        run: RunStart,
        only_sets: Sequence[str] | None,
        symbols: Sequence[str] | None,
        expected: Sequence[tuple[str, str]],
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
            frozenset(expected),
            background,
            self._sink,
        )
        return _Setup(run, sets, todo, prepared, ctx, (v0_tl, v3_tl), tuple(expected))

    def _run_parallel(
        self, st: _Setup, workers: int, reverse: bool = False
    ) -> dict[str, SetOutcome]:
        outcomes: dict[str, SetOutcome] = {}
        groups = submission_groups(st.ctx.sets, st.todo, reverse)
        self.last_submission = [list(g) for g in groups]  # 供测试核对提交顺序
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
        i17, missing = self._i17(st.ctx, st.prepared, outcomes, st.expected)
        self.last_outcomes = outcomes  # 供测试按检查键比较各次运行
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
            **_identity_field(self._replay.loader.reader),
            "timeline_sets": [v0_tl.set_id, v3_tl.set_id],
            "failed": failed,
            "invariant_violations": len(violations),
            "invariant_examples": violations[:20],
            "program_checks": checks[:50],
            "program_check_problem_count": len(checks),
            "i17_check": i17,
            "i17_ok": i17_ok(i17, st.expected, missing),
            "i17_expected_keys": [list(k) for k in st.expected],
            "i17_missing_keys": [list(k) for k in missing],
            "outputs": tables,
            "timelines": timelines["files"],
        }

    def _i17(
        self,
        ctx: WorkerContext,
        prepared: Sequence[Any],
        outcomes: Mapping[str, SetOutcome],
        expected: Sequence[tuple[str, str]],
    ) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
        """I17 ②：预期检查键上，单进程（不缓存）与多进程（缓存）的离散摘要与全语义摘要都相同。
        套不在结果中、对象未回放或多进程未返回摘要的键记为缺项（不再跳过）。"""
        out: list[dict[str, Any]] = []
        missing: list[tuple[str, str]] = []
        g = EXPLORATION_ASSUMPTIONS.gap_break
        by_label = {p.target.label: p for p in prepared}
        for set_id, label in expected:
            outcome = outcomes.get(set_id)
            p = by_label.get(label)
            if (
                outcome is None
                or p is None
                or label not in outcome.digests
                or label not in (outcome.semantic)
            ):
                missing.append((set_id, label))
                continue
            fresh = self._replay.prepare(p.target, g, cache=False)
            result = replay_prepared(fresh, params_of(ctx.sets[set_id]), ctx.calendar)
            out.append(
                {
                    "key": [set_id, label],
                    "set_id": set_id,
                    "target": label,
                    "equal": discrete_digest(result) == outcome.digests[label],
                    "semantic_equal": semantic_digest(result) == outcome.semantic[label],
                    "days": len(result.results),
                }
            )
        return out, missing


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
    "i17_expected_keys",
    "i17_missing_keys",
    "outputs",
    "timelines",
]


def i17_expected_keys(sets: Sequence[str], targets: Sequence[str]) -> list[tuple[str, str]]:
    """预期检查键（套, 对象），在运行开始时由两组常量生成。"""
    return [(s, t) for s in sets for t in targets]


def submission_groups(
    by_id: Mapping[str, VariantSet], todo: Sequence[str], reverse: bool = False
) -> list[list[str]]:
    """实际提交给进程池的组序列：按组排序之后的序列；reverse 只供专用测试倒序提交（T10-M0b）。"""
    groups = _groups(by_id, todo)
    return list(reversed(groups)) if reverse else groups


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


def _identity_field(reader: object) -> dict[str, object]:
    """规范数据时运行清单增加 data_identity（T11-D0a 第 7.4 节）；旧数据版本不增加字段。"""
    identity = data_identity(reader)
    return {} if identity is None else {"data_identity": identity}
