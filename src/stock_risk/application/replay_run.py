"""用例：一次回放运行（T04b、T04c）。生成 run_id 与运行清单（I16），输出到 output/replay/<run_id>/。

- reference12：12 组参考候选（I7）× 全部个股，
  写出 events、daily、episodes、chart 与 index、summary；

  每个结果读回做一致性检查，并运行不变量检查器。
- all：768 组全量，只写 summary；按个股多进程（I17 第 ② 项：固定抽样的检查键上，单进程不缓存、
单进程缓存与多进程缓存三种模式的离散摘要与全语义摘要都相同；T10-M0b 第 2 节）。
  预期检查键在运行开始时由 I17_SYMBOLS × I17_CANDIDATES 生成，实际检查的键必须与之完全相等。
  四模式与倒序提交由专用合成测试检查（tests/unit/application/test_i17_modes.py）。
工作区不干净或 git 不可用时 run_kind = debug；干净时为 formal（I16、P-26）。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

from stock_risk.application.progress import ProgressFn, quiet
from stock_risk.application.replay import (
    PreparedSymbol,
    ReplayService,
    ReplayTarget,
    SymbolReplay,
    replay_prepared,
)
from stock_risk.application.replay_chart_data import build_chart_data, drawdown_63
from stock_risk.application.replay_consistency import check_consistency
from stock_risk.application.replay_episodes import episode_csv_rows, episode_rows
from stock_risk.application.replay_outputs import daily_rows, event_rows
from stock_risk.application.replay_run_types import (
    ReplayOutputSink,
    RunReport,
    discrete_digest,
)
from stock_risk.application.replay_summary import (
    SUMMARY_END,
    SUMMARY_START,
    summarize,
    summarize_candidates,
)
from stock_risk.application.run_manifest import (
    GitState,
    RunEnvironment,
    RunStart,
    base_manifest,
    data_identity,
    finish_manifest,
    make_run_id,
    start_run,
)
from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.lifecycle.invariants import check_invariants
from stock_risk.domain.lifecycle.params import (
    EXPLORATION_ASSUMPTIONS,
    Candidate,
    EngineParams,
    candidate_grid,
    reference_candidates,
)
from stock_risk.domain.lifecycle.semantic_digest import semantic_digest

BACKGROUND_SYMBOLS = ("SPY", "QQQ")
I17_CANDIDATES = (1, 385, 768)
I17_SYMBOLS = ("AAPL", "NFLX", "VRX", "GE", "SHLD_corrected")
# GitState、RunEnvironment、make_run_id 定义在 run_manifest，这里转出供原有调用方使用
__all__ = [
    "GitState",
    "ProgressFn",
    "ReplayRunService",
    "ReplayOutputSink",
    "RunEnvironment",
    "RunReport",
    "discrete_digest",
    "make_run_id",
]


def _summarize_target(
    args: tuple[PreparedSymbol, TradingCalendar, tuple[tuple[int, Candidate], ...], frozenset[int]],
) -> dict[str, Any]:
    """多进程的工作单元：一个回放对象 × 一组候选。返回汇总行、不变量违反数，
    以及指定候选的离散摘要与全语义摘要（I17 ②）。"""
    prepared, calendar, cands, digest_ids = args
    rows: list[dict[str, Any]] = []
    digests: dict[int, tuple] = {}
    semantic: dict[int, str] = {}
    violations: list[str] = []
    failed: list[str] = []
    for cid, cand in cands:
        params = EngineParams(cand, EXPLORATION_ASSUMPTIONS)
        try:
            result = replay_prepared(prepared, params, calendar)
        except Exception as e:  # noqa: BLE001 —— 记录失败的候选，不吞掉：写入运行清单并使运行失败
            failed.append(f"c{cid:03d}/{prepared.target.label}: {type(e).__name__}: {e}")
            continue
        rep = SymbolReplay(prepared.target, cid, params, prepared, result)
        violations += [
            f"c{cid:03d}/{prepared.target.label}: {v}"
            for v in check_invariants(result.results, params)
        ]
        rows.append(summarize(rep, episode_rows(rep)))
        if cid in digest_ids:
            digests[cid] = discrete_digest(result)
            semantic[cid] = semantic_digest(result)
    return {
        "target": prepared.target.label,
        "rows": rows,
        "violations": violations,
        "failed": failed,
        "digests": digests,
        "semantic": semantic,
    }


def submission_order[T](items: Sequence[T], reverse: bool = False) -> list[T]:
    """实际提交给进程池的任务序列（T10-M0b 第 2 节）；reverse 只供专用测试倒序提交。"""
    return list(reversed(items)) if reverse else list(items)


def i17_expected_keys(symbols: Sequence[str], candidates: Sequence[int]) -> list[tuple[str, int]]:
    """预期检查键（对象, 候选），在运行开始时由两组常量生成。"""
    return [(s, c) for s in symbols for c in candidates]


def i17_ok(
    records: Sequence[Mapping[str, Any]], expected: Sequence[Any], missing: Sequence[Any]
) -> bool:
    """每条记录的离散与全语义比较都为真，没有缺项、没有重复，且检查不为空。"""
    keys = [tuple(r["key"]) for r in records]
    return (
        bool(records)
        and not missing
        and len(keys) == len(set(keys))
        and set(keys) == {tuple(k) for k in expected}
        and all(r["equal"] and r["semantic_equal"] for r in records)
    )


@dataclass(slots=True)
class _ReferenceAcc:
    rows: list[dict[str, Any]] = field(default_factory=list)
    cells: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)


def _report(run: RunStart, manifest: dict[str, Any]) -> RunReport:
    return RunReport(run.run_id, run.run_kind, run.run_dir, manifest)


def _target_record(p: PreparedSymbol) -> dict[str, Any]:
    return {
        "label": p.target.label,
        "symbol": p.target.symbol,
        "variant": p.target.variant,
        "replay_start": p.data.ranges.replay_start.isoformat(),
        "observation_end": p.data.ranges.observation_end.isoformat(),
        "right_end": p.data.ranges.right_end.value,
        "left_end": p.data.ranges.left_end.value,
    }


class ReplayRunService:
    def __init__(
        self,
        replay: ReplayService,
        sink: ReplayOutputSink,
        env: RunEnvironment,
        progress: ProgressFn = quiet,
    ) -> None:
        self._replay = replay
        self._sink = sink
        self._env = env
        self._progress = progress

    def _start(
        self, started: datetime, mode: str, symbols: Sequence[str] | None
    ) -> tuple[RunReport, list[PreparedSymbol]]:
        run = start_run(self._sink, self._env, started)
        g = EXPLORATION_ASSUMPTIONS.gap_break
        prepared = [self._replay.prepare(t, g) for t in self._replay.targets(symbols)]
        manifest = base_manifest(run, self._env, mode, started)
        manifest["calendar"] = {
            "source": "SPY 日历代理（I1），data/raw/SPY.csv",
            "sha256": self._env.file_hashes()["calendar_spy"]["sha256"],
        }
        manifest["summary_period"] = [SUMMARY_START.isoformat(), SUMMARY_END.isoformat()]
        identity = data_identity(self._replay.loader.reader)
        if identity is not None:
            manifest["data_identity"] = identity
        manifest["targets"] = [_target_record(p) for p in prepared]
        self._progress(f"run_id = {run.run_id}（{run.run_kind}），{len(prepared)} 个回放对象")
        return _report(run, manifest), prepared

    def _finish(self, report: RunReport, finished: datetime) -> RunReport:
        self._sink.write_manifest(report.run_dir, finish_manifest(report.manifest, finished))
        return report

    # ---- T04b ----
    def run_reference(
        self, started: datetime, clock: Callable[[], datetime], symbols: Sequence[str] | None = None
    ) -> RunReport:
        report, prepared = self._start(started, "reference12", symbols)
        cands = reference_candidates()
        report.manifest["candidates"] = [{"id": i, **asdict(c)} for i, c in cands]
        calendar = self._replay.loader.calendar()
        background = {
            s: drawdown_63(self._replay.loader.load(s).history.bars) for s in BACKGROUND_SYMBOLS
        }
        acc = _ReferenceAcc()
        for cid, cand in cands:
            params = EngineParams(cand, EXPLORATION_ASSUMPTIONS)
            label = f"{cand.confirm}/{cand.release_path}/{cand.atr_basis}"
            for p in prepared:
                self._reference_target(report, p, cid, params, label, calendar, background, acc)
            acc.completed.append(cid)
            self._progress(f"  候选 {cid} 完成")
        rows, cells, problems, violations = acc.rows, acc.cells, acc.problems, acc.violations
        completed = acc.completed
        self._sink.write_summary(report.run_dir, rows, summarize_candidates(rows))
        self._sink.write_index(report.run_dir, report.run_id, report.run_kind, cells)
        report.manifest.update(
            completed_candidates=completed,
            failed=[],
            invariant_violations=len(violations),
            invariant_examples=violations[:20],
            consistency_problems=problems[:50],
            consistency_problem_count=len(problems),
        )
        return self._finish(report, clock())

    def _reference_target(
        self,
        report: RunReport,
        p: PreparedSymbol,
        cid: int,
        params: EngineParams,
        label: str,
        calendar: TradingCalendar,
        background: dict[str, dict[date, float]],
        acc: _ReferenceAcc,
    ) -> None:
        """一个（候选, 对象）：回放、写出四种输出、读回做一致性检查、汇总。"""
        result = replay_prepared(p, params, calendar)
        rep = SymbolReplay(p.target, cid, params, p, result)
        where = f"c{cid:03d}/{p.target.label}"
        acc.violations += [f"{where}: {v}" for v in check_invariants(result.results, params)]
        eps = episode_rows(rep)
        ev, dy = event_rows(rep, report.run_id), daily_rows(rep, report.run_id)
        ep = episode_csv_rows(rep, report.run_id, eps)
        chart = build_chart_data(rep, report.run_id, background)
        self._sink.write_target(report.run_dir, cid, p.target.label, ev, dy, ep, chart)
        back = self._sink.read_target(report.run_dir, cid, p.target.label)
        acc.problems += [f"{where}: {x}" for x in check_consistency(*back)]
        acc.rows.append(summarize(rep, eps))
        acc.cells.append(
            {"target": p.target.label, "candidate_id": cid, "label": label, "episodes": len(eps)}
        )

    # ---- T04c ----
    def run_all(
        self,
        started: datetime,
        clock: Callable[[], datetime],
        symbols: Sequence[str] | None = None,
        workers: int | None = None,
        *,
        grid: Sequence[tuple[int, Candidate]] | None = None,
        i17_symbols: Sequence[str] = I17_SYMBOLS,
        i17_candidates: Sequence[int] = I17_CANDIDATES,
        reverse_submission: bool = False,
    ) -> RunReport:
        """grid、i17_symbols、i17_candidates、reverse_submission 只供专用合成测试使用；
        正式运行取默认值（768 组全量，固定抽样，正常提交顺序）。"""
        report, prepared = self._start(started, "all768", symbols)
        expected = i17_expected_keys(i17_symbols, i17_candidates)
        grid = tuple(grid) if grid is not None else candidate_grid()
        report.manifest["candidates"] = {
            "grid": "10 号第 4.1 节，字典序编号 1–768",
            "count": len(grid),
        }
        calendar = self._replay.loader.calendar()
        workers = workers or max(1, (os.cpu_count() or 2) - 1)
        digest_ids = frozenset(i17_candidates)
        tasks = [(p, calendar, grid, digest_ids) for p in prepared]
        submitted = submission_order(tasks, reverse_submission)
        self.last_submission = [t[0].target.label for t in submitted]  # 供测试核对提交顺序
        t0 = clock()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outs = []
            for out in pool.map(_summarize_target, submitted):
                outs.append(out)
                self._progress(f"  {out['target']}：{len(out['rows'])} 组完成")
        rows = [r for o in outs for r in o["rows"]]
        violations = [v for o in outs for v in o["violations"]]
        failed = [f for o in outs for f in o["failed"]]
        multi = {o["target"]: (o["digests"], o["semantic"]) for o in outs}
        self.last_multi = multi  # 供测试按检查键比较各次运行
        i17, missing = self._i17_check(prepared, calendar, multi, expected)
        self._sink.write_summary(report.run_dir, rows, summarize_candidates(rows))
        report.manifest.update(
            workers=workers,
            parallel_seconds=round((clock() - t0).total_seconds(), 1),
            completed_candidates=sorted({int(r["candidate_id"]) for r in rows}),
            summary_rows=len(rows),
            failed=failed,
            invariant_violations=len(violations),
            invariant_examples=violations[:20],
            i17_check=i17,
            i17_ok=i17_ok(i17, expected, missing),
            i17_expected_keys=[list(k) for k in expected],
            i17_missing_keys=[list(k) for k in missing],
        )
        return self._finish(report, clock())

    def _i17_check(
        self,
        prepared: Sequence[PreparedSymbol],
        calendar: TradingCalendar,
        multi: Mapping[str, tuple[Mapping[int, tuple], Mapping[int, str]]],
        expected: Sequence[tuple[str, int]],
    ) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
        """I17 ②：预期检查键上，单进程（不缓存、缓存）与多进程（缓存）的离散摘要与全语义摘要
        都完全相同。返回 (记录, 缺项)；对象未回放或多进程未返回的键记为缺项。"""
        grid = dict(candidate_grid())
        by_label = {p.target.label: p for p in prepared}
        fresh_cache: dict[str, PreparedSymbol] = {}
        out: list[dict[str, Any]] = []
        missing: list[tuple[str, int]] = []
        for label, cid in expected:
            p = by_label.get(label)
            discrete_m, semantic_m = multi.get(label, ({}, {}))
            if p is None or cid not in discrete_m or cid not in semantic_m or cid not in grid:
                missing.append((label, cid))
                continue
            if label not in fresh_cache:
                fresh_cache[label] = self._replay.prepare(
                    ReplayTarget(p.target.symbol, p.target.variant), p.gap_break, cache=False
                )
            params = EngineParams(grid[cid], EXPLORATION_ASSUMPTIONS)
            nocache = replay_prepared(fresh_cache[label], params, calendar)
            cached = replay_prepared(p, params, calendar)
            out.append(
                {
                    "key": [label, cid],
                    "target": label,
                    "candidate_id": cid,
                    "equal": discrete_digest(nocache) == discrete_digest(cached) == discrete_m[cid],
                    "semantic_equal": semantic_digest(nocache)
                    == semantic_digest(cached)
                    == semantic_m[cid],
                    "days": len(nocache.results),
                }
            )
        return out, missing
