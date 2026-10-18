"""T10-M0b 第 2 节：模式等价的专用合成测试（I-02）。

回放与对照回放各一组，覆盖四种模式：单进程不缓存、单进程缓存、多进程缓存、提交顺序倒序的多进程。
摘要按检查键对齐后比较，不按结果列表的位置比较。合成数据，只写临时目录，不访问网络。

正式回放为固定抽样的三模式检查；正式对照回放为固定抽样的两模式检查；
四模式与倒序提交由本文件的专用合成测试检查。
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from stock_risk.application.compare.service import (
    _MANIFEST_ORDER,
    CompareService,
    submission_groups,
)
from stock_risk.application.compare.variants import variant_sets
from stock_risk.application.compare.worker import params_of
from stock_risk.application.replay import CORRECTED, ReplayService, ReplayTarget, replay_prepared
from stock_risk.application.replay_run import (
    ReplayRunService,
    discrete_digest,
    i17_expected_keys,
    submission_order,
)
from stock_risk.domain.lifecycle.params import EXPLORATION_ASSUMPTIONS, EngineParams, candidate_grid
from stock_risk.domain.lifecycle.semantic_digest import semantic_digest
from stock_risk.infrastructure.replay.compare_files import CompareFileSink
from stock_risk.infrastructure.replay.files import ReplayFileSink
from tests.unit.application.test_replay_run import FakeEnv, service  # noqa: F401  (fixture)

T0 = datetime(2026, 10, 17, 9, 0, 0)
GRID = dict(candidate_grid())
CIDS = (1, 385)
SYMS = ("AAA", "BBB_corrected")  # 回放对象的标签（BBB 有登记的更正，标签带变体）
TARGETS = {"AAA": ReplayTarget("AAA"), "BBB_corrected": ReplayTarget("BBB", CORRECTED)}
G = EXPLORATION_ASSUMPTIONS.gap_break


def _clock() -> datetime:
    return T0 + timedelta(seconds=1)


# ---- 回放 ----


def _replay_run(svc: ReplayService, tmp: Path, reverse: bool) -> tuple[ReplayRunService, dict]:
    run = ReplayRunService(svc, ReplayFileSink(tmp), FakeEnv(False), progress=lambda _: None)
    report = run.run_all(
        T0,
        _clock,
        workers=2,
        grid=[(c, GRID[c]) for c in CIDS],
        i17_symbols=SYMS,
        i17_candidates=CIDS,
        reverse_submission=reverse,
    )
    return run, report.manifest


def test_replay_four_modes_and_reverse_submission(service, tmp_path: Path) -> None:  # noqa: F811
    normal, m1 = _replay_run(service, tmp_path / "a", reverse=False)
    rev, m2 = _replay_run(service, tmp_path / "b", reverse=True)
    expected = [list(k) for k in i17_expected_keys(SYMS, CIDS)]
    for m in (m1, m2):
        assert m["i17_ok"] is True and m["i17_missing_keys"] == []
        assert m["i17_expected_keys"] == expected
        assert sorted(r["key"] for r in m["i17_check"]) == sorted(expected)
        assert all(r["equal"] and r["semantic_equal"] for r in m["i17_check"])
    # 倒序作用于实际提交的任务序列：互为逆序，且不相同（至少两个不同的任务）
    assert len(set(normal.last_submission)) >= 2
    assert rev.last_submission == list(reversed(normal.last_submission))
    assert rev.last_submission != normal.last_submission
    # 四种模式按检查键对齐比较
    cal = service.loader.calendar()
    for label in SYMS:
        cached = service.prepare(TARGETS[label], G)
        fresh = service.prepare(TARGETS[label], G, cache=False)
        for cid in CIDS:
            p = EngineParams(GRID[cid], EXPLORATION_ASSUMPTIONS)
            modes = {
                "single_nocache": semantic_digest(replay_prepared(fresh, p, cal)),
                "single_cache": semantic_digest(replay_prepared(cached, p, cal)),
                "multi": normal.last_multi[label][1][cid],
                "multi_reversed": rev.last_multi[label][1][cid],
            }
            assert len(set(modes.values())) == 1, (label, cid, modes)


def test_submission_order_function() -> None:
    items = ["x", "y", "z"]
    assert submission_order(items) == items
    assert submission_order(items, reverse=True) == ["z", "y", "x"]


def test_replay_semantic_mismatch_and_missing_key(service, tmp_path: Path) -> None:  # noqa: F811
    run = ReplayRunService(
        service, ReplayFileSink(tmp_path), FakeEnv(False), progress=lambda _: None
    )
    cal = service.loader.calendar()
    prepared = [service.prepare(TARGETS[s], G) for s in SYMS]
    multi: dict = {}
    for p in prepared:
        disc, sem = {}, {}
        for cid in CIDS:
            result = replay_prepared(p, EngineParams(GRID[cid], EXPLORATION_ASSUMPTIONS), cal)
            disc[cid], sem[cid] = discrete_digest(result), semantic_digest(result)
            if p.target.label == "AAA" and cid == 385:
                # 人为让多进程模式的一个非离散字段不同：atr_ep 加最后一位
                i = next(k for k, r in enumerate(result.results) if r.record.atr_ep is not None)
                step = result.results[i]
                atr = step.record.atr_ep
                bumped = replace(
                    step, record=replace(step.record, atr_ep=math.nextafter(atr, math.inf))
                )
                changed = replace(
                    result, results=(*result.results[:i], bumped, *result.results[i + 1 :])
                )
                assert discrete_digest(changed) == disc[cid]
                sem[cid] = semantic_digest(changed)
        multi[p.target.label] = (disc, sem)
    expected = i17_expected_keys(SYMS, CIDS)
    records, missing = run._i17_check(prepared, cal, multi, expected)  # noqa: SLF001
    bad = next(r for r in records if r["key"] == ["AAA", 385])
    assert bad["equal"] is True and bad["semantic_equal"] is False
    from stock_risk.application.replay_run import i17_ok

    assert i17_ok(records, expected, missing) is False
    # 人为去掉一个预期检查键：多进程没有返回 BBB_corrected / 1
    del multi["BBB_corrected"][1][1]
    records2, missing2 = run._i17_check(prepared, cal, multi, expected)  # noqa: SLF001
    assert missing2 == [("BBB_corrected", 1)] and ("BBB_corrected", 1) not in [
        tuple(r["key"]) for r in records2
    ]
    assert i17_ok(records2, expected, missing2) is False
    assert i17_ok([], [], []) is False  # 检查为空


# ---- 对照回放 ----

SETS = ("V0_x0_x0_OFF_r2off_N20_s385", "V0_x0_x0_OFF_r2off_N20_s386")


def _compare_run(svc: ReplayService, tmp: Path, reverse: bool) -> tuple[CompareService, dict]:
    run = CompareService(svc, CompareFileSink(tmp), FakeEnv(False), progress=lambda _: None)
    m = run.run(
        T0,
        _clock,
        workers=2,
        only_sets=list(SETS),
        i17_sets=SETS,
        i17_targets=SYMS,
        reverse_submission=reverse,
    )
    return run, m


def test_compare_four_modes_and_reverse_submission(service, tmp_path: Path) -> None:  # noqa: F811
    normal, m1 = _compare_run(service, tmp_path / "a", reverse=False)
    rev, m2 = _compare_run(service, tmp_path / "b", reverse=True)
    expected = [[s, t] for s in SETS for t in SYMS]
    for m in (m1, m2):
        assert m["i17_ok"] is True and m["i17_missing_keys"] == []
        assert m["i17_expected_keys"] == expected
        assert all(r["equal"] and r["semantic_equal"] for r in m["i17_check"])
        written = json.loads((Path(m["run_dir"]) / "run_manifest.json").read_text("utf-8"))
        keys = list(written)
        assert keys.index("i17_expected_keys") == keys.index("i17_ok") + 1
        assert keys.index("i17_missing_keys") == keys.index("i17_ok") + 2
    assert len(normal.last_submission) >= 2  # 至少两个不同的组
    assert rev.last_submission == list(reversed(normal.last_submission))
    assert rev.last_submission != normal.last_submission
    by_id = {s.set_id: s for s in variant_sets()}
    cal = service.loader.calendar()
    for set_id in SETS:
        p = params_of(by_id[set_id])
        for label in SYMS:
            cached = service.prepare(TARGETS[label], G)
            fresh = service.prepare(TARGETS[label], G, cache=False)
            modes = {
                "single_nocache": semantic_digest(replay_prepared(fresh, p, cal)),
                "single_cache": semantic_digest(replay_prepared(cached, p, cal)),
                "multi": normal.last_outcomes[set_id].semantic[label],
                "multi_reversed": rev.last_outcomes[set_id].semantic[label],
            }
            assert len(set(modes.values())) == 1, (set_id, label, modes)


def test_compare_missing_key_recorded(service, tmp_path: Path) -> None:  # noqa: F811
    """预期键中的套没有运行：记为缺项（原写法为跳过），i17_ok 为假。"""
    run = CompareService(
        service, CompareFileSink(tmp_path), FakeEnv(False), progress=lambda _: None
    )
    m = run.run(
        T0,
        _clock,
        workers=1,
        only_sets=[SETS[0]],
        i17_sets=SETS,
        i17_targets=("AAA",),
    )
    assert m["i17_missing_keys"] == [[SETS[1], "AAA"]]
    assert m["i17_ok"] is False


def test_compare_submission_groups_and_manifest_order() -> None:
    by_id = {s.set_id: s for s in variant_sets()}
    groups = submission_groups(by_id, list(SETS))
    assert submission_groups(by_id, list(SETS), reverse=True) == list(reversed(groups))
    i = _MANIFEST_ORDER.index("i17_ok")
    assert _MANIFEST_ORDER[i + 1 : i + 3] == ["i17_expected_keys", "i17_missing_keys"]
