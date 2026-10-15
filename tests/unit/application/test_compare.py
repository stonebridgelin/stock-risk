"""T06b / T06c 对照回放（合成数据，只写临时目录）。"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from stock_risk.application.compare import analysis as an
from stock_risk.application.compare.service import CompareService
from stock_risk.application.compare.tables import pair_coverage
from stock_risk.application.compare.variants import (
    V3_REFERENCE,
    pairs,
    timeline_sets,
    variant_sets,
)
from stock_risk.application.compare.views import _status, build_view, window_after
from stock_risk.application.replay import ReplayTarget
from stock_risk.application.replay_run import GitState
from stock_risk.domain.lifecycle.model import DataStatus, Level
from stock_risk.infrastructure.replay.compare_files import CompareFileSink
from tests.unit.application.test_replay_run import FakeEnv, service  # noqa: F401  (fixture)


def test_variant_sets_and_ids() -> None:
    sets = variant_sets()
    assert len(sets) == 636
    assert Counter(s.group for s in sets) == {
        "V0": 12,
        "V1": 192,
        "V2": 12,
        "V3": 192,
        "V4": 192,
        "N14": 24,
        "V3_LOFF": 12,
    }
    ids = {s.set_id for s in sets}
    assert "V3_x8_x15_X1_r2off_N20_s385" in ids and "V0_x0_x0_OFF_r2off_N14_s1" in ids
    assert all(s.baseline_id in ids for s in sets)
    v0, v3 = timeline_sets(sets)
    assert v0.set_id == "V0_x0_x0_OFF_r2off_N20_s385"
    assert v3.switches == V3_REFERENCE and v3.candidate_id == 385
    kinds = Counter(p.kind for p in pairs(sets))
    assert kinds == {
        "r2_floor": 288,
        "long_path": 288,
        "x1": 288,
        "x2": 288,
        "release_hyst": 204,
        "rec_scale": 192,
        "N": 24,
        "long_path_off": 12,
    }
    assert "V3_x8_x15_OFF_r2off_N20_s385" in ids  # C-05：V3 参考取值但 L 关闭


def test_pair_coverage_ignores_not_computable() -> None:
    a = {"X": bytes([0, 1, 2, 9, 1])}
    b = {"X": bytes([1, 1, 0, 1, 9])}
    assert pair_coverage(a, b) == {
        "both": 1,
        "only_a": 1,
        "only_b": 1,
        "neither": 0,
        "common_days": 3,
    }


def views(
    svc,
    set_a: str = "V0_x0_x0_OFF_r2off_N20_s385",  # noqa: F811
    set_b: str = "V3_x8_x15_X1_r2off_N20_s385",
):  # type: ignore[no-untyped-def]
    from stock_risk.application.compare.worker import params_of

    sets = {s.set_id: s for s in variant_sets()}
    p = svc.prepare(ReplayTarget("AAA"), 10)
    out = []
    for sid in (set_a, set_b):
        vs = sets[sid]
        rep = svc.replay(p, vs.candidate_id, params_of(vs))
        out.append(build_view(rep, sid))
    return out


def test_coverage_cells_and_event_compare_cover_all_starts(service) -> None:  # noqa: F811
    base, new = views(service)
    rows, cells, excluded = an.coverage(base, new)
    assert (
        sum(cells[k] for k in ("both", "only_v0", "only_new", "neither"))
        == cells["common_days"]
        == len(rows)
    )
    assert all(r[2] for r in rows)
    cmp_rows = an.event_compare(base, new)
    for view, direction in ((base, "V0→新"), (new, "新→V0")):
        starts = {
            e.day.isoformat()
            for e in view.events
            if e.type.value == "START" and view.in_period(view.index[e.day])
        }
        listed = {
            r["date"]
            for r in cmp_rows
            if r["direction"] == direction and r["event_type"] == "START"
        }
        assert starts == listed
    assert all(r["tags"] is not None for r in cmp_rows)


def test_overlap_unique_partner() -> None:
    ov = {("v1", "n1"): [1, 2], ("v2", "n2"): [5], ("v2", "n3"): [7]}
    assert an.unique_partner(ov, "v1", 0) == "n1" and an.unique_partner(ov, "n1", 1) == "v1"
    assert an.unique_partner(ov, "v2", 0) is None and an.unique_partner(ov, "n2", 1) is None


def test_window_incomplete_at_end(service) -> None:  # noqa: F811
    base, _ = views(service)
    last = len(base.days) - 1
    w = window_after(base, last - 5, 20)
    assert not w.complete and len(w.day_indices) == 5
    assert window_after(base, 0, 20).complete


def test_compare_service_end_to_end(service, tmp_path: Path) -> None:  # noqa: F811
    sink = CompareFileSink(tmp_path / "out")
    svc = CompareService(service, sink, FakeEnv(False), progress=lambda _: None)
    t = datetime(2026, 10, 14, 12, 0, 0)
    only = ["V3_x8_x15_X1_r2off_N20_s385", "V3_x8_x15_X1_r2on_N20_s385"]
    m = svc.run(t, lambda: t + timedelta(seconds=1), workers=2, only_sets=only)
    assert m["run_kind"] == "debug"  # 只运行部分套时不得作为正式运行
    assert not m["failed"] and m["invariant_violations"] == 0
    assert m["program_check_problem_count"] == 0
    run_dir = Path(m["run_dir"])
    sdir = run_dir / "sets" / only[0]
    for name in (
        "events.csv",
        "episodes.csv",
        "daily_compact.csv.gz",
        f"coverage_daily_{only[0]}.csv.gz",
        f"event_compare_{only[0]}.csv",
        f"overlap_{only[0]}.csv",
    ):
        assert (sdir / name).exists(), name
    assert (sdir / "full" / "AAA" / "chart.html").exists()  # 时间线用的套有完整输出
    for name in (
        "coverage_summary.csv",
        "pairs_summary.csv",
        "burden.csv",
        "burden_by_set.csv",
        "special_r2_filtered.csv",
        "excluded_days.csv",
    ):
        assert (run_dir / name).exists(), name
    assert (run_dir / "timelines" / "index.html").exists()
    assert len(m["variant_sets"]) == 636 and m["variant_set_count"] == 636
    assert sorted(m["completed_sets"]) == sorted([*only, "V0_x0_x0_OFF_r2off_N20_s385"])
    pairs_rows = sink.read_csv(run_dir / "pairs_summary.csv")
    assert {r["pair_kind"] for r in pairs_rows} == {"r2_floor"}
    for r in pairs_rows:
        cells = sum(int(r[f"cov_{k}"]) for k in ("both", "only_a", "only_b", "neither"))
        assert cells == int(r["cov_common_days"])
    assert all(
        d <= date(2019, 12, 31).isoformat()
        for d in (row["date"] for row in sink.read_csv(sdir / "events.csv"))
    )


def test_git_state_formal_requires_full_run(service, tmp_path: Path) -> None:  # noqa: F811
    class Clean(FakeEnv):
        def git_state(self) -> GitState:
            return GitState("abc1234", False, (), None)

    svc = CompareService(
        service, CompareFileSink(tmp_path / "o"), Clean(False), progress=lambda _: None
    )
    t = datetime(2026, 10, 14)
    m = svc.run(t, lambda: t, workers=1, only_sets=["V2_x0_x0_OFF_r2off_N20_s385"])
    assert m["run_kind"] == "debug"


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.parametrize(
    ("level", "expect"),
    [
        (None, "缺价冻结（正常）"),
        (Level.WATCH, "缺价冻结（关注）"),
        (Level.ALERT, "缺价冻结（警戒）"),
    ],
)
def test_missing_day_status_keeps_frozen_level(level: Level | None, expect: str) -> None:
    """B-04：对照视图中缺价日的状态保留冻结时的等级。"""
    assert _status(DataStatus.MISSING, level) == expect
    assert _status(DataStatus.HAS_BAR, Level.ALERT) == "警戒"


def test_group_plan_runs_baseline_and_off_partner_once() -> None:
    """D-04：同一（结构组合, N）组内，V0 基准与 r2_floor OFF 配对各只运行一次。"""
    from stock_risk.application.compare.worker import group_key, group_plan

    sets = {s.set_id: s for s in variant_sets()}
    group = [sid for sid, s in sets.items() if group_key(s) == (385, 20)]
    assert len(group) == 1 + 16 + 1 + 16 + 16 + 1  # V0、V1、V2、V3、V4、V3_LOFF
    plan = group_plan(sets, group)
    order = [vs.set_id for vs, _ in plan.steps]
    assert sorted(order) == sorted(group) and all(emit for _, emit in plan.steps)
    assert order[0] == "V0_x0_x0_OFF_r2off_N20_s385"
    for vs, _ in plan.steps:
        if vs.switches.r2_floor:
            off = vs.set_id.replace("_r2on_", "_r2off_")
            assert order.index(off) < order.index(vs.set_id)
    assert plan.keep == {"V0_x0_x0_OFF_r2off_N20_s385"} | {
        sid.replace("_r2on_", "_r2off_") for sid in group if "_r2on_" in sid
    }
    # 只请求一个 ON 套：基准与 OFF 配对作为辅助运行，不输出
    part = group_plan(sets, ["V1_x8_x15_X1_r2on_N20_s385"])
    assert [(vs.set_id, emit) for vs, emit in part.steps] == [
        ("V0_x0_x0_OFF_r2off_N20_s385", False),
        ("V1_x8_x15_X1_r2off_N20_s385", False),
        ("V1_x8_x15_X1_r2on_N20_s385", True),
    ]
    with pytest.raises(ValueError):
        group_plan(sets, ["V0_x0_x0_OFF_r2off_N20_s385", "V0_x0_x0_OFF_r2off_N14_s1"])
