"""回放用例与输出（T04a–c）：合成数据，只写临时目录。"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from stock_risk.application.market_data_loader import MarketDataLoader
from stock_risk.application.replay import CORRECTED, UNCORRECTED, ReplayService, ReplayTarget
from stock_risk.application.replay_chart_data import build_chart_data, drawdown_63
from stock_risk.application.replay_outputs import (
    check_consistency,
    daily_rows,
    episode_csv_rows,
    episode_rows,
    event_rows,
)
from stock_risk.application.replay_run import (
    GitState,
    ReplayRunService,
    _summarize_target,
    discrete_digest,
    make_run_id,
)
from stock_risk.application.replay_summary import summarize
from stock_risk.domain.corrections import CorrectionMethod, PriceCorrection
from stock_risk.domain.lifecycle.model import EndReason
from stock_risk.domain.lifecycle.params import (
    EXPLORATION_ASSUMPTIONS,
    EngineParams,
    candidate_grid,
    reference_candidates,
)
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.universe import Role, UniverseMember
from stock_risk.infrastructure.replay.files import ReplayFileSink
from stock_risk.infrastructure.storage.csv_raw_reader import CsvRawDataReader
from stock_risk.infrastructure.storage.csv_raw_repository import BAR_COLUMNS


class Fixed:
    def __init__(self, value) -> None:
        self.value = value

    def load(self):
        return self.value


def weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DAYS = weekdays(date(2008, 1, 2), date(2011, 6, 30))


def closes(seed: int) -> list[float]:
    """有涨有跌的确定性路径（在 2009–2011 年产生多次预警）。"""
    out, c = [], 100.0
    for i in range(len(DAYS)):
        c *= 1 + (0.03 if (i // (17 + seed)) % 3 == 0 else -0.02) * (1 if i % 2 else 0.5)
        out.append(round(max(c, 1.0), 2))
    return out


def write(raw: Path, symbol: str, cs: list[float], skip: set[date] = frozenset()) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    lines, prev = [",".join(BAR_COLUMNS)], None
    for d, c in zip(DAYS, cs, strict=True):
        o = c if prev is None else prev
        prev = c
        if d in skip:
            continue
        h, lo = max(o, c) + 0.5, min(o, c) - 0.5
        lines.append(f"{d},{o},{h},{lo},{c},1000,{o},{h},{lo},{c}")
    (raw / f"{symbol}.csv").write_text("\n".join(lines) + "\n", "utf-8")
    (raw / f"{symbol}.actions.csv").write_text("date,type,value\n", "utf-8")


@pytest.fixture()
def service(tmp_path: Path) -> ReplayService:
    raw = tmp_path / "raw"
    for s in ("SPY", "QQQ"):
        write(raw, s, [100.0] * len(DAYS))
    write(raw, "AAA", closes(0), skip={date(2010, 3, 3)})
    write(raw, "BBB", closes(5))
    listings = {"BBB": ListingPeriod("BBB", listed_until=date(2011, 5, 31), source_url="u")}
    corr = {
        "BBB": (
            PriceCorrection(
                "BBB", date(2010, 6, 1), date(2010, 6, 30), CorrectionMethod.USE_RAW, "PM-T", "测试"
            ),
        )
    }
    loader = MarketDataLoader(CsvRawDataReader(raw), Fixed(listings), Fixed(corr))
    members = [UniverseMember(s, (s,), "x", Role.REFERENCE_ONLY) for s in ("SPY", "QQQ")]
    members += [UniverseMember(s, (s,), "x", Role.SIGNAL_TARGET) for s in ("AAA", "BBB")]
    return ReplayService(loader, Fixed(members))


P385 = EngineParams(dict(candidate_grid())[385], EXPLORATION_ASSUMPTIONS)


def test_targets_roles_and_variants(service: ReplayService) -> None:
    labels = [t.label for t in service.targets()]
    assert labels == ["AAA", "BBB_corrected", "BBB_uncorrected"]
    with pytest.raises(ValueError):
        service.targets(["SPY"])


def test_replay_ranges_censoring_and_outputs_consistent(service: ReplayService) -> None:
    rep = service.replay(service.prepare(ReplayTarget("AAA"), 10), 385, P385)
    recs = [r.record for r in rep.result.results]
    assert recs[0].day == date(2008, 1, 2) and not any(r.in_replay for r in recs[:252])
    assert all(not r.events for r in rep.result.results if r.record.day < date(2009, 1, 1))
    eps = episode_rows(rep)
    assert eps and all(e.end_reason is not None for e in eps)
    ev, dy = event_rows(rep, "rid"), daily_rows(rep, "rid")
    ep = episode_csv_rows(rep, "rid", eps)
    chart = build_chart_data(
        rep, "rid", {"SPY": drawdown_63(service.loader.load("SPY").history.bars)}
    )
    marks = [(e.day.isoformat(), e.type) for e in chart.events]
    assert check_consistency(ev, dy, ep, marks) == []
    # 篡改一个事件日期必须被发现
    bad = [dict(ev[0], date="2008-01-03"), *ev[1:]]
    assert check_consistency(bad, dy, ep, marks)
    assert any(r["data_status"] == "MISSING" for r in dy)  # 2010-03-03 缺价


def test_listing_end_censor_and_correction_variants(service: ReplayService) -> None:
    a = service.replay(service.prepare(ReplayTarget("BBB", CORRECTED), 10), 385, P385)
    b = service.replay(service.prepare(ReplayTarget("BBB", UNCORRECTED), 10), 385, P385)
    assert a.result.results[-1].record.day == date(2011, 5, 31)
    if a.result.censored_episode:
        assert a.result.results[-1].record.end_reason is EndReason.LISTING_END
    cut = date(2010, 6, 1)
    assert [r for r in a.result.results if r.record.day < cut] == [
        r for r in b.result.results if r.record.day < cut
    ]
    assert a.prepared.data.corrections and not b.prepared.data.corrections


def test_summary_occupancy_and_attribution(service: ReplayService) -> None:
    rep = service.replay(service.prepare(ReplayTarget("AAA"), 10), 385, P385)
    eps = episode_rows(rep)
    row = summarize(rep, eps)
    assert 0 <= row["occupancy"] <= 1
    assert row["missing_days"] == 1
    assert row["computable_days"] + row["missing_days"] + row["insufficient_days"] + row[
        "data_break_days"
    ] == sum(1 for d in DAYS if date(2010, 1, 1) <= d)
    dev = [e for e in eps if e.start_date >= date(2010, 1, 1)]
    assert row["episodes_dev"] == len(dev)
    assert row["release_n"] + row["data_break_n"] + row["censored_listing_end_n"] + row[
        "censored_exploration_end_n"
    ] == len(dev)


def test_worker_matches_single_process_and_cache(service: ReplayService) -> None:
    cal = service.loader.calendar()
    cands = tuple(c for c in candidate_grid() if c[0] in (1, 385, 768))
    cached = service.prepare(ReplayTarget("AAA"), 10)
    fresh = service.prepare(ReplayTarget("AAA"), 10, cache=False)
    out = _summarize_target((cached, cal, cands, frozenset({1, 385, 768})))
    assert out["failed"] == [] and out["violations"] == [] and len(out["rows"]) == 3
    for cid, cand in cands:
        p = EngineParams(cand, EXPLORATION_ASSUMPTIONS)
        single = discrete_digest(service.replay(fresh, cid, p).result)
        assert out["digests"][cid] == single


class FakeEnv:
    def __init__(self, dirty: bool) -> None:
        self.dirty = dirty

    def git_state(self) -> GitState:
        return GitState("abcdef1234567", self.dirty, ("M x",) if self.dirty else (), None)

    def file_hashes(self):
        return {"calendar_spy": {"path": "SPY.csv", "sha256": "h"}}

    def versions(self):
        return {"python": "3.13"}


@pytest.mark.parametrize(("dirty", "kind"), [(False, "formal"), (True, "debug")])
def test_reference_run_outputs_and_manifest(
    service: ReplayService, tmp_path: Path, dirty: bool, kind: str
) -> None:
    sink = ReplayFileSink(tmp_path / "out")
    run = ReplayRunService(service, sink, FakeEnv(dirty), progress=lambda _: None)
    t = datetime(2026, 10, 14, 12, 0, 0)
    report = run.run_reference(t, lambda: t + timedelta(seconds=5), symbols=["AAA"])
    assert report.run_kind == kind and report.run_id == make_run_id(t, "abcdef1234567")
    assert report.ok
    m = json.loads((report.run_dir / "run_manifest.json").read_text("utf-8"))
    assert m["run_kind"] == kind and m["invariant_violations"] == 0
    assert m["consistency_problem_count"] == 0
    assert [c["id"] for c in m["candidates"]] == [i for i, _ in reference_candidates()]
    assert m["parameters"]["exploration_assumptions_I6a"]["note"] == "探索假设，非发起人批准值"
    for cid, _ in reference_candidates():
        d = report.run_dir / f"c{cid:03d}" / "AAA"
        assert {p.name for p in d.iterdir()} == {
            "events.csv",
            "daily.csv",
            "episodes.csv",
            "chart.html",
        }
    assert (report.run_dir / "plotly.min.js").exists() and (report.run_dir / "index.html").exists()
    html = (report.run_dir / "c385" / "AAA" / "chart.html").read_text("utf-8")
    assert 'src="../../plotly.min.js"' in html  # 共用本地 plotly.js，不内嵌
    note = "未应用背景标签"
    assert note in html or json.dumps(note)[1:-1] in html  # 图表 JSON 中的中文为转义形式
    # 同一时刻再运行一次：按 run_id 分开，不覆盖
    again = run.run_reference(t, lambda: t, symbols=["AAA"])
    assert again.run_id == report.run_id + "-2"
