"""T04a 验收：真实数据上的单股串行回放（参考候选 SMA / TWO_STAGE / PEAK）。

需要本机 data/raw（不提交）；没有数据时跳过。只读取，不写任何文件。
"""

from __future__ import annotations

from datetime import date
from functools import cache

import pytest

from stock_risk.application.replay import (
    CORRECTED,
    UNCORRECTED,
    ReplayService,
    ReplayTarget,
    replay_prepared,
)
from stock_risk.bootstrap import build_replay_service
from stock_risk.config.settings import load_settings
from stock_risk.domain.lifecycle.invariants import check_invariants
from stock_risk.domain.lifecycle.model import DataStatus, EndReason, EventType
from stock_risk.domain.lifecycle.params import (
    EXPLORATION_ASSUMPTIONS,
    AtrBasis,
    Confirm,
    EngineParams,
    ReleasePath,
    reference_candidates,
)
from stock_risk.domain.observation import RightEnd

SETTINGS = load_settings()
pytestmark = pytest.mark.skipif(
    not (SETTINGS.raw_data_dir / "SPY.csv").exists(), reason="本机没有 data/raw"
)


def _reference() -> tuple[int, EngineParams]:
    for i, c in reference_candidates():
        if (c.confirm, c.release_path, c.atr_basis) == (
            Confirm.SMA,
            ReleasePath.TWO_STAGE,
            AtrBasis.PEAK,
        ):
            return i, EngineParams(c, EXPLORATION_ASSUMPTIONS)
    raise AssertionError("找不到参考候选")


CAND_ID, PARAMS = _reference()


@cache
def service() -> ReplayService:
    return build_replay_service(SETTINGS)


@cache
def replay(symbol: str, variant: str = ""):  # type: ignore[no-untyped-def]
    svc = service()
    prepared = svc.prepare(ReplayTarget(symbol, variant), PARAMS.assumptions.gap_break)
    return svc.replay(prepared, CAND_ID, PARAMS)


def test_reference_candidate_is_grid_385() -> None:
    assert CAND_ID == 385


def test_targets_are_27_signal_targets_with_shld_two_variants() -> None:
    targets = service().targets()
    labels = [t.label for t in targets]
    assert len({t.symbol for t in targets}) == 27
    assert "SHLD_corrected" in labels and "SHLD_uncorrected" in labels and "SHLD" not in labels
    assert not {"SPY", "QQQ", "SMH"} & {t.symbol for t in targets}  # ETF 不回放（I15）


def test_aapl_ranges_and_no_missing() -> None:
    r = replay("AAPL")
    rng = r.prepared.data.ranges
    assert (rng.replay_start, rng.observation_end, rng.right_end) == (
        date(2009, 1, 2),
        date(2019, 12, 31),
        RightEnd.EXPLORATION_END,
    )
    recs = [x.record for x in r.result.results]
    assert recs[-1].day == date(2019, 12, 31)
    assert all(rec.data_status is not DataStatus.MISSING for rec in recs)
    assert all(not x.events for x in r.result.results if x.record.day < date(2009, 1, 2))


def test_vrx_stitch_continuous() -> None:
    r = replay("VRX")
    by_day = {x.record.day: x for x in r.result.results}
    for d in (date(2018, 7, 13), date(2018, 7, 16)):
        assert by_day[d].record.data_status is not DataStatus.MISSING
    days = sorted(by_day)
    assert days[days.index(date(2018, 7, 13)) + 1] == date(2018, 7, 16)
    assert all(e.type is not EventType.DATA_ISSUE for e in r.result.events)


@pytest.mark.parametrize("variant", [CORRECTED, UNCORRECTED])
def test_shld_missing_day_and_listing_end(variant: str) -> None:
    r = replay("SHLD", variant)
    by_day = {x.record.day: x for x in r.result.results}
    gap = by_day[date(2017, 11, 8)]
    assert [e.type for e in gap.events if e.type is EventType.DATA_ISSUE] == [EventType.DATA_ISSUE]
    prev = by_day[date(2017, 11, 7)].record
    if prev.episode_id is not None:  # 当时有进行中的预警：观察重置一次
        assert gap.record.reset_reasons == ("DATA_GAP",)
        assert gap.record.cycle_no == prev.cycle_no + 1
    rng = r.prepared.data.ranges
    assert rng.observation_end == date(2018, 10, 23) and rng.right_end is RightEnd.LISTING_END
    last = r.result.results[-1].record
    assert last.day == date(2018, 10, 23)
    if r.result.censored_episode is not None:
        assert last.end_reason is EndReason.LISTING_END


def test_shld_corrected_equals_uncorrected_before_correction() -> None:
    a, b = replay("SHLD", CORRECTED).result.results, replay("SHLD", UNCORRECTED).result.results
    cut = date(2018, 3, 28)
    before_a = [x for x in a if x.record.day < cut]
    before_b = [x for x in b if x.record.day < cut]
    assert before_a == before_b and len(before_a) > 2000


def test_no_dates_in_sealed_period_and_invariants_for_all_targets() -> None:
    svc = service()
    total = 0
    for t in svc.targets():
        r = replay(t.symbol, t.variant)
        assert all(x.record.day <= date(2019, 12, 31) for x in r.result.results)
        assert all(e.day <= date(2019, 12, 31) for e in r.result.events)
        assert check_invariants(r.result.results, PARAMS) == [], t.label
        total += 1
    assert total == 28  # 27 只个股，SHLD 两套


@pytest.mark.parametrize("symbol", ["AAPL", "NFLX"])
def test_I17_no_lookahead_on_real_data(symbol: str) -> None:
    r = replay(symbol)
    full = r.result.results
    cal = service().loader.calendar()
    for t in (date(2011, 8, 8), date(2015, 8, 24), date(2018, 12, 24)):
        part = replay_prepared(r.prepared, PARAMS, cal, upto=t).results
        assert part[-1].record.day == t
        assert part == full[: len(part)]
