"""参数对象、历史段切分与引擎一致、回放起点、观察终点截尾、性能。"""

import random
import time
from dataclasses import fields

import pytest

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.lifecycle.history_segments import split_segments
from stock_risk.domain.lifecycle.model import DailyRecord, Display, EndReason, EventType
from stock_risk.domain.lifecycle.params import (
    EXPLORATION_ASSUMPTIONS,
    FIXED,
    AtrBasis,
    Confirm,
    ReleasePath,
    candidate_grid,
)
from stock_risk.domain.lifecycle.runner import run_bars
from stock_risk.domain.observation import DayPresence
from tests.unit.lifecycle.harness import SYMBOL, f_bars, params


def test_fixed_values_and_exploration_assumptions() -> None:
    assert (FIXED.w, FIXED.g, FIXED.sma_len) == (63, 3, 10)  # 10 号第 4.2 节
    a = EXPLORATION_ASSUMPTIONS  # I6 (a)：探索假设，非发起人批准值
    assert (a.phi, a.k_stab, a.gap_break) == (0.5, 8, 10)


def test_candidate_grid_numbering() -> None:
    grid = candidate_grid()
    assert len(grid) == 768 and [i for i, _ in grid] == list(range(1, 769))
    first, second, last = grid[0][1], grid[1][1], grid[-1][1]
    assert (first.n, first.k1, first.k2, first.k, first.d, first.u_rel) == (14, 2, 4, 3, 3, 3)
    assert (first.confirm, first.release_path, first.atr_basis) == (
        Confirm.SMA,
        ReleasePath.TWO_STAGE,
        AtrBasis.PEAK,
    )
    assert second.atr_basis is AtrBasis.MIN and second.release_path is ReleasePath.TWO_STAGE
    assert (last.n, last.k1, last.k2, last.k, last.d, last.u_rel) == (20, 3, 5, 5, 5, 5)
    assert (last.confirm, last.release_path, last.atr_basis) == (
        Confirm.BOX_ROLL,
        ReleasePath.DIRECT,
        AtrBasis.MIN,
    )
    assert grid[2][1].release_path is ReleasePath.DIRECT  # 第 3 号：解除路径变化
    assert grid[4][1].confirm is Confirm.BOX  # 第 5 号：确认方法变化
    assert len({c for _, c in grid}) == 768


def test_daily_record_covers_t04b_fields() -> None:
    names = {f.name for f in fields(DailyRecord)}
    required = {
        "data_status",
        "m",
        "display",
        "level",
        "episode_id",
        "p_ep",
        "p_ep_day",
        "atr_ep",
        "l_ep",
        "l_ep_day",
        "d_t",
        "d_ep",
        "r2_status",
        "r2_score",
        "cycle_no",
        "obs_start",
        "since_obs",
        "confirm_method",
        "confirm_line",
        "confirmed",
        "u",
        "strong_threshold",
        "strong",
        "evidence",
        "dc_cand",
        "rc_cand",
        "xc_cand",
        "dc",
        "rc",
        "xc",
        "h_d",
        "giveback",
        "reset",
        "reset_reasons",
        "end_reason",
    }
    assert required <= names


def gapped_presence(seed: int, n: int = 700) -> tuple[list, list[DayPresence]]:
    bars = f_bars([100 - (i % 7) for i in range(n - 90)])
    rng = random.Random(seed)
    presence, kept = [], []
    i = 0
    while i < len(bars):
        if rng.random() < 0.02:  # 制造 1–6 天的连续缺价
            for _ in range(rng.randint(1, 6)):
                if i < len(bars):
                    presence.append(DayPresence(bars[i].day, False))
                    i += 1
            continue
        presence.append(DayPresence(bars[i].day, True))
        kept.append(bars[i])
        i += 1
    return kept, presence


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_segments_match_engine(seed: int) -> None:
    p = params()  # G = 3
    bars, presence = gapped_presence(seed)
    cal = TradingCalendar.from_days(x.day for x in presence)
    segs = split_segments(presence, p.assumptions.gap_break)
    out = run_bars(SYMBOL, cal, presence, bars, p)
    engine_starts = sorted({r.record.seg_start for r in out.results if r.record.seg_start})
    assert engine_starts == [s.start for s in segs]
    assert len(segs) >= 2  # 种子保证出现过历史断点
    for r in out.results:  # 新历史段在满 W + N 条前不可计算
        if r.record.data_status.value == "HAS_BAR" or r.record.display is Display.INSUFFICIENT:
            seg = next(s for s in segs if s.start == r.record.seg_start)
            count = sum(1 for d in seg.bar_days if d <= r.record.day)
            assert (r.record.display is Display.INSUFFICIENT) == (count < FIXED.w + 20)


def test_replay_start_suppresses_everything_before() -> None:
    p = params()
    bars = f_bars([70, 69, 70, 72])
    presence = [DayPresence(b.day, True) for b in bars]
    presence[85] = DayPresence(bars[85].day, False)  # 回放起点前的缺价
    kept = [b for i, b in enumerate(bars) if i != 85]
    cal = TradingCalendar.from_days(b.day for b in bars)
    replay = bars[90].day  # 峰值后第一天（c0）
    out = run_bars(SYMBOL, cal, presence, kept, p, replay_start=replay)
    before = [r for r in out.results if r.record.day < replay]
    assert all(not r.events and not r.record.in_replay for r in before)
    assert before[85].record.m == 1  # A 段照常计数
    first = next(r for r in out.results if r.record.day == replay)
    assert first.record.in_replay and [e.type for e in first.events] == [EventType.START]


@pytest.mark.parametrize("reason", [EndReason.LISTING_END, EndReason.EXPLORATION_END])
def test_observation_end_censors_open_episode(reason: EndReason) -> None:
    p = params()
    bars = f_bars([70, 69, 70])
    cal = TradingCalendar.from_days(b.day for b in bars)
    presence = [DayPresence(b.day, True) for b in bars]
    out = run_bars(SYMBOL, cal, presence, bars, p, end_reason=reason)
    assert out.censored_episode == f"{SYMBOL}:{bars[90].day}"
    assert out.final_state.episode is None
    assert out.results[-1].record.end_reason is reason
    kinds = {e.type for e in out.events}
    assert EventType.RELEASE not in kinds and EventType.DATA_BREAK not in kinds  # 截尾不是事件


def test_censor_without_open_episode_is_noop() -> None:
    p = params()
    bars = f_bars([])
    cal = TradingCalendar.from_days(b.day for b in bars)
    out = run_bars(
        SYMBOL,
        cal,
        [DayPresence(b.day, True) for b in bars],
        bars,
        p,
        end_reason=EndReason.EXPLORATION_END,
    )
    assert out.censored_episode is None and out.results[-1].record.end_reason is None


def test_engine_cost_independent_of_history_length() -> None:
    """约 10.7 年日线（2790 天，止于 2019 年）单个候选的运行时间应很短
    （每日工作量与历史长度无关）。"""
    p = params(confirm=Confirm.BOX_ROLL, release=ReleasePath.DIRECT, basis=AtrBasis.MIN)
    rng = random.Random(7)
    closes, c = [], 100.0
    for _ in range(2700):
        c = max(5.0, c * (1 + rng.gauss(0, 0.02)))
        closes.append(round(c, 2))
    bars = f_bars(closes)
    cal = TradingCalendar.from_days(b.day for b in bars)
    presence = [DayPresence(b.day, True) for b in bars]
    t0 = time.perf_counter()
    out = run_bars(SYMBOL, cal, presence, bars, p)
    elapsed = time.perf_counter() - t0
    assert len(out.results) == len(bars) and elapsed < 5.0
    from stock_risk.domain.lifecycle.invariants import check_invariants

    assert check_invariants(out.results, p) == []
