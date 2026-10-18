"""T10-M0b 第 3 节第 2 项：前缀无前视的边界合成测试（I-01，M-05、M-06）。

合成日线用现有夹具工具的约定（13 号第 2 节：开 = 前一日收盘，高低为开收 ±1，原始价同复权价）。
完整运行带观察终点截尾（EXPLORATION_END）；前缀运行截到 T（严格早于观察终点），用截到 T 的日线、
cache=False 重建指标源。期望是一条关系：前缀运行等于"未截尾的对应前缀"
（results = full.results[:k]，final_state = full.results[k − 1].state，censored_episode 为空），
逐字段相同，全语义摘要也相同。截断点由构造确定；构造的前提（例如某日确为断点日）另行断言。
"""

from __future__ import annotations

from datetime import date

import pytest

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.lifecycle.model import DataStatus, Display, EndReason, EventType
from stock_risk.domain.lifecycle.params import AtrBasis, Confirm, ReleasePath
from stock_risk.domain.lifecycle.runner import RunResult, build_sources, run_bars
from stock_risk.domain.lifecycle.semantic_digest import semantic_digest
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.observation import DayPresence
from tests.unit.lifecycle.harness import FIXTURE_ASSUMPTIONS, START_DAY, SYMBOL, params, weekdays
from tests.unit.lifecycle.test_fixtures_f import PATHS

P = params(Confirm.SMA, ReleasePath.TWO_STAGE, AtrBasis.MIN)
G = FIXTURE_ASSUMPTIONS.gap_break  # 夹具约定 G = 3
SCALE_BARS = 63 + P.candidate.n  # W + N：新历史段满 W + N 条日线的第一日为尺度初始化日


def build(closes: list[float | None]) -> tuple[TradingCalendar, list[DayPresence], list[DailyBar]]:
    days = weekdays(START_DAY, len(closes))
    bars: list[DailyBar] = []
    prev: float | None = None
    for day, c in zip(days, closes, strict=True):
        if c is None:
            continue
        o = c if prev is None else prev
        h, lo = max(o, c) + 1, min(o, c) - 1
        bars.append(DailyBar(day, o, h, lo, c, 1_000_000, o, h, lo, c))
        prev = c
    presence = [DayPresence(d, c is not None) for d, c in zip(days, closes, strict=True)]
    return TradingCalendar.from_days(days), presence, bars


def full_run(closes: list[float | None]) -> RunResult:
    cal, presence, bars = build(closes)
    return run_bars(SYMBOL, cal, presence, bars, P, presence[0].day, EndReason.EXPLORATION_END)


def assert_prefix(closes: list[float | None], full: RunResult, i: int) -> None:
    cal, presence, bars = build(closes)
    t = presence[i].day
    assert t < presence[-1].day  # T 严格早于观察终点
    pres_k = presence[: i + 1]
    bars_k = [b for b in bars if b.day <= t]
    src = build_sources(cal, pres_k, bars_k, G, cache=False)
    part = run_bars(SYMBOL, cal, pres_k, bars_k, P, presence[0].day, None, src)
    k = len(part.results)
    expected = RunResult(full.results[:k], full.results[k - 1].state, None)
    assert part == expected, t
    assert semantic_digest(part) == semantic_digest(expected), t


# ---- 情形 A：F2 路径（有降级与解除），之后短缺价、历史断点与新历史段 ----

F2_AFTER = PATHS["F2"][2]
BASE = [100.0] * 90 + list(F2_AFTER)
LAST = BASE[-1]
SHORT_GAP = len(BASE) + 17  # 两个缺价日（≤ G）的第一日
AFTER_SHORT = SHORT_GAP + 2
BREAK_FIRST = AFTER_SHORT + 28  # G + 1 = 4 个连续缺价日的第一日
BREAK_DAY = BREAK_FIRST + G  # 第 G + 1 个缺价日：断点日
NEW_SEG = BREAK_DAY + 1  # 新历史段的第一日
A: list[float | None] = (
    BASE + [LAST] * 17 + [None, None] + [LAST] * 28 + [None] * (G + 1) + [LAST] * (SCALE_BARS + 8)
)


@pytest.fixture(scope="module")
def full_a() -> RunResult:
    return full_run(A)


def test_construction_preconditions(full_a: RunResult) -> None:
    rec = [r.record for r in full_a.results]
    assert rec[SHORT_GAP].data_status is DataStatus.MISSING
    assert rec[SHORT_GAP + 1].data_status is DataStatus.MISSING
    assert rec[BREAK_DAY - 1].data_status is DataStatus.MISSING
    assert rec[BREAK_DAY].data_status is DataStatus.DATA_BREAK
    assert rec[BREAK_DAY].display is Display.DATA_BREAK
    assert rec[NEW_SEG].data_status is DataStatus.INSUFFICIENT
    assert rec[NEW_SEG + SCALE_BARS - 2].display is Display.INSUFFICIENT
    assert rec[NEW_SEG + SCALE_BARS - 1].display is not Display.INSUFFICIENT
    assert rec[SCALE_BARS - 2].display is Display.INSUFFICIENT  # 第一个历史段
    assert rec[SCALE_BARS - 1].display is not Display.INSUFFICIENT


def _event_index(full: RunResult, kind: EventType) -> int:
    (i,) = [i for i, r in enumerate(full.results) for e in r.events if e.type is kind]
    return i


def boundary_cuts(full: RunResult) -> dict[str, list[int]]:
    """每种边界的截断点（结果序号）。"""
    down = _event_index(full, EventType.DOWNGRADE)
    rel = _event_index(full, EventType.RELEASE)
    return {
        "降级当日": [down],
        "解除当日": [rel],
        "短缺价：前一日、当日、后一日": [SHORT_GAP - 1, SHORT_GAP, SHORT_GAP + 1, AFTER_SHORT],
        "历史断点：前一日、断点当日、新历史段第一日": [BREAK_DAY - 1, BREAK_DAY, NEW_SEG],
        "尺度初始化日与前一日（第一个历史段）": [SCALE_BARS - 2, SCALE_BARS - 1],
        "尺度初始化日与前一日（新历史段）": [NEW_SEG + SCALE_BARS - 2, NEW_SEG + SCALE_BARS - 1],
    }


def test_prefix_equals_uncensored_prefix_at_boundaries(full_a: RunResult) -> None:
    assert full_a.censored_episode is None or isinstance(full_a.censored_episode, str)
    for name, cuts in boundary_cuts(full_a).items():
        for i in cuts:
            assert_prefix(A, full_a, i), name


# ---- 情形 C：预警进行中遇到历史断点（DATA_BREAK 事件；"数据中断"显示持续数日） ----

C_HOLD = 6  # F2 峰值后第 6 日：此时预警进行中（START 在峰值后第 1 日）
C_BREAK_FIRST = 90 + C_HOLD
C_MISSING = G + 3  # 连续缺价 6 日：第 G + 1 日为断点日，之后两日仍显示"数据中断"
C_BREAK_DAY = C_BREAK_FIRST + G
C_LAST_BREAK_DISPLAY = C_BREAK_FIRST + C_MISSING - 1
C_NEW_SEG = C_BREAK_FIRST + C_MISSING
C: list[float | None] = (
    [100.0] * 90
    + list(F2_AFTER[:C_HOLD])
    + [None] * C_MISSING
    + [F2_AFTER[C_HOLD - 1]] * (SCALE_BARS + 5)
)


def test_prefix_at_data_break_event_and_display() -> None:
    full = full_run(C)
    rec = [r.record for r in full.results]
    breaks = [
        i for i, r in enumerate(full.results) for e in r.events if e.type is EventType.DATA_BREAK
    ]
    assert breaks == [C_BREAK_DAY]  # 构造前提：预警进行中遇到断点，产生 DATA_BREAK 事件
    shown = [i for i, r in enumerate(rec) if r.display is Display.DATA_BREAK]
    assert (shown[0], shown[-1]) == (C_BREAK_DAY, C_LAST_BREAK_DISPLAY)
    for i in (C_BREAK_DAY - 1, C_BREAK_DAY, C_BREAK_DAY + 1, C_LAST_BREAK_DISPLAY, C_NEW_SEG):
        assert_prefix(C, full, i)


# ---- 情形 B：峰值滚出 63 日回看窗口 ----

PEAK = 89
B: list[float | None] = [100.0] * 90 + [96.0] * 80


def test_prefix_at_peak_rolling_out_of_window() -> None:
    assert max(c for c in B[PEAK + 1 : PEAK + 64] if c is not None) < B[PEAK]  # 构造前提
    full = full_run(B)
    for i in (PEAK + 63, PEAK + 64):  # 峰值滚出窗口的当日与次日
        assert_prefix(B, full, i)


def test_dates_before_2020() -> None:
    assert weekdays(START_DAY, len(A))[-1] < date(2020, 1, 1)
