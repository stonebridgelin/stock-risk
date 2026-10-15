"""指标接口（I5）两种实现与 R2（13 号 S19）。"""

from datetime import date, timedelta

import pytest

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.indicator_source import (
    ComputedIndicators,
    TableIndicators,
    TableRow,
    compute_r2,
)
from stock_risk.domain.indicators import CalcStatus
from stock_risk.domain.lifecycle.params import FIXED
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.numeric import InvalidInputError

G = FIXED.g  # v0.7 第 4.2 节固定值
D0 = date(2019, 3, 4)  # 周一
DAYS = [D0 + timedelta(days=i) for i in range(5)]  # r−2, r−1, r0, r1, r2（周一至周五）
CAL = TradingCalendar.from_days(DAYS)
RM2, RM1, R0, R1, R2_ = DAYS


def table(rows: dict[date, TableRow]) -> TableIndicators:
    # S19：ATR 每日给定为 2；历史长度视为充足
    return TableIndicators(CAL, rows, n=20, prior_valid_bars=1000)


def test_S19a_gap_triggers() -> None:
    src = table(
        {
            RM1: TableRow(100, 100, atr=2),
            R0: TableRow(100, 100, atr=2),
            R1: TableRow(93, 97, atr=2),
        }
    )
    r = compute_r2(src, R1, 20, G)
    assert (r.status, r.gap_score, r.triggered) == (CalcStatus.OK, -3.5, True)


def test_S19b_previous_trading_day_missing() -> None:
    # r0 缺价：不得取 r−1 的收盘
    src = table({RM1: TableRow(100, 100, atr=2), R1: TableRow(93, 97, atr=2)})
    r = compute_r2(src, R1, 20, G)
    assert (r.status, r.gap_score, r.triggered) == (CalcStatus.MISSING_DATA, None, False)
    assert src.window_max_close(R1, FIXED.w, None) == (100.0, RM1)  # D_T 的窗口最高来自 r−1


def test_S19c_gap_exactly_minus_g_triggers() -> None:
    src = table({R0: TableRow(100, 100, atr=2), R1: TableRow(94, 97, atr=2)})
    r = compute_r2(src, R1, 20, G)
    assert (r.status, r.gap_score, r.triggered) == (CalcStatus.OK, -3.0, True)


def test_r2_requires_close_below_previous_close() -> None:
    src = table({R0: TableRow(100, 100, atr=2), R1: TableRow(93, 100, atr=2)})
    assert compute_r2(src, R1, 20, G).triggered is False


def test_r2_insufficient_when_previous_atr_unavailable() -> None:
    src = table({R0: TableRow(100, 100, atr=None), R1: TableRow(93, 97, atr=2)})
    assert compute_r2(src, R1, 20, G).status is CalcStatus.INSUFFICIENT_HISTORY


def test_r2_first_calendar_day_is_missing_data() -> None:
    src = table({RM2: TableRow(100, 100, atr=2)})
    assert compute_r2(src, RM2, 20, G).status is CalcStatus.MISSING_DATA


def test_table_rejects_other_n_or_m_and_missing_day() -> None:
    src = table({R0: TableRow(100, 100, atr=2, sma=99)})
    assert src.sma(R0, 10) == 99
    with pytest.raises(InvalidInputError):
        src.atr(R0, 14)
    with pytest.raises(InvalidInputError):
        src.sma(R0, 5)
    with pytest.raises(InvalidInputError):
        src.close(R1)


def test_table_counts_and_windows() -> None:
    src = table({RM2: TableRow(1, 80), R0: TableRow(1, 82), R1: TableRow(1, 81)})
    assert src.valid_bar_count(R1) == 1003
    assert src.count_valid_after(RM2, R1) == 2  # since_obs：obs_start 之后的有效日
    assert src.max_close(RM2, R1) == 82
    assert src.min_close(RM2, R1) == (80, RM2)
    assert src.max_close_prior(R1, 2) == 82


def bars_on(days: list[date], closes: list[float]) -> list[DailyBar]:
    out, prev = [], None
    for d, c in zip(days, closes, strict=True):
        o = c if prev is None else prev
        out.append(
            DailyBar(d, o, max(o, c) + 1, min(o, c) - 1, c, 1, o, max(o, c) + 1, min(o, c) - 1, c)
        )
        prev = c
    return out


def test_computed_matches_definitions_and_skips_missing_day() -> None:
    # r0 缺价：I18 下 r1 的 TR 用 r−1 的收盘；R2 仍按日历前一交易日判定为 MISSING_DATA
    bars = bars_on([RM2, RM1, R1], [100, 100, 97])
    src = ComputedIndicators(CAL, bars)
    assert src.has_bar(RM1) and not src.has_bar(R0)
    assert src.valid_bar_count(R1) == 3
    assert src.atr(R1, 2) == (2 + max(101 - 96, abs(101 - 100), abs(96 - 100))) / 2
    assert src.sma(R1, 3) == pytest.approx(99.0)
    assert compute_r2(src, R1, 1, G).status is CalcStatus.MISSING_DATA
    assert src.window_max_close(R1, FIXED.w, None) == (100.0, RM1)


def test_computed_r2_ok_path() -> None:
    bars = bars_on([RM2, RM1, R0], [100, 100, 100])
    gap = DailyBar(R1, 93, 98, 92, 97, 1, 93, 98, 92, 97)
    src = ComputedIndicators(CAL, [*bars, gap])
    r = compute_r2(src, R1, 2, G)  # ATR_2(r0) = 2
    assert (r.status, r.gap_score, r.triggered) == (CalcStatus.OK, -3.5, True)


def test_computed_rejects_days_outside_calendar() -> None:
    with pytest.raises(InvalidInputError):
        ComputedIndicators(CAL, bars_on([date(2019, 3, 9)], [1]))


def test_range_max_window_matches_scan_with_ties() -> None:
    """ComputedIndicators 的稀疏表窗口最高与逐条扫描的规则相同（并列取最近；since 截断）。"""
    import random

    from stock_risk.domain import indicators as ind

    rng = random.Random(3)
    d0 = date(2018, 3, 5)  # 周一；400 个日历日内结束于 2019 年（日历构造时检查封存，T08a D-02）
    days = [d0 + timedelta(days=i) for i in range(400) if (d0 + timedelta(days=i)).weekday() < 5]
    closes = [float(rng.choice([95, 96, 97, 98, 99, 100])) for _ in days]
    bars = [
        DailyBar(d, c, c + 1, c - 1, c, 1, c, c + 1, c - 1, c)
        for d, c in zip(days, closes, strict=True)
    ]
    src = ComputedIndicators(TradingCalendar.from_days(days), bars)
    for i in range(0, len(days), 7):
        for w in (1, 5, 63, 252):
            since = days[max(0, i - rng.randint(0, 30))] if i % 2 else None
            assert src.window_max_close(days[i], w, since) == ind.window_max_close(
                bars, i, w, since
            )
