"""指标手算验收：13 号文档第 2 节 F 组构造、S21a / S21b；v0.4 第 2.1–2.3 节。"""

from datetime import date, timedelta

import pytest

from stock_risk.domain import indicators as ind
from stock_risk.domain.lifecycle.params import FIXED
from stock_risk.domain.market_data import DailyBar
from stock_risk.domain.numeric import InvalidInputError

W = FIXED.w  # v0.7 第 4.2 节固定值


def f_bars(after_peak: list[float]) -> list[DailyBar]:
    """13 号文档第 2 节：自 2009-01-05 起的连续工作日；峰值日（含）之前 90 根收盘 100；
    O_t = C_{t−1}（第一根 O = C）；H = max(O, C) + 1，L = min(O, C) − 1；复权价同原始价。"""
    closes = [100.0] * 90 + after_peak
    days, d = [], date(2009, 1, 5)
    while len(days) < len(closes):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    bars, prev = [], None
    for day, c in zip(days, closes, strict=True):
        o = c if prev is None else prev
        h, lo = max(o, c) + 1, min(o, c) - 1
        bars.append(DailyBar(day, o, h, lo, c, 1_000_000, o, h, lo, c))
        prev = c
    return bars


F1 = f_bars([70, 69, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90])  # c0 … c12
PEAK_F1, C0 = 89, 90  # c−1 与 c0 的下标
F2 = f_bars([98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 81, 88, 82, 83, 84, 85, 86, 87, 88, 89])


def test_F1_peak_day_atr_is_2_for_n20_and_n14() -> None:
    assert ind.atr(F1, PEAK_F1, 20) == 2.0
    assert ind.atr(F1, PEAK_F1, 14) == 2.0


def test_F1_c0_sma10_and_atr20() -> None:
    assert ind.true_range(F1, C0) == 32.0  # O = 100、H = 101、L = 69
    assert ind.sma(F1, C0, 10) == pytest.approx(97.0, abs=0)
    assert ind.atr(F1, C0, 20) == (19 * 2 + 32) / 20  # 3.5


def test_F1_peak_is_most_recent_100_and_low_is_earliest() -> None:
    assert ind.window_max_close(F1, C0, W) == (100.0, F1[PEAK_F1].day)
    # [c−1, c2] 最低 69 在 c1
    assert ind.min_close(F1, F1[PEAK_F1].day, C0 + 2) == (69.0, F1[C0 + 1].day)


def test_F2_e2_window_max_is_100_on_e0() -> None:
    e0, e2 = 89, 91
    assert ind.window_max_close(F2, e2, W) == (100.0, F2[e0].day)


def test_window_respects_since_and_tie_rules() -> None:
    i = len(F1) - 1  # c12 = 90
    since = F1[C0 + 5].day  # c5 起
    assert ind.window_max_close(F1, i, W, since) == (90.0, F1[i].day)
    # 最低收盘并列取最早：c0 = 70、c2 = 70，自 c0 起到 c2 的最低是 c1 = 69；自 c2 起只有 70、72…
    assert ind.min_close(F1, F1[C0 + 2].day, i) == (70.0, F1[C0 + 2].day)


def test_max_close_prior_excludes_today() -> None:
    i = C0 + 5  # c5 = 76；前 3 日 c2–c4 = 70、72、74
    assert ind.max_close_prior(F1, i, 3) == 74.0
    assert ind.max_close_prior(F1, 2, 3) is None


def test_box_range_max() -> None:
    assert ind.max_close(F1, F1[C0 + 2].day, F1[C0 + 4].day) == 74.0


@pytest.mark.parametrize(("n", "short", "enough"), [(20, 82, 83), (14, 76, 77)])
def test_S21a_S21b_history_length(n: int, short: int, enough: int) -> None:
    assert not ind.history_sufficient(short, W, n)
    assert ind.history_sufficient(enough, W, n)


def test_insufficient_history_returns_none() -> None:
    assert ind.atr(F1, 19, 20) is None and ind.atr(F1, 20, 20) == 2.0
    assert ind.sma(F1, 8, 10) is None and ind.sma(F1, 9, 10) == 100.0


def test_true_range_uses_previous_valid_bar() -> None:
    # I18：缺价日不占位，上一条有效日线即序列中的前一根
    bars = [F1[0], F1[C0]]
    assert ind.true_range(bars, 1) == max(101 - 69, abs(101 - 100), abs(69 - 100))


def test_atr_non_positive_rejected() -> None:
    d = date(2009, 1, 5)
    flat = [DailyBar(d + timedelta(days=i), 1, 1, 1, 1, 1, 1, 1, 1, 1) for i in range(3)]
    with pytest.raises(InvalidInputError):
        ind.atr(flat, 2, 2)
    with pytest.raises(InvalidInputError):
        ind.true_range(flat, 0)
