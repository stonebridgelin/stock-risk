"""T08a B-07：回放起点之前注入了预警的前态时抛错；缺价日的事件只在回放区间内产生。"""

from __future__ import annotations

import pytest

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.indicator_source import TableIndicators, TableRow
from stock_risk.domain.lifecycle.engine import episode_id_for
from stock_risk.domain.lifecycle.model import Episode, Level, SecurityState
from stock_risk.domain.lifecycle.runner import run_with_source
from stock_risk.domain.numeric import InvalidInputError
from tests.unit.lifecycle.harness import START_DAY, SYMBOL, params, weekdays


def _setup(with_bar: bool) -> tuple[list, TableIndicators, SecurityState]:
    days = weekdays(START_DAY, 3)
    rows = {days[0]: TableRow(100, 100, atr=2)}
    if with_bar:
        rows[days[1]] = TableRow(100, 85, atr=2, sma=90)
    rows[days[2]] = TableRow(85, 86, atr=2, sma=90)
    src = TableIndicators(TradingCalendar.from_days(days), rows, n=20, prior_valid_bars=10_000)
    ep = Episode(
        episode_id_for(SYMBOL, days[0]),
        days[0],
        Level.ALERT,
        100.0,
        days[0],
        2.0,
        80.0,
        days[0],
        obs_start=days[0],
        cycle_no=1,
        since_obs=5,
        prev_confirm=True,
        obs_n=1,
    )
    state = SecurityState(SYMBOL, seg_start=days[0], episode=ep)
    return days, src, state


@pytest.mark.parametrize("with_bar", [True, False])
def test_episode_before_replay_start_raises(with_bar: bool) -> None:
    """前态的预警出现在回放起点之前（有日线日与缺价日都检查）。"""
    days, src, state = _setup(with_bar)
    with pytest.raises(InvalidInputError, match="回放起点"):
        run_with_source(SYMBOL, days[1:], src, params(), replay_start=days[2], state=state)


def test_episode_inside_replay_runs() -> None:
    """同一前态从回放起点开始运行：缺价日撤销确认，属于回放区间内的事件。"""
    days, src, state = _setup(with_bar=False)
    out = run_with_source(SYMBOL, days[1:], src, params(), replay_start=days[1], state=state)
    kinds = sorted(e.type.value for e in out.results[0].events)
    assert kinds == ["DATA_ISSUE", "RECOVERY_REVOKED"]
