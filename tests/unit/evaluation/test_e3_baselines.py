"""T09-E3：18 号 C42a–f（基线 B1、B2、B3）。期望值逐项取自夹具 JSON。"""

from decimal import Decimal

import pytest

from stock_risk.domain.evaluation.baselines import b3_expected_s1, run_b1, run_b2, sma50
from stock_risk.domain.evaluation.method_series import AlertLevel
from stock_risk.domain.numeric import InvalidInputError
from tests.unit.evaluation.helpers import load_fixtures, make_series

# 18 号第 4 节：DATA_BREAK_END 表示预警因历史断点终止；方法序列中的事件类型为 DATA_BREAK（D-15）
KIND = {"DATA_BREAK_END": "DATA_BREAK"}


def c_case(case_id: str) -> dict:  # type: ignore[type-arg]
    return load_fixtures()["C"][case_id]


def assert_baseline(case_id: str, runner) -> None:  # type: ignore[no-untyped-def]
    case = c_case(case_id)
    series = make_series(case)
    run = runner(series, case["d"])
    method, days = run.method, series.calendar.days
    got = [(e.kind, days.index(e.day), e.episode_id) for e in method.events]
    expected = [(KIND.get(k, k), day, ep) for k, day, ep, *_ in case["expected_events"]]
    assert got == expected, case_id
    for kind, day, _, *rest in case["expected_events"]:
        if kind == "START":  # START 的等级取当日日末等级
            assert method.levels[day] is AlertLevel(rest[0]), (case_id, day)
    for key, (level, episode, l_ep) in case["expected_days"].items():
        i = int(key)
        assert method.levels[i] == (AlertLevel(level) if level else None), (case_id, i)
        assert method.episodes[i] == episode, (case_id, i)
        assert run.l_ep[i] == (Decimal(l_ep) if l_ep else None), (case_id, i)


def test_C42a_b1_two_stage_exit_and_escalate_reset() -> None:
    assert_baseline("C42a", run_b1)


def test_C42b_b1_missing_day_freezes_exit_count() -> None:
    assert_baseline("C42b", run_b1)


def test_C42c_b1_no_start_before_mask() -> None:
    assert_baseline("C42c", run_b1)


def test_C42d_b1_data_break_ends_episode() -> None:
    assert_baseline("C42d", run_b1)


def test_C42e_b2_sma50_start_and_release() -> None:
    assert_baseline("C42e", run_b2)
    case = c_case("C42e")
    averages = sma50(make_series(case))
    for key, value in case["expected_sma50"].items():
        assert averages[int(key)] == Decimal(value), key


def test_C42e_b2_release_carries_l_ep() -> None:
    case = c_case("C42e")
    run = run_b2(make_series(case), case["d"])
    (release,) = run.method.events_of("RELEASE")
    assert release.l_ep == Decimal("97")  # 启动以来最低收盘（第 102 日）


def test_C42f_b3_expected_s1() -> None:
    case = c_case("C42f")
    series = make_series(load_fixtures()["A"][case["use_bars_of"]])
    days = series.calendar.days
    p, k = case["event"]["p"], case["event"]["k"]
    assert b3_expected_s1(series, days[p], days[k]) == Decimal(case["expected_S1"])


@pytest.mark.parametrize("d", [0, -1, True])
def test_baseline_rejects_invalid_d(d: int) -> None:
    series = make_series(c_case("C42a"))
    with pytest.raises(InvalidInputError):
        run_b1(series, d)
    with pytest.raises(InvalidInputError):
        run_b2(series, d)
