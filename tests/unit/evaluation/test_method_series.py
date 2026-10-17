"""T09 E2 适配器测试：由生命周期引擎的 RunResult 构造 MethodSeries（D-15）。

用 13 号的夹具构造方法（S 组的按表指标，G = 3）：
- S1：s9 以 RELEASE 解除；
- 自建短序列：启动后短缺价（冻结等级），再连续缺价 G + 1 = 4 日形成 DATA_BREAK。
期望值按引擎规则推导，见各断言旁的注释。
"""

from decimal import Decimal

import pytest

from stock_risk.domain.evaluation.method_series import (
    AlertLevel,
    MethodEvent,
    MethodSeries,
    from_run_result,
)
from stock_risk.domain.numeric import InvalidInputError
from tests.unit.lifecycle.harness import SYMBOL, params, s_run
from tests.unit.lifecycle.test_fixtures_s1_s9 import S1_ROWS


def test_release_day_not_in_alert_and_episode_ids() -> None:
    run = s_run(S1_ROWS, params())
    days = [r.record.day for r in run.result.results]
    m = from_run_result("S1", days, run.result)
    lab = run.labels
    eid = f"{SYMBOL}:{lab['s0'].isoformat()}"
    i = {k: days.index(d) for k, d in lab.items()}
    assert not m.in_alert(i["p"])  # 峰值日无预警
    assert m.levels[i["s0"]] is AlertLevel.ALERT and m.episodes[i["s0"]] == eid  # 以警戒启动
    assert m.levels[i["s6"]] is AlertLevel.WATCH  # s6 降级
    assert m.in_alert(i["s8"]) and m.episodes[i["s8"]] == eid
    assert not m.in_alert(i["s9"]) and m.episodes[i["s9"]] is None  # 解除当日不在警
    release = m.events_of("RELEASE")
    # RELEASE 带被解除预警当日的 L_ep：s1 新低 78
    assert release == (MethodEvent("RELEASE", lab["s9"], eid, Decimal("78.000000")),)
    # 按日期：s0 START、s4 RECOVERY_OBS（STABILIZED 不计入通知）、s6 DOWNGRADE、s9 RELEASE
    assert [e.kind for e in m.events] == ["START", "RECOVERY_OBS", "DOWNGRADE", "RELEASE"]


def test_missing_days_keep_frozen_level_and_data_break_not_in_alert() -> None:
    rows = [
        ("a0", 80, None),  # D_T = 10：以警戒启动
        ("a1", None, None),  # 短缺价：冻结警戒
        ("a2", 81, 83),
        ("a3", None, None),
        ("a4", None, None),
        ("a5", None, None),  # 连续缺价第 3 日（= G）仍冻结
        ("a6", None, None),  # 第 4 日 = G + 1：DATA_BREAK
    ]
    run = s_run(rows, params())
    days = [r.record.day for r in run.result.results]
    m = from_run_result("BREAK", days, run.result)
    i = {k: days.index(d) for k, d in run.labels.items()}
    eid = f"{SYMBOL}:{run.labels['a0'].isoformat()}"
    for label in ("a0", "a1", "a2", "a3", "a4", "a5"):
        assert m.levels[i[label]] is AlertLevel.ALERT and m.episodes[i[label]] == eid, label
    assert not m.in_alert(i["a6"]) and m.episodes[i["a6"]] is None
    assert [(e.kind, e.day) for e in m.events] == [
        ("START", run.labels["a0"]),
        ("DATA_BREAK", run.labels["a6"]),
    ]


def test_method_series_rejects_inconsistent_input() -> None:
    run = s_run(S1_ROWS, params())
    days = [r.record.day for r in run.result.results]
    with pytest.raises(InvalidInputError):
        from_run_result("X", days[1:], run.result)
    with pytest.raises(InvalidInputError):
        MethodSeries("X", (days[0],), (AlertLevel.WATCH,), (None,), ())
