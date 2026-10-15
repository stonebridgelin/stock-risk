"""T08a B-03 补充测试。期望值由编码助手按 10 号规则推导（推导过程见 T08a 回执），
待产品经理确认后写入 13 号；不由引擎输出回填。

共同前提同 13 号第 1 节：P_ep = 100、ATR_ep = 2、k₁ = 2、k₂ = 4、K = 3、d = 3、u_rel = 3；
TWO_STAGE。D_ep = (100 − C) / 2；警戒的 a ⇔ C > 92；强恢复门槛 L_ep + min(6, 0.5 × (100 − L_ep))。
"""

from __future__ import annotations

import pytest

from stock_risk.domain.lifecycle.model import EventType, Level
from stock_risk.domain.lifecycle.params import AtrBasis, Confirm, LongPath, ReleaseHyst, ReleasePath
from tests.unit.lifecycle.fields import check_days, started
from tests.unit.lifecycle.harness import SYMBOL, params, s_run
from tests.unit.lifecycle.test_fixtures_v08_f import F2_PATH, F2X_SW, F3X_PATH, dec, f6_path, sw
from tests.unit.lifecycle.v08_harness import A, B, Day, Table, f_run_decimal, run_table
from tests.unit.lifecycle.v08_harness import params as v08_params

E = EventType


def test_B03_box_revoked_back_into_box() -> None:
    """BOX：b1–b3 的最高收盘 83 在 b3（since_obs = K）冻结；b4 收盘 84 > 83 确认；
    b5 收盘 83 不高于 box_high，确认消失，撤销原因 BACK_INTO_BOX。"""
    rows = [("b0", 80, None), ("b1", 81, None), ("b2", 82, None), ("b3", 83, None)]
    rows += [("b4", 84, None), ("b5", 83, None)]
    run = s_run(rows, params(Confirm.BOX))
    eid = f"{SYMBOL}:{run.labels['b0']}"
    assert run.rec("b3").box_high == 83.0 and run.rec("b3").confirmed is False  # since 不大于 K
    assert (run.rec("b4").confirmed, run.rec("b4").confirm_line) == (True, 83.0)
    revoked = run.event("b5", E.RECOVERY_REVOKED)
    assert revoked.identity == (eid, 1, 1)
    assert revoked.basis["reason"] == "BACK_INTO_BOX" and revoked.basis["line"] == 83.0
    check_days(
        run,
        {
            "b0": started("b0", "A"),
            "b3": {"box_high": 83.0, "events": [("STAB", 1)]},
            "b4": {"prev_confirm": True, "obs_issued": True, "dc": 1, "events": [("OBS", 1, 1)]},
            "b5": {"prev_confirm": False, "dc": 0, "events": [("REV", 1, 1)]},
        },
    )
    assert run.invariants() == []


def test_B03_box_roll_revoked_when_window_high_moves_up() -> None:
    """BOX_ROLL：确认线为今日之前 3 个有效交易日的最高收盘。
    r4：线 = max(81, 82, 83) = 83，84 > 83 确认；r5：线 = 84，85 > 84 仍确认；
    r6：收盘 85 与前一日相同（未跌），但线上移到 max(83, 84, 85) = 85，85 不高于 85，
    确认消失，撤销原因 BELOW_ROLLING_HIGH。"""
    rows = [("r0", 80, None), ("r1", 81, None), ("r2", 82, None), ("r3", 83, None)]
    rows += [("r4", 84, None), ("r5", 85, None), ("r6", 85, None)]
    run = s_run(rows, params(Confirm.BOX_ROLL))
    eid = f"{SYMBOL}:{run.labels['r0']}"
    lines = [run.rec(x).confirm_line for x in ("r4", "r5", "r6")]
    assert lines == [83.0, 84.0, 85.0]
    assert [run.rec(x).confirmed for x in ("r4", "r5", "r6")] == [True, True, False]
    revoked = run.event("r6", E.RECOVERY_REVOKED)
    assert revoked.identity == (eid, 1, 1)
    assert revoked.basis["reason"] == "BELOW_ROLLING_HIGH" and revoked.basis["close"] == 85.0
    check_days(
        run,
        {
            "r0": started("r0", "A"),
            "r3": {"events": [("STAB", 1)]},
            "r4": {"prev_confirm": True, "obs_issued": True, "dc": 1, "events": [("OBS", 1, 1)]},
            "r5": {"dc": 2},
            "r6": {"prev_confirm": False, "dc": 0, "events": [("REV", 1, 1)]},
        },
    )
    assert run.invariants() == []


def test_B03_s_and_l_same_day_short_peak_wins() -> None:
    """A 组（x₁ 8%、x₂ 15%、long X1，x_L = 8%）。
    长窗口：pl 收盘 110，距 T 81 个有效交易日（在 252 内、63 外）；
    短窗口最高 100（h79，并列取最近）。
    T 收盘 91：D_T = 4.5、DD_T = 9% → S 成立；DD^L = 1 − 91/110 ≈ 17.27% → L 也成立。
    按 S > R > L 取 S：P_ep = 100、ATR_ep = 2、D_ep = 4.5、DD_ep = 9% < 15% → 关注，原因 R1_GE_K1；
    entry_path = S，Δ_start = 9；dd_l 照记。
    （若误取 L：P_ep = 110、D_ep = 9.5、DD 17.3% → 警戒。）"""
    hist = [Day("pl", 110)] + [Day(f"h{i}", 100) for i in range(80)]
    run = run_table(Table([Day("T", 91)], history=hist), v08_params(A))
    rec = run.rec("T")
    assert rec.s_ok and rec.l_ok and not rec.r_ok
    assert rec.dd_t == pytest.approx(0.09) and rec.dd_l == pytest.approx(1 - 91 / 110)
    start = run.event("T", E.START)
    assert start.basis["reasons"] == ["R1_GE_K1"] and start.transition == "T7"
    assert (rec.level, rec.p_ep, rec.p_ep_day, rec.atr_ep) == (
        Level.WATCH,
        100.0,
        run.labels["h79"],
        2.0,
    )
    assert (rec.entry_path, rec.delta_start) == ("S", 9.0)
    assert rec.d_ep == pytest.approx(4.5) and rec.dd_ep == pytest.approx(0.09)
    v08 = start.basis["v08"]
    assert v08["entry_path"] == "S" and v08["dd_l"] == pytest.approx(1 - 91 / 110)
    assert v08["dd_t"] == pytest.approx(0.09) and (v08["x1"], v08["x2"]) == (0.08, 0.15)
    assert "peak_atr_available" not in v08  # L 路径的 A_start 字段只在 L 被选中时记录
    check_days(run, {"T": started("T", "W")})
    assert run.invariants() == []


# I17 ①：v0.8 开关打开时，只回放到第 upto 根与回放全段后截取到 upto，逐日结果完全相同。
# upto 为日线序号（峰值日为第 89 根）；每条路径抽 3 个日期，覆盖启动、加重或恢复、解除前后。
I17_V08 = {
    "F2x": (F2_PATH, F2X_SW, Confirm.SMA, ReleasePath.DIRECT, (93, 104, 107)),
    "F3x": (F3X_PATH, B, Confirm.BOX, ReleasePath.TWO_STAGE, (96, 102, 105)),
    "F6": (
        f6_path("0.1", 300),
        sw(LongPath.X1, 0.08, 0.15, ReleaseHyst.EFFECTIVE),
        Confirm.SMA,
        ReleasePath.TWO_STAGE,
        (169, 239, 339),
    ),
    "F7": (
        f6_path("0.03", 460),
        sw(LongPath.X1, 0.08, 0.15, ReleaseHyst.OLD),
        Confirm.SMA,
        ReleasePath.TWO_STAGE,
        (389, 536, 544),
    ),
}


@pytest.mark.parametrize("name", list(I17_V08))
def test_B03_I17_no_lookahead_v08(name: str) -> None:
    path, switches, confirm, release, uptos = I17_V08[name]
    closes = dec(path) if name in ("F2x", "F3x") else path
    p = v08_params(switches, confirm, release, AtrBasis.MIN)
    full = f_run_decimal("t", closes, p).result.results
    assert any(r.events for r in full)
    for upto in uptos:
        part = f_run_decimal("t", closes, p, upto=upto).result.results
        assert len(part) == upto + 1
        assert part == full[: upto + 1], f"{name} upto={upto}"
