"""F1–F5 完整行情端到端验收（13 号第 2 节构造；期望以 10 号第 7.3 节为准），
以及 S14、S20（在 F3、F4 上追加断言）与 I17 第 ① 项（无前视）。

由日线计算指标，不跳过历史长度检查，不注入前态。
参数：N = 20、k₁ = 2、k₂ = 4、K = 3、d = 3、u_rel = 3、
K_stab = 8、G = 3、φ = 0.5；atr_basis 取 PEAK 与 MIN 各跑一次，两者结果必须相同。
"""

import pytest

from stock_risk.domain.lifecycle.model import Display, EventType, Level
from stock_risk.domain.lifecycle.params import FIXED, AtrBasis, Confirm, ReleasePath
from tests.unit.lifecycle.fields import check_days, released, reset_at, started
from tests.unit.lifecycle.harness import SYMBOL, Run, f_run, params

E = EventType
SMA, BOX, ROLL = Confirm.SMA, Confirm.BOX, Confirm.BOX_ROLL
TWO, DIRECT = ReleasePath.TWO_STAGE, ReleasePath.DIRECT

# 价格路径（12 号第 7.2 节；F2 按 13 号第 2 节延长 e21 = 90、e22 = 91、e23 = 92）
PATHS: dict[str, tuple[str, int, list[float]]] = {
    # 名称: (标签前缀, 峰值日标签序号, 峰值后收盘)
    "F1": ("c", -1, [70, 69, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90]),
    "F2": (
        "e",
        0,
        [
            98,
            96,
            94,
            92,
            90,
            88,
            86,
            84,
            82,
            80,
            81,
            88,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
        ],
    ),
    "F3": ("f", 0, [80, 78, 79, 80, 79.5, 81, 77]),
    "F4": ("g", 0, [98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 86, 81, 82, 84, 84.5, 79]),
    "F5": ("h", 0, [70, 69, 70, 70.5, 71, 72, 72.5, 73, 73.5, 74]),
}

COMMON = {
    "F1": {"c0": [E.START], "c4": [E.STABILIZED]},
    "F2": {"e2": [E.START], "e4": [E.ESCALATE], "e13": [E.STABILIZED]},
    "F3": {"f1": [E.START], "f5": [E.STABILIZED]},
    "F4": {"g2": [E.START], "g4": [E.ESCALATE], "g13": [E.STABILIZED]},
    "F5": {"h1": [E.START], "h5": [E.STABILIZED]},
}
OBS, DOWN, REL, FAIL = E.RECOVERY_OBS, E.DOWNGRADE, E.RELEASE, E.RECOVERY_FAIL
# 10 号第 7.3 节；同日多事件按规则推导（例如 F2 BOX e20 恢复观察与降级同日；直接解除路径中的降级）
SPECIFIC = {
    ("F1", SMA, TWO): {"c7": [OBS], "c9": [DOWN], "c12": [REL]},
    ("F1", SMA, DIRECT): {"c7": [OBS], "c9": [REL]},
    ("F1", BOX, TWO): {"c5": [OBS], "c7": [DOWN], "c10": [REL]},
    ("F1", BOX, DIRECT): {"c5": [OBS], "c7": [REL]},
    ("F1", ROLL, TWO): {"c5": [OBS], "c7": [DOWN], "c10": [REL]},
    ("F1", ROLL, DIRECT): {"c5": [OBS], "c7": [REL]},
    ("F2", SMA, TWO): {"e15": [OBS], "e17": [DOWN], "e20": [REL]},
    ("F2", SMA, DIRECT): {"e15": [OBS], "e17": [DOWN], "e19": [REL]},
    ("F2", BOX, TWO): {"e20": [OBS, DOWN], "e23": [REL]},
    ("F2", BOX, DIRECT): {"e20": [OBS, DOWN], "e22": [REL]},
    ("F2", ROLL, TWO): {"e16": [OBS], "e18": [DOWN], "e21": [REL]},
    ("F2", ROLL, DIRECT): {"e16": [OBS], "e18": [DOWN], "e19": [REL]},
    ("F3", SMA, TWO): {},
    ("F3", BOX, TWO): {"f6": [OBS], "f7": [FAIL]},
    ("F3", ROLL, TWO): {"f6": [OBS], "f7": [FAIL]},
    ("F4", SMA, TWO): {"g15": [OBS], "g16": [FAIL]},
    ("F4", BOX, TWO): {},
    ("F4", ROLL, TWO): {"g15": [OBS], "g16": [FAIL]},
    ("F5", SMA, TWO): {"h10": [OBS]},
    ("F5", BOX, TWO): {"h6": [OBS], "h8": [DOWN]},
    ("F5", ROLL, TWO): {"h6": [OBS], "h8": [DOWN]},
}
for _path in ("F3", "F4", "F5"):  # 这三条路径中两种解除路径结果相同（10 号第 7.3 节）
    for _m in (SMA, BOX, ROLL):
        SPECIFIC[(_path, _m, DIRECT)] = SPECIFIC[(_path, _m, TWO)]

# 恢复观察口径：距谷底日数 / 回升占比 / 距低点涨幅（10 号第 7.3 节）
OBS_METRICS = {
    ("F1", SMA): (6, 0.355, 0.159),
    ("F1", BOX): (4, 0.226, 0.101),
    ("F1", ROLL): (4, 0.226, 0.101),
    ("F2", SMA): (5, 0.20, 0.050),
    ("F2", BOX): (10, 0.45, 0.1125),
    ("F2", ROLL): (6, 0.25, 0.0625),
    ("F5", SMA): (8, 0.161, 0.072),
    ("F5", BOX): (4, 0.097, 0.043),
    ("F5", ROLL): (4, 0.097, 0.043),
}


# ---- B-01 逐日全字段（推导） ----
# 记法见 tests/unit/lifecycle/fields.py。推导所用的量：
# - F1：c1 新低后 L_ep = 69，强恢复门槛 69 + min(6, 15.5) = 75；SMA10 在 c4–c12 依次为
#   85.5、83.1、80.9、78.9、77.1、75.5、77.1、79.0、81.0；BOX 区间 c2–c4，box_high = 74。
# - F2：e10 后 L_ep = 80，门槛 80 + min(6, 10) = 86；SMA10 在 e13–e23 依次为 85.3、84.4、83.8、
#   83.5、83.5、83.8、84.4、85.3、86.2、86.5、87.5；BOX 区间 e11–e13，box_high = 88；
#   e18 起 since_obs ≥ 8（证据 c）。
# - F3：f2 后 L_ep = 78，门槛 84（不可达）；SMA10 在 f6 为 87.75；box_high = 80（f3–f5）。
# - F4：g10 后 L_ep = 80，门槛 86（不可达）；SMA10 在 g14、g15 为 84.3、83.75；box_high = 86。
# - F5：h2 后 L_ep = 69，门槛 75（不可达）；SMA10 在 h9、h10 为 74.15、71.55；box_high = 71；
#   h10 since_obs = 8。
# BOX_ROLL 的确认线是前 3 个有效交易日的最高收盘，各路径中与 BOX 同日首次成立（F2 除外，见下）。
def _cycles(prefix: str, first: int, last: int, cycle: int) -> dict:
    """连续新低：每日观察重置，周期依次加一。"""
    return {
        f"{prefix}{i}": reset_at(f"{prefix}{i}", cycle + i - first) for i in range(first, last + 1)
    }


def _down(h_d: float, **more) -> dict:
    return {
        "level": "W",
        "downgraded": True,
        "h_d": h_d,
        "dc": 0,
        "rc": 0,
        "events": [("DOWN",)],
        **more,
    }


def _obs(cycle: int, dc: int, **more) -> dict:
    return {
        "prev_confirm": True,
        "obs_issued": True,
        "dc": dc,
        "events": [("OBS", cycle, 1)],
        **more,
    }


def _f1(confirm: Confirm, release: ReleasePath) -> dict:
    days = {"c0": started("c0", "A"), "c1": reset_at("c1", 2), "c4": {"events": [("STAB", 2)]}}
    direct = release is DIRECT
    if confirm is SMA:  # c7 确认；c9 dc 达到 3（b、c）降级；DIRECT 时 c7 起强恢复，c9 xc 达到 3
        days |= {"c7": _obs(2, 1, xc=int(direct)), "c8": {"dc": 2, "xc": 2 * direct}}
        if direct:
            return days | {"c9": released("c9")}
        return days | {
            "c9": _down(84.0),
            "c10": {"h_d": 86.0, "rc": 1},
            "c11": {"h_d": 88.0, "rc": 2},
            "c12": released("c12"),
        }
    if confirm is BOX:
        days["c4"] = {"box_high": 74.0, "events": [("STAB", 2)]}
    days |= {"c5": _obs(2, 1, xc=int(direct)), "c6": {"dc": 2, "xc": 2 * direct}}
    if direct:
        return days | {"c7": released("c7")}
    return days | {
        "c7": _down(80.0),
        "c8": {"h_d": 82.0, "rc": 1},
        "c9": {"h_d": 84.0, "rc": 2},
        "c10": released("c10"),
    }


def _f2(confirm: Confirm, release: ReleasePath) -> dict:
    days = {
        "e2": started("e2", "W"),
        "e3": reset_at("e3", 2),
        "e4": {**reset_at("e4", 3), "level": "A", "events": [("ESC",)]},
        **_cycles("e", 5, 10, 4),
        "e13": {"events": [("STAB", 9)]},
    }
    direct = release is DIRECT
    if confirm is SMA:  # e15 确认；e17 降级（强恢复自 e17 起，86 ≥ 86）
        days |= {"e15": _obs(9, 1), "e16": {"dc": 2}, "e17": _down(86.0, xc=int(direct))}
        if direct:
            return days | {"e18": {"h_d": 87.0, "rc": 1, "xc": 2}, "e19": released("e19")}
        return days | {
            "e18": {"h_d": 87.0, "rc": 1},
            "e19": {"h_d": 88.0, "rc": 2},
            "e20": released("e20"),
        }
    if confirm is BOX:  # e18 起证据 c；e20 首次突破 88，确认与降级同日（b、c）
        days["e13"] = {"box_high": 88.0, "events": [("STAB", 9)]}
        days |= {
            "e18": {"dc": 1},
            "e19": {"dc": 2},
            "e20": {
                **_down(89.0, xc=int(direct)),
                "prev_confirm": True,
                "obs_issued": True,
                "events": [("OBS", 9, 1), ("DOWN",)],
            },
        }
        if direct:
            return days | {"e21": {"h_d": 90.0, "rc": 1, "xc": 2}, "e22": released("e22")}
        return days | {
            "e21": {"h_d": 90.0, "rc": 1},
            "e22": {"h_d": 91.0, "rc": 2},
            "e23": released("e23"),
        }
    # BOX_ROLL：e16 收盘 85 高于 e13–e15 的最高 84，首次确认；e18 降级（b、c）
    days |= {
        "e16": _obs(9, 1),
        "e17": {"dc": 2, "xc": int(direct)},
        "e18": _down(87.0, xc=2 * direct),
    }
    if direct:
        return days | {"e19": released("e19")}
    return days | {
        "e19": {"h_d": 88.0, "rc": 1},
        "e20": {"h_d": 89.0, "rc": 2},
        "e21": released("e21"),
    }


def _f3(confirm: Confirm) -> dict:
    days = {
        "f1": started("f1", "A"),
        "f2": reset_at("f2", 2),
        "f5": {"events": [("STAB", 2)]},
        "f7": reset_at("f7", 3),
    }
    if confirm is SMA:
        return days
    if confirm is BOX:
        days["f5"] = {"box_high": 80.0, "events": [("STAB", 2)]}
    return days | {"f6": _obs(2, 1), "f7": {**reset_at("f7", 3), "events": [("FAIL", 2)]}}


def _f4(confirm: Confirm) -> dict:
    days = {
        "g2": started("g2", "W"),
        "g3": reset_at("g3", 2),
        "g4": {**reset_at("g4", 3), "level": "A", "events": [("ESC",)]},
        **_cycles("g", 5, 10, 4),
        "g13": {"events": [("STAB", 9)]},
        "g16": reset_at("g16", 10),
    }
    if confirm is BOX:  # g14、g15 未突破 86
        days["g13"] = {"box_high": 86.0, "events": [("STAB", 9)]}
        return days
    return days | {"g15": _obs(9, 1), "g16": {**reset_at("g16", 10), "events": [("FAIL", 9)]}}


def _f5(confirm: Confirm) -> dict:
    days = {"h1": started("h1", "A"), "h2": reset_at("h2", 2), "h5": {"events": [("STAB", 2)]}}
    if confirm is SMA:  # h10 确认，同日 since_obs = 8，证据 b、c
        return days | {"h10": _obs(2, 1)}
    if confirm is BOX:
        days["h5"] = {"box_high": 71.0, "events": [("STAB", 2)]}
    # h6 确认；h8 降级 H_d = 73；此后无 a′、b′，rc 为 0，H_d 随收盘上移
    return days | {
        "h6": _obs(2, 1),
        "h7": {"dc": 2},
        "h8": _down(73.0),
        "h9": {"h_d": 73.5},
        "h10": {"h_d": 74.0},
    }


def expected_days(path: str, confirm: Confirm, release: ReleasePath) -> dict:
    if path == "F1":
        return _f1(confirm, release)
    if path == "F2":
        return _f2(confirm, release)
    return {"F3": _f3, "F4": _f4, "F5": _f5}[path](confirm)  # 两种解除路径相同（不可达强恢复）


def run_path(
    name: str, confirm: Confirm, release: ReleasePath, basis: AtrBasis, upto: int | None = None
) -> Run:
    prefix, peak_label, closes = PATHS[name]
    run = f_run(prefix, closes, params(confirm, release, basis), upto=upto)
    if peak_label == -1:  # F1：峰值日记为 c−1，峰值后第一天为 c0
        shifted = {
            f"{prefix}{i - 1}": d
            for i, d in ((int(k[len(prefix) :]), d) for k, d in run.labels.items())
        }
        run = Run(shifted, run.result, run.params)
    return run


def label_of(run: Run, day) -> str:
    return next(k for k, d in run.labels.items() if d == day)


CASES = [
    (p, m, r, b)
    for p in PATHS
    for m in (SMA, BOX, ROLL)
    for r in (TWO, DIRECT)
    for b in (AtrBasis.PEAK, AtrBasis.MIN)
]


@pytest.mark.parametrize(
    ("path", "confirm", "release", "basis"), CASES, ids=[f"{p}-{m}-{r}-{b}" for p, m, r, b in CASES]
)
def test_F_paths_events(path: str, confirm: Confirm, release: ReleasePath, basis: AtrBasis) -> None:
    run = run_path(path, confirm, release, basis)
    expected = {**COMMON[path]}
    for label, kinds in SPECIFIC[(path, confirm, release)].items():
        expected[label] = expected.get(label, []) + kinds
    actual = {
        label_of(run, r.record.day): sorted(e.type for e in r.events)
        for r in run.result.results
        if r.events
    }
    assert actual == {k: sorted(v) for k, v in expected.items()}
    # 不跳过历史长度检查：前 W + N − 1 = 82 根日线不可计算，峰值日已可计算
    recs = [r.record for r in run.result.results]
    need = FIXED.w + 20  # W + N
    assert all(rec.display is Display.INSUFFICIENT for rec in recs[: need - 1])
    assert recs[need - 1].display is Display.NORMAL and recs[89].display is Display.NORMAL
    check_days(run, expected_days(path, confirm, release))
    assert run.invariants() == []


@pytest.mark.parametrize(("path", "confirm"), list(OBS_METRICS))
def test_F_recovery_observation_metrics(path: str, confirm: Confirm) -> None:
    run = run_path(path, confirm, TWO, AtrBasis.PEAK)
    obs_day = next(r for r in run.result.results if any(e.type is OBS for e in r.events))
    rec = obs_day.record
    days = [r.record.day for r in run.result.results]
    days_from_trough = days.index(rec.day) - days.index(rec.l_ep_day)
    close = next(e for e in obs_day.events if e.type is OBS).basis["close"]
    exp_days, exp_share, exp_gain = OBS_METRICS[(path, confirm)]
    assert days_from_trough == exp_days
    assert (close - rec.l_ep) / (rec.p_ep - rec.l_ep) == pytest.approx(exp_share, abs=6e-4)
    assert close / rec.l_ep - 1 == pytest.approx(exp_gain, abs=6e-4)


@pytest.mark.parametrize(
    ("path", "confirm", "release"),
    [(p, m, r) for p in PATHS for m in (SMA, BOX, ROLL) for r in (TWO, DIRECT)],
)
def test_F_peak_and_min_identical(path: str, confirm: Confirm, release: ReleasePath) -> None:
    a = run_path(path, confirm, release, AtrBasis.PEAK).result
    b = run_path(path, confirm, release, AtrBasis.MIN).result
    assert [e.key for e in a.events] == [e.key for e in b.events]


def test_F_start_details() -> None:
    f1 = run_path("F1", SMA, TWO, AtrBasis.PEAK)
    start = f1.event("c0", E.START)
    assert start.basis["level"] == "ALERT" and start.basis["d_t"] == pytest.approx(15.0)
    assert (f1.rec("c0").p_ep, f1.rec("c0").atr_ep, f1.rec("c0").p_ep_day) == (
        100.0,
        2.0,
        f1.labels["c-1"],
    )
    f2 = run_path("F2", SMA, TWO, AtrBasis.PEAK)
    assert f2.event("e2", E.START).basis["level"] == "WATCH"
    assert f2.event("e2", E.START).basis["d_t"] == pytest.approx(2.0)
    assert f2.event("e4", E.ESCALATE).basis["reason"] == "D_EP_GE_K2"
    assert f2.rec("e4").d_ep == pytest.approx(4.0)
    # F2 BOX：e13 区间最高 88（e11–e13）冻结；e19 不大于 88；e20 降级原因 b、c
    box = run_path("F2", BOX, TWO, AtrBasis.PEAK)
    assert box.rec("e13").box_high == 88.0 and box.rec("e19").confirmed is False
    assert box.event("e20", E.DOWNGRADE).basis["reasons"] == ["b", "c"]
    assert [box.rec(x).dc for x in ("e18", "e19")] == [1, 2]


def test_S14_new_low_and_escalation_same_day_cycle_plus_one() -> None:
    run = run_path("F4", SMA, TWO, AtrBasis.PEAK)
    eid = f"{SYMBOL}:{run.labels['g2']}"
    cycles = {f"g{i}": run.rec(f"g{i}").cycle_no for i in range(2, 14)}
    assert cycles["g2"] == 1 and cycles["g3"] == 2 and cycles["g4"] == 3
    assert [cycles[f"g{i}"] for i in range(5, 11)] == [4, 5, 6, 7, 8, 9]
    assert run.rec("g4").reset_reasons == ("NEW_LOW", "ESCALATE")
    assert run.rec("g4").level is Level.ALERT
    assert run.event("g13", E.STABILIZED).identity == (eid, 9)


@pytest.mark.parametrize("confirm", [BOX, SMA])
def test_S20_F3_cycle_identity(confirm: Confirm) -> None:
    run = run_path("F3", confirm, TWO, AtrBasis.PEAK)
    eid = f"{SYMBOL}:{run.labels['f1']}"
    assert run.rec("f1").cycle_no == 1 and run.rec("f2").cycle_no == 2
    assert run.event("f5", E.STABILIZED).identity == (eid, 2)
    assert run.rec("f7").cycle_no == 3
    if confirm is BOX:
        assert run.rec("f5").box_high == 80.0 and run.rec("f6").box_high == 80.0  # f3–f5，f5 冻结
        assert run.event("f6", OBS).identity == (eid, 2, 1)
        assert run.event("f7", FAIL).identity == (eid, 2)
    else:
        assert all(e.type is not OBS for e in run.result.events)
        assert all(e.type is not FAIL for e in run.result.events)


@pytest.mark.parametrize("confirm", [SMA, BOX])
def test_S20_F4_cycle_identity(confirm: Confirm) -> None:
    run = run_path("F4", confirm, TWO, AtrBasis.PEAK)
    eid = f"{SYMBOL}:{run.labels['g2']}"
    assert run.rec("g13").cycle_no == 9 and run.rec("g16").cycle_no == 10
    if confirm is SMA:
        assert run.event("g15", OBS).identity == (eid, 9, 1)
        assert run.event("g16", FAIL).identity == (eid, 9)
    else:
        assert run.rec("g13").box_high == 86.0  # g11–g13
        assert run.rec("g14").confirmed is False and run.rec("g15").confirmed is False
        assert all(e.type is not FAIL for e in run.result.events)


@pytest.mark.parametrize("path", list(PATHS))
@pytest.mark.parametrize("confirm", [SMA, BOX, ROLL])
def test_I17_no_lookahead(path: str, confirm: Confirm) -> None:
    """只回放到 T 与回放全段后截取到 T，状态、事件与诊断记录完全相同（每条路径抽 3 个日期）。"""
    full = run_path(path, confirm, DIRECT, AtrBasis.MIN).result.results
    n = len(full)
    for upto in (85, 89 + (n - 89) // 2, n - 2):
        part = run_path(path, confirm, DIRECT, AtrBasis.MIN, upto=upto).result.results
        assert len(part) == upto + 1
        assert part == full[: upto + 1]
