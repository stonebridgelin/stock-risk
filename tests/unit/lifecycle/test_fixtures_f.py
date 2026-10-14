"""F1–F5 完整行情端到端验收（13 号第 2 节构造；期望以 10 号第 7.3 节为准），
以及 S14、S20（在 F3、F4 上追加断言）与 I17 第 ① 项（无前视）。

由日线计算指标，不跳过历史长度检查，不注入前态。
参数：N = 20、k₁ = 2、k₂ = 4、K = 3、d = 3、u_rel = 3、
K_stab = 8、G = 3、φ = 0.5；atr_basis 取 PEAK 与 MIN 各跑一次，两者结果必须相同。
"""

import pytest

from stock_risk.domain.lifecycle.model import Display, EventType, Level
from stock_risk.domain.lifecycle.params import FIXED, AtrBasis, Confirm, ReleasePath
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
