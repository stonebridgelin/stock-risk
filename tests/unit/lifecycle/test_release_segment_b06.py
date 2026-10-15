"""T08a B-06：只由 xc 达到 d 解除时，RELEASE 的 segment（rc 计数段）写空，xc 计数段另列为
xc_segment。期望由夹具推导：xc 计数段为 xc 连续计数期间每日的证据集合（警戒日为 E_down，关注日为
E_rel）。"""

from __future__ import annotations

from stock_risk.domain.lifecycle.model import EventType
from stock_risk.domain.lifecycle.params import ReleasePath
from tests.unit.lifecycle.harness import params, s_run

E = EventType
S10_ROWS = [("d1", 80, None), ("d2", 81, 84), ("d3", 82, 83), ("d4", 83, 82.5)]
S10_ROWS += [("d5", 87, 83), ("d6", 89, 83.5), ("d7", 90, 84)]


def test_xc_only_release_has_empty_segment() -> None:
    """S10 甲：d5、d6 警戒日强恢复（证据 {b}），d6 降级保留 xc = 2；d7 关注日（{b′}），
    xc 达到 3、rc 只到 1，只由 xc 解除。"""
    run = s_run(S10_ROWS, params(release=ReleasePath.DIRECT))
    rel = run.event("d7", E.RELEASE)
    assert rel.basis["reasons"] == ["DIRECT_STRONG_RECOVERY"]
    assert rel.basis["segment"] == []
    assert rel.basis["xc_segment"] == [["b"], ["b"], ["b′"]]


def test_t14_release_in_alert() -> None:
    """S6 DIRECT：w3–w5 警戒日强恢复，T14 解除；segment 原本就为空。"""
    rows = [("w0", 80, None), ("w1", 81, 83), ("w2", 82, 82.5), ("w3", 87, 82.8)]
    rows += [("w4", 88, 83.5), ("w5", 89, 84.2)]
    run = s_run(rows, params(release=ReleasePath.DIRECT))
    rel = run.event("w5", E.RELEASE)
    assert rel.transition == "T14" and rel.basis["segment"] == []
    assert rel.basis["xc_segment"] == [["b"], ["b"], ["b"]]


def test_xc_and_rc_release_keeps_rc_segment() -> None:
    """S16：d8–d10 关注日 {b′}，rc 与 xc 同日达到 3，segment 仍为 rc 计数段。"""
    rows = [*S10_ROWS[:3], ("d4", 83, 82.5), ("d5", 87, 83), ("d6", 89, 83.5)]
    rows += [("d7", 88, 89), ("d8", 91, 89.5), ("d9", 92, 90), ("d10", 93, 90.5)]
    run = s_run(rows, params(release=ReleasePath.DIRECT))
    rel = run.event("d10", E.RELEASE)
    assert rel.basis["reasons"] == ["DIRECT_STRONG_RECOVERY", "b′"]
    assert rel.basis["segment"] == [["b′"], ["b′"], ["b′"]]
    assert rel.basis["xc_segment"] == [["b′"], ["b′"], ["b′"]]


def test_two_stage_release_has_no_xc_segment() -> None:
    rows = [("w0", 80, None), ("w1", 81, 83), ("w2", 82, 82.5), ("w3", 87, 82.8)]
    rows += [("w4", 88, 83.5), ("w5", 89, 84.2), ("w6", 90, 85), ("w7", 91, 85.8)]
    rows += [("w8", 92, 86.6)]
    run = s_run(rows, params())
    rel = run.event("w8", E.RELEASE)
    assert rel.basis["segment"] == [["b′"], ["b′"], ["b′"]] and "xc_segment" not in rel.basis
