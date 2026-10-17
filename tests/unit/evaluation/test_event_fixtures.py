"""18 号 A 组：每个夹具独立测试，逐事件核对全部机读字段。"""

from tests.unit.evaluation.helpers import assert_case


def test_A01_deep_plateau() -> None:
    assert_case("A01")


def test_A02_quiet_and_successor() -> None:
    assert_case("A02")


def test_A03_peak_end_successor() -> None:
    assert_case("A03")


def test_A04_cap_continuation() -> None:
    assert_case("A04")


def test_A05_frozen_atr() -> None:
    assert_case("A05")


def test_A06_merge() -> None:
    assert_case("A06")


def test_A07_replay_successor() -> None:
    assert_case("A07")


def test_A08_early_confirmation() -> None:
    assert_case("A08")


def test_A09_confirmation_day_peak() -> None:
    assert_case("A09")


def test_A10a_confirmation_cap() -> None:
    assert_case("A10a")


def test_A10b_merge_on_cap() -> None:
    assert_case("A10b")


def test_A11_peak_before_cap() -> None:
    assert_case("A11")


def test_A12_latest_anchor_earliest_trough() -> None:
    assert_case("A12")


def test_A19c_strict_quiet_boundary() -> None:
    assert_case("A19c")


def test_A19d1_rounding_equal() -> None:
    assert_case("A19d1")


def test_A19d2_rounding_below() -> None:
    assert_case("A19d2")


def test_A24_rolling_recovery_high() -> None:
    assert_case("A24")


def test_A25_current_atr_on_replay() -> None:
    assert_case("A25")


def test_A29a_merge_without_successors() -> None:
    assert_case("A29a")


def test_A29b_multiple_replayed_successors() -> None:
    assert_case("A29b")


def test_A30_dependency_pending() -> None:
    assert_case("A30")


def test_A31_warmup() -> None:
    assert_case("A31")


def test_A39a_missing_without_warmup() -> None:
    assert_case("A39a")


def test_A39b_missing_and_warmup() -> None:
    assert_case("A39b")


def test_A_DB_break_resets_chain() -> None:
    assert_case("A_DB")


def test_A_P10_medium_decline() -> None:
    assert_case("A_P10")


def test_A_M1_missing_after_provisional_trough() -> None:
    assert_case("A_M1")  # T09-E1R：暂定谷底之后有一个缺价日


def test_A_M2_trailing_missing_before_data_end() -> None:
    assert_case("A_M2")  # T09-E1R：数据终点前有尾部缺价


def test_A_M3_missing_in_pending_confirmation() -> None:
    assert_case("A_M3")  # T09-E1R：待确认期内有缺价，数据在确认前结束


def test_all_A_cases_have_a_test() -> None:
    from tests.unit.evaluation.helpers import load_fixtures

    names = {n for n in globals() if n.startswith("test_A")}
    for case_id in load_fixtures()["A"]:
        assert any(n.startswith(f"test_{case_id}_") for n in names), case_id
