"""T08a C-06：距上一次解除的有效交易日数（domain.lifecycle.restart）。"""

from __future__ import annotations

from datetime import date

import pytest

from stock_risk.domain.lifecycle.model import EndReason
from stock_risk.domain.lifecycle.restart import EpisodeSpan, days_since_prev_release, restart_gaps

D = [date(2010, 1, d) for d in (4, 5, 6, 7, 8, 11, 12, 13)]
# 1 月 7 日缺价（不在有日线的交易日中）
BARS = [d for d in D if d != date(2010, 1, 7)]
REL = EndReason.RELEASE


def test_counts_bar_days_after_release_up_to_start() -> None:
    prev = EpisodeSpan(D[0], D[1], REL)  # 1 月 5 日解除
    # 区间 (1/5, 1/11]：1/6、1/8、1/11（1/7 缺价不计）
    assert days_since_prev_release(prev, D[5], BARS) == 3
    assert days_since_prev_release(prev, D[2], BARS) == 1


@pytest.mark.parametrize(
    "reason", [EndReason.DATA_BREAK, EndReason.LISTING_END, EndReason.EXPLORATION_END, None]
)
def test_previous_must_end_in_release(reason: EndReason | None) -> None:
    assert days_since_prev_release(EpisodeSpan(D[0], D[1], reason), D[5], BARS) is None
    assert days_since_prev_release(None, D[5], BARS) is None


def test_only_immediately_preceding_episode() -> None:
    """更早的一次以解除结束，但紧邻的前一次以断点结束：为空（不回溯到更早的解除）。"""
    spans = [
        EpisodeSpan(D[2], D[3], EndReason.DATA_BREAK),
        EpisodeSpan(D[0], D[1], REL),
        EpisodeSpan(D[5], None, None),
    ]
    # 按启动日排序：解除（1/4–1/5）→ 断点（1/6–1/7）→ 1/11 启动
    assert restart_gaps(spans, BARS) == [None, 1, None]
    assert restart_gaps(spans[1:], BARS) == [None, 3]


def test_release_not_before_start_rejected() -> None:
    with pytest.raises(ValueError):
        days_since_prev_release(EpisodeSpan(D[0], D[5], REL), D[5], BARS)
