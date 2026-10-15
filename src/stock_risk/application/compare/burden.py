"""T06b 第 6 项：通知负担（每套每证券）。统计期间 2010-01-01 至 2019-12-31。纯函数。

- 每有效证券年 = 期间内有日线的交易日数 / 252。
- START 分为首次与短期再启动（紧邻的前一次预警以解除结束，且距其解除 ≤ 20 个有日线的交易日；
  口径见 domain.lifecycle.restart）。
- 占用率 = 可计算日中处于关注或警戒的日数 ÷ 可计算日数（与 15 号第 5 节相同）。
- 持续 ≥ 200 个交易日的预警：只统计开发期预警（启动日在期间内）。
- 解除后 20、60 个有效交易日内的最低收盘、相对解除收盘的跌幅、是否跌破所解除预警的 L_ep；
  窗口不完整的单列。未解除的预警（截尾或至观察终点）单列。
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from stock_risk.application.compare.views import RunView, window_after
from stock_risk.domain.lifecycle.model import EventType
from stock_risk.domain.lifecycle.restart import EpisodeSpan, restart_gaps

YEAR_DAYS = 252
SHORT_RESTART = 20
LONG_EPISODE = 200
EVENT_KINDS = (
    EventType.ESCALATE,
    EventType.DOWNGRADE,
    EventType.RELEASE,
    EventType.RECOVERY_OBS,
    EventType.RECOVERY_REVOKED,
    EventType.RECOVERY_FAIL,
)


def release_followups(view: RunView) -> list[dict[str, Any]]:
    rows = []
    for ep in view.episodes:
        if ep.end_reason is None or ep.end_reason.value != "RELEASE" or ep.end_date is None:
            continue
        i = view.index[ep.end_date]
        if not view.in_period(i):
            continue
        c0 = view.closes.get(ep.end_date)
        row: dict[str, Any] = {
            "target": view.target,
            "episode_id": ep.episode_id,
            "release_date": ep.end_date.isoformat(),
            "release_close": c0,
            "l_ep": ep.low_close,
        }
        for n in (20, 60):
            w = window_after(view, i, n)
            lows = [view.closes[view.days[j]] for j in w.day_indices]
            low = min(lows) if lows else None
            row[f"complete_{n}"] = w.complete
            row[f"min_close_{n}"] = low if low is not None else ""
            row[f"drop_{n}"] = round(low / c0 - 1, 6) if low is not None and c0 else ""
            row[f"below_l_ep_{n}"] = (
                low < ep.low_close if low is not None and ep.low_close is not None else ""
            )
        rows.append(row)
    return rows


def unreleased(view: RunView) -> list[dict[str, Any]]:
    return [
        {
            "target": view.target,
            "episode_id": ep.episode_id,
            "start_date": ep.start_date.isoformat(),
            "end_reason": ep.end_reason.value if ep.end_reason else "至观察终点",
            "end_date": ep.end_date.isoformat() if ep.end_date else "",
            "duration_trading_days": ep.duration_trading_days,
        }
        for ep in view.episodes
        if (ep.end_reason is None or ep.end_reason.value in ("LISTING_END", "EXPLORATION_END"))
        and ep.end_date is not None
        and view.in_period(view.index[ep.end_date])
    ]


def burden_row(view: RunView, followups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    period = [i for i in range(len(view.days)) if view.in_period(i)]
    valid_days = sum(1 for i in period if view.valid[i])
    years = valid_days / YEAR_DAYS if valid_days else 0.0
    row: dict[str, Any] = {
        "target": view.target,
        "valid_days": valid_days,
        "security_years": round(years, 6),
    }
    # 距紧邻前一次预警解除的有效交易日数（domain.lifecycle.restart，T08a C-06）
    spans = [EpisodeSpan(e.start_date, e.end_date, e.end_reason) for e in view.episodes]
    bars = [d for d, ok in zip(view.days, view.valid, strict=True) if ok]
    gaps = dict(zip(sorted(s.start_day for s in spans), restart_gaps(spans, bars), strict=True))
    first = short = 0
    for e in view.events_of(EventType.START):
        i = view.index[e.day]
        if not view.in_period(i):
            continue
        gap = gaps[e.day]
        if gap is not None and gap <= SHORT_RESTART:
            short += 1
        else:
            first += 1
    counts = {"start": first + short, "start_first": first, "start_short_restart": short}
    for kind in EVENT_KINDS:
        counts[kind.value.lower()] = sum(
            1 for e in view.events_of(kind) if view.in_period(view.index[e.day])
        )
    for k, v in counts.items():
        row[f"n_{k}"] = v
        row[f"per_year_{k}"] = round(v / years, 6) if years else ""
    comp = [i for i in period if view.computable[i]]
    warned = sum(1 for i in comp if view.level[i] > 0)
    row["computable_days"] = len(comp)
    row["warning_days"] = warned
    row["occupancy"] = round(warned / len(comp), 6) if comp else ""
    dev = [e for e in view.episodes if view.in_period(view.index[e.start_date])]
    long_eps = [e for e in dev if e.duration_trading_days >= LONG_EPISODE]
    row["n_long_episodes"] = len(long_eps)
    row["long_episode_days"] = sum(e.duration_trading_days for e in long_eps)
    for n in (20, 60):
        done = [f for f in followups if f[f"complete_{n}"]]
        row[f"releases_complete_{n}"] = len(done)
        row[f"releases_incomplete_{n}"] = len(followups) - len(done)
        row[f"below_l_ep_{n}"] = sum(1 for f in done if f[f"below_l_ep_{n}"] is True)
        drops = [f[f"drop_{n}"] for f in done if f[f"drop_{n}"] != ""]
        row[f"median_drop_{n}"] = round(statistics.median(drops), 6) if drops else ""
    row["n_unreleased"] = len(unreleased(view))
    return row


SUM_KEYS = (
    "valid_days",
    "computable_days",
    "warning_days",
    "n_long_episodes",
    "long_episode_days",
    "n_unreleased",
    "releases_complete_20",
    "releases_complete_60",
    "releases_incomplete_20",
    "releases_incomplete_60",
    "below_l_ep_20",
    "below_l_ep_60",
)


def burden_by_set(set_id: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """按套合并：次数之和 ÷ 有效证券年之和；占用率两种口径（合并证券日、跨证券中位数）。"""
    out: dict[str, Any] = {"set_id": set_id, "targets": len(rows)}
    years = sum(r["security_years"] for r in rows)
    out["security_years"] = round(years, 6)
    for k in [k for k in rows[0] if k.startswith("n_") and k not in SUM_KEYS]:
        total = sum(r[k] for r in rows)
        out[k] = total
        out[f"per_year_{k[2:]}"] = round(total / years, 6) if years else ""
    for k in SUM_KEYS:
        out[k] = sum(r[k] for r in rows)
    out["occupancy_pooled"] = (
        round(out["warning_days"] / out["computable_days"], 6) if out["computable_days"] else ""
    )
    occ = [r["occupancy"] for r in rows if r["occupancy"] != ""]
    out["occupancy_median_across_securities"] = round(statistics.median(occ), 6) if occ else ""
    return out
