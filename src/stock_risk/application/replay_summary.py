"""summary.csv 的统计（03 计划 T04c，P-16、P-25）。只描述，不设门槛，不排名。纯函数。

- 时间范围只取 2010-01-01 至 2019-12-31；2009 年只作预热，不进入汇总。
- 预警按启动日期归属：启动日 ≥ 2010-01-01 为开发期；2009 年启动、延续到 2010 年的为"预热期启动"，
  单独报告个数与时长；2009 年内启动并结束的不计入。
- 开发期预警按终止原因分开：RELEASE、DATA_BREAK、截尾（LISTING_END、EXPLORATION_END）。
  持续时间只对 RELEASE 计算均值与分位数；DATA_BREAK 与截尾报告个数与已观察的有效交易日数。
- 占用率 = 可计算日中处于关注或警戒的日数 ÷ 可计算日数（2010-01-01 起）。
  可计算日 = 有日线且历史段充足；

  缺价日不进分子也不进分母，其中等级冻结为关注 / 警戒的缺价日另报；数据中断、不可计算日数分别列出。
- 恢复观察、撤销、恢复失败等事件次数按事件日期计（2010-01-01 起）。
- 解除后再次启动的间隔：解除日在汇总期内、紧邻的下一次启动的间隔，按有日线的交易日计
  （domain.lifecycle.restart，T08a C-06）。
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

from stock_risk.application.replay import SymbolReplay
from stock_risk.application.replay_episodes import EpisodeRow
from stock_risk.domain.lifecycle.model import DataStatus, EndReason, EventType
from stock_risk.domain.lifecycle.restart import EpisodeSpan, bar_days, restart_gaps

SUMMARY_START = date(2010, 1, 1)
SUMMARY_END = date(2019, 12, 31)
QUANTILES = ("p25", "p50", "p75", "p90")  # 描述用分位数（线性插值，statistics.quantiles inclusive）


def describe(values: Sequence[float], prefix: str) -> dict[str, Any]:
    """个数、均值、分位数、最小、最大；没有数据时为空。"""
    out: dict[str, Any] = {f"{prefix}_n": len(values)}
    keys = ["mean", *QUANTILES, "min", "max"]
    if not values:
        return out | {f"{prefix}_{k}": "" for k in keys}
    v = sorted(values)
    q4 = statistics.quantiles(v, n=4, method="inclusive") if len(v) > 1 else [v[0]] * 3
    q10 = statistics.quantiles(v, n=10, method="inclusive") if len(v) > 1 else [v[0]] * 9
    stats = {
        "mean": statistics.fmean(v),
        "p25": q4[0],
        "p50": q4[1],
        "p75": q4[2],
        "p90": q10[8],
        "min": v[0],
        "max": v[-1],
    }
    return out | {f"{prefix}_{k}": round(stats[k], 4) for k in keys}


def _in_period(d: date | None) -> bool:
    return d is not None and SUMMARY_START <= d <= SUMMARY_END


COUNTED_EVENTS = (
    EventType.START,
    EventType.ESCALATE,
    EventType.DOWNGRADE,
    EventType.RELEASE,
    EventType.STABILIZED,
    EventType.RECOVERY_OBS,
    EventType.RECOVERY_REVOKED,
    EventType.RECOVERY_FAIL,
    EventType.DATA_ISSUE,
)


def summarize(rep: SymbolReplay, episodes: Iterable[EpisodeRow]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": rep.candidate_id,
        "target": rep.target.label,
        "symbol": rep.target.symbol,
        "variant": rep.target.variant,
    }
    eps = list(episodes)
    row |= _episode_stats(eps)
    row |= _day_stats(rep)
    for kind in COUNTED_EVENTS:
        row[f"n_{kind.value.lower()}"] = sum(
            1 for e in rep.result.events if e.type is kind and _in_period(e.day)
        )
    gaps = _restart_gaps(rep, eps)
    row |= describe(gaps, "restart_gap")
    row["_restart_gaps"] = gaps
    return row


def _episode_stats(eps: list[EpisodeRow]) -> dict[str, Any]:
    """开发期预警按终止原因分开；预热期启动的单列。"""
    row: dict[str, Any] = {}
    dev = [e for e in eps if e.start_date >= SUMMARY_START]
    warm = [
        e
        for e in eps
        if e.start_date < SUMMARY_START and (e.end_date is None or e.end_date >= SUMMARY_START)
    ]
    by_reason = {r: [e for e in dev if e.end_reason is r] for r in EndReason}
    row["episodes_dev"] = len(dev)
    row["release_n"] = len(by_reason[EndReason.RELEASE])
    durations = [e.duration_trading_days for e in by_reason[EndReason.RELEASE]]
    row |= describe(durations, "release_duration")
    row["_release_durations"] = durations  # 以下划线开头的键不写入 CSV，只用于按候选合并
    row["data_break_n"] = len(by_reason[EndReason.DATA_BREAK])
    row["data_break_observed_days"] = sum(
        e.observed_valid_days for e in by_reason[EndReason.DATA_BREAK]
    )
    censored = by_reason[EndReason.LISTING_END] + by_reason[EndReason.EXPLORATION_END]
    row["censored_listing_end_n"] = len(by_reason[EndReason.LISTING_END])
    row["censored_exploration_end_n"] = len(by_reason[EndReason.EXPLORATION_END])
    row["censored_observed_days"] = sum(e.observed_valid_days for e in censored)
    row["warmup_started_n"] = len(warm)
    row["warmup_started_durations"] = ";".join(
        f"{e.duration_trading_days}{'' if e.end_reason is EndReason.RELEASE else '+'}" for e in warm
    )
    return row


def _day_stats(rep: SymbolReplay) -> dict[str, Any]:
    """可计算日、占用率、缺价日、数据中断与不可计算日数（汇总期内）。"""
    in_period = [r.record for r in rep.result.results if _in_period(r.record.day)]
    computable = [x for x in in_period if x.data_status is DataStatus.HAS_BAR and x.in_replay]
    warned = [x for x in computable if x.level is not None]
    missing = [x for x in in_period if x.data_status is DataStatus.MISSING]
    return {
        "computable_days": len(computable),
        "warning_days": len(warned),
        "occupancy": round(len(warned) / len(computable), 6) if computable else "",
        "missing_days": len(missing),
        "missing_days_frozen_warning": sum(1 for x in missing if x.level is not None),
        "data_break_days": sum(1 for x in in_period if x.data_status is DataStatus.DATA_BREAK),
        "insufficient_days": sum(1 for x in in_period if x.data_status is DataStatus.INSUFFICIENT),
    }


def _restart_gaps(rep: SymbolReplay, eps: list[EpisodeRow]) -> list[int]:
    """解除日在汇总期内的再次启动间隔（domain.lifecycle.restart，T08a C-06）。"""
    ordered_eps = sorted(eps, key=lambda e: e.start_date)
    spans = [EpisodeSpan(e.start_date, e.end_date, e.end_reason) for e in ordered_eps]
    all_gaps = restart_gaps(spans, bar_days([r.record for r in rep.result.results]))
    return [
        g
        for prev, g in zip(ordered_eps, all_gaps[1:], strict=False)
        if g is not None and _in_period(prev.end_date)
    ]


def summarize_candidates(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按候选汇总（SHLD 只计更正口径，避免重复）。占用率为合并后的分子 ÷ 分母。"""
    out: list[dict[str, Any]] = []
    by_cand: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        if r["variant"] != "uncorrected":
            by_cand.setdefault(int(r["candidate_id"]), []).append(r)
    sums = [
        "episodes_dev",
        "release_n",
        "data_break_n",
        "censored_listing_end_n",
        "censored_exploration_end_n",
        "warmup_started_n",
        "computable_days",
        "warning_days",
        "missing_days",
        "missing_days_frozen_warning",
        "data_break_days",
        "insufficient_days",
        "n_start",
        "n_escalate",
        "n_downgrade",
        "n_release",
        "n_stabilized",
        "n_recovery_obs",
        "n_recovery_revoked",
        "n_recovery_fail",
        "n_data_issue",
    ]
    for cid in sorted(by_cand):
        group = by_cand[cid]
        agg: dict[str, Any] = {"candidate_id": cid, "targets": len(group)}
        for k in sums:
            agg[k] = sum(int(g[k]) for g in group)
        agg["occupancy"] = (
            round(agg["warning_days"] / agg["computable_days"], 6) if agg["computable_days"] else ""
        )
        agg |= describe([x for g in group for x in g["_release_durations"]], "release_duration")
        agg |= describe([x for g in group for x in g["_restart_gaps"]], "restart_gap")
        out.append(agg)
    return out
