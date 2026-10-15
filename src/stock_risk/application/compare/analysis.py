"""T06b 对照分析（一个回放对象上，新变体对基准 V0）。纯函数，只描述，不设门槛、不排名、不下结论。

- 逐日覆盖：只取双方都可计算的日子；缺价、不可计算、数据中断另列。
- 事件对照：每条记录可带多个标签（分号分隔），不做互斥分类；不称为"延迟"。
- 重叠：双方日末等级为关注或警戒的日子；解除日日末已为正常，不算重叠。
全部限于统计期间（2010-01-01 至 2019-12-31）。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from stock_risk.application.compare.views import ALERT, RunView, window_after
from stock_risk.domain.lifecycle.model import Event, EventType, Level

WINDOWS = (20, 60)


# ---------------- 1. 逐日覆盖 ----------------


def coverage(base: RunView, new: RunView) -> tuple[list[tuple], dict[str, int], list[tuple]]:
    """返回（逐日行、四格计数、排除日行）。两侧日历相同（同一对象的观察区间）。"""
    if base.days != new.days:
        raise ValueError(f"{base.target}：两套的日历不同")
    rows, excluded = [], []
    cells = Counter({"both": 0, "only_v0": 0, "only_new": 0, "neither": 0})
    for i, d in enumerate(base.days):
        if not base.in_period(i):
            continue
        if base.computable[i] and new.computable[i]:
            rows.append(
                (
                    base.target,
                    d.isoformat(),
                    base.status[i],
                    base.level[i],
                    base.episode[i] or "",
                    new.status[i],
                    new.level[i],
                    new.episode[i] or "",
                )
            )
            a, b = base.level[i] > 0, new.level[i] > 0
            cells["both" if a and b else "only_v0" if a else "only_new" if b else "neither"] += 1
        else:
            excluded.append((new.set_id, base.target, d.isoformat(), base.status[i], new.status[i]))
    cells["common_days"] = len(rows)
    return rows, dict(cells), excluded


# ---------------- 3. 重叠 ----------------


def overlaps(base: RunView, new: RunView) -> dict[tuple[str, str], list[int]]:
    """(V0 episode, 新变体 episode) → 重叠日下标。"""
    out: dict[tuple[str, str], list[int]] = {}
    for i in range(len(base.days)):
        if base.in_period(i) and base.level[i] > 0 and new.level[i] > 0:
            out.setdefault((base.episode[i], new.episode[i]), []).append(i)  # type: ignore[arg-type]
    return out


def overlap_rows(base: RunView, new: RunView, ov: dict[tuple[str, str], list[int]]) -> list[tuple]:
    return [
        (base.target, a, b, len(ix), base.days[ix[0]].isoformat(), base.days[ix[-1]].isoformat())
        for (a, b), ix in sorted(ov.items())
    ]


def unique_partner(ov: dict[tuple[str, str], list[int]], eid: str, side: int) -> str | None:
    """互为唯一重叠时返回对方 episode_id（side 0：eid 为 V0 一侧；1：新变体一侧）。"""
    mine = [k for k in ov if k[side] == eid]
    if len(mine) != 1:
        return None
    other = mine[0][1 - side]
    theirs = [k for k in ov if k[1 - side] == other]
    return other if len(theirs) == 1 else None


# ---------------- 2. 事件对照 ----------------


def _first_event(view: RunView, idx: Iterable[int], pred) -> int | None:  # type: ignore[no-untyped-def]
    wanted = set(idx)
    for e in view.events:
        i = view.index[e.day]
        if i in wanted and pred(e, i):
            return i
    return None


def _wait_fields(src: RunView, other: RunView, i: int, pred) -> tuple[dict[str, Any], list[str]]:  # type: ignore[no-untyped-def]
    """事件日 i 之后 20、60 个有效交易日内，对方首次满足 pred 的日子、等待日数与期间路径。"""
    out: dict[str, Any] = {}
    tags: list[str] = []
    c0 = src.closes.get(src.days[i])
    for n in WINDOWS:
        w = window_after(other, i, n)
        hit = _first_event(other, w.day_indices, pred)
        span = w.day_indices if hit is None else tuple(j for j in w.day_indices if j <= hit)
        lows = [other.closes[other.days[j]] for j in span if other.days[j] in other.closes]
        out[f"first_{n}"] = other.days[hit].isoformat() if hit is not None else ""
        out[f"wait_{n}"] = w.day_indices.index(hit) + 1 if hit is not None else ""
        out[f"min_close_rel_{n}"] = round(min(lows) / c0 - 1, 6) if lows and c0 else ""
        end = w.end_index
        out[f"end_close_rel_{n}"] = (
            round(other.closes[other.days[end]] / c0 - 1, 6)
            if end is not None and other.days[end] in other.closes and c0
            else ""
        )
        if hit is None:
            tags.append(f"{n} 日内无通知" if w.complete else f"窗口不完整（{n} 日）")
    return out, tags


def _is_start(e: Event, i: int) -> bool:
    return e.type is EventType.START


def _enters_alert(e: Event, i: int) -> bool:
    return e.type is EventType.ESCALATE or (
        e.type is EventType.START and e.basis.get("level") == Level.ALERT.name
    )


def event_compare(base: RunView, new: RunView) -> list[dict[str, Any]]:
    ov = overlaps(base, new)
    rows: list[dict[str, Any]] = []
    kinds = (EventType.START, EventType.ESCALATE, EventType.RELEASE)
    for src, other, direction, side in ((base, new, "V0→新", 0), (new, base, "新→V0", 1)):
        for e in src.events:
            i = src.index[e.day]
            if not src.in_period(i) or e.type not in kinds:
                continue
            if e.type is EventType.ESCALATE and direction != "V0→新":
                continue
            row: dict[str, Any] = {
                "target": src.target,
                "direction": direction,
                "event_type": e.type.value,
                "date": e.day.isoformat(),
                "episode_id": e.identity[0],
                "close": src.closes.get(e.day, ""),
                "other_status": other.status[i],
                "other_episode_id": other.episode[i] or "",
            }
            tags: list[str] = []
            if e.type is EventType.START:
                if other.level[i] > 0:
                    tags.append(f"对方当日已在预警（{other.status[i]}）")
                else:
                    fields, more = _wait_fields(src, other, i, _is_start)
                    row |= fields
                    tags += more
            elif e.type is EventType.ESCALATE:
                if other.status[i] != ALERT:
                    fields, more = _wait_fields(src, other, i, _enters_alert)
                    row |= fields
                    tags += [t.replace("无通知", "未进入警戒") for t in more]
                else:
                    tags.append("对方当日为警戒")
            else:  # RELEASE
                tags += _release_partner(src, other, ov, e, side, row)
            row["tags"] = ";".join(tags)
            rows.append(row)
    return rows


def _release_partner(
    src: RunView,
    other: RunView,
    ov: dict[tuple[str, str], list[int]],
    e: Event,
    side: int,
    row: dict[str, Any],
) -> list[str]:
    """解除事件：找唯一的重叠对方预警，记录其终止日与原因、两侧解除日之差。返回标签。"""
    partner = unique_partner(ov, e.identity[0], side)
    if partner is None:
        return ["多对多" if any(k[side] == e.identity[0] for k in ov) else "无重叠"]
    row["partner_episode_id"] = partner
    p_end = next((x for x in other.episodes if x.episode_id == partner), None)
    if p_end is not None and p_end.end_reason is not None and p_end.end_date:
        row["partner_end_date"] = p_end.end_date.isoformat()
        row["partner_end_reason"] = p_end.end_reason.value
        if p_end.end_reason.value == "RELEASE":
            diff = _valid_between(src, e.day, p_end.end_date)
            row["release_day_diff"] = diff  # 对方解除日 − 本方解除日（有效交易日）
    return []


def _valid_between(view: RunView, a, b) -> int:  # type: ignore[no-untyped-def]
    """b − a 的有效交易日数（b 早于 a 时为负）。"""
    ia, ib = view.index[a], view.index[b]
    lo, hi = sorted((ia, ib))
    n = sum(1 for j in range(lo + 1, hi + 1) if view.valid[j])
    return n if ib >= ia else -n


# ---------------- 5. 专项清单 ----------------


def cancelled_escalations(base: RunView, new: RunView) -> list[dict[str, Any]]:
    """V0 有、新变体当日没有的进入警戒（ESCALATE）。"""
    rows = []
    for e in base.events_of(EventType.ESCALATE):
        i = base.index[e.day]
        if not base.in_period(i):
            continue
        if any(_enters_alert(x, i) and x.day == e.day for x in new.events):
            continue
        fields, tags = _wait_fields(base, new, i, _enters_alert)
        rows.append(
            {
                "target": base.target,
                "date": e.day.isoformat(),
                "reason": e.basis.get("reason"),
                "d_ep": base.d_ep[i],
                "dd_ep": base.dd_ep[i],
                "new_status": new.status[i],
                **fields,
                "tags": ";".join(t.replace("无通知", "未进入警戒") for t in tags),
            }
        )
    return rows


def l_path_starts(base: RunView, new: RunView) -> list[dict[str, Any]]:
    """只有 L 路径成立的 START（原因只有 R1L_CUMULATIVE）。"""
    rows = []
    for e in new.events_of(EventType.START):
        i = new.index[e.day]
        if not new.in_period(i) or e.basis.get("reasons") != ["R1L_CUMULATIVE"]:
            continue
        fields, _ = _wait_fields(new, new, i, lambda x, j: False)  # 只取此后路径
        rows.append(
            {
                "target": new.target,
                "date": e.day.isoformat(),
                "episode_id": e.identity[0],
                "dd_l": (e.basis.get("v08") or {}).get("dd_l"),
                "close": new.closes.get(e.day, ""),
                "v0_status": base.status[i],
                **{k: v for k, v in fields.items() if k.startswith(("min_", "end_"))},
            }
        )
    return rows


def l_release_without_rise(new: RunView) -> list[dict[str, Any]]:
    """V1 中 L 启动后不涨即解除（10 号第 1A.5 节已知副作用）：解除收盘不高于启动收盘。"""
    rows = []
    for ep in new.episodes:
        if (
            ep.entry_path == "L"
            and ep.end_reason is not None
            and ep.end_reason.value == "RELEASE"
            and ep.end_close is not None
            and ep.start_close is not None
            and ep.end_close <= ep.start_close
            and ep.start_date >= new.days[0]
            and new.in_period(new.index[ep.start_date])
        ):
            rows.append(
                {
                    "target": new.target,
                    "episode_id": ep.episode_id,
                    "start_date": ep.start_date.isoformat(),
                    "start_close": ep.start_close,
                    "release_date": ep.end_date.isoformat() if ep.end_date else "",
                    "release_close": ep.end_close,
                    "duration_trading_days": ep.duration_trading_days,
                }
            )
    return rows


def r2_filtered(off: RunView, on: RunView) -> tuple[list[dict[str, Any]], int]:
    """r2_floor OFF / ON 配对中被过滤的 R2。返回（逐条清单，被过滤的 R2 日数）。

    标签：(i) 当日 GAP 证据被过滤，但同日或随后由其他路径启动（记日期差）；
    (ii) OFF 中的 START 在 ON 中被取消（记 ON 的后续通知，或"至观察终点无后续通知"）。
    """
    rows, n_days = [], 0
    on_starts = [on.index[e.day] for e in on.events_of(EventType.START)]
    off_start_days = {e.day for e in off.events_of(EventType.START)}
    for i, flag in enumerate(on.r2_filtered):
        if not flag or not on.in_period(i):
            continue
        n_days += 1
        d = on.days[i]
        later = [j for j in on_starts if j >= i]
        tags, row = [], {"target": on.target, "date": d.isoformat(), "close": on.closes.get(d, "")}
        if later:
            j = later[0]
            row["on_next_start"] = on.days[j].isoformat()
            row["on_start_gap_days"] = _valid_between(on, d, on.days[j])
            tags.append("(i) 其他路径启动" if j == i else "(i) 随后由其他路径启动")
        if d in off_start_days and i not in on_starts:
            tags.append("(ii) OFF 的 START 在 ON 中被取消")
            if not later:
                tags.append("至观察终点无后续通知")
        row["tags"] = ";".join(tags)
        rows.append(row)
    return rows, n_days
