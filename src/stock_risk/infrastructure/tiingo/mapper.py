"""Tiingo EOD JSON → 领域对象。字段说明见 docs/develop_docs/02-数据获取.md。"""

from __future__ import annotations

from datetime import date
from typing import Any

from stock_risk.domain.market_data import ActionType, CorporateAction, DailyBar, PriceHistory


def _f(row: dict[str, Any], key: str) -> float:
    v = row.get(key)
    if v is None:
        raise ValueError(f"缺少字段 {key}: {row.get('date')}")
    return float(v)


def to_history(rows: list[dict[str, Any]]) -> PriceHistory:
    bars: list[DailyBar] = []
    actions: list[CorporateAction] = []
    seen: set[date] = set()
    for r in rows:
        d = date.fromisoformat(str(r["date"])[:10])
        if d in seen:
            raise ValueError(f"重复日期 {d}")
        seen.add(d)
        bars.append(
            DailyBar(
                day=d,
                open=_f(r, "open"),
                high=_f(r, "high"),
                low=_f(r, "low"),
                close=_f(r, "close"),
                volume=_f(r, "volume"),
                adj_open=_f(r, "adjOpen"),
                adj_high=_f(r, "adjHigh"),
                adj_low=_f(r, "adjLow"),
                adj_close=_f(r, "adjClose"),
            )
        )
        div = float(r.get("divCash") or 0.0)
        if div != 0.0:
            actions.append(CorporateAction(d, ActionType.DIVIDEND, div))
        split = r.get("splitFactor")
        if split is not None and float(split) != 1.0:
            actions.append(CorporateAction(d, ActionType.SPLIT, float(split)))
    bars.sort(key=lambda b: b.day)
    actions.sort(key=lambda a: (a.day, a.type))
    return PriceHistory(tuple(bars), tuple(actions))
