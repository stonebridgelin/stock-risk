"""中转服务数据核验（T08d-2c 第 3 节）：价格逐日对账与参考数据比对。纯函数，照实统计，不作判定。

- 价格：中转服务未复权日线 vs Tiingo（data/raw）未复权收盘，区间 [2010-01-01, 2019-12-31]；
  差值 = 中转 − Tiingo，按 6 位小数取整后比较，"超过 0.01"为严格大于；
- 成交量相对差 = (中转 − Tiingo) ÷ Tiingo（Tiingo 成交量为 0 或缺失的日子不计，另计数），
  报告其中位数与绝对值的第 95 百分位（第 7 型）；只报告，不作判定标准；
- 参考数据：与 T08d-2a 官方免费档快照按同一查询逐条比对白名单字段。
判定标准（第 4 节）由产品经理执行，本模块不判定。
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from typing import Any

from stock_risk.domain.massive_paid import Bar, ReferenceBar
from stock_risk.domain.seal import EXPLORATION_END

VERIFY_FROM = date(2010, 1, 1)
VERIFY_TO = EXPLORATION_END
CLOSE_TOL = 0.01  # 第 3.1 节：收盘价之差超过 0.01 的天数
TOP_DIFFS = 5
ONLY_EXAMPLES = 20  # 第 3.2 节：只在一边出现的代码各列 20 个例子
FIELD_EXAMPLES = 10  # 第 3.2 节：字段不同各列 10 个例子
REF_FIELDS = (
    "ticker",
    "primary_exchange",
    "type",
    "active",
    "cik",
    "composite_figi",
    "list_date",
    "delisted_utc",
)
DELISTED_AFTER_2019 = "AFTER_2019"  # 封存后 delisted_utc 晚于截止日的统一写法


def _r6(x: float) -> float:
    return round(x, 6)


def quantile7(values: Sequence[float], p: float) -> float | None:
    """第 7 型分位数（线性插值）；空序列为 None。"""
    if not values:
        return None
    x = sorted(values)
    h = (len(x) - 1) * p
    lo = math.floor(h)
    if lo + 1 >= len(x):
        return x[-1]
    return x[lo] + (h - lo) * (x[lo + 1] - x[lo])


def compare_daily(
    relay: Sequence[Bar],
    tiingo: Sequence[ReferenceBar],
    calendar: Iterable[date],
    start: date = VERIFY_FROM,
    end: date = VERIFY_TO,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """逐日对账：返回 (逐日行, 汇总)。逐日行包含两边任一边有的日子。"""
    cal = set(calendar)
    mine = {b.day: b for b in relay if start <= b.day <= end}
    ref = {b.day: b for b in tiingo if start <= b.day <= end}
    rows: list[dict[str, Any]] = []
    diffs: list[tuple[date, float]] = []
    vol_rel: list[float] = []
    no_volume = relay_close_missing = 0
    for day in sorted(mine.keys() | ref.keys()):
        m, t = mine.get(day), ref.get(day)
        side = "both" if m and t else ("relay_only" if m else "tiingo_only")
        row: dict[str, Any] = {
            "date": day.isoformat(),
            "side": side,
            "in_spy_calendar": day in cal,
            "relay_close": m.close if m else None,
            "tiingo_close": t.close if t else None,
            "close_diff": None,
            "relay_volume": m.volume if m else None,
            "tiingo_volume": t.volume if t else None,
            "volume_rel_diff": None,
        }
        if m and t:
            if m.close is None:
                relay_close_missing += 1
            else:
                diff = _r6(m.close - t.close)
                row["close_diff"] = diff
                diffs.append((day, diff))
            if m.volume is None or not t.volume:
                no_volume += 1
            else:
                rel = (m.volume - t.volume) / t.volume
                row["volume_rel_diff"] = _r6(rel)
                vol_rel.append(rel)
        rows.append(row)
    return rows, _price_summary(rows, diffs, vol_rel, no_volume, relay_close_missing)


def _price_summary(
    rows: Sequence[Mapping[str, Any]],
    diffs: Sequence[tuple[date, float]],
    vol_rel: Sequence[float],
    no_volume: int,
    relay_close_missing: int,
) -> dict[str, Any]:
    relay_only = [r["date"] for r in rows if r["side"] == "relay_only"]
    tiingo_only = [r["date"] for r in rows if r["side"] == "tiingo_only"]
    over = [(d, x) for d, x in diffs if abs(x) > CLOSE_TOL]
    top = sorted(diffs, key=lambda dx: (-abs(dx[1]), dx[0]))[:TOP_DIFFS]
    p95 = quantile7([abs(v) for v in vol_rel], 0.95)
    return {
        "relay_rows": sum(1 for r in rows if r["side"] != "tiingo_only"),
        "tiingo_rows": sum(1 for r in rows if r["side"] != "relay_only"),
        "common_days": sum(1 for r in rows if r["side"] == "both"),
        "relay_only_days": len(relay_only),
        "relay_only_dates": "|".join(relay_only),
        "tiingo_only_days": len(tiingo_only),
        "tiingo_only_dates": "|".join(tiingo_only),
        "relay_close_missing": relay_close_missing,
        "max_abs_close_diff": max((abs(x) for _, x in diffs), default=None),
        "days_over_0_01": len(over),
        "share_over_0_01": _r6(len(over) / len(diffs)) if diffs else None,
        "top5_abs_diff": "|".join(f"{d.isoformat()}:{x}" for d, x in top),
        "median_close_diff": _r6(statistics.median(x for _, x in diffs)) if diffs else None,
        "volume_rel_diff_median": _r6(statistics.median(vol_rel)) if vol_rel else None,
        "volume_abs_rel_diff_p95": _r6(p95) if p95 is not None else None,
        "volume_days_skipped": no_volume,
    }


# ---- 参考数据 ----


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def ref_view(row: Mapping[str, Any]) -> dict[str, str]:
    """封存后的清单行（T08d-2a 的 lists_<日期>.csv 行或 SealedTicker.as_row()）
    → 白名单字段的字符串。

    delisted_utc 取封存后的 delisted；晚于截止日的记为 AFTER_2019。
    """
    after = _cell(row.get("delisted_after_2019")) == "true"
    out = {f: _cell(row.get(f)) for f in REF_FIELDS if f != "delisted_utc"}
    out["delisted_utc"] = DELISTED_AFTER_2019 if after else _cell(row.get("delisted"))
    return out


def _keyed(rows_by_query: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[tuple, dict]:  # type: ignore[type-arg]
    """(查询, 代码, 同一查询内同代码的出现序号) → 白名单字段。"""
    out: dict[tuple, dict] = {}  # type: ignore[type-arg]
    for query, rows in rows_by_query.items():
        seen: Counter[str] = Counter()
        for row in rows:
            view = ref_view(row)
            seen[view["ticker"]] += 1
            out[(query, view["ticker"], seen[view["ticker"]])] = view
    return out


def compare_snapshot(
    day: date,
    official: Mapping[str, Sequence[Mapping[str, Any]]],
    relay: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """一个快照日期：按查询逐条比对。返回 (汇总行, 例子行)。

    只在一边出现的比例与字段不同的比例都以官方快照条数为分母（另报中转条数）。
    """
    a, b = _keyed(official), _keyed(relay)
    only_a = sorted(a.keys() - b.keys())
    only_b = sorted(b.keys() - a.keys())
    common = sorted(a.keys() & b.keys())
    by_field: dict[str, list[tuple]] = {f: [] for f in REF_FIELDS}  # type: ignore[type-arg]
    rows_differ = 0
    for k in common:
        fields = [f for f in REF_FIELDS if a[k][f] != b[k][f]]
        rows_differ += bool(fields)
        for f in fields:
            by_field[f].append(k)
    n = len(a)
    summary: dict[str, Any] = {
        "date": day.isoformat(),
        "official_rows": n,
        "relay_rows": len(b),
        "only_official": len(only_a),
        "only_relay": len(only_b),
        "only_either_share": _r6((len(only_a) + len(only_b)) / n) if n else None,
        "rows_with_field_diff": rows_differ,
        "field_diff_share": _r6(rows_differ / n) if n else None,
        **{f"diff_{f}": len(ks) for f, ks in by_field.items()},
    }
    examples = [_only("only_official", k) for k in only_a[:ONLY_EXAMPLES]]
    examples += [_only("only_relay", k) for k in only_b[:ONLY_EXAMPLES]]
    for f, ks in by_field.items():
        for k in ks[:FIELD_EXAMPLES]:
            examples.append(
                {
                    "kind": "field_diff",
                    "field": f,
                    "query": k[0],
                    "ticker": k[1],
                    "official_value": a[k][f],
                    "relay_value": b[k][f],
                }
            )
    return summary, [{"date": day.isoformat(), **e} for e in examples]


def _only(kind: str, key: tuple) -> dict[str, Any]:  # type: ignore[type-arg]
    return {"kind": kind, "field": "", "query": key[0], "ticker": key[1]}
