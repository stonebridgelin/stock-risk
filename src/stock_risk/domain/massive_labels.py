"""Massive 付费档验证第 3 节：无标签证券的分类。纯函数。

清单逐行保存（白名单字段，封存同 T08d-2a 第 2.3 节）；对主交易所或 type 为空的记录，按顺序尝试：
- 规则 1：同一 composite_figi 在其他快照中有非空取值；
- 规则 2：没有 FIGI 时，同一 CIK 且同一代码在其他快照中的非空取值；
- 规则 3：只用于 type，按代码后缀判断（SUFFIX_TYPES）。
其他快照有多个不同取值时记为矛盾，补值取日期最近的快照（同样近时取较晚的日期）。照实统计，不作结论。

**只用于事后诊断，不得用于正式资格判定**（T09-C1 C-06）：补标签会用较晚的快照补较早的记录，
用于季度资格判定会引入前视。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from stock_risk.domain.massive_followup import EMPTY
from stock_risk.domain.massive_probe import QUERY_DATES, ListQuery
from stock_risk.domain.seal import EXPLORATION_END, assert_dates_not_sealed

LIST_FIELDS = (
    "ticker",
    "market",
    "locale",
    "primary_exchange",
    "type",
    "active",
    "currency_name",
    "cik",
    "composite_figi",
    "share_class_figi",
)
UNFILLED_EXAMPLES = 20
CONFLICT_EXAMPLES = 10
SAMPLES_PER_FIELD = 10

# 规则 3（只用于 type）：Massive 代码后缀 → type。依据任务卡举例，
# 并按 T08d-2a 输出中已有 type 的记录核对；小写 p 的已有记录中 PFD 约 2/3、SP 约 1/3，
# 按任务卡记为 PFD。产品经理未给完整对照表，回执中待确认。
SUFFIX_TYPES: tuple[tuple[str, str], ...] = (
    (r"[A-Z0-9]p[A-Z]*$", "PFD"),  # 小写 p：优先股，如 ABRpA
    (r"\.WS(\.[A-Z])?$|\.W$", "WARRANT"),  # .WS / .W：认股权证
    (r"\.U$", "UNIT"),  # .U：单位
    (r"\.R$|[A-Z]rw?$", "RIGHT"),  # .R、小写 r / rw：认购权
)


def paid_list_queries() -> tuple[ListQuery, ...]:
    """4 个日期 × market=stocks × active=true，加 2019-06-03 × active=false。

    均不传交易所与 type。
    """
    active = [ListQuery(d, None, None, True, market="stocks") for d in QUERY_DATES]
    return (*active, ListQuery(QUERY_DATES[-1], None, None, False, market="stocks"))


def snapshot_key(q: ListQuery) -> str:
    return f"{q.day.isoformat()}_{'active' if q.active else 'inactive'}"


def _blank(v: Any) -> bool:
    return v is None or v == ""


def _day(raw: Any) -> date | None:
    return None if _blank(raw) else date.fromisoformat(str(raw)[:10])


def seal_list_row(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    """清单一行的封存处理：list_date 晚于截止日的丢弃（返回 None）；delisted_utc 晚于截止日的
    写为空并标 delisted_after_2019；只保留白名单字段。"""
    listed = _day(raw.get("list_date"))
    if listed is not None and listed > EXPLORATION_END:
        return None
    delisted = _day(raw.get("delisted_utc"))
    after = delisted is not None and delisted > EXPLORATION_END
    if after:
        delisted = None
    assert_dates_not_sealed("Massive 清单行", listed, delisted)
    row: dict[str, Any] = {f: raw.get(f) for f in LIST_FIELDS}
    row["list_date"] = listed.isoformat() if listed else None
    row["delisted_utc"] = delisted.isoformat() if delisted else None
    row["delisted_after_2019"] = after
    return row


def suffix_type(ticker: str) -> str | None:
    for pattern, t in SUFFIX_TYPES:
        if re.search(pattern, ticker):
            return t
    return None


@dataclass(frozen=True, slots=True)
class Snapshot:
    key: str
    day: date
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class Filled:
    value: str | None
    rule: str  # original / rule1 / rule2 / rule3 / none
    conflict: tuple[str, ...]  # 其他快照中的不同取值（多于一个时为矛盾）


def _index(snaps: Sequence[Snapshot], field: str) -> tuple[dict, dict]:
    """(figi → [(快照, 取值)]) 与 ((cik, ticker) → [(快照, 取值)])，只收非空取值。"""
    by_figi: dict[str, list[tuple[Snapshot, str]]] = {}
    by_cik: dict[tuple[str, str], list[tuple[Snapshot, str]]] = {}
    for s in snaps:
        for r in s.rows:
            v = r.get(field)
            if _blank(v):
                continue
            if not _blank(r.get("composite_figi")):
                by_figi.setdefault(str(r["composite_figi"]), []).append((s, str(v)))
            if not _blank(r.get("cik")):
                by_cik.setdefault((str(r["cik"]), str(r["ticker"])), []).append((s, str(v)))
    return by_figi, by_cik


def _pick(own: Snapshot, cands: Sequence[tuple[Snapshot, str]]) -> tuple[str, tuple[str, ...]]:
    others = [(s, v) for s, v in cands if s.key != own.key]
    s, v = min(others, key=lambda sv: (abs((sv[0].day - own.day).days), -sv[0].day.toordinal()))
    return v, tuple(sorted({v for _, v in others}))


def fill_field(snaps: Sequence[Snapshot], field: str) -> dict[tuple[str, int], Filled]:
    """对每个快照的每一行补 field（primary_exchange 或 type）。键为（快照键, 行号）。"""
    by_figi, by_cik = _index(snaps, field)
    out: dict[tuple[str, int], Filled] = {}
    for s in snaps:
        for i, r in enumerate(s.rows):
            v = r.get(field)
            if not _blank(v):
                out[(s.key, i)] = Filled(str(v), "original", ())
                continue
            figi, cik, ticker = r.get("composite_figi"), r.get("cik"), str(r.get("ticker"))
            if not _blank(figi):
                cands = [c for c in by_figi.get(str(figi), []) if c[0].key != s.key]
                if cands:
                    val, seen = _pick(s, cands)
                    out[(s.key, i)] = Filled(val, "rule1", seen)
                    continue
            elif not _blank(cik):
                cands = [c for c in by_cik.get((str(cik), ticker), []) if c[0].key != s.key]
                if cands:
                    val, seen = _pick(s, cands)
                    out[(s.key, i)] = Filled(val, "rule2", seen)
                    continue
            guess = suffix_type(ticker) if field == "type" else None
            out[(s.key, i)] = Filled(guess, "rule3", ()) if guess else Filled(None, "none", ())
    return out


def _count(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def _example(s: Snapshot, r: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "date": s.day.isoformat(),
        "snapshot": s.key,
        "ticker": r.get("ticker"),
        "cik": r.get("cik") or "",
        "composite_figi": r.get("composite_figi") or "",
    }


def label_report(
    snaps: Sequence[Snapshot],
    exchange: Mapping[tuple[str, int], Filled],
    types: Mapping[tuple[str, int], Filled],
) -> dict[str, Any]:
    """按快照分组的补标签报告（第 3 节"报告"各项）。"""
    out: dict[str, Any] = {}
    for s in snaps:
        part: dict[str, Any] = {"rows": len(s.rows)}
        for name, filled, rules in (
            ("primary_exchange", exchange, ("rule1", "rule2")),
            ("type", types, ("rule1", "rule2", "rule3")),
        ):
            fs = [(i, filled[(s.key, i)]) for i in range(len(s.rows))]
            empty = [(i, f) for i, f in fs if f.rule != "original"]
            done = [(i, f) for i, f in empty if f.rule != "none"]
            left = [i for i, f in empty if f.rule == "none"]
            conflicts = [(i, f) for i, f in done if len(f.conflict) > 1]
            part[name] = {
                "empty": len(empty),
                "filled_by": {r: sum(1 for _, f in done if f.rule == r) for r in rules},
                "filled_values": _count(str(f.value) for _, f in done),
                "unfilled": len(left),
                "conflicts": len(conflicts),
                "conflict_examples": [
                    {**_example(s, s.rows[i]), "rule": f.rule, "values": list(f.conflict)}
                    for i, f in conflicts[:CONFLICT_EXAMPLES]
                ],
                "unfilled_examples": [_example(s, s.rows[i]) for i in left[:UNFILLED_EXAMPLES]],
            }
        out[s.key] = part
    return out


def exchange_share(report: Mapping[str, Any], key: str, value: str = "XNAS") -> dict[str, Any]:
    """某快照中规则 1、2 补出的交易所里，value 所占比例（第 3 节 2010 年一项）。"""
    vals = report[key]["primary_exchange"]["filled_values"]
    total = sum(vals.values())
    return {
        "snapshot": key,
        "filled": total,
        value: vals.get(value, 0),
        "share": round(vals.get(value, 0) / total, 4) if total else None,
    }


def unfilled_rows(
    snaps: Sequence[Snapshot], filled: Mapping[tuple[str, int], Filled]
) -> list[tuple[Snapshot, Mapping[str, Any]]]:
    return [
        (s, r) for s in snaps for i, r in enumerate(s.rows) if filled[(s.key, i)].rule == "none"
    ]


def pick_samples(
    no_exchange: Sequence[tuple[Snapshot, Mapping[str, Any]]],
    no_type: Sequence[tuple[Snapshot, Mapping[str, Any]]],
    n: int = SAMPLES_PER_FIELD,
) -> list[dict[str, Any]]:
    """抽样：主交易所补不上的、type 补不上的各取 n 个。

    在按（日期, 代码）排序的列表中等距取，两组不重复。
    """
    picked: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group, rows in (("primary_exchange", no_exchange), ("type", no_type)):
        pool = sorted(
            ((s, r) for s, r in rows if (s.key, str(r.get("ticker"))) not in seen),
            key=lambda sr: (sr[0].day, str(sr[1].get("ticker")), sr[0].key),
        )
        if not pool:
            continue
        step = max(len(pool) / n, 1.0)
        idx = sorted({int(k * step) for k in range(min(n, len(pool)))})
        for k in idx:
            s, r = pool[k]
            seen.add((s.key, str(r.get("ticker"))))
            picked.append({"group": group, **_example(s, r)})
    return picked


def label_display(v: Any) -> str:
    return EMPTY if _blank(v) else str(v)
