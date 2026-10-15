"""Massive 数据源试验（T08d-2a 第 2 节）：查询计划、已知案例、封存处理与汇总。纯函数。

封存处理（第 2.3 节，照 probe-tickers）：
- 请求参数 date 不得晚于 2019-12-31，违反即拒绝执行（check_query_date）；
- list_date 晚于 2019-12-31 的行丢弃，不计数；
- delisted_utc 晚于 2019-12-31 的写为空，另记 delisted_after_2019 = 是，不保留真实日期；
- 只保留白名单字段（ADMITTED_FIELDS 加两个日期），name、last_updated_utc、description 等一律不保存。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from stock_risk.domain.seal import EXPLORATION_END, SealViolation, assert_dates_not_sealed

QUERY_DATES = (date(2010, 1, 4), date(2012, 6, 1), date(2015, 6, 1), date(2019, 6, 3))
EXCHANGES = ("XNYS", "XNAS")
TYPES = ("CS", "ADRC")
ACTIVE_FLAGS = (True, False)
ANY = "any"  # 查询键中表示"不传该参数"

# 原样保留的字段；日期字段（list_date、delisted_utc）另行封存处理
ADMITTED_FIELDS = (
    "ticker",
    "market",
    "locale",
    "primary_exchange",
    "type",
    "active",
    "cik",
    "composite_figi",
    "share_class_figi",
)


class MassiveFormatError(ValueError):
    """响应格式与约定不符（例如日期无法解析）。"""


def check_query_date(d: date) -> None:
    """请求参数 date 不得晚于探索期截止日（用 domain/seal.py 的常量）。"""
    if d > EXPLORATION_END:
        raise SealViolation(f"Massive 查询日期 {d} 晚于 {EXPLORATION_END}，拒绝执行")


@dataclass(frozen=True, slots=True)
class ListQuery:
    """All Tickers 的一种查询组合。

    exchange、type、market 为 None 时请求中不传该参数（第 2.4 节）。
    """

    day: date
    exchange: str | None
    type: str | None
    active: bool
    market: str | None = None

    def __post_init__(self) -> None:
        check_query_date(self.day)

    @property
    def key(self) -> str:
        flag = "active" if self.active else "inactive"
        key = f"{self.day.isoformat()}_{self.exchange or ANY}_{self.type or ANY}_{flag}"
        return f"{key}_{self.market}" if self.market else key


def list_queries() -> tuple[ListQuery, ...]:
    """第 2.1 节：4 个日期 × 2 个交易所 × 2 种类型 × active 真 / 假，共 32 种。"""
    return tuple(
        ListQuery(d, x, t, a)
        for d in QUERY_DATES
        for x in EXCHANGES
        for t in TYPES
        for a in ACTIVE_FLAGS
    )


@dataclass(frozen=True, slots=True)
class Case:
    """第 2.2 节的已知案例（产品经理提供）。note 只供对照，不作为测试期望。"""

    category: str
    ticker: str
    day: date
    note: str

    def __post_init__(self) -> None:
        check_query_date(self.day)

    @property
    def key(self) -> str:
        return f"{self.ticker}_{self.day.isoformat()}"


def _c(category: str, ticker: str, day: str, note: str) -> Case:
    return Case(category, ticker, date.fromisoformat(day), note)


CASES: tuple[Case, ...] = (
    _c("破产后转场外", "LEH", "2008-06-02", "雷曼兄弟，NYSE，2008 年破产"),
    _c(
        "破产后转场外",
        "GM",
        "2009-03-02",
        "旧通用汽车，NYSE，2009 年破产；新通用汽车 2010 年用同一代码上市",
    ),
    _c("破产后转场外", "EK", "2011-06-01", "柯达，NYSE，2012 年初摘牌"),
    _c("破产后转场外", "RSH", "2014-06-02", "RadioShack，NYSE，2015 年摘牌"),
    _c("破产后转场外", "SHLD", "2015-06-01", "西尔斯控股，Nasdaq，2018 年摘牌"),
    _c("破产后转场外", "WFT", "2018-06-01", "Weatherford，NYSE，2019 年摘牌"),
    _c("破产后转场外", "BTU", "2015-06-01", "皮博迪能源，NYSE，2016 年破产"),
    _c("破产后转场外", "SUNE", "2015-06-01", "SunEdison，NYSE，2016 年破产"),
    _c("破产后转场外", "DNDN", "2014-01-02", "Dendreon，Nasdaq，2014 年破产"),
    _c("破产后转场外", "GTAT", "2014-06-02", "GT Advanced，Nasdaq，2014 年破产"),
    _c("被收购", "HNZ", "2013-01-02", "亨氏，NYSE，2013 年被收购"),
    _c(
        "被收购",
        "DELL",
        "2013-01-02",
        "戴尔，Nasdaq，2013 年私有化；DELL 代码 2018 年起另属 Dell Technologies（NYSE）",
    ),
    _c("被收购", "MON", "2017-06-01", "孟山都，NYSE，2018 年被收购"),
    _c("被收购", "LLTC", "2016-06-01", "凌力尔特，Nasdaq，2017 年被收购"),
    _c("代码复用", "GM", "2011-06-01", "新通用汽车（与旧 GM 对照 FIGI 与 CIK）"),
    _c("代码复用", "DELL", "2019-06-03", "Dell Technologies（与 2013 年的 DELL 对照）"),
    _c("代码复用", "ADT", "2013-06-03", "2012 年分拆的 ADT"),
    _c("代码复用", "ADT", "2019-06-03", "2018 年重新上市的 ADT"),
    _c("改名", "FB", "2015-06-01", "Facebook，Nasdaq。只核对是否查得到，不记录名称字段"),
    _c("换代码", "VRX", "2016-06-01", "Valeant，NYSE（2018 年改为 BHC）"),
    _c("对照", "MSFT", "2012-06-01", "正常情形"),
)


@dataclass(frozen=True, slots=True)
class SealedTicker:
    """封存处理后的一条代码记录。日期都不晚于 2019-12-31。"""

    fields: tuple[tuple[str, Any], ...]  # ADMITTED_FIELDS 中响应给出的字段
    list_date: date | None
    delisted: date | None
    delisted_after_2019: bool

    def __post_init__(self) -> None:
        assert_dates_not_sealed("Massive 代码记录", self.list_date, self.delisted)

    def get(self, name: str) -> Any:
        return dict(self.fields).get(name)

    def as_row(self) -> dict[str, Any]:
        out: dict[str, Any] = {f: self.get(f) for f in ADMITTED_FIELDS}
        out["list_date"] = self.list_date.isoformat() if self.list_date else None
        out["delisted"] = self.delisted.isoformat() if self.delisted else None
        out["delisted_after_2019"] = self.delisted_after_2019
        return out


def _day(raw: Any, name: str) -> date | None:
    if raw is None or raw == "":
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError as e:
        raise MassiveFormatError(f"{name} 不是日期") from e


def seal_ticker(raw: Mapping[str, Any]) -> SealedTicker | None:
    """封存处理一条响应记录（只在内存中）。list_date 晚于截止日时返回 None（丢弃，不计数）。"""
    list_date = _day(raw.get("list_date"), "list_date")
    if list_date is not None and list_date > EXPLORATION_END:
        return None
    delisted = _day(raw.get("delisted_utc"), "delisted_utc")
    after = delisted is not None and delisted > EXPLORATION_END
    return SealedTicker(
        fields=tuple((f, raw[f]) for f in ADMITTED_FIELDS if f in raw),
        list_date=list_date,
        delisted=None if after else delisted,
        delisted_after_2019=after,
    )


def seal_all(raws: Iterable[Mapping[str, Any]]) -> list[SealedTicker]:
    return [s for s in (seal_ticker(r) for r in raws) if s is not None]


def exchange_changes(
    by_date: Mapping[date, Sequence[SealedTicker]], examples: int = 20
) -> dict[str, Any]:
    """同一 composite_figi 在不同查询日期的主交易所是否变化（第 2.1 节）。"""
    seen: dict[str, dict[str, set[str]]] = {}
    tickers: dict[str, set[str]] = {}
    for d in sorted(by_date):
        for s in by_date[d]:
            figi, exch = s.get("composite_figi"), s.get("primary_exchange")
            if not figi or not exch:
                continue
            seen.setdefault(figi, {}).setdefault(d.isoformat(), set()).add(str(exch))
            tickers.setdefault(figi, set()).add(str(s.get("ticker")))
    multi = {f: v for f, v in seen.items() if len(v) > 1}
    changed = {f: v for f, v in multi.items() if len({e for es in v.values() for e in es}) > 1}
    return {
        "figi_seen_on_multiple_dates": len(multi),
        "figi_with_exchange_change": len(changed),
        "examples": [
            {
                "composite_figi": f,
                "tickers": sorted(tickers[f]),
                "primary_exchange_by_date": {d: sorted(es) for d, es in sorted(v.items())},
            }
            for f, v in sorted(changed.items())[:examples]
        ],
    }
