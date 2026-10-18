"""读取分段与上市期间配置（实现 SegmentProvider、ListingPeriodProvider）。

config/symbol_segments.csv：symbol,source_ticker,from,to,note
    同一 symbol 的各行按文件顺序组成分段；from / to 为空表示该端不设边界。
config/listing_periods.csv：symbol,listed_from,listed_until,venue,source_url,verified_on,note
    只登记需要截断的代码；日期为空表示该端不设边界（找不到可靠来源时也留空，并在 note 说明）。
config/price_corrections.csv：symbol,from,to,method,ruling,evidence[,d0a_status]
    复权价更正（I11）；from / to 必填，method 目前只有 use_raw，未知方法报错。
以 # 开头的行为注释。文件不存在视为没有配置。
"""

from __future__ import annotations

import csv
import hashlib
from datetime import date
from pathlib import Path

from stock_risk.domain.adjustment.inputs import (
    Verification,
    check_verification_ids,
    parse_verification,
)
from stock_risk.domain.corrections import (
    CorrectionError,
    PriceCorrection,
    RawPriceCorrection,
    parse_method,
    parse_raw_price_correction,
)
from stock_risk.domain.listing import ListingPeriod
from stock_risk.domain.segments import SymbolSegment, validate_segments


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(lines)]


def _day(text: str) -> date | None:
    return date.fromisoformat(text) if text else None


class CsvSegmentProvider:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[str, tuple[SymbolSegment, ...]]:
        grouped: dict[str, list[SymbolSegment]] = {}
        for r in _rows(self._path):
            seg = SymbolSegment(
                symbol=r["symbol"].upper(),
                source_ticker=r["source_ticker"].upper(),
                start=_day(r["from"]),
                end=_day(r["to"]),
                note=r.get("note", ""),
            )
            grouped.setdefault(seg.symbol, []).append(seg)
        return {s: validate_segments(segs) for s, segs in grouped.items()}


class CsvPriceCorrectionProvider:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[str, tuple[PriceCorrection, ...]]:
        grouped: dict[str, list[PriceCorrection]] = {}
        for r in _rows(self._path):
            start, end = _day(r["from"]), _day(r["to"])
            if start is None or end is None:
                raise CorrectionError(f"{r['symbol']}: 更正的 from / to 不能为空")
            c = PriceCorrection(
                symbol=r["symbol"].upper(),
                start=start,
                end=end,
                method=parse_method(r["method"]),
                ruling=r.get("ruling", ""),
                evidence=r.get("evidence", ""),
                d0a_status=r.get("d0a_status") or None,
            )
            grouped.setdefault(c.symbol, []).append(c)
        return {s: tuple(cs) for s, cs in grouped.items()}


class CsvListingPeriodProvider:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[str, ListingPeriod]:
        out: dict[str, ListingPeriod] = {}
        for r in _rows(self._path):
            p = ListingPeriod(
                symbol=r["symbol"].upper(),
                listed_from=_day(r["listed_from"]),
                listed_until=_day(r["listed_until"]),
                venue=r.get("venue", ""),
                source_url=r.get("source_url", ""),
                verified_on=_day(r.get("verified_on", "")),
                note=r.get("note", ""),
            )
            if p.symbol in out:
                raise ValueError(f"上市期间配置中代码重复: {p.symbol}")
            out[p.symbol] = p
        return out


# ---- T11-D0a：核实登记与原始价更正（第 5、6 节） ----


def file_sha256(path: Path) -> str | None:
    """配置文件内容的 SHA-256；文件不存在为 None。写入数据版本。"""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


class CsvAdjustmentVerificationProvider:
    """config/adjustment_verifications.csv：id,target_type,target,source_sha256,kind,value,ruling,evidence。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> tuple[Verification, ...]:
        out = tuple(parse_verification(r) for r in _rows(self._path))
        check_verification_ids(out)
        return out

    def sha256(self) -> str | None:
        return file_sha256(self._path)


class CsvRawPriceCorrectionProvider:
    """config/raw_price_corrections.csv：id,security,day,field,value,ruling,evidence,version。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> tuple[RawPriceCorrection, ...]:
        out = tuple(parse_raw_price_correction(r) for r in _rows(self._path))
        ids = [c.id for c in out]
        if len(set(ids)) != len(ids):
            raise CorrectionError("原始价更正编号重复")
        return out

    def sha256(self) -> str | None:
        return file_sha256(self._path)


# ---- T12-D0b：Massive 源代码映射（PM-D0b-01） ----


class CsvMassiveSourceTickerProvider:
    """config/massive_source_tickers.csv：symbol,source_ticker,from,to,ruling,note。

    只由 fetch-massive-d0b 与 build-canonical 读取；旧数据（Tiingo）路径不读。每行必须有裁定编号。
    文件不存在视为没有登记。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> dict[str, tuple[SymbolSegment, ...]]:
        return {s: segs for s, (segs, _) in self.load_with_rulings().items()}

    def load_with_rulings(self) -> dict[str, tuple[tuple[SymbolSegment, ...], str]]:
        grouped: dict[str, list[SymbolSegment]] = {}
        rulings: dict[str, set[str]] = {}
        for r in _rows(self._path):
            ruling = r.get("ruling", "")
            if not ruling:
                raise ValueError(f"{self._path.name}: {r.get('symbol')} 的映射没有裁定编号")
            seg = SymbolSegment(
                symbol=r["symbol"].upper(),
                source_ticker=r["source_ticker"].upper(),
                start=_day(r["from"]),
                end=_day(r["to"]),
                note=r.get("note", ""),
            )
            grouped.setdefault(seg.symbol, []).append(seg)
            rulings.setdefault(seg.symbol, set()).add(ruling)
        return {
            s: (validate_segments(segs), "、".join(sorted(rulings[s])))
            for s, segs in grouped.items()
        }

    def sha256(self) -> str | None:
        return file_sha256(self._path)
