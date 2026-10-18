"""构建规范数据版本（T11-D0a 第 7 节）：组装输入，调用领域复权函数，交给存储写出。

- 数据版本号 = "d0a-" + _dataset.json 规范 JSON 的 SHA-256 前 12 位；_dataset.json 不含版本号字段。
- 凡会改变输出的输入，都写入 _dataset.json：规格版本、裁决文件、参数（含 G 集合与 ATR 窗口）、
  交易日历内容哈希与起止日、证券身份与拼接、各输入文件哈希与封存行计数、配置文件哈希、代码提交号。
- 输出目录已存在的，拒绝写入（不覆盖、不合并）。
- 本卡只在测试的临时目录上运行，不生成正式数据。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from stock_risk.domain.adjustment import AdjustmentParams, compute_canonical
from stock_risk.domain.adjustment.inputs import DividendInput, SplitInput, Verification
from stock_risk.domain.adjustment.model import (
    ASSUMPTION_CODES,
    AdjustmentResult,
    Eligibility,
)
from stock_risk.domain.adjustment.params import RULING_FILE, UNREGISTERED_GAP_BREAKS
from stock_risk.domain.corrections import RawPriceCorrection
from stock_risk.domain.evaluation.params import ATR_PERIOD
from stock_risk.domain.lifecycle.params import GRID_DIMENSIONS, REFERENCE_NUMERIC
from stock_risk.domain.raw_text import RawBarText
from stock_risk.domain.segments import SymbolSegment

VERSION_PREFIX = "d0a-"


class CanonicalBuildError(RuntimeError):
    """构建停止：输出目录已存在、输入缺失等。"""


@dataclass(frozen=True, slots=True)
class TextInput[T]:
    """一个输入文件读到的行：rows 只含探索期的行；sealed_rows 为只计数并跳过的封存期行数。"""

    rows: tuple[T, ...]
    sealed_rows: int
    sha256: str | None  # 文件不存在为 None
    source: str


class CanonicalInputSource(Protocol):
    def bars(self, ticker: str) -> TextInput[RawBarText]: ...

    def splits(self, ticker: str) -> TextInput[SplitInput]: ...

    def dividends(self, ticker: str) -> TextInput[DividendInput]: ...


@dataclass(frozen=True, slots=True)
class CanonicalConfig:
    calendar: TextInput[date]
    segments: Mapping[str, tuple[SymbolSegment, ...]]
    verifications: tuple[Verification, ...]
    verifications_sha256: str | None
    raw_corrections: tuple[RawPriceCorrection, ...]
    raw_corrections_sha256: str | None
    price_corrections_sha256: str | None


@dataclass(frozen=True, slots=True)
class CanonicalOutput:
    version: str
    dataset: dict[str, Any]
    dataset_text: str  # 规范 JSON（不含末尾换行）
    results: tuple[AdjustmentResult, ...]
    unrelated_verification_ids: tuple[str, ...]
    meta: dict[str, Any]


class CanonicalSink(Protocol):
    def exists(self, version: str) -> bool: ...

    def write(self, output: CanonicalOutput) -> Path: ...


@dataclass(frozen=True, slots=True)
class BuildSummary:
    version: str
    out_dir: Path
    total: int
    usable: int
    not_usable_by_reason: dict[str, int]
    assumption_counts: dict[str, int]
    terminal_pending: int
    unrelated_verification_ids: tuple[str, ...]
    sealed_rows_skipped: int
    results: tuple[AdjustmentResult, ...] = field(repr=False, default=())


def atr_windows() -> tuple[int, ...]:
    """ATR 预检的窗口集合（第 4.7 节）：参数网格中 n 的全部取值、当前配置的 n、评价的 ATR_PERIOD。

    尚无冻结的候选配置；"当前配置的 n"取参考组合 REFERENCE_NUMERIC 的 n（I7），它也在网格之内。"""
    grid_n = dict(GRID_DIMENSIONS)["n"]
    values = {int(v) for v in grid_n} | {int(REFERENCE_NUMERIC["n"]), ATR_PERIOD}  # type: ignore[call-overload]
    return tuple(sorted(values))


def default_params(gap_breaks: tuple[int, ...] = UNREGISTERED_GAP_BREAKS) -> AdjustmentParams:
    """G 尚未登记（SR-M1）：预检对 {10, 20} 两个取值都做。"""
    return AdjustmentParams(atr_windows=atr_windows(), gap_breaks=gap_breaks)


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dataset_version(dataset_text: str) -> str:
    return VERSION_PREFIX + hashlib.sha256(dataset_text.encode("utf-8")).hexdigest()[:12]


def _segment_record(s: SymbolSegment) -> dict[str, Any]:
    return {
        "symbol": s.symbol,
        "source_ticker": s.source_ticker,
        "from": s.start.isoformat() if s.start else None,
        "to": s.end.isoformat() if s.end else None,
    }


def _calendar_record(cal: TextInput[date]) -> dict[str, Any]:
    text = "\n".join(d.isoformat() for d in cal.rows)
    return {
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "first": cal.rows[0].isoformat() if cal.rows else None,
        "last": cal.rows[-1].isoformat() if cal.rows else None,
        "days": len(cal.rows),
        "file_sha256": cal.sha256,
        "sealed_rows_skipped": cal.sealed_rows,
    }


@dataclass(frozen=True, slots=True)
class _SecurityInputs:
    bars: tuple[RawBarText, ...]
    splits: tuple[SplitInput, ...]
    dividends: tuple[DividendInput, ...]
    record: dict[str, Any]
    sealed: int


def _gather(
    symbol: str, segments: tuple[SymbolSegment, ...], source: CanonicalInputSource
) -> _SecurityInputs:
    bars: list[RawBarText] = []
    splits: list[SplitInput] = []
    dividends: list[DividendInput] = []
    parts: list[dict[str, Any]] = []
    sealed = 0
    for seg in segments:
        b, s, d = (
            source.bars(seg.source_ticker),
            source.splits(seg.source_ticker),
            (source.dividends(seg.source_ticker)),
        )
        if b.sha256 is None:
            raise CanonicalBuildError(f"{symbol}: 缺少 {seg.source_ticker} 的日线文件")
        bars += [r for r in b.rows if seg.contains(r.day)]
        splits += [r for r in s.rows if seg.contains(r.ex_date)]
        dividends += [r for r in d.rows if seg.contains(r.ex_date)]
        sealed += b.sealed_rows + s.sealed_rows + d.sealed_rows
        parts.append(
            {
                **_segment_record(seg),
                "bars": {
                    "source": b.source,
                    "sha256": b.sha256,
                    "sealed_rows_skipped": b.sealed_rows,
                },
                "splits": {
                    "source": s.source,
                    "sha256": s.sha256,
                    "sealed_rows_skipped": s.sealed_rows,
                },
                "dividends": {
                    "source": d.source,
                    "sha256": d.sha256,
                    "sealed_rows_skipped": d.sealed_rows,
                },
            }
        )
    bars.sort(key=lambda r: r.day)
    return _SecurityInputs(
        tuple(bars), tuple(splits), tuple(dividends), {"segments": parts}, sealed
    )


def build_canonical(
    symbols: Sequence[str],
    source: CanonicalInputSource,
    config: CanonicalConfig,
    sink: CanonicalSink,
    *,
    code_commit: str | None,
    params: AdjustmentParams | None = None,
    clock: Callable[[], datetime],
    args: Mapping[str, Any] | None = None,
) -> BuildSummary:
    started = clock()
    params = params or default_params()
    if not config.calendar.rows:
        raise CanonicalBuildError("交易日历为空")
    syms = sorted({s.upper() for s in symbols})
    if not syms:
        raise CanonicalBuildError("没有要构建的证券")
    gathered: dict[str, _SecurityInputs] = {}
    segment_rows: list[dict[str, Any]] = []
    for sym in syms:
        segs = config.segments.get(sym) or (SymbolSegment(sym, sym),)
        if sym in config.segments:
            segment_rows += [_segment_record(s) for s in segs]
        gathered[sym] = _gather(sym, segs, source)
    results: list[AdjustmentResult] = []
    for sym in syms:
        g = gathered[sym]
        results.append(
            compute_canonical(
                sym,
                g.bars,
                g.splits,
                g.dividends,
                config.calendar.rows,
                config.verifications,
                config.raw_corrections,
                params,
            )
        )
    securities = {
        r.security: {**gathered[r.security].record, "anchor": r.anchor.isoformat()} for r in results
    }
    segments_text = canonical_json(segment_rows)
    sealed_total = config.calendar.sealed_rows + sum(g.sealed for g in gathered.values())
    dataset: dict[str, Any] = {
        "spec_version": params.spec_version,
        "ruling_file": RULING_FILE,
        "params": params.as_record(),
        "calendar": _calendar_record(config.calendar),
        "segments": {
            "rows": segment_rows,
            "content_sha256": hashlib.sha256(segments_text.encode("utf-8")).hexdigest(),
        },
        "securities": securities,
        "config": {
            "adjustment_verifications_sha256": config.verifications_sha256,
            "raw_price_corrections_sha256": config.raw_corrections_sha256,
            "price_corrections_sha256": config.price_corrections_sha256,
        },
        "code_commit": code_commit,
        "sealed_rows_skipped": sealed_total,
    }
    text = canonical_json(dataset)
    version = dataset_version(text)
    if sink.exists(version):
        raise CanonicalBuildError(f"规范数据目录 {version} 已存在，拒绝写入（不覆盖、不合并）")
    # 所有证券都用到的登记才算"与本次构建无关"
    unrelated = set.intersection(*(set(r.unrelated_verification_ids) for r in results))
    used = set().union(*(set(r.verification_ids_used) for r in results))
    unrelated_ids = tuple(sorted(unrelated - used))
    meta = {
        "started_at": started.isoformat(),
        "args": dict(args or {}),
    }
    meta["finished_at"] = clock().isoformat()
    output = CanonicalOutput(version, dataset, text, tuple(results), unrelated_ids, meta)
    out_dir = sink.write(output)
    by_reason: dict[str, int] = {}
    counts: dict[str, int] = {str(c): 0 for c in sorted(ASSUMPTION_CODES)}
    for r in results:
        for code in r.reasons:
            by_reason[code] = by_reason.get(code, 0) + 1
        for a in r.assumptions:
            counts[a.code] = counts.get(a.code, 0) + 1
    return BuildSummary(
        version=version,
        out_dir=out_dir,
        total=len(results),
        usable=sum(1 for r in results if r.eligibility is Eligibility.USABLE),
        not_usable_by_reason=dict(sorted(by_reason.items())),
        assumption_counts=counts,
        terminal_pending=sum(len(r.terminal_pending) for r in results),
        unrelated_verification_ids=unrelated_ids,
        sealed_rows_skipped=sealed_total,
        results=tuple(results),
    )
