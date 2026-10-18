"""D0b P1 的工作文件（T12-D0b 第 2.1、2.2、2.5、2.7 节）：output/d0b/ 下的身份清单、缺口清单、
映射比较与日历文件，以及中转服务核验文件（verify/prices_<T>.csv）的只读摘要。

只读 data/massive_probe/paid/verify/ 下的核验文件
（只取日期、是否有中转值与 Tiingo 值，不读价格数值），
只写 output/d0b/。
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

from stock_risk.application.d0b_inventory import IdentityRow, MappingCompare, VerifyEvidence
from stock_risk.application.fetch_massive_d0b import RequestSpec

IDENTITY_COLUMNS = ["symbol", "segments", "listing", "range", "evidence"]
GAP_COLUMNS = ["symbol", "source_ticker", "category", "start", "end", "reason"]
COMPARE_COLUMNS = ["symbol", "identity", "builder", "same"]


def read_verify_evidence(verify_dir: Path, symbols: Iterable[str]) -> dict[str, VerifyEvidence]:
    """每只证券的 verify/prices_<T>.csv：中转、Tiingo 各有值的行数与起止日。

    文件不存在的不返回。"""
    out: dict[str, VerifyEvidence] = {}
    for sym in symbols:
        path = verify_dir / f"prices_{sym}.csv"
        if not path.exists():
            continue
        relay: list[date] = []
        tiingo: list[date] = []
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                d = date.fromisoformat(r["date"])
                if r.get("relay_close"):
                    relay.append(d)
                if r.get("tiingo_close"):
                    tiingo.append(d)
        out[sym] = VerifyEvidence(
            f"verify/prices_{sym}.csv",
            len(relay),
            min(relay) if relay else None,
            max(relay) if relay else None,
            len(tiingo),
            max(tiingo) if tiingo else None,
        )
    return out


def _write(path: Path, header: Sequence[str], rows: Iterable[Sequence[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def write_identity(path: Path, rows: Sequence[IdentityRow]) -> Path:
    return _write(path, IDENTITY_COLUMNS, ([r.record()[c] for c in IDENTITY_COLUMNS] for r in rows))


def read_identity(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {r["symbol"]: r for r in csv.DictReader(f)}


def write_compare(path: Path, rows: Sequence[MappingCompare]) -> Path:
    return _write(
        path,
        COMPARE_COLUMNS,
        ([r.symbol, r.identity, r.builder, str(r.same).lower()] for r in rows),
    )


def write_gaps(path: Path, specs: Sequence[RequestSpec], reasons: dict[str, str]) -> Path:
    return _write(
        path,
        GAP_COLUMNS,
        (
            [
                s.symbol,
                s.ticker,
                s.category,
                s.start.isoformat(),
                s.end.isoformat(),
                reasons[s.category],
            ]
            for s in specs
        ),
    )


def read_gaps(path: Path) -> list[RequestSpec]:
    with path.open(encoding="utf-8", newline="") as f:
        return [
            RequestSpec(
                r["symbol"],
                r["source_ticker"],
                r["category"],
                date.fromisoformat(r["start"]),
                date.fromisoformat(r["end"]),
            )
            for r in csv.DictReader(f)
        ]


def write_calendar(path: Path, days: Sequence[date]) -> str:
    """交易日历文件（一列 date）；返回文件的 SHA-256。"""
    _write(path, ["date"], ([d.isoformat()] for d in days))
    return hashlib.sha256(path.read_bytes()).hexdigest()
