"""D0b 补取的原文记录与文件（T12-D0b 第 2.3 节；G-D0b-01、DBr-01、DBr-03）。

- 数据路径：响应（以 client.parse_json_keep_numbers 解析，数字为原文字符串）
  → 本模块的原文记录 → 文件。
  全程不构造 domain.massive_paid 的 Bar、Split、Dividend，不调用 float()。
- 封存只按日期判断（domain.seal 的常量）：日线按日期、拆股按执行日、分红按除息日决定整行是否保留，
  越界行在写文件之前过滤并计数；保留的分红中，公告日、登记日、派息日晚于 2019-12-31 的写为空，
  字段名记入 masked_after_2019（与现有 seal_dividend 的规则相同）。
- 文件格式：日线、拆股与 massive_paid_files 相同；分红在其列之后加 currency：
  响应中有币种字段的写原值，确实没有的写空，并在账本中计数（"来源无币种字段"）。
- 账本 _ledger.json：请求身份 = (主机, 类别, 源代码, 请求起止日, adjusted)，每个请求记状态、页数、
  行数、封存过滤数、输出文件哈希；"完成" = 成功且分页取完。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from stock_risk.application.fetch_massive_d0b import ADJ, DIVIDENDS, RAW, SPLITS, SavedFile
from stock_risk.domain.seal import EXPLORATION_END
from stock_risk.infrastructure.storage.massive_paid_files import (
    BAR_COLUMNS,
    DIVIDEND_COLUMNS,
    SPLIT_COLUMNS,
)
from stock_risk.infrastructure.storage.massive_source_host import (
    check_source_host,
    write_host_csv,
    write_host_json,
)

D0B_DIVIDEND_COLUMNS = [*DIVIDEND_COLUMNS, "currency"]
DIVIDEND_OTHER_DATES = ("declaration_date", "record_date", "pay_date")
LEDGER_FILE = "_ledger.json"


class RawValueError(ValueError):
    """响应中出现了已解析为 float 的数值：必须在首次解码时保留原文（DBr-01）。"""


def text(v: Any) -> str:
    """原文 → 文件文本。数字应已是字符串；float 说明解析方式不对，报错。"""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        raise RawValueError(f"数值已被解析为 float（{v!r}），原文丢失")
    return str(v)


def _date(v: Any) -> date | None:
    if v is None or v == "":
        return None
    return date.fromisoformat(str(v)[:10])


def bar_day(t_ms: Any) -> date:
    """t 为窗口起点的 Unix 毫秒时间戳（美东零点），UTC 日期即交易日。t 为整数原文。"""
    return datetime.fromtimestamp(int(str(t_ms)) / 1000, tz=UTC).date()


@dataclass(frozen=True, slots=True)
class Records:
    rows: list[dict[str, str]]
    sealed_dropped: int
    masked: int = 0  # 分红：其他日期字段被遮蔽的行数
    currency_missing: int = 0  # 分红：响应中没有币种字段的行数


def bar_records(results: Sequence[Mapping[str, Any]]) -> Records:
    rows: list[dict[str, str]] = []
    dropped = 0
    for r in results:
        d = bar_day(r["t"])
        if d > EXPLORATION_END:
            dropped += 1
            continue
        rows.append(
            {
                "date": d.isoformat(),
                "open": text(r.get("o")),
                "high": text(r.get("h")),
                "low": text(r.get("l")),
                "close": text(r.get("c")),
                "volume": text(r.get("v")),
                "vwap": text(r.get("vw")),
                "trades": text(r.get("n")),
            }
        )
    rows.sort(key=lambda x: x["date"])
    return Records(rows, dropped)


def split_records(results: Sequence[Mapping[str, Any]]) -> Records:
    rows: list[dict[str, str]] = []
    dropped = 0
    for r in results:
        d = _date(r.get("execution_date"))
        if d is None or d > EXPLORATION_END:
            dropped += 1
            continue
        rows.append(
            {
                "execution_date": d.isoformat(),
                "split_from": text(r.get("split_from")),
                "split_to": text(r.get("split_to")),
            }
        )
    rows.sort(key=lambda x: x["execution_date"])
    return Records(rows, dropped)


def dividend_records(results: Sequence[Mapping[str, Any]]) -> Records:
    rows: list[dict[str, str]] = []
    dropped = masked_rows = missing = 0
    for r in results:
        d = _date(r.get("ex_dividend_date"))
        if d is None or d > EXPLORATION_END:
            dropped += 1
            continue
        row = {
            "ex_dividend_date": d.isoformat(),
            "cash_amount": text(r.get("cash_amount")),
            "dividend_type": text(r.get("dividend_type")),
        }
        masked: list[str] = []
        for f in DIVIDEND_OTHER_DATES:
            v = _date(r.get(f))
            if v is not None and v > EXPLORATION_END:
                masked.append(f)
                v = None
            row[f] = "" if v is None else v.isoformat()
        row["masked_after_2019"] = "|".join(masked)
        if "currency" in r:
            row["currency"] = text(r["currency"])
        else:
            row["currency"] = ""
            missing += 1
        masked_rows += bool(masked)
        rows.append(row)
    rows.sort(key=lambda x: x["ex_dividend_date"])
    return Records(rows, dropped, masked_rows, missing)


def relative_path(category: str, ticker: str) -> str:
    return {
        RAW: f"aggs/{ticker}_raw.csv",
        ADJ: f"vendor_adj/{ticker}_adj.csv",
        SPLITS: f"actions/{ticker}_splits.csv",
        DIVIDENDS: f"actions/{ticker}_dividends.csv",
    }[category]


COLUMNS = {
    RAW: BAR_COLUMNS,
    ADJ: BAR_COLUMNS,
    SPLITS: SPLIT_COLUMNS,
    DIVIDENDS: D0B_DIVIDEND_COLUMNS,
}
CONVERT = {RAW: bar_records, ADJ: bar_records, SPLITS: split_records, DIVIDENDS: dividend_records}


class MassiveD0bFiles:
    """补取命令的输出目录（data/massive_d0b/download/）与账本。

    目录中已有其他主机的记录时，构造时拒绝（massive_source_host）。"""

    def __init__(self, root: Path, source_host: str) -> None:
        check_source_host(root, source_host)
        self._root = root
        self._host = source_host

    @property
    def root(self) -> Path:
        return self._root

    def load_ledger(self) -> dict[str, Any]:
        p = self._root / LEDGER_FILE
        if not p.exists():
            return {"requests": {}}
        data = json.loads(p.read_text("utf-8"))
        data.pop("source_host", None)
        return data

    def save_ledger(self, ledger: Mapping[str, Any]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        write_host_json(self._root / LEDGER_FILE, dict(ledger), self._host)

    def save(self, category: str, ticker: str, results: Sequence[Mapping[str, Any]]) -> SavedFile:
        """响应结果 → 原文记录 → 文件；返回文件身份与计数。"""
        records = CONVERT[category](results)
        rel = relative_path(category, ticker)
        path = self._root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        write_host_csv(path, COLUMNS[category], records.rows, self._host)
        return SavedFile(
            rel,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            len(records.rows),
            records.sealed_dropped,
            records.masked,
            records.currency_missing,
        )

    def file_sha256(self, rel: str) -> str | None:
        p = self._root / rel
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
