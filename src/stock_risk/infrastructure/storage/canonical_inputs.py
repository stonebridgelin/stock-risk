"""规范复权构建的输入文件读取（T11-D0a 第 3.2 节）。只读，不改任何文件。

- Massive 日线 aggs/<T>_raw.csv、拆股 actions/<T>_splits.csv、分红 actions/<T>_dividends.csv，
  列名按 massive_paid_files 的 BAR_COLUMNS、SPLIT_COLUMNS、DIVIDEND_COLUMNS；
  读原文，不调用 load_actions（它产出 float）。
- 交易日历：一列 date 的 CSV。
- 封存期的行（日期晚于 2019-12-31）：先判日期，只计数并跳过，同行的数值不解析；
  这只适用于本构建入口，现有正式读取入口遇到封存行即拒绝的行为不变（T11R-04）。
- 数据行号从表头之后的 1 起算，封存行也占行号。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import date
from pathlib import Path

from stock_risk.application.build_canonical import CanonicalBuildError, TextInput
from stock_risk.domain.adjustment.inputs import (
    DividendInput,
    SplitInput,
    dividend_from_text,
    split_from_text,
)
from stock_risk.domain.raw_text import RawBarText
from stock_risk.domain.seal import EXPLORATION_END
from stock_risk.infrastructure.storage.massive_paid_files import (
    BAR_COLUMNS as MASSIVE_BAR_COLUMNS,
)
from stock_risk.infrastructure.storage.massive_paid_files import (
    DIVIDEND_COLUMNS,
    SPLIT_COLUMNS,
)


class InputFileError(ValueError):
    """输入文件缺列或格式不对。"""


def _read(path: Path, columns: list[str]) -> tuple[list[dict[str, str]], str | None]:
    if not path.exists():
        return [], None
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    missing = [c for c in columns if c not in (reader.fieldnames or [])]
    if missing:
        raise InputFileError(f"{path.name} 缺少列 {missing}")
    return list(reader), hashlib.sha256(data).hexdigest()


def _day(text: str, path: Path, row: int) -> date:
    try:
        return date.fromisoformat(text.strip())
    except ValueError as e:
        raise InputFileError(f"{path.name} 第 {row} 行日期不对: {text!r}") from e


def read_massive_bars(root: Path, ticker: str) -> TextInput[RawBarText]:
    """aggs/<T>_raw.csv（adjusted = false）。

    复权价列不存在，按原始价文本填入（复权计算不读它们）。"""
    path = root / "aggs" / f"{ticker}_raw.csv"
    rows, sha = _read(path, MASSIVE_BAR_COLUMNS)
    out: list[RawBarText] = []
    sealed = 0
    for i, r in enumerate(rows, start=1):
        d = _day(r["date"], path, i)
        if d > EXPLORATION_END:
            sealed += 1
            continue
        o, h, lo, c = r["open"], r["high"], r["low"], r["close"]
        out.append(RawBarText(d, o, h, lo, c, r["volume"], o, h, lo, c))
    return TextInput(tuple(out), sealed, sha, f"massive:aggs/{ticker}_raw.csv")


def read_massive_splits(root: Path, ticker: str) -> TextInput[SplitInput]:
    rel = f"actions/{ticker}_splits.csv"
    path = root / rel
    rows, sha = _read(path, SPLIT_COLUMNS)
    source = f"massive:{rel}"
    out: list[SplitInput] = []
    sealed = 0
    for i, r in enumerate(rows, start=1):
        d = _day(r["execution_date"], path, i)
        if d > EXPLORATION_END:
            sealed += 1
            continue
        out.append(split_from_text(d, r["split_from"], r["split_to"], source, i, sha or ""))
    return TextInput(tuple(out), sealed, sha, source)


def read_massive_dividends(root: Path, ticker: str) -> TextInput[DividendInput]:
    """现有分红文件没有币种列与单位列：currency 为 None（裁决 A-3），单位固定为 PER_SHARE。"""
    rel = f"actions/{ticker}_dividends.csv"
    path = root / rel
    rows, sha = _read(path, DIVIDEND_COLUMNS)
    source = f"massive:{rel}"
    out: list[DividendInput] = []
    sealed = 0
    for i, r in enumerate(rows, start=1):
        d = _day(r["ex_dividend_date"], path, i)
        if d > EXPLORATION_END:
            sealed += 1
            continue
        decl = r["declaration_date"].strip()
        out.append(
            dividend_from_text(
                d,
                r["cash_amount"],
                r["dividend_type"],
                decl,
                r["masked_after_2019"],
                source,
                i,
                sha or "",
                currency=r.get("currency") or None,
            )
        )
    return TextInput(tuple(out), sealed, sha, source)


def read_calendar(path: Path) -> TextInput[date]:
    """交易日历文件：一列 date。"""
    rows, sha = _read(path, ["date"])
    days: list[date] = []
    sealed = 0
    for i, r in enumerate(rows, start=1):
        d = _day(r["date"], path, i)
        if d > EXPLORATION_END:
            sealed += 1
            continue
        days.append(d)
    if sha is None:
        raise InputFileError(f"交易日历文件不存在: {path}")
    return TextInput(tuple(days), sealed, sha, f"calendar:{path.name}")


INPUTS_STATUS_FILE = "INPUTS_STATUS.json"


class MassiveCanonicalSource:
    """实现 build_canonical.CanonicalInputSource：从冻结的输入快照读原文
    （T12-D0b 第 2.6 节第 1 项）。

    快照中必须有 INPUTS_STATUS.json，格式：
        {"files": {"aggs/<T>_raw.csv": {"status": "ok", "complete": true, "sha256": "...", ...},
                   "actions/<T>_splits.csv": {...}, "actions/<T>_dividends.csv": {...}}}
    任何一个文件（原始日线、拆股、分红）没有登记、不是"成功且完整"、文件缺失或哈希不符，
    都抛 CanonicalBuildError 停止构建，不得把缺失读成空。
    合法的空行动（成功、完整、零条，文件只有表头）正常。
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        path = root / INPUTS_STATUS_FILE
        if not path.exists():
            raise CanonicalBuildError(f"输入快照中没有 {INPUTS_STATUS_FILE}：{root}")
        data = path.read_bytes()
        self._status_sha256 = hashlib.sha256(data).hexdigest()
        parsed = json.loads(data.decode("utf-8"))
        files = parsed.get("files") if isinstance(parsed, dict) else None
        if not isinstance(files, dict):
            raise CanonicalBuildError(f"{INPUTS_STATUS_FILE} 格式不对：缺少 files")
        self._files: dict[str, dict[str, object]] = files

    def _checked[T](self, rel: str, got: TextInput[T]) -> TextInput[T]:
        entry = self._files.get(rel)
        if entry is None:
            raise CanonicalBuildError(f"{rel}: {INPUTS_STATUS_FILE} 中没有登记")
        if entry.get("status") != "ok" or entry.get("complete") is not True:
            raise CanonicalBuildError(f"{rel}: 请求不是成功且完整（{entry.get('status')}）")
        if got.sha256 is None:
            raise CanonicalBuildError(f"{rel}: 文件缺失")
        if got.sha256 != entry.get("sha256"):
            raise CanonicalBuildError(f"{rel}: 文件哈希与 {INPUTS_STATUS_FILE} 不符")
        return got

    def bars(self, ticker: str) -> TextInput[RawBarText]:
        return self._checked(f"aggs/{ticker}_raw.csv", read_massive_bars(self._root, ticker))

    def splits(self, ticker: str) -> TextInput[SplitInput]:
        return self._checked(
            f"actions/{ticker}_splits.csv", read_massive_splits(self._root, ticker)
        )

    def dividends(self, ticker: str) -> TextInput[DividendInput]:
        rel = f"actions/{ticker}_dividends.csv"
        return self._checked(rel, read_massive_dividends(self._root, ticker))

    def status_sha256(self) -> str | None:
        return self._status_sha256
