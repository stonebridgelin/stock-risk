"""读取 data/raw 中的原始行情（实现 application.ports.RawDataReader）。

两种读法：
- load：审计用。晚于探索期截止日的行只计数，不解析数值、不返回；数值按 float 解析，不量化。
- load_text：统一取数入口用（I12）。只解析日期，数值保持文本，封存检查与量化由入口完成。

adj_volume（T11-D0a 第 2 节）：缺列或空单元格为 None（未知）；"0" 为真实的零；非法值报错。
目录中有规范数据的标志（_dataset.json 等）时，不按本类读取（规范模式见 canonical_reader）。
"""

from __future__ import annotations

import csv
import hashlib
from datetime import date
from pathlib import Path

from stock_risk.application.ports import RawFile, RawTextFile
from stock_risk.domain.audit import count_sealed_rows
from stock_risk.domain.market_data import ActionType, CorporateAction, DailyBar, PriceHistory
from stock_risk.domain.raw_text import RawActionText, RawBarText, parse_adj_volume
from stock_risk.domain.seal import EXPLORATION_END

MANIFEST_FILE = "MANIFEST.sha256"
MERGED_ACTIONS_FILE = "corporate_actions.csv"
DATASET_FILE = "_dataset.json"
ELIGIBILITY_FILE = "_eligibility.csv"
DIAGNOSTIC_DIR = "diagnostic"


class CanonicalModeError(ValueError):
    """规范数据目录不合规，或旧模式读取器遇到规范数据的标志（T11-D0a 第 7.4 节）。"""


def canonical_markers(root: Path) -> list[str]:
    """旧模式下不应出现的规范数据标志：_eligibility.csv、*.adjlog.json、diagnostic/。"""
    if not root.is_dir():
        return []
    found = []
    if (root / ELIGIBILITY_FILE).exists():
        found.append(ELIGIBILITY_FILE)
    if any(root.glob("*.adjlog.json")):
        found.append("*.adjlog.json")
    if (root / DIAGNOSTIC_DIR).exists():
        found.append(f"{DIAGNOSTIC_DIR}/")
    return found


def is_daily_file(name: str) -> bool:
    """按文件名枚举证券：<SYMBOL>.csv，排除以 _ 开头的文件、公司行动文件与合并文件。"""
    return (
        name.endswith(".csv")
        and not name.startswith("_")
        and not name.endswith(".actions.csv")
        and name != MERGED_ACTIONS_FILE
    )


def read_bars(path: Path, symbol: str) -> tuple[list[DailyBar], list[date]]:
    """审计读法：返回 (探索期日线, 越界日期)。"""
    bars: list[DailyBar] = []
    sealed_days: list[date] = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            d = date.fromisoformat(r["date"])
            if d > EXPLORATION_END:
                sealed_days.append(d)
                continue
            bars.append(
                DailyBar(
                    day=d,
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r["volume"]),
                    adj_open=float(r["adj_open"]),
                    adj_high=float(r["adj_high"]),
                    adj_low=float(r["adj_low"]),
                    adj_close=float(r["adj_close"]),
                    adj_volume=parse_adj_volume(r.get("adj_volume"), f"{symbol} {d} adj_volume"),
                )
            )
    days = [b.day for b in bars]
    if any(a >= b for a, b in zip(days, days[1:], strict=False)):
        raise ValueError(f"{path.name} 日期未严格升序（重复或乱序）")
    return bars, sealed_days


def read_bars_text(path: Path) -> list[RawBarText]:
    with path.open(encoding="utf-8", newline="") as f:
        return [
            RawBarText(
                day=date.fromisoformat(r["date"]),
                open=r["open"],
                high=r["high"],
                low=r["low"],
                close=r["close"],
                volume=r["volume"],
                adj_open=r["adj_open"],
                adj_high=r["adj_high"],
                adj_low=r["adj_low"],
                adj_close=r["adj_close"],
                adj_volume=r.get("adj_volume") or None,  # 缺列或空单元格为 None（未知）
            )
            for r in csv.DictReader(f)
        ]


class CsvRawDataReader:
    def __init__(self, root: Path) -> None:
        if (root / DATASET_FILE).exists():
            raise CanonicalModeError(f"{root} 是规范数据目录，应按规范模式读取")
        markers = canonical_markers(root)
        if markers:
            raise CanonicalModeError(
                f"{root} 没有 {DATASET_FILE}，却有规范数据的文件 {markers}；不按旧模式读取"
            )
        self._root = root

    def exists(self, symbol: str) -> bool:
        return (self._root / f"{symbol}.csv").exists()

    def load(self, symbol: str) -> RawFile:
        bars, sealed_days = read_bars(self._root / f"{symbol}.csv", symbol)
        actions: list[CorporateAction] = []
        action_path = self._root / f"{symbol}.actions.csv"
        if action_path.exists():
            with action_path.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    d = date.fromisoformat(r["date"])
                    if d > EXPLORATION_END:
                        sealed_days.append(d)
                        continue
                    actions.append(CorporateAction(d, ActionType(r["type"]), float(r["value"])))
        return RawFile(
            symbol=symbol,
            history=PriceHistory(tuple(bars), tuple(actions)),
            sealed_rows=count_sealed_rows(sealed_days),
        )

    def load_text(self, symbol: str) -> RawTextFile:
        bars = read_bars_text(self._root / f"{symbol}.csv")
        actions: list[RawActionText] = []
        action_path = self._root / f"{symbol}.actions.csv"
        if action_path.exists():
            with action_path.open(encoding="utf-8", newline="") as f:
                actions = [
                    RawActionText(date.fromisoformat(r["date"]), r["type"], r["value"])
                    for r in csv.DictReader(f)
                ]
        return RawTextFile(symbol, tuple(bars), tuple(actions))

    def manifest_sha256(self) -> str | None:
        p = self._root / MANIFEST_FILE
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

    def manifest_entries(self) -> dict[str, str]:
        """MANIFEST.sha256 中的条目（文件名 → 哈希）；格式为每行“<哈希>  <文件名>”。"""
        return read_manifest(self._root / MANIFEST_FILE)

    def daily_file_hashes(self) -> dict[str, str]:
        """目录中各日线文件（<SYMBOL>.csv，不含公司行动文件与以 _ 开头的文件）的 SHA-256。"""
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(self._root.glob("*.csv"))
            if is_daily_file(p.name)
        }


def read_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.strip().partition("  ")
        if not name:
            raise ValueError(f"{MANIFEST_FILE} 行格式不正确: {line!r}")
        out[name.strip()] = digest.strip()
    return out
