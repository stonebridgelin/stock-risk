"""规范数据版本的写出与读取（T11-D0a 第 7 节）。

目录 data/canonical/<版本号>/：
  <SYMBOL>.csv            仅 USABLE 的证券：原始列为原始价（已作用原始价更正），
                          adj_* 为 F，adj_volume 为 F_V
  <SYMBOL>.adjlog.json    复权日志（全部证券）：原始行动与计算组两张表，
                          semantic_digest.canonical，末尾一个换行
  diagnostic/<SYMBOL>.csv NOT_USABLE 证券的诊断序列，第一列 status 为"复权未完成"；正式读取入口不读
  _eligibility.csv        每个证券一行（含不可正式使用的）
  _assumptions.csv        每个假设实例一行
  _issues.csv             每个问题一行
  _dataset.json           数据版本的全部输入；不含版本号字段
  _calendar.csv          构建所用的交易日历（一列 date；T12-D0b 第 2.6 节第 3 项），
                          规范模式的读取器从这里取日历
  MANIFEST.sha256         上述全部文件的哈希（不含它自己与 _build_meta.json）
  _build_meta.json        开始、结束时间与命令行参数；不在清单与逐字节确定性比较之内

读取（第 7.4 节）：有 _dataset.json 的为规范模式；任何条件不满足即报错，不回退旧模式。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from collections.abc import Iterable, Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from stock_risk.application.build_canonical import CanonicalOutput, canonical_json, dataset_version
from stock_risk.application.ports import RawFile, RawTextFile
from stock_risk.domain.adjustment.model import (
    APPROVED_ASSUMPTIONS,
    APPROVED_LABEL,
    ASSUMPTION_CODES,
    RULE_LABEL,
    AdjustedBar,
    AdjustmentResult,
    Eligibility,
)
from stock_risk.domain.audit import count_sealed_rows
from stock_risk.domain.lifecycle.semantic_digest import canonical
from stock_risk.domain.market_data import PriceHistory
from stock_risk.infrastructure.storage.csv_raw_reader import (
    DATASET_FILE,
    DIAGNOSTIC_DIR,
    ELIGIBILITY_FILE,
    MANIFEST_FILE,
    CanonicalModeError,
    CsvRawDataReader,
    is_daily_file,
    read_bars,
    read_bars_text,
    read_manifest,
)
from stock_risk.infrastructure.storage.csv_raw_repository import BAR_COLUMNS

BUILD_META_FILE = "_build_meta.json"
ASSUMPTIONS_FILE = "_assumptions.csv"
ISSUES_FILE = "_issues.csv"
CALENDAR_FILE = "_calendar.csv"
DIAGNOSTIC_STATUS = "复权未完成"
DATASET_FIELDS = (
    "spec_version",
    "ruling_file",
    "params",
    "calendar",
    "segments",
    "securities",
    "config",
    "code_commit",
    "sealed_rows_skipped",
)
ASSUMPTION_COUNT_CODES = tuple(sorted(str(c) for c in ASSUMPTION_CODES))
ELIGIBILITY_COLUMNS = [
    "security",
    "eligibility",
    "reasons",
    "anchor",
    "terminal_pending",
    "amplitude_collapse_days",
    *(f"n_{c}" for c in ASSUMPTION_COUNT_CODES),
]
ASSUMPTION_COLUMNS = [
    "security",
    "code",
    "level",
    "target_id",
    "original_ex_date",
    "effective_day",
    "source",
    "k",
    "label",
]
ISSUE_COLUMNS = ["security", "code", "day", "field", "value", "note"]


# ---- 写出 ----


def _text(x: Decimal | None) -> str:
    return "" if x is None else str(x)


def _csv(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _bar_row(b: AdjustedBar) -> list[str]:
    return [
        b.day.isoformat(),
        str(b.open),
        str(b.high),
        str(b.low),
        str(b.close),
        str(b.volume),
        _text(b.f_open),
        _text(b.f_high),
        _text(b.f_low),
        _text(b.f_close),
        _text(b.f_volume),
    ]


def adjustment_log(r: AdjustmentResult) -> dict[str, Any]:
    return {
        "security": r.security,
        "anchor": r.anchor,
        "actions": r.actions,
        "groups": r.groups,
        "raw_correction_ids": r.raw_correction_ids,
        "verification_ids_used": r.verification_ids_used,
        "terminal_pending": r.terminal_pending,
    }


def render_files(output: CanonicalOutput) -> dict[str, bytes]:
    """除 MANIFEST.sha256 与 _build_meta.json 以外的全部文件（相对路径 → 内容）。"""
    files: dict[str, bytes] = {}
    elig_rows, assumption_rows, issue_rows = [], [], []
    for r in output.results:
        if r.eligibility is Eligibility.USABLE:
            files[f"{r.security}.csv"] = _csv(BAR_COLUMNS, (_bar_row(b) for b in r.bars))
        else:
            files[f"{DIAGNOSTIC_DIR}/{r.security}.csv"] = _csv(
                ["status", *BAR_COLUMNS], ([DIAGNOSTIC_STATUS, *_bar_row(b)] for b in r.bars)
            )
        files[f"{r.security}.adjlog.json"] = (canonical(adjustment_log(r)) + "\n").encode("utf-8")
        counts = {c: 0 for c in ASSUMPTION_COUNT_CODES}
        for a in r.assumptions:
            counts[a.code] += 1
            assumption_rows.append(
                [
                    a.security,
                    a.code,
                    a.level,
                    a.target_id,
                    a.original_ex_date.isoformat(),
                    a.effective_day.isoformat() if a.effective_day else "",
                    a.source,
                    "" if a.k is None else str(a.k),
                    APPROVED_LABEL if a.code in APPROVED_ASSUMPTIONS else RULE_LABEL,
                ]
            )
        elig_rows.append(
            [
                r.security,
                r.eligibility.value,
                "|".join(r.reasons),
                r.anchor.isoformat(),
                str(len(r.terminal_pending)),
                str(len(r.amplitude_collapse_days)),
                *(str(counts[c]) for c in ASSUMPTION_COUNT_CODES),
            ]
        )
        issue_rows += [
            [i.security, i.code, i.day.isoformat(), i.field, i.value, i.note] for i in r.issues
        ]
    files[ELIGIBILITY_FILE] = _csv(ELIGIBILITY_COLUMNS, elig_rows)
    files[ASSUMPTIONS_FILE] = _csv(ASSUMPTION_COLUMNS, assumption_rows)
    files[ISSUES_FILE] = _csv(ISSUE_COLUMNS, issue_rows)
    files[DATASET_FILE] = (output.dataset_text + "\n").encode("utf-8")
    if output.calendar_days:
        # 实际使用的交易日（封存期行已在读取时跳过），内容与 _dataset.json 的日历哈希对应
        files[CALENDAR_FILE] = _csv(["date"], ([d.isoformat()] for d in output.calendar_days))
    return files


def manifest_text(files: dict[str, bytes]) -> bytes:
    lines = [f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in sorted(files.items())]
    return ("\n".join(lines) + "\n").encode("utf-8")


class CanonicalFileSink:
    """实现 build_canonical.CanonicalSink。目录已存在的拒绝写入；先写临时目录再整体改名。"""

    def __init__(self, root: Path) -> None:
        self._root = root

    def exists(self, version: str) -> bool:
        return (self._root / version).exists()

    def write(self, output: CanonicalOutput) -> Path:
        final = self._root / output.version
        if final.exists():
            raise FileExistsError(f"{final} 已存在，拒绝写入")
        files = render_files(output)
        files[MANIFEST_FILE] = manifest_text(files)
        self._root.mkdir(parents=True, exist_ok=True)
        tmp = self._root / f".{output.version}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        for name, data in files.items():
            p = tmp / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        meta = json.dumps(output.meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        (tmp / BUILD_META_FILE).write_text(meta, "utf-8")
        tmp.rename(final)
        return final


# ---- 读取 ----


def _actual_files(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if rel in (MANIFEST_FILE, BUILD_META_FILE):
                continue
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class CanonicalDataReader:
    """规范模式的读取（实现 application.ports.RawDataReader）。构造时完整校验，不符即报错。"""

    is_canonical = True

    def __init__(self, root: Path) -> None:
        self._root = root
        ds_path = root / DATASET_FILE
        if not ds_path.exists():
            raise CanonicalModeError(f"{root}: 没有 {DATASET_FILE}")
        dataset = json.loads(ds_path.read_text("utf-8"))
        missing = [k for k in DATASET_FIELDS if k not in dataset]
        if missing:
            raise CanonicalModeError(f"{DATASET_FILE} 缺字段 {missing}")
        self._version = dataset_version(canonical_json(dataset))
        if self._version != root.name:
            raise CanonicalModeError(f"版本号 {self._version} 与目录名 {root.name} 不一致")
        manifest = read_manifest(root / MANIFEST_FILE)
        if not manifest:
            raise CanonicalModeError(f"{root}: 没有 {MANIFEST_FILE} 或清单为空")
        actual = _actual_files(root)
        if set(manifest) != set(actual):
            extra = sorted(set(actual) - set(manifest))
            lost = sorted(set(manifest) - set(actual))
            raise CanonicalModeError(f"{MANIFEST_FILE} 与目录不符：多出 {extra}，缺少 {lost}")
        bad = sorted(n for n in manifest if manifest[n] != actual[n])
        if bad:
            raise CanonicalModeError(f"{MANIFEST_FILE} 哈希不符: {bad}")
        if not (root / ELIGIBILITY_FILE).exists():
            raise CanonicalModeError(f"{root}: 没有 {ELIGIBILITY_FILE}")
        with (root / ELIGIBILITY_FILE).open(encoding="utf-8", newline="") as f:
            self._eligibility: list[dict[str, str]] = list(csv.DictReader(f))
        self._dataset_sha256 = hashlib.sha256(ds_path.read_bytes()).hexdigest()

    # ---- 身份 ----
    @property
    def data_version(self) -> str:
        return self._version

    @property
    def dataset_sha256(self) -> str:
        return self._dataset_sha256

    def calendar_days(self) -> tuple[date, ...]:
        """规范模式的交易日历：版本目录中的 _calendar.csv（不从 SPY 日线生成）。"""
        path = self._root / CALENDAR_FILE
        if not path.exists():
            raise CanonicalModeError(f"{self._root}: 没有 {CALENDAR_FILE}")
        with path.open(encoding="utf-8", newline="") as f:
            return tuple(date.fromisoformat(r["date"]) for r in csv.DictReader(f))

    def identity(self) -> dict[str, str | None]:
        """运行清单记录的数据身份（第 7.4 节）。"""
        return {
            "data_version": self._version,
            "dataset_sha256": self._dataset_sha256,
            "manifest_sha256": self.manifest_sha256(),
        }

    # ---- RawDataReader ----
    def _require(self, symbol: str) -> Path:
        rows = [r for r in self._eligibility if r["security"] == symbol]
        if len(rows) != 1:
            raise CanonicalModeError(
                f"{symbol}: {ELIGIBILITY_FILE} 中应恰好一行，实际 {len(rows)} 行"
            )
        if rows[0]["eligibility"] != Eligibility.USABLE.value:
            raise CanonicalModeError(f"{symbol}: 不可正式使用（{rows[0]['reasons']}）")
        path = self._root / f"{symbol}.csv"
        if not path.exists() or not (self._root / f"{symbol}.adjlog.json").exists():
            raise CanonicalModeError(f"{symbol}: 缺少日线文件或复权日志")
        return path

    def exists(self, symbol: str) -> bool:
        try:
            self._require(symbol)
        except CanonicalModeError:
            return False
        return True

    def load(self, symbol: str) -> RawFile:
        bars, sealed_days = read_bars(self._require(symbol), symbol)
        return RawFile(symbol, PriceHistory(tuple(bars), ()), count_sealed_rows(sealed_days))

    def load_text(self, symbol: str) -> RawTextFile:
        return RawTextFile(symbol, tuple(read_bars_text(self._require(symbol))), ())

    def manifest_sha256(self) -> str | None:
        return hashlib.sha256((self._root / MANIFEST_FILE).read_bytes()).hexdigest()

    def manifest_entries(self) -> dict[str, str]:
        return {
            n: h
            for n, h in read_manifest(self._root / MANIFEST_FILE).items()
            if "/" not in n and is_daily_file(n)
        }

    def daily_file_hashes(self) -> dict[str, str]:
        return {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(self._root.glob("*.csv"))
            if is_daily_file(p.name)
        }


def open_market_reader(root: Path) -> CsvRawDataReader | CanonicalDataReader:
    """按目录识别模式：有 _dataset.json 为规范模式，否则为旧模式（旧模式下有规范标志即报错）。"""
    if (root / DATASET_FILE).exists():
        return CanonicalDataReader(root)
    return CsvRawDataReader(root)


# ---- 独立审计路径：读取复权日志（T12-D0b 第 2.6 节第 2 项，DBr-02） ----


def _decode(node: object) -> object:
    """semantic_digest.canonical 的 JSON → 普通 Python 结构（数据类为字段字典，日期为 date）。"""
    if isinstance(node, dict):
        if "dc" in node:
            return {k: _decode(v) for k, v in node["fields"]}  # type: ignore[index]
        if "map" in node:
            return {_decode(k): _decode(v) for k, v in node["map"]}  # type: ignore[index]
        if "tuple" in node or "list" in node:
            return [_decode(v) for v in node.get("tuple", node.get("list"))]  # type: ignore[union-attr]
        if "date" in node:
            return date.fromisoformat(node["date"])
        if "enum" in node:
            return str(node["enum"]).rsplit(".", 1)[1]
        if "i" in node:
            return int(node["i"])
        if "d" in node:
            return Decimal(node["d"])
        if "f" in node:
            return float.fromhex(node["f"])
    return node


def read_adjustment_log(root: Path, symbol: str) -> dict[str, object]:
    """版本目录中 <SYMBOL>.adjlog.json 的内容（原始行动表、计算组表等）。不经过信号读取路径。"""
    data = json.loads((root / f"{symbol}.adjlog.json").read_text("utf-8"))
    out = _decode(data)
    assert isinstance(out, dict)
    return out
