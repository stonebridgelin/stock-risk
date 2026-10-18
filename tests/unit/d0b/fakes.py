"""T12-D0b 测试的假传输层：按请求路径返回事先写好的 JSON 原文，经 MassiveReference 的真实解析路径。

不发任何网络请求；响应正文是文本，经 client.parse_json_keep_numbers 首次解码（DBr-01）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from stock_risk.application.fetch_massive_d0b import RequestSpec
from stock_risk.infrastructure.massive.client import (
    RELAY_BASE_URL,
    MassiveReference,
    parse_json_keep_numbers,
)

HOST = "relaynex.vip"


def ms(d: date) -> str:
    """美东零点（UTC 05:00）的毫秒时间戳原文。"""
    return str(int(datetime(d.year, d.month, d.day, 5, tzinfo=UTC).timestamp() * 1000))


def bar_json(d: date, o: str, h: str, lo: str, c: str, v: str = "1000") -> str:
    return f'{{"t":{ms(d)},"o":{o},"h":{h},"l":{lo},"c":{c},"v":{v},"vw":{c},"n":10}}'


def ok(results: list[str]) -> str:
    return '{"status":"OK","results":[' + ",".join(results) + "]}"


class FakeMassive:
    """responses：(类别, 代码) → 正文文本或 HTTP 状态码。类别：raw / adj / splits / dividends。"""

    def __init__(self, responses: dict[tuple[str, str], str | int]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def transport(self, req: urllib.request.Request) -> Any:
        parts = urllib.parse.urlsplit(req.full_url)
        assert parts.hostname == HOST
        q = dict(urllib.parse.parse_qsl(parts.query))
        if parts.path.startswith("/v2/aggs/ticker/"):
            ticker = urllib.parse.unquote(parts.path.split("/")[4])
            key = ("adj" if q["adjusted"] == "true" else "raw", ticker)
        elif parts.path == "/v3/reference/splits":
            key = ("splits", q["ticker"])
        elif parts.path == "/v3/reference/dividends":
            key = ("dividends", q["ticker"])
        else:  # pragma: no cover
            raise AssertionError(parts.path)
        self.calls.append(key)
        got = self.responses.get(key, ok([]))
        if isinstance(got, int):
            raise urllib.error.HTTPError(
                req.full_url, got, "error", {}, BytesIO(b'{"message":"x"}')
            )  # type: ignore[arg-type]
        return parse_json_keep_numbers(got)

    def source(self) -> MassiveReference:
        return MassiveReference(
            "test-key",
            base_url=RELAY_BASE_URL,
            per_minute=600,
            transport=self.transport,
            sleep=lambda _: None,
        )


def spec(
    symbol: str, category: str, start: date, end: date, ticker: str | None = None
) -> RequestSpec:
    return RequestSpec(symbol, ticker or symbol, category, start, end)


def snapshot(download: Path, dest: Path) -> Path:
    """测试用的输入快照：复制下载目录中的构建输入，按账本写 INPUTS_STATUS.json。

    P3 的正式快照另做。"""
    ledger = json.loads((download / "_ledger.json").read_text("utf-8"))
    files: dict[str, dict[str, object]] = {}
    for entry in ledger["requests"].values():
        if entry["category"] == "adj" or entry.get("status") != "ok":
            continue
        src = download / entry["file"]
        dst = dest / entry["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files[entry["file"]] = {
            "status": entry["status"],
            "complete": entry["complete"],
            "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        }
    (dest / "INPUTS_STATUS.json").write_text(json.dumps({"files": files}), "utf-8")
    return dest


def weekdays(start: date, end: date) -> list[date]:
    from datetime import timedelta

    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def write_config(cfg: Path, listing_rows: list[str] = (), massive_rows: list[str] = ()) -> Path:  # type: ignore[assignment]
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "symbol_segments.csv").write_text("symbol,source_ticker,from,to,note\n", "utf-8")
    (cfg / "adjustment_verifications.csv").write_text(
        "id,target_type,target,source_sha256,kind,value,ruling,evidence\n", "utf-8"
    )
    (cfg / "raw_price_corrections.csv").write_text(
        "id,security,day,field,value,ruling,evidence,version\n", "utf-8"
    )
    (cfg / "price_corrections.csv").write_text(
        "symbol,from,to,method,ruling,evidence,d0a_status\n", "utf-8"
    )
    (cfg / "listing_periods.csv").write_text(
        "symbol,listed_from,listed_until,venue,source_url,verified_on,note\n"
        + "".join(r + "\n" for r in listing_rows),
        "utf-8",
    )
    (cfg / "massive_source_tickers.csv").write_text(
        "symbol,source_ticker,from,to,ruling,note\n" + "".join(r + "\n" for r in massive_rows),
        "utf-8",
    )
    return cfg


def write_calendar(path: Path, days: list[date]) -> Path:
    path.write_text("date\n" + "".join(f"{d.isoformat()}\n" for d in days), "utf-8")
    return path


Factory = Callable[[], MassiveReference]
