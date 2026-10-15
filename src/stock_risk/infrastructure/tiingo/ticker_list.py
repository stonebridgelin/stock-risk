"""Tiingo 公开的代码清单 supported_tickers.zip（实现 application.probe_tickers.TickerListSource）。

只发一个请求，不带密钥。压缩包只在内存中解开，不写盘；逐行交给领域层做封存处理。
"""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from collections.abc import Callable, Iterator, Mapping

from stock_risk.domain.universe_probe import ProbeFormatError, check_fields

TICKER_LIST_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"

Download = Callable[[str], bytes]


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 —— 固定的 https 地址
        return resp.read()


class TiingoTickerList:
    def __init__(self, url: str = TICKER_LIST_URL, download: Download = _download) -> None:
        self._url = url
        self._download = download

    def rows(self) -> Iterator[Mapping[str, str]]:
        data = self._download(self._url)
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as e:
            raise ProbeFormatError("代码清单不是有效的 zip") from e
        with archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if len(names) != 1:
                raise ProbeFormatError(f"代码清单压缩包中应有一个 CSV，实际 {names}")
            with archive.open(names[0]) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                check_fields(reader.fieldnames)
                yield from reader
