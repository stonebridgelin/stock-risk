"""T08d-1：代码清单的封存处理、筛选、汇总与覆盖对照（构造的小清单）。"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pytest

from stock_risk.application.probe_tickers import ProbeTickersService
from stock_risk.domain.seal import EXPLORATION_END
from stock_risk.domain.universe_probe import (
    CoverageRequest,
    ProbeFormatError,
    coverage,
    seal_rows,
    summarize,
)
from stock_risk.infrastructure.storage.universe_probe_files import ProbeFileWriter
from stock_risk.infrastructure.tiingo.ticker_list import TiingoTickerList

HEADER = "ticker,exchange,assetType,priceCurrency,startDate,endDate"
ROWS = [
    "AAA,NYSE,Stock,USD,2005-01-03,2015-06-30",
    "BBB,NASDAQ,Stock,USD,2010-02-01,2021-03-01",  # 延续到 2019 年之后
    "CCC,NYSE,Stock,USD,2020-05-01,2022-01-01",  # 开始日晚于截止日：丢弃，不计数
    "DDD,NYSE,Stock,USD,1999-01-04,2007-12-31",  # 结束日早于 2008-01-02：丢弃
    "EEE,NYSE MKT,Stock,USD,2009-01-02,",  # 其他交易所，结束日为空
    "FFF,NYSE ARCA,ETF,USD,2009-01-02,2019-12-31",
    "AAA,NASDAQ,Stock,USD,2016-01-04,",  # 代码复用
    "GGG,NYSE,Stock,CNY,2011-01-03,2018-01-02",
    "HHH,NYSE,Stock,USD,,",  # 没有开始日
]


def raw_rows() -> list[dict[str, str]]:
    keys = HEADER.split(",")
    return [dict(zip(keys, r.split(","), strict=True)) for r in ROWS]


def test_seal_drops_and_rewrites_dates() -> None:
    sealed = seal_rows(raw_rows())
    tickers = [r.ticker for r in sealed.rows]
    assert "CCC" not in tickers and "DDD" not in tickers and "HHH" not in tickers
    assert sealed.dropped_ended_before_2008 == 1 and sealed.dropped_no_start == 1
    assert all(r.end <= EXPLORATION_END and r.start <= EXPLORATION_END for r in sealed.rows)
    bbb = next(r for r in sealed.rows if r.ticker == "BBB")
    assert bbb.end == EXPLORATION_END and bbb.active_after_2019
    aaa = [r for r in sealed.rows if r.ticker == "AAA"]
    assert [(r.end, r.active_after_2019) for r in aaa] == [
        (date(2015, 6, 30), False),
        (EXPLORATION_END, True),
    ]


def test_summary_counts_selected_only() -> None:
    s = summarize(seal_rows(raw_rows()))
    assert s["total"] == 3 and s["unique_tickers"] == 2  # AAA×2、BBB
    assert s["by_exchange"] == {"NASDAQ": 2, "NYSE": 1}
    assert s["by_start_year"] == {"2005": 1, "2010": 1, "2016": 1}
    assert (s["ended_by_2019"], s["active_after_2019"]) == (1, 2)
    assert s["reused_tickers"] == 1 and list(s["reused_examples"]) == ["AAA"]
    assert s["stock_usd_other_exchanges"] == {"NYSE MKT": 1}
    assert s["all_asset_type_values"] == {"Stock": 5, "ETF": 1}
    text = json.dumps(s)
    assert "2020" not in text and "2021" not in text and "2022" not in text


def test_coverage_check() -> None:
    sealed = seal_rows(raw_rows())
    out = coverage(
        sealed,
        [
            CoverageRequest("AAA", date(2008, 1, 2), date(2019, 12, 31)),
            CoverageRequest("BBB", date(2009, 1, 2), date(2015, 1, 2)),
            CoverageRequest("EEE"),
            CoverageRequest("ZZZ", date(2010, 1, 4), date(2011, 1, 3)),
        ],
    )
    by = {r["ticker"]: r for r in out}
    assert by["AAA"]["in_list"] and by["AAA"]["covered"] is False  # 两段各自都不覆盖
    assert by["BBB"]["covered"] is False  # 2010 年才开始
    assert not by["EEE"]["in_list"] and by["EEE"]["other_listing"] == "NYSE MKT/Stock/USD"
    assert by["EEE"]["covered"] == ""
    assert not by["ZZZ"]["in_list"] and by["ZZZ"]["covered"] is False
    with pytest.raises(Exception, match="晚于"):
        CoverageRequest("AAA", date(2019, 1, 2), date(2020, 1, 2))


def _zip_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("supported_tickers.csv", text)
    return buf.getvalue()


def test_ticker_list_in_memory_and_service_outputs(tmp_path: Path, monkeypatch) -> None:
    """压缩包只在内存中解开：运行前后工作目录没有新文件；输出只含封存处理后的日期。"""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    calls = []

    def download(url: str) -> bytes:
        calls.append(url)
        return _zip_bytes("\n".join([HEADER, *ROWS]) + "\n")

    out = tmp_path / "probe"
    svc = ProbeTickersService(TiingoTickerList(download=download), ProbeFileWriter())
    result = svc.run(out, [CoverageRequest("AAA")])
    assert len(calls) == 1 and list(work.iterdir()) == []
    assert sorted(p.name for p in out.iterdir()) == [
        "coverage_check.csv",
        "summary.json",
        "tickers_2008_2019.csv",
    ]
    text = "".join(p.read_text("utf-8") for p in out.iterdir())
    assert "2020" not in text and "2021" not in text and "2022" not in text
    lines = (out / "tickers_2008_2019.csv").read_text("utf-8").splitlines()
    assert lines[0] == "ticker,exchange,start_date,end_date,active_after_2019"
    assert "BBB,NASDAQ,2010-02-01,2019-12-31,true" in lines
    assert result.summary["total"] == 3


def test_ticker_list_format_errors() -> None:
    bad = TiingoTickerList(download=lambda _: _zip_bytes("ticker,exchange\nA,NYSE\n"))
    with pytest.raises(ProbeFormatError, match="缺少字段"):
        list(bad.rows())
    with pytest.raises(ProbeFormatError, match="zip"):
        list(TiingoTickerList(download=lambda _: b"not a zip").rows())
    junk = "\n".join([HEADER, "AAA,NYSE,Stock,USD,yesterday,"])
    with pytest.raises(ProbeFormatError, match="不是日期"):
        seal_rows(TiingoTickerList(download=lambda _: _zip_bytes(junk)).rows())
