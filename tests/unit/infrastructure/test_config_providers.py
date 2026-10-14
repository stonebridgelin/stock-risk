from datetime import date
from pathlib import Path

import pytest

from stock_risk.config.settings import PROJECT_ROOT
from stock_risk.domain.segments import SegmentError
from stock_risk.infrastructure.storage.csv_config_providers import (
    CsvListingPeriodProvider,
    CsvSegmentProvider,
)


def test_segment_provider_groups_rows(tmp_path: Path) -> None:
    p = tmp_path / "s.csv"
    p.write_text(
        "# c\nsymbol,source_ticker,from,to,note\nvrx,vrx,,2018-07-13,a\nVRX,BHC,2018-07-16,,b\n",
        "utf-8",
    )
    segs = CsvSegmentProvider(p).load()["VRX"]
    assert [(s.source_ticker, s.start, s.end) for s in segs] == [
        ("VRX", None, date(2018, 7, 13)),
        ("BHC", date(2018, 7, 16), None),
    ]


def test_segment_provider_rejects_overlap(tmp_path: Path) -> None:
    p = tmp_path / "s.csv"
    p.write_text("symbol,source_ticker,from,to,note\nV,V,,2018-07-16,\nV,B,2018-07-16,,\n", "utf-8")
    with pytest.raises(SegmentError):
        CsvSegmentProvider(p).load()


def test_missing_files_mean_no_config(tmp_path: Path) -> None:
    assert CsvSegmentProvider(tmp_path / "none.csv").load() == {}
    assert CsvListingPeriodProvider(tmp_path / "none.csv").load() == {}


def test_listing_provider_rejects_dates_without_source(tmp_path: Path) -> None:
    p = tmp_path / "l.csv"
    p.write_text(
        "symbol,listed_from,listed_until,venue,source_url,verified_on,note\n"
        "SHLD,,2018-10-23,Nasdaq,,,\n",
        "utf-8",
    )
    with pytest.raises(ValueError):
        CsvListingPeriodProvider(p).load()


def test_project_config_files_are_valid() -> None:
    segs = CsvSegmentProvider(PROJECT_ROOT / "config" / "symbol_segments.csv").load()
    assert [s.source_ticker for s in segs["VRX"]] == ["VRX", "BHC"]
    periods = CsvListingPeriodProvider(PROJECT_ROOT / "config" / "listing_periods.csv").load()
    assert set(periods) == {"SHLD", "WFT", "SMH"}
    assert all(p.source_url.startswith("https://") for p in periods.values())
