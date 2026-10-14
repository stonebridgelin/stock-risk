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
    assert set(periods) == {"SHLD", "WFT", "SMH", "GMCR"}
    assert all(p.source_url.startswith("https://") for p in periods.values())


def test_price_correction_provider(tmp_path: Path) -> None:
    from stock_risk.domain.corrections import CorrectionMethod
    from stock_risk.infrastructure.storage.csv_config_providers import CsvPriceCorrectionProvider

    p = tmp_path / "c.csv"
    p.write_text(
        "# c\nsymbol,from,to,method,ruling,evidence\nshld,2018-03-28,2018-10-23,use_raw,PM-03,x\n",
        "utf-8",
    )
    (c,) = CsvPriceCorrectionProvider(p).load()["SHLD"]
    assert (c.start, c.end, c.method, c.ruling) == (
        date(2018, 3, 28),
        date(2018, 10, 23),
        CorrectionMethod.USE_RAW,
        "PM-03",
    )


def test_price_correction_provider_rejects_unknown_method(tmp_path: Path) -> None:
    from stock_risk.domain.corrections import CorrectionError
    from stock_risk.infrastructure.storage.csv_config_providers import CsvPriceCorrectionProvider

    p = tmp_path / "c.csv"
    p.write_text(
        "symbol,from,to,method,ruling,evidence\nX,2018-03-28,2018-10-23,scale,PM,x\n", "utf-8"
    )
    with pytest.raises(CorrectionError):
        CsvPriceCorrectionProvider(p).load()


def test_project_price_corrections_file_is_valid() -> None:
    from stock_risk.infrastructure.storage.csv_config_providers import CsvPriceCorrectionProvider

    corr = CsvPriceCorrectionProvider(PROJECT_ROOT / "config" / "price_corrections.csv").load()
    (c,) = corr["SHLD"]
    assert (c.start, c.end, c.ruling) == (date(2018, 3, 28), date(2018, 10, 23), "PM-03")
    periods = CsvListingPeriodProvider(PROJECT_ROOT / "config" / "listing_periods.csv").load()
    assert periods["SHLD"].contains(c.start) and periods["SHLD"].contains(c.end)


def test_reader_manifest_entries_and_hashes(tmp_path: Path) -> None:
    import hashlib

    from stock_risk.infrastructure.storage.csv_raw_reader import CsvRawDataReader

    (tmp_path / "A.csv").write_bytes(b"x\n")
    (tmp_path / "A.actions.csv").write_text("y\n", "utf-8")
    (tmp_path / "corporate_actions.csv").write_text("z\n", "utf-8")
    digest = hashlib.sha256(b"x\n").hexdigest()
    (tmp_path / "MANIFEST.sha256").write_text(f"{digest}  A.csv\n", "utf-8")
    r = CsvRawDataReader(tmp_path)
    assert r.manifest_entries() == {"A.csv": digest}
    assert r.daily_file_hashes() == {"A.csv": digest}
