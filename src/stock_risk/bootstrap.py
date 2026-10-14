"""组装根：唯一允许同时引用 config、application、infrastructure 的地方（类似 Spring 的配置类）。"""

from __future__ import annotations

from stock_risk.application.audit_raw_data import AuditRawDataService
from stock_risk.application.check_calendar import CheckCalendarService
from stock_risk.application.fetch_raw_data import FetchRawDataService
from stock_risk.application.market_data_loader import MarketDataLoader
from stock_risk.application.show_ranges import ShowRangesService
from stock_risk.config.settings import Settings
from stock_risk.infrastructure.calendars.xnys import ExchangeCalendarsXnys
from stock_risk.infrastructure.storage.audit_report_writer import AuditReportFileWriter
from stock_risk.infrastructure.storage.csv_config_providers import (
    CsvListingPeriodProvider,
    CsvPriceCorrectionProvider,
    CsvSegmentProvider,
)
from stock_risk.infrastructure.storage.csv_raw_reader import CsvRawDataReader
from stock_risk.infrastructure.storage.csv_raw_repository import CsvRawDataRepository
from stock_risk.infrastructure.storage.csv_universe_provider import CsvUniverseProvider
from stock_risk.infrastructure.storage.ranges_calendar_writers import (
    CalendarCheckFileWriter,
    RangesFileWriter,
)
from stock_risk.infrastructure.tiingo.client import TiingoPriceSource


def build_fetch_raw_data_service(settings: Settings) -> FetchRawDataService:
    if not settings.tiingo_api_key:
        raise SystemExit("未配置 TIINGO_API_KEY（项目根目录 .env 或环境变量）")
    return FetchRawDataService(
        source=TiingoPriceSource(settings.tiingo_api_key),
        universe=CsvUniverseProvider(settings.exploration_universe_file),
        repository=CsvRawDataRepository(settings.raw_data_dir),
        segments=CsvSegmentProvider(settings.symbol_segments_file),
    )


def build_audit_raw_data_service(settings: Settings) -> AuditRawDataService:
    return AuditRawDataService(
        loader=build_market_data_loader(settings),
        universe=CsvUniverseProvider(settings.exploration_universe_file),
        segments=CsvSegmentProvider(settings.symbol_segments_file),
        writer=AuditReportFileWriter(),
    )


def build_market_data_loader(settings: Settings) -> MarketDataLoader:
    """下游读取原始数据的统一入口：应用已登记的复权价更正（I11）。"""
    return MarketDataLoader(
        reader=CsvRawDataReader(settings.raw_data_dir),
        listings=CsvListingPeriodProvider(settings.listing_periods_file),
        corrections=CsvPriceCorrectionProvider(settings.price_corrections_file),
    )


def build_check_calendar_service(settings: Settings) -> CheckCalendarService:
    return CheckCalendarService(
        loader=build_market_data_loader(settings),
        reference=ExchangeCalendarsXnys(),
        writer=CalendarCheckFileWriter(),
    )


def build_show_ranges_service(settings: Settings) -> ShowRangesService:
    return ShowRangesService(
        loader=build_market_data_loader(settings),
        universe=CsvUniverseProvider(settings.exploration_universe_file),
        writer=RangesFileWriter(),
    )
