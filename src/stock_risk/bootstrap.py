"""组装根：唯一允许同时引用 config、application、infrastructure 的地方（类似 Spring 的配置类）。"""

from __future__ import annotations

from stock_risk.application.fetch_raw_data import FetchRawDataService
from stock_risk.config.settings import Settings
from stock_risk.infrastructure.storage.csv_raw_repository import CsvRawDataRepository
from stock_risk.infrastructure.storage.csv_universe_provider import CsvUniverseProvider
from stock_risk.infrastructure.tiingo.client import TiingoPriceSource


def build_fetch_raw_data_service(settings: Settings) -> FetchRawDataService:
    if not settings.tiingo_api_key:
        raise SystemExit("未配置 TIINGO_API_KEY（项目根目录 .env 或环境变量）")
    return FetchRawDataService(
        source=TiingoPriceSource(settings.tiingo_api_key),
        universe=CsvUniverseProvider(settings.exploration_universe_file),
        repository=CsvRawDataRepository(settings.raw_data_dir),
    )
