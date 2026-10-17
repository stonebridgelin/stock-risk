"""组装根：唯一允许同时引用 config、application、infrastructure 的地方（类似 Spring 的配置类）。"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from stock_risk.application.audit_raw_data import AuditRawDataService
from stock_risk.application.check_calendar import CheckCalendarService
from stock_risk.application.compare.service import CompareService
from stock_risk.application.fetch_batch import BatchItem, FetchBatchService, QuotaLimits
from stock_risk.application.fetch_raw_data import FetchRawDataService
from stock_risk.application.market_data_loader import MarketDataLoader
from stock_risk.application.probe_massive import ProbeMassiveService
from stock_risk.application.probe_massive_followup import ProbeMassiveFollowupService
from stock_risk.application.probe_massive_paid import ProbeMassivePaidService
from stock_risk.application.probe_tickers import ProbeTickersService
from stock_risk.application.progress import ProgressFn, quiet
from stock_risk.application.replay import ReplayService
from stock_risk.application.replay_run import ReplayRunService
from stock_risk.application.show_ranges import ShowRangesService
from stock_risk.application.verify_massive_relay import VerifyMassiveRelayService
from stock_risk.config.settings import Settings
from stock_risk.domain.massive_paid import COMPARE_TICKERS
from stock_risk.domain.massive_probe import list_queries
from stock_risk.domain.universe import Role, UniverseMember
from stock_risk.domain.universe_probe import CoverageRequest
from stock_risk.infrastructure.calendars.xnys import ExchangeCalendarsXnys
from stock_risk.infrastructure.massive.client import ALLOWED_BASE_URLS as MASSIVE_ALLOWED_BASE_URLS
from stock_risk.infrastructure.massive.client import BASE_URL as MASSIVE_BASE_URL
from stock_risk.infrastructure.massive.client import MassiveReference
from stock_risk.infrastructure.replay.compare_files import CompareFileSink
from stock_risk.infrastructure.replay.environment import LocalRunEnvironment
from stock_risk.infrastructure.replay.files import ReplayFileSink
from stock_risk.infrastructure.storage.audit_report_writer import AuditReportFileWriter
from stock_risk.infrastructure.storage.csv_config_providers import (
    CsvListingPeriodProvider,
    CsvPriceCorrectionProvider,
    CsvSegmentProvider,
)
from stock_risk.infrastructure.storage.csv_raw_reader import CsvRawDataReader
from stock_risk.infrastructure.storage.csv_raw_repository import (
    CsvRawDataRepository,
    confirmed_no_data,
)
from stock_risk.infrastructure.storage.csv_universe_provider import CsvUniverseProvider
from stock_risk.infrastructure.storage.fetch_batch_files import JsonQuotaStore, read_batch_list
from stock_risk.infrastructure.storage.massive_followup_files import (
    MassiveFollowupFiles,
    read_reference_spans,
)
from stock_risk.infrastructure.storage.massive_paid_files import (
    MassivePaidFiles,
    read_calendar,
    read_tiingo_actions,
    read_tiingo_prices,
)
from stock_risk.infrastructure.storage.massive_probe_files import MassiveProbeFiles
from stock_risk.infrastructure.storage.massive_source_host import SourceHostConflict
from stock_risk.infrastructure.storage.massive_verify_files import (
    MassiveVerifyFiles,
    read_official_snapshots,
)
from stock_risk.infrastructure.storage.ranges_calendar_writers import (
    CalendarCheckFileWriter,
    RangesFileWriter,
)
from stock_risk.infrastructure.storage.universe_probe_files import (
    ProbeFileWriter,
    read_coverage_requests,
)
from stock_risk.infrastructure.tiingo.client import TiingoPriceSource
from stock_risk.infrastructure.tiingo.ticker_list import TiingoTickerList


def build_fetch_raw_data_service(
    settings: Settings, progress: ProgressFn = quiet
) -> FetchRawDataService:
    if not settings.tiingo_api_key:
        raise SystemExit("未配置 TIINGO_API_KEY（项目根目录 .env 或环境变量）")
    return FetchRawDataService(
        source=TiingoPriceSource(settings.tiingo_api_key.reveal()),
        universe=CsvUniverseProvider(settings.exploration_universe_file),
        repository=CsvRawDataRepository(settings.raw_data_dir),
        progress=progress,
        segments=CsvSegmentProvider(settings.symbol_segments_file),
    )


def build_audit_raw_data_service(settings: Settings) -> AuditRawDataService:
    return AuditRawDataService(
        loader=build_market_data_loader(settings),
        universe=CsvUniverseProvider(settings.exploration_universe_file),
        segments=CsvSegmentProvider(settings.symbol_segments_file),
        writer=AuditReportFileWriter(),
    )


def build_probe_tickers_service() -> ProbeTickersService:
    """代码清单探查（T08d-1）：公开清单，不用密钥，不下载价格。"""
    return ProbeTickersService(source=TiingoTickerList(), writer=ProbeFileWriter())


def massive_free_key(settings: Settings) -> str:
    """免费档命令（T08d-2a）的密钥：只用 MASSIVE_API_KEY。"""
    if not settings.massive_api_key:
        raise SystemExit("未配置 MASSIVE_API_KEY（项目根目录 .env 或环境变量）")
    return settings.massive_api_key.reveal()


def massive_paid_key(settings: Settings) -> str:
    """付费档命令（T08d-2b 起）的密钥：只用 MASSIVE_ADVANCED_API_KEY，不回退到 MASSIVE_API_KEY。"""
    key = settings.massive_advanced_api_key.reveal()
    if not key.strip():  # 未设置、空串、只有空白字符都视为没有密钥（T08d-2b 第 0.2a 节）
        raise SystemExit("未配置 MASSIVE_ADVANCED_API_KEY（项目根目录 .env 或环境变量）")
    return key


def massive_paid_base_url(settings: Settings) -> str:
    """付费档命令的主机（T08d-2c 第 1 节）：未设置或空白取官方主机；
    只允许两个取值，其他值在入口停止。"""
    value = settings.massive_advanced_base_url.strip()
    if not value:
        return MASSIVE_BASE_URL
    if value not in MASSIVE_ALLOWED_BASE_URLS:
        allowed = "、".join(MASSIVE_ALLOWED_BASE_URLS)
        raise SystemExit(f"MASSIVE_ADVANCED_BASE_URL 只允许 {allowed}")
    return value


def build_probe_massive_service(
    settings: Settings,
    out_dir: Path,
    per_minute: int | None = None,
    progress: ProgressFn = quiet,
) -> ProbeMassiveService:
    """Massive 数据源试验（T08d-2a）：只读参考数据，不下载价格。限频：命令行 > .env > 5 次/分钟。"""
    key = massive_free_key(settings)
    rate = per_minute or settings.massive_per_minute or 5
    return ProbeMassiveService(
        source=MassiveReference(key, per_minute=rate),
        store=MassiveProbeFiles(out_dir),
        clock=time.monotonic,
        progress=progress,
    )


def build_probe_massive_followup_service(
    settings: Settings,
    out_dir: Path,
    per_minute: int | None = None,
    progress: ProgressFn = quiet,
) -> ProbeMassiveFollowupService:
    """Massive 追加探查（T08d-2a 第 2.4 节）。D 读取 T08d-1 的 Tiingo 探查输出，不发请求。"""
    key = massive_free_key(settings)
    tiingo = settings.universe_probe_dir / "tickers_2008_2019.csv"
    if not tiingo.exists():
        raise SystemExit(f"缺少 T08d-1 的 Tiingo 探查输出：{tiingo}（先运行 probe-tickers）")
    rate = per_minute or settings.massive_per_minute or 5
    return ProbeMassiveFollowupService(
        source=MassiveReference(key, per_minute=rate),
        store=MassiveFollowupFiles(out_dir),
        tiingo=read_reference_spans(tiingo),
        clock=time.monotonic,
        progress=progress,
    )


def load_batch_list(path: Path) -> list[BatchItem]:
    """读取下载清单 CSV（symbol,source_ticker,from,to）。入口层经此读取，不直接依赖存储实现。"""
    return read_batch_list(path)


def load_coverage_requests(path: Path) -> list[CoverageRequest]:
    """读取覆盖对照 CSV（一列 ticker，可附 from,to）。"""
    return read_coverage_requests(path)


def quota_limits(
    settings: Settings,
    per_hour: int | None = None,
    per_day: int | None = None,
    unique_per_month: int | None = None,
    max_symbols: int | None = None,
) -> QuotaLimits:
    """额度：命令行 > .env / 环境变量 > 默认（免费档）。"""
    d = QuotaLimits()
    return QuotaLimits(
        per_hour=per_hour or settings.tiingo_per_hour or d.per_hour,
        per_day=per_day or settings.tiingo_per_day or d.per_day,
        unique_per_month=unique_per_month or settings.tiingo_unique_per_month or d.unique_per_month,
        max_symbols=max_symbols,
    )


def build_fetch_batch_service(
    settings: Settings, out_dir: Path, limits: QuotaLimits, progress: ProgressFn = quiet
) -> FetchBatchService:
    if not settings.tiingo_api_key:
        raise SystemExit("未配置 TIINGO_API_KEY（项目根目录 .env 或环境变量）")
    source = TiingoPriceSource(
        settings.tiingo_api_key.reveal(), min_interval_sec=3600 / limits.per_hour
    )
    return FetchBatchService(
        source=source,
        repository=CsvRawDataRepository(out_dir, title="stock-risk 选参集原始行情"),
        quota=JsonQuotaStore(out_dir),
        now=lambda: datetime.now(UTC),
        progress=progress,
    )


class _StaticUniverse:
    def __init__(self, members: Sequence[UniverseMember]) -> None:
        self._members = tuple(members)

    def load(self) -> tuple[UniverseMember, ...]:
        return self._members


class _NoConfig:
    def load(self) -> dict:
        return {}


def build_audit_dir_service(
    settings: Settings, raw_dir: Path, items: Sequence[BatchItem]
) -> AuditRawDataService:
    """审计其他目录（如 data/raw_selection）：代码取自下载清单；日历代理仍用 data/raw 的 SPY；
    上市期间沿用配置，不应用探索集的更正与分段。_fetch_status.json 中已确认无数据的代码
    不算缺失（B-04）。"""
    loader = MarketDataLoader(
        reader=CsvRawDataReader(raw_dir),
        listings=CsvListingPeriodProvider(settings.listing_periods_file),
        corrections=_NoConfig(),
    )
    return AuditRawDataService(
        loader=loader,
        universe=_StaticUniverse([i.member() for i in items]),
        segments=_NoConfig(),
        writer=AuditReportFileWriter(),
        calendar_reader=CsvRawDataReader(settings.raw_data_dir),
        confirmed_no_data=confirmed_no_data(raw_dir),
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


def build_replay_service(settings: Settings) -> ReplayService:
    return ReplayService(
        loader=build_market_data_loader(settings),
        universe=CsvUniverseProvider(settings.exploration_universe_file),
    )


def build_replay_run_service(settings: Settings, progress: ProgressFn = quiet) -> ReplayRunService:
    return ReplayRunService(
        replay=build_replay_service(settings),
        sink=ReplayFileSink(settings.replay_output_dir),
        env=LocalRunEnvironment(settings.project_root),
        progress=progress,
    )


def build_v0_regression(settings: Settings) -> tuple[ReplayService, ReplayFileSink]:
    """V0 真实数据回归（T06a）所需的回放用例与表格读写。"""
    return build_replay_service(settings), ReplayFileSink(settings.replay_output_dir)


def build_compare_service(settings: Settings, progress: ProgressFn = quiet) -> CompareService:
    return CompareService(
        replay=build_replay_service(settings),
        sink=CompareFileSink(settings.compare_output_dir),
        env=LocalRunEnvironment(settings.project_root),
        progress=progress,
    )


def build_probe_massive_paid_service(
    settings: Settings,
    out_dir: Path,
    per_minute: int | None = None,
    progress: ProgressFn = quiet,
) -> ProbeMassivePaidService:
    """Massive 付费档验证（T08d-2b）。

    密钥只用 MASSIVE_ADVANCED_API_KEY；主机取 MASSIVE_ADVANCED_BASE_URL（T08d-2c）；
    限频：命令行 > .env > 100 次/分钟。
    """
    key = massive_paid_key(settings)
    spy = settings.raw_data_dir / "SPY.csv"
    if not spy.exists():
        raise SystemExit(f"缺少交易日历：{spy}")
    rate = per_minute or settings.massive_per_minute or 100
    tickers = list(COMPARE_TICKERS)
    source = MassiveReference(key, base_url=massive_paid_base_url(settings), per_minute=rate)
    return ProbeMassivePaidService(
        source=source,
        store=_host_checked(lambda: MassivePaidFiles(out_dir, source.source_host)),
        calendar=read_calendar(spy),
        tiingo_prices=read_tiingo_prices(settings.selection_raw_dir, tickers),
        tiingo_actions=read_tiingo_actions(settings.selection_raw_dir, tickers),
        clock=time.monotonic,
        progress=progress,
    )


def _host_checked[T](make: Callable[[], T]) -> T:
    """构造付费档的输出存储；目录中已有其他主机的记录时在入口停止（T08d-2c 第 1 节）。"""
    try:
        return make()
    except SourceHostConflict as e:
        raise SystemExit(str(e)) from None


def build_verify_massive_relay_service(
    settings: Settings,
    per_minute: int | None = None,
    progress: ProgressFn = quiet,
) -> VerifyMassiveRelayService:
    """中转服务核验（T08d-2c 第 2、3 节）。

    密钥只用 MASSIVE_ADVANCED_API_KEY，主机取 MASSIVE_ADVANCED_BASE_URL；
    输出 data/massive_probe/paid/verify/。
    价格对照 data/raw 中 27 只探索股（role = signal_target）的 Tiingo 未复权价；
    参考数据对照 T08d-2a 的官方快照 data/massive_probe/lists_<日期>.csv（不发官方请求）。
    """
    key = massive_paid_key(settings)
    spy = settings.raw_data_dir / "SPY.csv"
    if not spy.exists():
        raise SystemExit(f"缺少交易日历：{spy}")
    rate = per_minute or settings.massive_per_minute or 100
    source = MassiveReference(key, base_url=massive_paid_base_url(settings), per_minute=rate)
    members = CsvUniverseProvider(settings.exploration_universe_file).load()
    tickers = [m.symbol for m in members if m.role is Role.SIGNAL_TARGET]
    queries = list_queries()
    return VerifyMassiveRelayService(
        source=source,
        store=_host_checked(
            lambda: MassiveVerifyFiles(settings.massive_paid_dir, source.source_host)
        ),
        calendar=read_calendar(spy),
        tickers=tickers,
        tiingo_prices=read_tiingo_prices(settings.raw_data_dir, tickers),
        official_lists=read_official_snapshots(settings.massive_probe_dir, queries),
        clock=time.monotonic,
        progress=progress,
        queries=queries,
    )
