"""Massive（原 Polygon.io）参考数据只读适配器（实现 application.probe_massive.ReferenceSource）。

- All Tickers `/v3/reference/tickers`（分页用 next_url）与
  Ticker Overview `/v3/reference/tickers/{ticker}`；
- 认证：请求头 `Authorization: Bearer <key>`（Massive 文档 quickstart），密钥不进 URL；
  next_url 中若带 apiKey 参数一律去掉，且只接受与本实例主机相同的 https 地址；
  请求头带固定的 `User-Agent: stock-risk/1.0`（免费档与付费档相同；
  不用 urllib 默认的 Python-urllib）；
- 主机（T08d-2c）：默认 api.massive.com；付费档可配置为中转服务 relaynex.vip，只允许这两个取值。
  一个实例只连一个主机，密钥只发往该主机；用哪个主机、哪个密钥由 bootstrap 决定；
- 日志、异常信息中不出现 URL 与密钥；服务端错误原文中若出现密钥也替换为 ***；
- 限频：默认每分钟 5 次（免费档），两次请求最小间隔 60 / 上限 秒；遇 429 退避（60 秒 × 次数），
  连续 3 次 429 抛 RateLimitError；
- 封存：请求参数 date 不得晚于 2019-12-31（domain.massive_probe.check_query_date）；
- 查询的 exchange、type、market 为 None 时请求中不传该参数（第 2.4 节）；
- 付费档（T08d-2b）：日线 Aggregates、拆股 /stocks/v1/splits、分红 /stocks/v1/dividends；
  日线的 to 不得晚于 2019-12-31（domain.massive_paid.check_price_range）。
  适配器不读配置：用哪个密钥由 bootstrap 决定。
只用标准库 urllib；原始响应只在内存中解析后返回，不写盘。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date
from typing import Any

from stock_risk.application.ports import PriceSourceError, QueryRejected, RateLimitError
from stock_risk.domain.massive_followup import TickerPages
from stock_risk.domain.massive_paid import check_price_range
from stock_risk.domain.massive_probe import ListQuery, check_query_date

HOST = "api.massive.com"
BASE_URL = f"https://{HOST}"
RELAY_BASE_URL = "https://relaynex.vip"  # T08d-2c：第三方中转服务，只用于付费档
ALLOWED_BASE_URLS = (BASE_URL, RELAY_BASE_URL)
# 固定的请求标识（T08d-2c）：中转服务前的 Cloudflare 拦截 urllib 默认的 Python-urllib（403，1010）
USER_AGENT = "stock-risk/1.0"
PAGE_LIMIT = 1000  # All Tickers 每页上限（文档）
MAX_PAGES = 200  # 防止 next_url 循环

Transport = Callable[[urllib.request.Request], Any]


def _default_transport(req: urllib.request.Request) -> Any:
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 —— 只请求固定主机
        return json.loads(resp.read().decode("utf-8"))


class MassiveReference:
    name = "Massive"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        per_minute: int = 5,
        max_retries: int = 3,
        backoff_sec: float = 60.0,
        transport: Transport = _default_transport,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("缺少 MASSIVE_API_KEY")
        if per_minute < 1:
            raise ValueError("per_minute 必须为正整数")
        if base_url not in ALLOWED_BASE_URLS:
            raise ValueError("Massive 主机只允许 https://api.massive.com 或 https://relaynex.vip")
        self._base = base_url
        self._host = urllib.parse.urlsplit(base_url).hostname or ""
        self._key = api_key
        self._min_interval = 60.0 / per_minute
        self._max_retries = max_retries
        self._backoff = backoff_sec
        self._transport = transport
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None
        self._requests = 0

    @property
    def source_host(self) -> str:
        """本实例连接的主机（写入付费档的来源记录）。"""
        return self._host

    @property
    def request_count(self) -> int:
        """已发出的请求数（含重试）。"""
        return self._requests

    def list_tickers(self, query: ListQuery) -> list[dict[str, Any]]:
        got = self.list_tickers_pages(query, MAX_PAGES)
        if not got.complete:
            raise PriceSourceError(f"分页超过 {MAX_PAGES} 页，停止")
        return got.rows

    def list_tickers_pages(self, query: ListQuery, max_pages: int) -> TickerPages:
        """翻页至多 max_pages 页；达到上限时仍有下一页则 complete 为假（第 2.4 节 B 的页数上限）。
        exchange、type、market 为 None 时请求中不出现该参数。"""
        check_query_date(query.day)
        params = {
            "market": query.market,
            "exchange": query.exchange,
            "type": query.type,
            "active": "true" if query.active else "false",
            "date": query.day.isoformat(),
            "limit": str(PAGE_LIMIT),
            "order": "asc",
            "sort": "ticker",
        }
        qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        got = self._paginate(f"{self._base}/v3/reference/tickers?{qs}", max_pages, "All Tickers")
        if got is None:
            raise QueryRejected(404, "not found")
        return got

    # ---- 付费档（T08d-2b）：日线、拆股、分红 ----

    def aggs(self, ticker: str, start: date, end: date, adjusted: bool) -> TickerPages | None:
        """日线 /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}；to 不得晚于 2019-12-31。
        查不到（404）时返回 None。原始行只在内存中返回，由调用方做封存处理。"""
        check_price_range(start, end)
        path = (
            f"/v2/aggs/ticker/{urllib.parse.quote(ticker, safe='')}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}"
        )
        params = {"adjusted": "true" if adjusted else "false", "sort": "asc", "limit": "50000"}
        url = f"{self._base}{path}?{urllib.parse.urlencode(params)}"
        return self._paginate(url, MAX_PAGES, "Aggregates")

    def splits(self, ticker: str, until: date) -> TickerPages:
        """拆股 /stocks/v1/splits：ticker、execution_date.lte（不晚于 2019-12-31）。"""
        check_query_date(until)
        params = {
            "ticker": ticker,
            "execution_date.lte": until.isoformat(),
            "limit": "5000",
            "sort": "execution_date.asc",
        }
        return self._reference("/stocks/v1/splits", params, "Splits")

    def dividends(self, ticker: str, until: date) -> TickerPages:
        """分红 /stocks/v1/dividends：ticker、ex_dividend_date.lte（不晚于 2019-12-31）。"""
        check_query_date(until)
        params = {
            "ticker": ticker,
            "ex_dividend_date.lte": until.isoformat(),
            "limit": "5000",
            "sort": "ex_dividend_date.asc",
        }
        return self._reference("/stocks/v1/dividends", params, "Dividends")

    def _reference(self, path: str, params: dict[str, str], what: str) -> TickerPages:
        url = f"{self._base}{path}?{urllib.parse.urlencode(params)}"
        got = self._paginate(url, MAX_PAGES, what)
        if got is None:
            raise QueryRejected(404, "not found")
        return got

    def _paginate(self, url: str | None, max_pages: int, what: str) -> TickerPages | None:
        """翻页至多 max_pages 页；首页 404 返回 None。达到上限时仍有下一页则 complete 为假。"""
        out: list[dict[str, Any]] = []
        pages = 0
        while url is not None:
            if pages >= max_pages:
                return TickerPages(out, pages, complete=False)
            data = self._get(url)
            if data is None:
                if pages == 0:
                    return None
                raise QueryRejected(404, "not found")
            results = data.get("results") or []
            if not isinstance(results, list):
                raise PriceSourceError(f"{what} 返回格式异常（results 不是列表）")
            out.extend(r for r in results if isinstance(r, dict))
            pages += 1
            url = self._next(data.get("next_url"))
        return TickerPages(out, pages, complete=True)

    def overview(self, ticker: str, day: date) -> dict[str, Any] | None:
        check_query_date(day)
        path = f"/v3/reference/tickers/{urllib.parse.quote(ticker, safe='')}"
        data = self._get(f"{self._base}{path}?{urllib.parse.urlencode({'date': day.isoformat()})}")
        if data is None:
            return None
        results = data.get("results")
        return results if isinstance(results, dict) else None

    # ---- 内部 ----

    def _next(self, next_url: Any) -> str | None:
        """next_url 只接受本实例主机的 https 地址；去掉其中可能带的 apiKey 参数。"""
        if not next_url:
            return None
        parts = urllib.parse.urlsplit(str(next_url))
        if parts.scheme != "https" or parts.hostname != self._host:
            raise PriceSourceError(f"next_url 不是 {self._host} 的 https 地址，停止")
        query = [
            (k, v)
            for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() != "apikey"
        ]
        return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))

    def _clean(self, text: str) -> str:
        return text.replace(self._key, "***")[:300]

    def _throttle(self) -> None:
        if self._last is not None:
            wait = self._min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def _error_message(self, e: urllib.error.HTTPError) -> str:
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 —— 读取错误正文失败时只记状态
            return ""
        try:
            data = json.loads(body)
        except ValueError:
            return self._clean(body.strip())
        if isinstance(data, dict):
            parts = [str(data[k]) for k in ("status", "message", "error") if data.get(k)]
            return self._clean("；".join(parts))
        return self._clean(body.strip())

    def _get(self, url: str) -> dict[str, Any] | None:
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        for attempt in range(1, self._max_retries + 1):
            self._throttle()
            self._requests += 1
            try:
                data = self._transport(req)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code == 429:
                    if attempt < self._max_retries:
                        self._sleep(self._backoff * attempt)
                        continue
                    raise RateLimitError(f"HTTP 429（连续 {attempt} 次）") from None
                raise QueryRejected(e.code, self._error_message(e)) from None
            except urllib.error.URLError as e:
                if attempt < self._max_retries:
                    self._sleep(30)
                    continue
                raise PriceSourceError(f"网络错误: {self._clean(str(e.reason))}") from None
            if not isinstance(data, dict):
                raise PriceSourceError("返回格式异常（不是 JSON 对象）")
            status = str(data.get("status", "")).upper()
            if status in ("ERROR", "NOT_AUTHORIZED"):
                raise QueryRejected(200, self._clean(f"{status}；{data.get('message', '')}"))
            return data
        raise PriceSourceError("重试次数耗尽")
