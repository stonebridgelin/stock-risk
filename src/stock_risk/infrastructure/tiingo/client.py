"""Tiingo EOD 数据源适配器（实现 application.ports.PriceSource）。

限频：免费档约 50 次/小时，默认两次请求间隔 75 秒；遇 429 退避重试。
只使用标准库 urllib，避免为一次性下载引入依赖。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date
from typing import Any

from stock_risk.application.ports import PriceSourceError, RateLimitError
from stock_risk.domain.market_data import PriceHistory
from stock_risk.infrastructure.tiingo.mapper import to_history

BASE_URL = "https://api.tiingo.com/tiingo/daily"

Transport = Callable[[urllib.request.Request], Any]


def _default_transport(req: urllib.request.Request) -> Any:
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TiingoPriceSource:
    name = "Tiingo"

    def __init__(
        self,
        api_key: str,
        *,
        min_interval_sec: float = 75.0,
        max_retries: int = 3,
        backoff_sec: float = 600.0,
        transport: Transport = _default_transport,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("缺少 TIINGO_API_KEY")
        self._api_key = api_key
        self._min_interval = min_interval_sec
        self._max_retries = max_retries
        self._backoff = backoff_sec
        self._transport = transport
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self._requests = 0

    @property
    def request_count(self) -> int:
        """已发出的请求数（含重试）。"""
        return self._requests

    def fetch(self, ticker: str, start: date, end: date) -> PriceHistory | None:
        url = (
            f"{BASE_URL}/{ticker.lower()}/prices"
            f"?startDate={start.isoformat()}&endDate={end.isoformat()}"
            "&format=json&resampleFreq=daily"
        )
        data = self._request(url)
        if data is None:
            return None
        if isinstance(data, dict) and "not found" in str(data.get("detail", "")).lower():
            return None
        if not isinstance(data, list):
            raise PriceSourceError(f"返回格式异常: {str(data)[:200]}")
        try:
            return to_history(data)
        except (KeyError, ValueError) as e:
            raise PriceSourceError(f"解析失败: {e}") from e

    def _throttle(self) -> None:
        if self._last_request is not None:
            wait = self._min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _request(self, url: str) -> Any:
        req = urllib.request.Request(
            url,
            headers={"Content-Type": "application/json", "Authorization": f"Token {self._api_key}"},
        )
        for attempt in range(1, self._max_retries + 1):
            self._throttle()
            self._requests += 1
            try:
                return self._transport(req)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code == 429 and attempt < self._max_retries:
                    self._sleep(self._backoff * attempt)
                    continue
                if e.code == 429:  # 连续 max_retries 次 429
                    raise RateLimitError(f"HTTP 429（连续 {attempt} 次）") from e
                raise PriceSourceError(f"HTTP {e.code}") from e
            except urllib.error.URLError as e:
                if attempt < self._max_retries:
                    self._sleep(30)
                    continue
                raise PriceSourceError(f"网络错误: {e.reason}") from e
        raise PriceSourceError("重试次数耗尽")
