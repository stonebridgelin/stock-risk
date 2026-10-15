"""Massive（原 Polygon.io）参考数据只读适配器（实现 application.probe_massive.ReferenceSource）。

- All Tickers `/v3/reference/tickers`（分页用 next_url）与
  Ticker Overview `/v3/reference/tickers/{ticker}`；
- 认证：请求头 `Authorization: Bearer <key>`（Massive 文档 quickstart），密钥不进 URL；
  next_url 中若带 apiKey 参数一律去掉，且只接受 api.massive.com 的 https 地址；
- 日志、异常信息中不出现 URL 与密钥；服务端错误原文中若出现密钥也替换为 ***；
- 限频：默认每分钟 5 次（免费档），两次请求最小间隔 60 / 上限 秒；遇 429 退避（60 秒 × 次数），
  连续 3 次 429 抛 RateLimitError；
- 封存：请求参数 date 不得晚于 2019-12-31（domain.massive_probe.check_query_date）。
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
from stock_risk.domain.massive_probe import ListQuery, check_query_date

HOST = "api.massive.com"
BASE_URL = f"https://{HOST}"
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
    def request_count(self) -> int:
        """已发出的请求数（含重试）。"""
        return self._requests

    def list_tickers(self, query: ListQuery) -> list[dict[str, Any]]:
        check_query_date(query.day)
        params = {
            "exchange": query.exchange,
            "type": query.type,
            "active": "true" if query.active else "false",
            "date": query.day.isoformat(),
            "limit": str(PAGE_LIMIT),
            "order": "asc",
            "sort": "ticker",
        }
        url: str | None = f"{BASE_URL}/v3/reference/tickers?{urllib.parse.urlencode(params)}"
        out: list[dict[str, Any]] = []
        for _ in range(MAX_PAGES):
            if url is None:
                return out
            data = self._get(url)
            if data is None:
                raise QueryRejected(404, "not found")
            results = data.get("results") or []
            if not isinstance(results, list):
                raise PriceSourceError("All Tickers 返回格式异常（results 不是列表）")
            out.extend(r for r in results if isinstance(r, dict))
            url = self._next(data.get("next_url"))
        raise PriceSourceError(f"分页超过 {MAX_PAGES} 页，停止")

    def overview(self, ticker: str, day: date) -> dict[str, Any] | None:
        check_query_date(day)
        path = f"/v3/reference/tickers/{urllib.parse.quote(ticker, safe='')}"
        data = self._get(f"{BASE_URL}{path}?{urllib.parse.urlencode({'date': day.isoformat()})}")
        if data is None:
            return None
        results = data.get("results")
        return results if isinstance(results, dict) else None

    # ---- 内部 ----

    def _next(self, next_url: Any) -> str | None:
        """next_url 只接受本主机的 https 地址；去掉其中可能带的 apiKey 参数。"""
        if not next_url:
            return None
        parts = urllib.parse.urlsplit(str(next_url))
        if parts.scheme != "https" or parts.hostname != HOST:
            raise PriceSourceError("next_url 不是 api.massive.com 的 https 地址，停止")
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
            url, headers={"Authorization": f"Bearer {self._key}", "Accept": "application/json"}
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
