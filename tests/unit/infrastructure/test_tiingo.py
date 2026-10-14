import io
import urllib.error
from datetime import date

import pytest

from stock_risk.application.ports import PriceSourceError
from stock_risk.domain.market_data import ActionType
from stock_risk.infrastructure.tiingo.client import TiingoPriceSource
from stock_risk.infrastructure.tiingo.mapper import to_history


def row(d: str, div: float = 0.0, split: float = 1.0) -> dict:
    return {
        "date": f"{d}T00:00:00.000Z",
        "open": 10,
        "high": 11,
        "low": 9,
        "close": 10.5,
        "volume": 1000,
        "adjOpen": 5,
        "adjHigh": 5.5,
        "adjLow": 4.5,
        "adjClose": 5.25,
        "divCash": div,
        "splitFactor": split,
    }


def test_mapper_parses_bars_and_actions() -> None:
    h = to_history([row("2014-06-09", split=7.0), row("2014-08-07", div=0.47), row("2014-06-06")])
    assert [b.day for b in h.bars] == [date(2014, 6, 6), date(2014, 6, 9), date(2014, 8, 7)]
    assert h.bars[0].close == 10.5 and h.bars[0].adj_close == 5.25
    assert [(a.type, a.value) for a in h.actions] == [
        (ActionType.SPLIT, 7.0),
        (ActionType.DIVIDEND, 0.47),
    ]


def test_mapper_rejects_duplicate_dates() -> None:
    with pytest.raises(ValueError):
        to_history([row("2014-06-09"), row("2014-06-09")])


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b""))  # type: ignore[arg-type]


def make_source(responses: list, clock: FakeClock) -> tuple[TiingoPriceSource, list]:
    seen: list = []

    def transport(req):
        seen.append(req)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    src = TiingoPriceSource(
        "k",
        min_interval_sec=75,
        backoff_sec=600,
        transport=transport,
        sleep=clock.sleep,
        clock=clock.now,
    )
    return src, seen


def test_client_throttles_and_sends_token_header() -> None:
    clock = FakeClock()
    src, seen = make_source([[row("2019-01-02")], [row("2019-01-02")]], clock)
    src.fetch("AAPL", date(2019, 1, 1), date(2019, 12, 31))
    src.fetch("MSFT", date(2019, 1, 1), date(2019, 12, 31))
    assert clock.sleeps == [75]
    assert seen[0].get_header("Authorization") == "Token k"
    assert "/aapl/prices?startDate=2019-01-01&endDate=2019-12-31" in seen[0].full_url


def test_client_404_and_not_found_return_none() -> None:
    clock = FakeClock()
    src, _ = make_source([_http_error(404), {"detail": "Error: Ticker 'X' not found"}], clock)
    assert src.fetch("X", date(2019, 1, 1), date(2019, 12, 31)) is None
    assert src.fetch("X", date(2019, 1, 1), date(2019, 12, 31)) is None


def test_client_retries_on_429_then_succeeds() -> None:
    clock = FakeClock()
    src, _ = make_source([_http_error(429), [row("2019-01-02")]], clock)
    h = src.fetch("AAPL", date(2019, 1, 1), date(2019, 12, 31))
    assert h is not None and len(h.bars) == 1
    assert 600 in clock.sleeps


def test_client_auth_error_raises() -> None:
    clock = FakeClock()
    src, _ = make_source([_http_error(401)], clock)
    with pytest.raises(PriceSourceError):
        src.fetch("AAPL", date(2019, 1, 1), date(2019, 12, 31))
