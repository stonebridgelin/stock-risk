"""T08d-2c 第 1 节：付费档主机配置、密钥不交叉与来源记录（假密钥、假传输层，不访问网络）。

- MASSIVE_ADVANCED_BASE_URL 未设置或空白取官方主机；只允许两个取值，其他值在入口停止；
- next_url 主机与配置主机不同时停止；
- 免费档命令不受影响：仍只连 api.massive.com、只用 MASSIVE_API_KEY；
- 两把密钥各自只发往自己的主机，且只在请求头中；
- 付费档输出带 source_host；已有其他主机记录的目录拒绝运行。
"""

from __future__ import annotations

import csv
import json
import urllib.parse
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from stock_risk import bootstrap
from stock_risk.application.ports import PriceSourceError
from stock_risk.bootstrap import massive_paid_base_url
from stock_risk.config.settings import load_settings
from stock_risk.domain.massive_probe import ListQuery
from stock_risk.infrastructure.massive.client import MassiveReference
from stock_risk.infrastructure.storage.massive_paid_files import MassivePaidFiles
from stock_risk.infrastructure.storage.massive_source_host import (
    SourceHostConflict,
    recorded_hosts,
)

FREE, PAID = "fake-free-key-111", "fake-paid-key-222"
OFFICIAL, RELAY = "https://api.massive.com", "https://relaynex.vip"
VARS = (
    "MASSIVE_API_KEY",
    "MASSIVE_ADVANCED_API_KEY",
    "MASSIVE_ADVANCED_BASE_URL",
    "MASSIVE_PER_MINUTE",
)
Q = ListQuery(date(2015, 6, 1), "XNYS", "CS", True)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for v in VARS:
        monkeypatch.delenv(v, raising=False)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "SPY.csv").write_text("date,close\n2008-01-02,1\n", "utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "universe_exploration.csv").write_text(
        "symbol,source_candidates,category,role,note\nMSFT,MSFT,x,signal_target,\n", "utf-8"
    )
    return tmp_path


def _env(root: Path, **values: str) -> None:
    (root / ".env").write_text("".join(f"{k}={v}\n" for k, v in values.items()), "utf-8")


class Capture:
    """假传输层：记录请求，返回给定的响应。"""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []

    def __call__(self, req):  # type: ignore[no-untyped-def]
        self.requests.append(req)
        return self.responses.pop(0) if self.responses else {"results": []}


def _wire(client: Any, transport: Capture) -> None:
    client._transport = transport  # noqa: SLF001 —— 替换为假传输层
    client._sleep = lambda s: None  # noqa: SLF001
    client._clock = lambda: 0.0  # noqa: SLF001


# ---- 取值 ----


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_default_is_official_host(root: Path, value: str | None) -> None:
    if value is None:
        _env(root, MASSIVE_ADVANCED_API_KEY=PAID)
    else:
        _env(root, MASSIVE_ADVANCED_API_KEY=PAID, MASSIVE_ADVANCED_BASE_URL=value)
    s = load_settings(root)
    assert massive_paid_base_url(s) == OFFICIAL
    svc = bootstrap.build_probe_massive_paid_service(s, root / "out")
    assert svc._source.source_host == "api.massive.com"  # type: ignore[attr-defined]  # noqa: SLF001


@pytest.mark.parametrize("value", [OFFICIAL, RELAY])
def test_allowed_values(root: Path, value: str) -> None:
    _env(root, MASSIVE_ADVANCED_API_KEY=PAID, MASSIVE_ADVANCED_BASE_URL=value)
    s = load_settings(root)
    assert massive_paid_base_url(s) == value
    svc = bootstrap.build_verify_massive_relay_service(s)
    assert svc.source_host == urllib.parse.urlsplit(value).hostname


@pytest.mark.parametrize(
    "value",
    ["https://evil.example", "http://relaynex.vip", "https://relaynex.vip/", "relaynex.vip"],
)
def test_illegal_value_stops_at_entry(root: Path, value: str) -> None:
    _env(root, MASSIVE_ADVANCED_API_KEY=PAID, MASSIVE_ADVANCED_BASE_URL=value)
    s = load_settings(root)
    with pytest.raises(SystemExit, match="MASSIVE_ADVANCED_BASE_URL"):
        massive_paid_base_url(s)
    with pytest.raises(SystemExit, match="MASSIVE_ADVANCED_BASE_URL"):
        bootstrap.build_probe_massive_paid_service(s, root / "out")
    with pytest.raises(SystemExit, match="MASSIVE_ADVANCED_BASE_URL"):
        bootstrap.build_verify_massive_relay_service(s)
    with pytest.raises(ValueError):
        MassiveReference(PAID, base_url=value)


# ---- 翻页地址 ----


def test_next_url_other_host_stops() -> None:
    t = Capture({"results": [{"ticker": "A"}], "next_url": f"{OFFICIAL}/v3/reference/tickers?x=1"})
    c = MassiveReference(PAID, base_url=RELAY)
    _wire(c, t)
    with pytest.raises(PriceSourceError, match="relaynex.vip"):
        c.list_tickers(Q)
    assert len(t.requests) == 1  # 不向官方主机发请求


def test_next_url_same_host_drops_api_key() -> None:
    t = Capture(
        {
            "results": [{"ticker": "A"}],
            "next_url": f"{RELAY}/v3/reference/tickers?cursor=c&apiKey=k",
        },
        {"results": [{"ticker": "B"}]},
    )
    c = MassiveReference(PAID, base_url=RELAY)
    _wire(c, t)
    assert [r["ticker"] for r in c.list_tickers(Q)] == ["A", "B"]
    second = urllib.parse.urlsplit(t.requests[1].full_url)
    assert second.hostname == "relaynex.vip" and "apikey" not in second.query.lower()


# ---- 密钥不交叉 ----


def test_each_key_only_to_its_own_host(root: Path) -> None:
    _env(
        root,
        MASSIVE_API_KEY=FREE,
        MASSIVE_ADVANCED_API_KEY=PAID,
        MASSIVE_ADVANCED_BASE_URL=RELAY,
    )
    s = load_settings(root)
    free = bootstrap.build_probe_massive_service(s, root / "free")._source  # noqa: SLF001
    paid = bootstrap.build_verify_massive_relay_service(s)._source  # noqa: SLF001
    tf, tp = Capture(), Capture()
    _wire(free, tf)
    _wire(paid, tp)
    free.list_tickers(Q)
    paid.list_tickers(Q)
    paid.overview("MSFT", date(2019, 6, 3))
    for transport, host, key, other in (
        (tf, "api.massive.com", FREE, PAID),
        (tp, "relaynex.vip", PAID, FREE),
    ):
        assert transport.requests
        for req in transport.requests:
            assert urllib.parse.urlsplit(req.full_url).hostname == host
            assert req.get_header("Authorization") == f"Bearer {key}"
            assert key not in req.full_url and other not in req.full_url


def test_every_request_has_fixed_user_agent(root: Path) -> None:
    """中转服务前的 Cloudflare 拦截 Python-urllib（403，error code 1010）：
    免费档与付费档都带固定标识。"""
    _env(
        root,
        MASSIVE_API_KEY=FREE,
        MASSIVE_ADVANCED_API_KEY=PAID,
        MASSIVE_ADVANCED_BASE_URL=RELAY,
    )
    s = load_settings(root)
    free = bootstrap.build_probe_massive_service(s, root / "free")._source  # noqa: SLF001
    paid = bootstrap.build_verify_massive_relay_service(s)._source  # noqa: SLF001
    tf = Capture()
    tp = Capture(
        {"results": [{"ticker": "A"}], "next_url": f"{RELAY}/v3/reference/tickers?cursor=c"},
        {"results": []},
    )
    _wire(free, tf)
    _wire(paid, tp)
    free.list_tickers(Q)
    paid.list_tickers(Q)  # 含翻页请求
    paid.aggs("MSFT", date(2019, 1, 2), date(2019, 1, 31), False)
    paid.overview("MSFT", date(2019, 6, 3))
    requests = tf.requests + tp.requests
    assert len(requests) == 5
    for req in requests:
        assert req.get_header("User-agent") == "stock-risk/1.0"
        headers = " ".join(f"{k}: {v}" for k, v in req.header_items())
        assert "Python-urllib" not in headers
        assert req.get_header("Authorization") in (f"Bearer {FREE}", f"Bearer {PAID}")


def test_free_commands_ignore_base_url(root: Path) -> None:
    _env(root, MASSIVE_API_KEY=FREE, MASSIVE_ADVANCED_BASE_URL=RELAY)
    s = load_settings(root)
    probe = root / "data" / "universe_probe"
    probe.mkdir(parents=True)
    (probe / "tickers_2008_2019.csv").write_text("ticker\n", "utf-8")  # 追加探查的输入
    for build in (
        bootstrap.build_probe_massive_service,
        bootstrap.build_probe_massive_followup_service,
    ):
        client = build(s, root / "free")._source  # type: ignore[attr-defined]  # noqa: SLF001
        assert client.source_host == "api.massive.com"
        assert client._key == FREE  # noqa: SLF001


# ---- 来源记录 ----


def test_paid_files_record_source_host(tmp_path: Path) -> None:
    store = MassivePaidFiles(tmp_path, "relaynex.vip")
    store.save_progress({"prices": {}})
    store.save_list("2019-06-03_active", [{"ticker": "A"}])
    store.write_outputs({}, [], {"x": 1})
    progress = json.loads((tmp_path / "_progress.json").read_text("utf-8"))
    assert progress["source_host"] == "relaynex.vip"
    with (tmp_path / "lists" / "lists_2019-06-03_active.csv").open(encoding="utf-8") as f:
        assert [r["source_host"] for r in csv.DictReader(f)] == ["relaynex.vip"]
    assert recorded_hosts(tmp_path) == {"relaynex.vip"}


def test_directory_with_other_host_refused(root: Path) -> None:
    paid = root / "data" / "massive_probe" / "paid"
    MassivePaidFiles(paid, "api.massive.com").save_progress({})
    with pytest.raises(SourceHostConflict):
        MassivePaidFiles(paid, "relaynex.vip")
    MassivePaidFiles(paid, "api.massive.com")  # 同一主机可以续写
    _env(root, MASSIVE_ADVANCED_API_KEY=PAID, MASSIVE_ADVANCED_BASE_URL=RELAY)
    s = load_settings(root)
    with pytest.raises(SystemExit, match="api.massive.com"):
        bootstrap.build_probe_massive_paid_service(s, paid)
    with pytest.raises(SystemExit, match="api.massive.com"):
        bootstrap.build_verify_massive_relay_service(s)  # verify/ 在付费档目录内，整体检查


def test_verify_subdir_record_blocks_paid_dir(root: Path) -> None:
    """核验输出（verify/ 子目录）中的记录同样阻止付费档目录换主机。"""
    verify = root / "data" / "massive_probe" / "paid" / "verify"
    verify.mkdir(parents=True)
    (verify / "coverage.csv").write_text("name,source_host\naggs_raw,relaynex.vip\n", "utf-8")
    _env(root, MASSIVE_ADVANCED_API_KEY=PAID)  # 官方主机
    with pytest.raises(SystemExit, match="relaynex.vip"):
        bootstrap.build_probe_massive_paid_service(
            load_settings(root), root / "data" / "massive_probe" / "paid"
        )


def test_legacy_files_without_host_are_not_records(tmp_path: Path) -> None:
    """T08d-2c 之前写出的文件（例如改名保存的 401 进度）没有 source_host，不视为任何主机的记录。"""
    (tmp_path / "_progress.401-20261017.json").write_text('{"requests": 1}', "utf-8")
    (tmp_path / "samples.csv").write_text("group,ticker\n", "utf-8")
    assert recorded_hosts(tmp_path) == set()
    MassivePaidFiles(tmp_path, "relaynex.vip")
