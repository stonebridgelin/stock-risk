"""T09-C1 C-01：默认传输层不跟随重定向（真实的默认传输层，只连 127.0.0.1 上的本地服务）。

第一个本地服务对 301、302、303、307、308 都返回指向第二个本地服务的 Location；
断言：抛出数据源错误；第二个服务没有收到任何请求；错误信息中不含认证头或密钥。
不发任何外部网络请求。
"""

from __future__ import annotations

import threading
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from stock_risk.application.ports import PriceSourceError
from stock_risk.infrastructure.massive import client as massive_client

KEY = "fake-redirect-key-777"


class _Target(BaseHTTPRequestHandler):
    hits: list[str] = []

    def do_GET(self) -> None:  # noqa: N802
        type(self).hits.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "OK", "results": []}')

    def log_message(self, *args: object) -> None:  # 不输出访问日志
        return


def _redirector(target_port: int) -> type[BaseHTTPRequestHandler]:
    class _Redirect(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            code = int(self.path.strip("/").split("?")[0])
            self.send_response(code)
            self.send_header("Location", f"http://127.0.0.1:{target_port}/landing?x=1")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            return

    return _Redirect


def _serve(handler: type[BaseHTTPRequestHandler]) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def servers() -> Iterator[tuple[HTTPServer, HTTPServer]]:
    _Target.hits = []
    target = _serve(_Target)
    source = _serve(_redirector(target.server_address[1]))
    yield source, target
    for s in (source, target):
        s.shutdown()
        s.server_close()


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_default_transport_refuses_redirect(
    servers: tuple[HTTPServer, HTTPServer], code: int
) -> None:
    source, _ = servers
    req = urllib.request.Request(
        f"http://127.0.0.1:{source.server_address[1]}/{code}",
        headers={
            "Authorization": f"Bearer {KEY}",
            "Accept": "application/json",
            "User-Agent": massive_client.USER_AGENT,
        },
    )
    with pytest.raises(PriceSourceError) as e:
        massive_client._default_transport(req)  # noqa: SLF001 —— 真实的默认传输层
    assert isinstance(e.value, massive_client.RedirectRefused)
    assert e.value.http_status == code and e.value.target_host == "127.0.0.1"
    assert _Target.hits == []  # 第二个服务没有收到任何请求
    text = str(e.value) + repr(e.value)
    assert KEY not in text and "Authorization" not in text and "Bearer" not in text
    assert str(code) in text


def test_target_server_reachable_directly(servers: tuple[HTTPServer, HTTPServer]) -> None:
    """对照：第二个服务本身可以访问，说明上面的 0 次请求不是因为它不可达。"""
    _, target = servers
    req = urllib.request.Request(f"http://127.0.0.1:{target.server_address[1]}/direct")
    assert massive_client._default_transport(req) == {"status": "OK", "results": []}  # noqa: SLF001
    assert _Target.hits == ["/direct"]
