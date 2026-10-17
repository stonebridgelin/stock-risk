"""T08d-2b 第 0.4 节：Massive 免费档与付费档密钥的配置对齐（假密钥，不访问网络）。

付费档命令只用 MASSIVE_ADVANCED_API_KEY、不回退；免费档命令照旧用 MASSIVE_API_KEY；
环境变量优先于 .env；
repr、日志与异常中不出现密钥；.env.example 的变量集合等于 load_settings 读取的集合（用代码比对）。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from stock_risk import bootstrap
from stock_risk.bootstrap import massive_free_key, massive_paid_key
from stock_risk.config import settings as settings_module
from stock_risk.config.settings import load_settings

FREE, PAID = "fake-free-key-111", "fake-paid-key-222"
MASSIVE_VARS = ("MASSIVE_API_KEY", "MASSIVE_ADVANCED_API_KEY", "MASSIVE_PER_MINUTE")
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for v in MASSIVE_VARS:
        monkeypatch.delenv(v, raising=False)
    return tmp_path


def _env(root: Path, **values: str) -> None:
    (root / ".env").write_text("".join(f"{k}={v}\n" for k, v in values.items()), "utf-8")


def test_only_advanced_key(root: Path) -> None:
    _env(root, MASSIVE_ADVANCED_API_KEY=PAID)
    s = load_settings(root)
    assert massive_paid_key(s) == PAID
    assert not s.massive_api_key
    with pytest.raises(SystemExit) as e:
        massive_free_key(s)
    assert "MASSIVE_API_KEY" in str(e.value) and PAID not in str(e.value)


def test_only_free_key_paid_stops_without_fallback(root: Path) -> None:
    _env(root, MASSIVE_API_KEY=FREE)
    s = load_settings(root)
    with pytest.raises(SystemExit) as e:
        massive_paid_key(s)
    msg = str(e.value)
    assert "MASSIVE_ADVANCED_API_KEY" in msg and FREE not in msg
    assert massive_free_key(s) == FREE


@pytest.mark.parametrize("value", [None, "", "   "], ids=["unset", "empty", "blank"])
def test_blank_advanced_key_stops_at_entry(
    root: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    """第 0.2a 节：未设置、空串、只有空白字符都视为没有密钥，在命令入口报错停止，信息只含变量名。"""
    _env(root, MASSIVE_API_KEY=FREE)
    if value is not None:
        monkeypatch.setenv("MASSIVE_ADVANCED_API_KEY", value)
    s = load_settings(root)
    with pytest.raises(SystemExit) as e:
        massive_paid_key(s)
    assert "MASSIVE_ADVANCED_API_KEY" in str(e.value) and FREE not in str(e.value)
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "SPY.csv").write_text("date,close\n2008-01-02,1\n", "utf-8")
    with pytest.raises(SystemExit, match="MASSIVE_ADVANCED_API_KEY"):
        bootstrap.build_probe_massive_paid_service(s, root / "out")
    assert massive_free_key(s) == FREE  # 其他密钥变量的行为不变


def test_both_keys_each_command_uses_its_own(root: Path) -> None:
    _env(root, MASSIVE_API_KEY=FREE, MASSIVE_ADVANCED_API_KEY=PAID)
    s = load_settings(root)
    assert massive_paid_key(s) == PAID and massive_free_key(s) == FREE


def test_environment_overrides_env_file(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(root, MASSIVE_API_KEY=FREE, MASSIVE_ADVANCED_API_KEY=PAID)
    monkeypatch.setenv("MASSIVE_ADVANCED_API_KEY", "fake-paid-from-env")
    monkeypatch.setenv("MASSIVE_API_KEY", "fake-free-from-env")
    s = load_settings(root)
    assert massive_paid_key(s) == "fake-paid-from-env"
    assert massive_free_key(s) == "fake-free-from-env"


def test_keys_not_in_repr(root: Path) -> None:
    _env(root, MASSIVE_API_KEY=FREE, MASSIVE_ADVANCED_API_KEY=PAID)
    s = load_settings(root)
    text = repr(s) + str(s.massive_advanced_api_key) + f"{s.massive_advanced_api_key}"
    assert FREE not in text and PAID not in text


def test_paid_service_uses_advanced_key_only(root: Path) -> None:
    """bootstrap 组装付费档服务时只把 MASSIVE_ADVANCED_API_KEY 交给适配器。"""
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "SPY.csv").write_text("date,close\n2008-01-02,1\n", "utf-8")
    _env(root, MASSIVE_API_KEY=FREE, MASSIVE_ADVANCED_API_KEY=PAID)
    svc = bootstrap.build_probe_massive_paid_service(load_settings(root), root / "out")
    client = svc._source  # noqa: SLF001 —— 只核对传给适配器的密钥
    assert client._key == PAID  # type: ignore[attr-defined]  # noqa: SLF001
    _env(root, MASSIVE_API_KEY=FREE)
    with pytest.raises(SystemExit, match="MASSIVE_ADVANCED_API_KEY"):
        bootstrap.build_probe_massive_paid_service(load_settings(root), root / "out")


def _keys_read_by_load_settings() -> set[str]:
    """从 load_settings 的源码中取出 get(...) / get_int(...) 的变量名（不手写清单）。"""
    tree = ast.parse(inspect.getsource(settings_module.load_settings))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("get", "get_int")
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }


def test_env_example_lists_exactly_the_loaded_variables() -> None:
    example = {
        line.split("=", 1)[0].strip()
        for line in (REPO / ".env.example").read_text("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    values = [
        line.split("=", 1)[1].strip()
        for line in (REPO / ".env.example").read_text("utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    loaded = _keys_read_by_load_settings()
    assert "MASSIVE_ADVANCED_API_KEY" in loaded
    assert example == loaded
    assert all(v == "" for v in values)  # 样例只写空值
