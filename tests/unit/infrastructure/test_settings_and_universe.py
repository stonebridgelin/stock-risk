from pathlib import Path

import pytest

from stock_risk.config.settings import PROJECT_ROOT, load_settings
from stock_risk.infrastructure.storage.csv_universe_provider import CsvUniverseProvider


def test_settings_reads_env_file_and_env_overrides(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text('TIINGO_API_KEY="abc"\nFRED_API_KEY=f\n# c\n', "utf-8")
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.setenv("FRED_API_KEY", "env")
    s = load_settings(tmp_path)
    assert s.tiingo_api_key.reveal() == "abc" and s.fred_api_key.reveal() == "env"
    assert "abc" not in repr(s)


def test_secret_never_leaks(tmp_path: Path, monkeypatch) -> None:
    """D-05：repr、str、格式化、asdict、json 都不泄露密钥（测试值为虚构字符串）。"""
    import json
    import pickle
    from dataclasses import asdict

    from stock_risk.config.settings import Secret

    for key in ("TIINGO_API_KEY", "FRED_API_KEY", "MYSQL_VERIFY_URL"):
        monkeypatch.setenv(key, f"fake-{key.lower()}-value")
    s = load_settings(tmp_path)
    secrets = [f"fake-{k}-value" for k in ("tiingo_api_key", "fred_api_key", "mysql_verify_url")]
    d = asdict(s)
    texts = [
        repr(s),
        str(s),
        repr(d),
        str(d),
        f"{s.tiingo_api_key}",
        f"{s.fred_api_key:>10}",
        json.dumps(d, default=str),
    ]
    for text in texts:
        assert not any(x in text for x in secrets), text
    assert d["tiingo_api_key"] == s.tiingo_api_key and str(d["tiingo_api_key"]) == "***"
    assert json.dumps({"k": Secret("fake")}, default=str) == '{"k": "***"}'
    with pytest.raises(TypeError):
        json.dumps({"k": Secret("fake")})  # 不能直接序列化，不会写出明文
    with pytest.raises(TypeError):
        pickle.dumps(Secret("fake"))
    assert bool(Secret("")) is False and bool(Secret("x")) is True


def test_exploration_universe_file_is_valid() -> None:
    members = CsvUniverseProvider(PROJECT_ROOT / "config" / "universe_exploration.csv").load()
    assert len(members) == 41
    fb = next(m for m in members if m.symbol == "FB")
    assert fb.source_candidates == ("FB", "META")


def test_universe_rejects_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "u.csv"
    p.write_text(
        "symbol,source_candidates,category,role,note\nA,A,x,signal_target,\nA,A,x,signal_target,\n",
        "utf-8",
    )
    with pytest.raises(ValueError):
        CsvUniverseProvider(p).load()


def test_universe_roles_are_14_reference_and_27_signal() -> None:
    from stock_risk.domain.universe import Role

    members = CsvUniverseProvider(PROJECT_ROOT / "config" / "universe_exploration.csv").load()
    roles = {m.symbol: m.role for m in members}
    reference = {s for s, r in roles.items() if r is Role.REFERENCE_ONLY}
    assert len(reference) == 14 and sum(r is Role.SIGNAL_TARGET for r in roles.values()) == 27
    assert reference == {m.symbol for m in members if m.category == "etf"}


@pytest.mark.parametrize("role", ["", "etf", "SIGNAL_TARGET"])
def test_universe_rejects_bad_role(tmp_path: Path, role: str) -> None:
    p = tmp_path / "u.csv"
    p.write_text(f"symbol,source_candidates,category,role,note\nA,A,x,{role},\n", "utf-8")
    with pytest.raises(ValueError):
        CsvUniverseProvider(p).load()
