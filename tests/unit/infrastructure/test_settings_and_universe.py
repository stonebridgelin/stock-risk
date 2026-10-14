from pathlib import Path

import pytest

from stock_risk.config.settings import PROJECT_ROOT, load_settings
from stock_risk.infrastructure.storage.csv_universe_provider import CsvUniverseProvider


def test_settings_reads_env_file_and_env_overrides(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text('TIINGO_API_KEY="abc"\nFRED_API_KEY=f\n# c\n', "utf-8")
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.setenv("FRED_API_KEY", "env")
    s = load_settings(tmp_path)
    assert s.tiingo_api_key == "abc" and s.fred_api_key == "env"
    assert "abc" not in repr(s)


def test_exploration_universe_file_is_valid() -> None:
    members = CsvUniverseProvider(PROJECT_ROOT / "config" / "universe_exploration.csv").load()
    assert len(members) == 41
    fb = next(m for m in members if m.symbol == "FB")
    assert fb.source_candidates == ("FB", "META")


def test_universe_rejects_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "u.csv"
    p.write_text("symbol,source_candidates,category,note\nA,A,x,\nA,A,x,\n", "utf-8")
    with pytest.raises(ValueError):
        CsvUniverseProvider(p).load()
