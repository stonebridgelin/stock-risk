"""运行配置：读取项目根目录 .env，环境变量优先。密钥只在此处读取，不得写入日志或文件。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        values[key.strip()] = val
    return values


@dataclass(frozen=True)
class Settings:
    project_root: Path
    tiingo_api_key: str = field(repr=False)
    fred_api_key: str = field(repr=False)
    mysql_verify_url: str = field(repr=False)

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def exploration_universe_file(self) -> Path:
        return self.project_root / "config" / "universe_exploration.csv"

    @property
    def symbol_segments_file(self) -> Path:
        return self.project_root / "config" / "symbol_segments.csv"

    @property
    def listing_periods_file(self) -> Path:
        return self.project_root / "config" / "listing_periods.csv"

    @property
    def audit_output_dir(self) -> Path:
        return self.project_root / "output" / "audit"


def load_settings(project_root: Path = PROJECT_ROOT) -> Settings:
    file_values = _parse_env_file(project_root / ".env")

    def get(key: str) -> str:
        return os.environ.get(key) or file_values.get(key, "")

    return Settings(
        project_root=project_root,
        tiingo_api_key=get("TIINGO_API_KEY"),
        fred_api_key=get("FRED_API_KEY"),
        mysql_verify_url=get("MYSQL_VERIFY_URL"),
    )
