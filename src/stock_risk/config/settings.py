"""运行配置：读取项目根目录 .env，环境变量优先。密钥只在此处读取，不得写入日志或文件。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


class Secret:
    """密钥（T08a D-05）：repr、str、格式化与 dataclasses.asdict 后的副本都只显示 ***。

    取值只能显式调用 reveal()。json 不能直接序列化本类型（报 TypeError，不会写出明文）；
    以 default=str 序列化时写出 ***。
    """

    __slots__ = ("_value",)
    MASK = "***"

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __repr__(self) -> str:
        return self.MASK

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return self.MASK

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Secret) and other._value == self._value

    def __hash__(self) -> int:
        return hash((Secret, self._value))

    def __reduce__(self) -> tuple[Any, ...]:
        raise TypeError("密钥不得被序列化")

    def __deepcopy__(self, memo: dict[int, Any]) -> Secret:
        return self

    def __copy__(self) -> Secret:
        return self


@dataclass(frozen=True)
class Settings:
    project_root: Path
    tiingo_api_key: Secret = field(repr=False)
    fred_api_key: Secret = field(repr=False)
    mysql_verify_url: Secret = field(repr=False)

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
    def price_corrections_file(self) -> Path:
        return self.project_root / "config" / "price_corrections.csv"

    @property
    def audit_output_dir(self) -> Path:
        return self.project_root / "output" / "audit"

    @property
    def calendar_check_output_dir(self) -> Path:
        return self.project_root / "output" / "calendar_check"

    @property
    def compare_output_dir(self) -> Path:
        return self.project_root / "output" / "compare"

    @property
    def replay_output_dir(self) -> Path:
        return self.project_root / "output" / "replay"

    @property
    def ranges_output_dir(self) -> Path:
        return self.project_root / "output" / "ranges"


def load_settings(project_root: Path = PROJECT_ROOT) -> Settings:
    file_values = _parse_env_file(project_root / ".env")

    def get(key: str) -> str:
        return os.environ.get(key) or file_values.get(key, "")

    return Settings(
        project_root=project_root,
        tiingo_api_key=Secret(get("TIINGO_API_KEY")),
        fred_api_key=Secret(get("FRED_API_KEY")),
        mysql_verify_url=Secret(get("MYSQL_VERIFY_URL")),
    )
