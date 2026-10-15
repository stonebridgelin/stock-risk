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
    # fetch-batch 额度（T08d-1）：.env 或环境变量 TIINGO_PER_HOUR / TIINGO_PER_DAY /
    # TIINGO_UNIQUE_PER_MONTH；未设置时由命令行或默认值（免费档）决定
    tiingo_per_hour: int | None = None
    tiingo_per_day: int | None = None
    tiingo_unique_per_month: int | None = None
    # Massive 试验（T08d-2a）：密钥 MASSIVE_API_KEY；限频 MASSIVE_PER_MINUTE（未设置时为免费档 5）
    massive_api_key: Secret = field(default_factory=lambda: Secret(""), repr=False)
    massive_per_minute: int | None = None

    @property
    def massive_probe_dir(self) -> Path:
        return self.project_root / "data" / "massive_probe"

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def selection_raw_dir(self) -> Path:
        """选参集日线的默认目录（与探索集的 data/raw 分开）。"""
        return self.project_root / "data" / "raw_selection"

    @property
    def universe_probe_dir(self) -> Path:
        return self.project_root / "data" / "universe_probe"

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

    def get_int(key: str) -> int | None:
        v = get(key).strip()
        return int(v) if v else None

    return Settings(
        project_root=project_root,
        tiingo_api_key=Secret(get("TIINGO_API_KEY")),
        fred_api_key=Secret(get("FRED_API_KEY")),
        mysql_verify_url=Secret(get("MYSQL_VERIFY_URL")),
        tiingo_per_hour=get_int("TIINGO_PER_HOUR"),
        tiingo_per_day=get_int("TIINGO_PER_DAY"),
        tiingo_unique_per_month=get_int("TIINGO_UNIQUE_PER_MONTH"),
        massive_api_key=Secret(get("MASSIVE_API_KEY")),
        massive_per_minute=get_int("MASSIVE_PER_MINUTE"),
    )
