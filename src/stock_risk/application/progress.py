"""进度回调（T08a D-03）：application 层默认不输出，由 CLI 注入 print。"""

from __future__ import annotations

from collections.abc import Callable

ProgressFn = Callable[[str], None]


def quiet(_: str) -> None:
    """默认进度回调：不输出。"""
