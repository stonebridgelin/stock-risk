"""一次回放（一个对象 × 一套）的紧凑视图，供对照分析使用。纯函数。

统计期间与 15 号文档第 5 节相同：2010-01-01 至 2019-12-31（2009 年为预热期）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from stock_risk.application.replay import SymbolReplay
from stock_risk.application.replay_outputs import EpisodeRow, closes_by_day, episode_rows
from stock_risk.domain.lifecycle.model import DataStatus, Event, EventType, Level

PERIOD_START = date(2010, 1, 1)
PERIOD_END = date(2019, 12, 31)

NORMAL, WATCH, ALERT = "正常", "关注", "警戒"
INSUFFICIENT, MISSING, BREAK = "不可计算", "缺价冻结", "数据中断"


@dataclass(frozen=True, slots=True)
class RunView:
    target: str
    set_id: str
    days: tuple[date, ...]
    index: dict[date, int]
    status: tuple[str, ...]
    level: tuple[int, ...]  # 0 无预警、1 关注、2 警戒（日末）
    episode: tuple[str | None, ...]  # 日末有预警时的 episode_id
    computable: tuple[bool, ...]  # 有日线、历史段充足、在回放区间内
    valid: tuple[bool, ...]  # 当日有日线
    data_break: tuple[bool, ...]
    d_ep: tuple[float | None, ...]
    dd_ep: tuple[float | None, ...]
    r2_filtered: tuple[bool, ...]
    closes: dict[date, float]
    events: tuple[Event, ...]
    episodes: tuple[EpisodeRow, ...]

    def in_period(self, i: int) -> bool:
        return PERIOD_START <= self.days[i] <= PERIOD_END

    def events_of(self, kind: EventType) -> list[Event]:
        return [e for e in self.events if e.type is kind]


def _status(data_status: DataStatus, level: Level | None) -> str:
    if data_status is DataStatus.MISSING:
        return MISSING
    if data_status is DataStatus.DATA_BREAK:
        return BREAK
    if data_status is DataStatus.INSUFFICIENT:
        return INSUFFICIENT
    if level is Level.ALERT:
        return ALERT
    if level is Level.WATCH:
        return WATCH
    return NORMAL


def build_view(rep: SymbolReplay, set_id: str) -> RunView:
    recs = [r.record for r in rep.result.results]
    days = tuple(r.day for r in recs)
    return RunView(
        target=rep.target.label,
        set_id=set_id,
        days=days,
        index={d: i for i, d in enumerate(days)},
        status=tuple(_status(r.data_status, r.level) for r in recs),
        level=tuple(int(r.level) if r.level is not None else 0 for r in recs),
        episode=tuple(r.episode_id if r.level is not None else None for r in recs),
        computable=tuple(r.data_status is DataStatus.HAS_BAR and r.in_replay for r in recs),
        valid=tuple(r.data_status in (DataStatus.HAS_BAR, DataStatus.INSUFFICIENT) for r in recs),
        data_break=tuple(r.data_status is DataStatus.DATA_BREAK for r in recs),
        d_ep=tuple(r.d_ep for r in recs),
        dd_ep=tuple(r.dd_ep for r in recs),
        r2_filtered=tuple(bool(r.r2_filtered) for r in recs),
        closes=closes_by_day(rep),
        events=rep.result.events,
        episodes=tuple(episode_rows(rep)),
    )


@dataclass(frozen=True, slots=True)
class Window:
    """事件日之后 n 个有效交易日的窗口（不含事件日）。"""

    end_index: int | None  # 窗口最后一日的下标；窗口不完整时为能到达的最后一日
    complete: bool
    day_indices: tuple[int, ...]


def window_after(view: RunView, i: int, n: int) -> Window:
    """窗口不完整：观察终点之前不足 n 个有效交易日，或窗口内出现数据中断。"""
    picked: list[int] = []
    j = i + 1
    while j < len(view.days) and len(picked) < n:
        if view.data_break[j]:
            return Window(picked[-1] if picked else None, False, tuple(picked))
        if view.valid[j]:
            picked.append(j)
        j += 1
    return Window(picked[-1] if picked else None, len(picked) == n, tuple(picked))
