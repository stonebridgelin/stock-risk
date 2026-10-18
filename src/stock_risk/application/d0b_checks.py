"""规范数据与下游读取的一致性核对（T12-D0b 第 2.6 节第 2 项，G-D0b-03、DBr-02）。

两项分开核对，不一致即停止：
- 日线：规范数据经 MarketDataLoader.load 读取后，日线的日期与全部字段与规范数据完全一致
  （没有被二次截掉）；
- 行动：信号读取路径按既定契约返回空的行动序列，不得往里补行动；改由独立的审计路径
  读取输入行动与复权日志，核对原始行动、计算组、生效日与待处理状态。原始除权日早于观察区间起点、
  已按锚日规则处理的行动，不计入比较。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from stock_risk.application.market_data_loader import MarketDataLoader
from stock_risk.application.ports import RawDataReader


@dataclass(frozen=True, slots=True)
class InputAction:
    action_id: str
    original_ex_date: date


def check_bars_consistent(
    loader: MarketDataLoader, reader: RawDataReader, symbol: str
) -> list[str]:
    """返回不一致的说明（空 = 一致）。"""
    problems: list[str] = []
    observed = loader.load(symbol)
    stored = reader.load(symbol).history.bars
    got = observed.history.bars
    if [b.day for b in got] != [b.day for b in stored]:
        missing = sorted({b.day for b in stored} - {b.day for b in got})
        problems.append(f"{symbol}: 下游读取的日线日期与规范数据不同（少 {len(missing)} 天）")
        return problems
    for a, b in zip(got, stored, strict=True):
        if a != b:
            problems.append(f"{symbol} {a.day}: 下游读取的字段与规范数据不同")
    if observed.history.actions:
        problems.append(f"{symbol}: 信号读取路径返回了行动（规范数据的契约是空序列）")
    return problems


def check_actions_consistent(
    symbol: str,
    inputs: Sequence[InputAction],
    log: Mapping[str, Any],
    observation_start: date,
    last_bar: date,
) -> list[str]:
    """输入行动（上市期间截取之后）与复权日志的核对。"""
    problems: list[str] = []
    logged = {a["action_id"]: a for a in log["actions"]}
    wanted = {a.action_id: a for a in inputs if a.original_ex_date >= observation_start}
    for aid in sorted(set(wanted) - set(logged)):
        problems.append(f"{symbol}: 输入行动 {aid} 不在复权日志中")
    extra = [aid for aid in logged if aid not in {a.action_id for a in inputs}]
    for aid in sorted(extra):
        problems.append(f"{symbol}: 复权日志中的行动 {aid} 不在输入中")
    for aid, a in wanted.items():
        rec = logged.get(aid)
        if rec is not None and rec["original_ex_date"] != a.original_ex_date:
            problems.append(f"{symbol}: {aid} 的原始除权日与输入不同")
    groups = {g["group_id"]: g for g in log["groups"]}
    for aid, rec in logged.items():
        gid = rec.get("group_id")
        if gid is not None and aid not in groups.get(gid, {}).get("members", []):
            problems.append(f"{symbol}: {aid} 的计算组 {gid} 不含它")
    for g in groups.values():
        e, status = g["effective_day"], g["status"]
        if status == "PENDING":
            if e is not None or g["original_ex_date"] <= last_bar:
                problems.append(f"{symbol}: 待处理组 {g['group_id']} 的生效日或除权日不对")
        elif e is None or e < g["original_ex_date"] or e > last_bar:
            problems.append(f"{symbol}: 组 {g['group_id']} 的生效日 {e} 不对")
    return problems
