"""17 §1.1 / D-30：季度筛选的纯函数部分。

筛选日 D 的信息截至 D 的前一个市场交易日。处理顺序：
1. 只在生成开发期选参集时排除给定发行人（由调用方传入；验证集传空集合）；
2. 逐只证券检查资格（D-30 ①–⑥，按此顺序记第一个不满足的原因）；
3. 按发行人确定代表股类：已固定的只用该股类（不合格即发行人当季缺席，不换股类）；
   首次出现的取成交额中位数最高的合格股类，并列按稳定标识字典序，此后固定；
4. 代表证券按成交额中位数降序、稳定标识升序排名，取前 N_u 只（不足全部入选）。
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, localcontext
from enum import StrEnum

from stock_risk.domain.calendar import TradingCalendar
from stock_risk.domain.evaluation.decimal_ratio import EVALUATION_CONTEXT
from stock_risk.domain.evaluation.vol_groups import median
from stock_risk.domain.lifecycle.history_segments import split_segments
from stock_risk.domain.numeric import InvalidInputError
from stock_risk.domain.observation import DayPresence

STALE_MAX_DAYS = 5  # 17 §1.1：最近一条有效日线距筛选日不超过 5 个市场交易日
ACTIVE_WINDOW = 63  # 17 §1.1：最近 63 个市场交易日 / 最近 63 个有效日
ACTIVE_MIN_BARS = 57  # 17 §1.1：其中有效日线至少 57 个（90%）
MIN_PRICE = Decimal(5)  # 17 §1.1：原始收盘中位数 ≥ 5 美元


class Eligibility(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    EXCLUDED_ISSUER = "EXCLUDED_ISSUER"
    CLASS = "CLASS"  # ① 证券类别不合格
    STALE_LAST_BAR = "STALE_LAST_BAR"  # ②
    FEW_VALID_IN_63 = "FEW_VALID_IN_63"  # ③
    WARMUP = "WARMUP"  # ④ 本历史段有效日线不足 M_mask
    PRICE_LT_5 = "PRICE_LT_5"  # ⑤
    ZERO_DOLLAR_VOLUME = "ZERO_DOLLAR_VOLUME"  # ⑥


@dataclass(frozen=True, slots=True)
class ScreenBar:
    """一条有效日线的原始收盘与原始成交量。"""

    day: date
    raw_close: Decimal
    raw_volume: Decimal


@dataclass(frozen=True, slots=True)
class ScreenSecurity:
    security_id: str  # 稳定标识
    issuer: str
    class_ok: bool  # ① 由调用方按当时已知信息判定
    bars: tuple[ScreenBar, ...]  # 按日期升序

    def __post_init__(self) -> None:
        days = [b.day for b in self.bars]
        if any(a >= b for a, b in zip(days, days[1:], strict=False)):
            raise InvalidInputError(f"{self.security_id}: 日线须按日期严格升序")


@dataclass(frozen=True, slots=True)
class ScreenResult:
    eligibility: Mapping[str, Eligibility]
    dollar_volume: Mapping[str, Decimal]  # 合格证券的成交额中位数
    ranking: tuple[tuple[str, Decimal], ...]  # 代表证券及其成交额中位数，按名次
    selected: tuple[str, ...]
    rep_map: Mapping[str, str]  # 筛选后的发行人 → 代表股类


def _check(
    sec: ScreenSecurity,
    calendar: TradingCalendar,
    screen_i: int,
    gap_break: int,
    mask_min: int,
) -> tuple[Eligibility, Decimal | None]:
    if not sec.class_ok:
        return Eligibility.CLASS, None
    cutoff = calendar.days[screen_i - 1]
    bars = [b for b in sec.bars if b.day <= cutoff]
    known = {b.day for b in bars}
    if any(not calendar.contains(d) for d in known):
        raise InvalidInputError(f"{sec.security_id}: 日线日期不在市场日历内")
    last_i = calendar.days.index(bars[-1].day) if bars else None
    if last_i is None or screen_i - last_i > STALE_MAX_DAYS:
        return Eligibility.STALE_LAST_BAR, None
    recent = calendar.days[max(0, screen_i - ACTIVE_WINDOW) : screen_i]
    if sum(d in known for d in recent) < ACTIVE_MIN_BARS:
        return Eligibility.FEW_VALID_IN_63, None
    presence = [DayPresence(d, d in known) for d in calendar.days[:screen_i]]
    segments = split_segments(presence, gap_break)
    # 尾部缺价达到断点时，前一交易日已不在任何历史段内（新历史段尚未开始）
    in_break = (screen_i - 1) - last_i >= gap_break + 1
    if in_break or len(segments[-1].bar_days) < mask_min:
        return Eligibility.WARMUP, None
    last = bars[-ACTIVE_WINDOW:]
    if median([b.raw_close for b in last]) < MIN_PRICE:
        return Eligibility.PRICE_LT_5, None
    with localcontext(EVALUATION_CONTEXT):
        dollar = median([b.raw_close * b.raw_volume for b in last])
    if dollar <= 0:
        return Eligibility.ZERO_DOLLAR_VOLUME, None
    return Eligibility.ELIGIBLE, dollar


def screen_quarter(
    calendar: TradingCalendar,
    screen_day: date,
    securities: Sequence[ScreenSecurity],
    n_u: int,
    excluded_issuers: Collection[str],
    rep_map: Mapping[str, str],
    gap_break: int,
    mask_min: int,
) -> ScreenResult:
    """一个筛选日的筛选。n_u、gap_break、mask_min 由调用方显式传入（N_u 拟定，待批准）。"""
    for name, value in (("n_u", n_u), ("gap_break", gap_break), ("mask_min", mask_min)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidInputError(f"{name} 必须为正整数")
    if not calendar.contains(screen_day) or screen_day == calendar.first:
        raise InvalidInputError("筛选日须为日历内、且有前一交易日的交易日")
    if len({s.security_id for s in securities}) != len(securities):
        raise InvalidInputError("稳定标识重复")
    screen_i = calendar.days.index(screen_day)
    eligibility: dict[str, Eligibility] = {}
    dollar: dict[str, Decimal] = {}
    for sec in securities:
        if sec.issuer in excluded_issuers:
            eligibility[sec.security_id] = Eligibility.EXCLUDED_ISSUER
            continue
        status, value = _check(sec, calendar, screen_i, gap_break, mask_min)
        eligibility[sec.security_id] = status
        if value is not None:
            dollar[sec.security_id] = value
    after = dict(rep_map)
    representatives: list[str] = []
    for issuer in sorted({s.issuer for s in securities if s.security_id in dollar}):
        candidates = [s.security_id for s in securities if s.issuer == issuer]
        if issuer in after:
            if after[issuer] in dollar:
                representatives.append(after[issuer])
            continue  # 已固定的股类不合格：发行人当季缺席，不换股类
        eligible = [c for c in candidates if c in dollar]
        after[issuer] = min(eligible, key=lambda c: (-dollar[c], c))
        representatives.append(after[issuer])
    ranking = tuple((c, dollar[c]) for c in sorted(representatives, key=lambda c: (-dollar[c], c)))
    return ScreenResult(
        eligibility,
        dollar,
        ranking,
        tuple(c for c, _ in ranking[:n_u]),
        dict(sorted(after.items())),
    )


__all__ = ["Eligibility", "ScreenBar", "ScreenResult", "ScreenSecurity", "screen_quarter"]
