"""前缀不变性：固定种子的合成历史（T11-D0a 第 9.3 节）。

用标准库 random.Random(种子) 生成 240 段合成历史，随机含：拆股、合股、分红（含大额、同日多笔）、
同日行动、缺价（含超过 G 的长缺口，即历史断点）、周末除权日、最后一根日线之后的行动（终点待处理）。
每段历史在全部截断日上比较截至 T 的视图（见 prefix.py）。
"""

from __future__ import annotations

import random
from datetime import timedelta

from tests.unit.adjustment.helpers import CAL, bar, div, split
from tests.unit.adjustment.prefix import check_prefix

SEED = 20261018
HISTORIES = 240
SPLIT_RATIOS = (("1", "2"), ("2", "1"), ("1", "3"), ("10", "1"), ("3", "2"), ("0", "1"))


def _history(rng: random.Random) -> tuple[list, list, list]:
    start = rng.randrange(300, len(CAL) - 200)
    n = rng.randint(25, 70)
    bars = []
    price = rng.uniform(5, 150)
    i = start
    while len(bars) < n:
        r = rng.random()
        if r < 0.03:
            i += rng.randint(11, 25)  # 长缺口：历史断点
        elif r < 0.18:
            i += rng.randint(1, 7)  # 缺价
        d = CAL[i]
        price = max(0.5, price * rng.uniform(0.9, 1.1))
        c = round(price, 4)
        h, lo = round(c * rng.uniform(1.0, 1.05), 4), round(c * rng.uniform(0.95, 1.0), 4)
        o = round(rng.uniform(lo, h), 4)
        bars.append(bar(d, c, open_=o, high=h, low=lo))
        i += 1
    first, last = bars[0].day, bars[-1].day
    span = (last - first).days + 10
    splits, divs = [], []
    for row in range(1, rng.randint(0, 3) + 1):
        d = first + timedelta(days=rng.randint(-2, span))
        frm, to = rng.choice(SPLIT_RATIOS)
        splits.append(split(d, frm, to, row))
    for row in range(1, rng.randint(0, 5) + 1):
        if divs and rng.random() < 0.2:
            d = divs[-1].ex_date  # 同日多笔
        elif splits and rng.random() < 0.15:
            d = splits[-1].ex_date  # 与拆股同日
        else:
            d = first + timedelta(days=rng.randint(-2, span))
        # 40：在小价格上产生异常或非法因子
        amount = f"{rng.choice([0.05, 0.3, 1.0, 2.5, 40.0, rng.uniform(0.01, 3.0)]):.4f}"
        kind = rng.choice(["CD", "CD", "CD", "SC", "LT"])
        currency = rng.choice(["USD", "USD", None])
        divs.append(div(d, amount, row, dividend_type=kind, currency=currency))
    return bars, splits, divs


def test_prefix_invariance_on_random_histories() -> None:
    rng = random.Random(SEED)
    checked = 0
    for _ in range(HISTORIES):
        bars, splits, divs = _history(rng)
        checked += check_prefix(bars, splits, divs)
    assert checked > HISTORIES * 30  # 截断日总数（回执中报告）
