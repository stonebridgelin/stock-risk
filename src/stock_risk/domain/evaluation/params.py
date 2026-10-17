"""17 §7 / T09 D-31：拟定，待 SR-M1′ 批准；仅供探索与夹具显式传入。"""

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class ExplorationValues:
    x_pct: Decimal
    x_pct10: Decimal
    x_atr: tuple[Decimal, ...]
    rho_end: Decimal
    w_q: int
    rho: Decimal
    y_max: int
    y_cap: int
    q: int
    q_l: int
    n_min: int
    mask_min: int


PROPOSED_EXPLORATION = ExplorationValues(
    Decimal("0.20"),
    Decimal("0.10"),
    tuple(map(Decimal, (6, 8, 10, 12))),
    Decimal("0.5"),
    20,
    Decimal("0.5"),
    252,
    60,
    20,
    60,
    10,
    83,
)

# 17 §2.1–2.2 固定的事件规则；与上方拟定参数分别传递。
PEAK_WINDOW = 63
EVENT_CAP = 252
QUIET_DAYS = 10
CONFIRM_DAYS = 20
ATR_PERIOD = 20
