"""全语义摘要（T10-M0b 第 1 节，F-07、I-18）。纯函数，不做 I/O。

`canonical(obj)` 把生命周期输出规范序列化为紧凑 JSON；`semantic_digest(result)` 对 RunResult 的全部
内容（results 中每一步的 state、events（含 basis）、record，以及 final_state、censored_episode）
取 SHA-256。与 replay_run_types.discrete_digest（只含日期、显示状态、等级、预警编号、周期、事件键）
不同，任何字段改变都会改变摘要。

类型判定顺序（取第一个匹配的行；见任务卡第 1 节的表）：
datetime（拒绝）→ None → 枚举 → bool → int → float → Decimal → str → date → 数据类 → tuple → list
→ frozenset → set → dict（只限 dict 本身）→ 其他（拒绝）。
非有限的 float、Decimal 拒绝；拒绝时异常信息写出字段路径。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

DIGEST_VERSION = "semantic-digest/1\n"


class CanonicalError(ValueError):
    """规范序列化拒绝的值（带字段路径）。"""


def _type_name(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


# 字段路径惰性表示：(上一级, 本级片段) 的链，只在拒绝时才拼成字符串
_Path = tuple[Any, str] | None


def _path_str(path: _Path) -> str:
    parts: list[str] = []
    while path is not None:
        path, seg = path
        parts.append(seg)
    text = "".join(reversed(parts))
    return text.lstrip(".") or "<根>"


def _reject(path: _Path, why: str) -> CanonicalError:
    return CanonicalError(f"{_path_str(path)}：{why}")


_FIELD_NAMES: dict[type, tuple[str, ...]] = {}


def _field_names(cls: type) -> tuple[str, ...]:
    names = _FIELD_NAMES.get(cls)
    if names is None:
        names = _FIELD_NAMES[cls] = tuple(f.name for f in dataclasses.fields(cls))
    return names


def _enc(obj: Any, path: _Path) -> Any:  # noqa: C901, PLR0911, PLR0912 —— 判定顺序即规格表
    # 快速路径：精确类型。这些类型不可能匹配表中更靠前的行（datetime、None、枚举），
    # 所以结果与按表逐行判定相同；子类仍走下面的完整判定顺序。
    t = type(obj)
    if t is float:
        if not math.isfinite(obj):
            raise _reject(path, f"非有限的 float：{obj!r}")
        return {"f": obj.hex()}
    if t is str:
        return obj
    if t is int:
        return {"i": str(obj)}
    if t is bool:
        return obj
    if t is date:
        return {"date": obj.isoformat()}
    if isinstance(obj, datetime):
        raise _reject(path, "不接受时间戳（datetime）")
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return {"enum": f"{_type_name(obj)}.{obj.name}"}
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return {"i": str(obj)}
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise _reject(path, f"非有限的 float：{obj!r}")
        return {"f": obj.hex()}
    if isinstance(obj, Decimal):
        if not obj.is_finite():
            raise _reject(path, f"非有限的 Decimal：{obj!r}")
        return {"d": str(obj)}
    if isinstance(obj, str):
        return obj
    if isinstance(obj, date):
        return {"date": obj.isoformat()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = [[name, _enc(getattr(obj, name), (path, f".{name}"))] for name in _field_names(t)]
        return {"dc": _type_name(obj), "fields": fields}
    if isinstance(obj, tuple):
        return {"tuple": [_enc(v, (path, f"[{i}]")) for i, v in enumerate(obj)]}
    if isinstance(obj, list):
        return {"list": [_enc(v, (path, f"[{i}]")) for i, v in enumerate(obj)]}
    if isinstance(obj, frozenset):
        return {"frozenset": _sorted_items(obj, path)}
    if isinstance(obj, set):
        return {"set": _sorted_items(obj, path)}
    if t is dict:
        items = []
        for k, v in obj.items():
            ek = _enc(k, (path, "{键}"))
            items.append((_dump(ek), ek, _enc(v, (path, f"[{k!r}]"))))
        items.sort(key=lambda x: x[0])
        return {"map": [[ek, ev] for _, ek, ev in items]}
    raise _reject(path, f"不支持的类型 {_type_name(obj)}")


def _sorted_items(items: frozenset[Any] | set[Any], path: _Path) -> list[Any]:
    encoded = [_enc(v, (path, "{元素}")) for v in items]
    return sorted(encoded, key=_dump)


def _dump(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def canonical(obj: Any) -> str:
    """规范序列化：紧凑 JSON（separators=(",", ":")，ensure_ascii=False）。"""
    return _dump(_enc(obj, None))


def semantic_digest(result: Any) -> str:
    """RunResult 的全语义摘要：

    SHA-256("semantic-digest/1\\n" + canonical(result) 的 UTF-8 字节)，十六进制小写。"""
    data = DIGEST_VERSION.encode("utf-8") + canonical(result).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


__all__ = ["DIGEST_VERSION", "CanonicalError", "canonical", "semantic_digest"]
