"""Massive 数据源试验命令（T08d-2a）：probe-massive。只解析参数与输出结果。

uv run stock-risk probe-massive [--out data/massive_probe] [--per-minute 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stock_risk.bootstrap import build_probe_massive_service
from stock_risk.config.settings import load_settings


def _cmd_probe_massive(args: argparse.Namespace) -> int:
    settings = load_settings()
    out = Path(args.out) if args.out else settings.massive_probe_dir
    service = build_probe_massive_service(settings, out, args.per_minute, progress=print)
    try:
        r = service.run()
    except KeyboardInterrupt:
        print("\n已中断：已完成的查询已保存进度；再次运行会跳过已完成项。")
        return 130
    s = r.summary
    statuses: dict[str, int] = {}
    for v in s["list_queries"].values():
        statuses[v["status"]] = statuses.get(v["status"], 0) + 1
    print(f"\n清单查询 {len(s['list_queries'])} 种：{statuses}")
    print(f"案例 {len(s['case_status'])} 个：{_count(s['case_status'].values())}")
    print(
        f"本次请求 {r.requests_now} 次；累计请求 {s['requests_total']} 次，"
        f"累计耗时 {s['elapsed_seconds_total']} 秒"
    )
    if r.rate_limited:
        print("停止原因：连续 3 次 429，进度已保存，稍后再运行")
    if r.remaining:
        print(f"尚未完成 {len(r.remaining)} 项")
    for p in r.paths:
        print(f"  {p}")
    return 1 if r.rate_limited else 0


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def add_parsers(sub: Any) -> None:
    p = sub.add_parser(
        "probe-massive", help="Massive 数据源试验：历史日期清单与已知案例（不下载价格；T08d-2a）"
    )
    p.add_argument("--out", default="", help="输出目录，默认 data/massive_probe/")
    p.add_argument("--per-minute", type=int, default=None, help="每分钟请求上限，默认 5（免费档）")
    p.set_defaults(func=_cmd_probe_massive)
