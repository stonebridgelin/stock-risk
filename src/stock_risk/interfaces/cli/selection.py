"""选参集数据的命令（T08d-1）：probe-tickers、fetch-batch。只解析参数与输出结果。

uv run stock-risk probe-tickers [--check names.csv]      代码清单探查（一个请求，不下载价格）
uv run stock-risk fetch-batch --list list.csv [--out data/raw_selection]
      [--per-hour 50] [--per-day 1000] [--unique-per-month 500] [--max-symbols N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stock_risk.bootstrap import (
    build_fetch_batch_service,
    build_probe_tickers_service,
    quota_limits,
)
from stock_risk.config.settings import load_settings
from stock_risk.infrastructure.storage.fetch_batch_files import read_batch_list
from stock_risk.infrastructure.storage.universe_probe_files import read_coverage_requests

STOP_TEXT = {
    "max_symbols": "已达本次处理上限（--max-symbols）",
    "per_day": "今日请求额度已用完，明天再运行",
    "unique_per_month": "本月新代码额度已用完，下月再运行",
    "rate_limited": "额度已用尽（连续 3 次 429），稍后再运行",
}


def _cmd_probe_tickers(args: argparse.Namespace) -> int:
    settings = load_settings()
    check = read_coverage_requests(Path(args.check)) if args.check else None
    out = Path(args.out) if args.out else settings.universe_probe_dir
    result = build_probe_tickers_service().run(out, check)
    s = result.summary
    print(
        f"NYSE / NASDAQ 股票（USD）{s['total']} 行，{s['unique_tickers']} 个代码；"
        f"2019-12-31 前结束 {s['ended_by_2019']}，延续到之后 {s['active_after_2019']}；"
        f"代码复用 {s['reused_tickers']} 个"
    )
    if result.checked is not None:
        n_in = sum(1 for r in result.checked if r["in_list"])
        print(f"覆盖对照：{len(result.checked)} 个代码，在清单中 {n_in} 个")
    for p in result.paths:
        print(f"  {p}")
    return 0


def _cmd_fetch_batch(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        items = read_batch_list(Path(args.list))
    except ValueError as e:
        print(f"拒绝执行：{e}")
        return 2
    out = Path(args.out) if args.out else settings.selection_raw_dir
    limits = quota_limits(
        settings, args.per_hour, args.per_day, args.unique_per_month, args.max_symbols
    )
    service = build_fetch_batch_service(settings, out, limits, progress=print)
    try:
        r = service.run(items, limits)
    except KeyboardInterrupt:
        print("\n已中断：已完成的代码已落盘，额度已记账；再次运行会跳过已完成的代码。")
        return 130
    _print_report(r, limits)
    print(f"输出目录：{out}（可运行 audit-raw --dir {out} --list {args.list} 审计）")
    return 1 if r.rate_limited else 0


def _print_report(r: Any, limits: Any) -> None:
    print(
        f"\n本次完成 {len(r.done_now)} 个（失败 {len(r.failed_now)} 个，跳过已完成 "
        f"{r.skipped_done} 个）；累计完成 {r.cumulative_done} / {r.total}，剩余 {r.remaining} 个"
    )
    print(
        f"额度：今日已用 {r.day_requests} / {limits.per_day} 次请求，"
        f"本月新代码 {r.month_unique} / {limits.unique_per_month} 个"
    )
    if r.stop_reason:
        print(f"停止原因：{STOP_TEXT[r.stop_reason]}")
    if r.remaining:
        print(f"按当前额度，预计还需运行约 {r.estimated_runs} 次")


def add_parsers(sub: Any) -> None:
    p = sub.add_parser("probe-tickers", help="代码清单探查（只一个请求，不下载价格；T08d-1）")
    p.add_argument("--check", default="", help="覆盖对照：CSV，一列 ticker，可附 from,to")
    p.add_argument("--out", default="", help="输出目录，默认 data/universe_probe/")
    p.set_defaults(func=_cmd_probe_tickers)

    b = sub.add_parser("fetch-batch", help="选参集批量下载（额度保护、断点续传；T08d-1）")
    b.add_argument("--list", required=True, help="CSV：symbol,source_ticker,from,to")
    b.add_argument("--out", default="", help="输出目录，默认 data/raw_selection/")
    b.add_argument("--per-hour", type=int, default=None, help="每小时请求上限，默认 50")
    b.add_argument("--per-day", type=int, default=None, help="每日请求上限，默认 1000")
    b.add_argument(
        "--unique-per-month", type=int, default=None, help="每个自然月新代码上限，默认 500"
    )
    b.add_argument("--max-symbols", type=int, default=None, help="本次最多处理的代码数")
    b.set_defaults(func=_cmd_fetch_batch)
