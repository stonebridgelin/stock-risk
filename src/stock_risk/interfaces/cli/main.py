"""命令行入口。

用法（项目根目录）：
    uv run stock-risk fetch-raw                       下载探索期原始日线（断点续传）
    uv run stock-risk fetch-raw --force               忽略已下载记录，全部重新下载
    uv run stock-risk fetch-raw --force --symbols VRX 只重新下载指定代码
    uv run stock-risk audit-raw                       审计 data/raw，输出到 output/audit/<日期>/
    uv run stock-risk check-calendar                  核对交易日历代理（output/calendar_check/）
    uv run stock-risk show-ranges                     三个区间与缺价日（output/ranges/）
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from stock_risk.application.market_data_loader import DOWNLOAD_START
from stock_risk.bootstrap import (
    build_audit_raw_data_service,
    build_check_calendar_service,
    build_fetch_raw_data_service,
    build_show_ranges_service,
)
from stock_risk.config.settings import load_settings

DEFAULT_START = DOWNLOAD_START


def _cmd_fetch_raw(args: argparse.Namespace) -> int:
    service = build_fetch_raw_data_service(load_settings())
    symbols = [s for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    report = service.run(start=date.fromisoformat(args.start), force=args.force, symbols=symbols)
    print(f"\n完成：成功 {len(report.succeeded)} 个，缺失 {len(report.failed)} 个。")
    print("详情见 data/raw/README.txt")
    return 0


def _cmd_audit_raw(args: argparse.Namespace) -> int:
    settings = load_settings()
    service = build_audit_raw_data_service(settings)
    report, paths = service.run(date.today(), settings.audit_output_dir)
    print(
        f"审计 {len(report.symbols)} 个代码：明确失败 {report.total_failures} 项，"
        f"封存越界 {report.total_sealed_rows} 行，缺少原始文件 {len(report.missing)} 个。"
    )
    for p in paths:
        print(f"  {p}")
    return 0


def _cmd_check_calendar(args: argparse.Namespace) -> int:
    settings = load_settings()
    result, paths = build_check_calendar_service(settings).run(
        date.today(), settings.calendar_check_output_dir
    )
    print(
        f"{result.proxy_symbol} {result.proxy_days} 天 vs {result.reference_library} "
        f"{result.reference_version} {result.reference_calendar} {result.reference_days} 天"
        f"（{result.start} ~ {result.end}）"
    )
    only_proxy, only_ref = len(result.only_in_proxy), len(result.only_in_reference)
    print(f"只在代理中: {only_proxy} 天；只在参照中: {only_ref} 天")
    print("结论: 完全一致" if result.identical else "结论: 存在差异，交产品经理裁定")
    for p in paths:
        print(f"  {p}")
    return 0 if result.identical else 1


def _cmd_show_ranges(args: argparse.Namespace) -> int:
    settings = load_settings()
    rows, paths = build_show_ranges_service(settings).run(date.today(), settings.ranges_output_dir)
    total_missing = sum(len(r.missing_days) for r in rows)
    print(f"{len(rows)} 个代码；观察区间内缺价日合计 {total_missing} 天。")
    for p in paths:
        print(f"  {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stock-risk")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-raw", help="下载探索期原始日线（截至 2019-12-31）")
    p.add_argument("--start", default=DEFAULT_START.isoformat(), help="开始日期，默认 2008-01-01")
    p.add_argument("--force", action="store_true", help="重新下载已完成的代码")
    p.add_argument("--symbols", default="", help="只处理这些代码，逗号分隔（须在研究样本中）")
    p.set_defaults(func=_cmd_fetch_raw)

    a = sub.add_parser("audit-raw", help="审计 data/raw 原始数据（只报告，不修改）")
    a.set_defaults(func=_cmd_audit_raw)

    c = sub.add_parser("check-calendar", help="用 XNYS 独立日历核对 SPY 日历代理（I13）")
    c.set_defaults(func=_cmd_check_calendar)

    r = sub.add_parser("show-ranges", help="输出各代码的三个区间、端点类型与缺价日（I2、I3）")
    r.set_defaults(func=_cmd_show_ranges)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
