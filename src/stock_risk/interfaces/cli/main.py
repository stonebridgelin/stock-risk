"""命令行入口。

用法（项目根目录）：
    uv run stock-risk fetch-raw            下载探索期原始日线（断点续传）
    uv run stock-risk fetch-raw --force    忽略已下载记录，全部重新下载
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from stock_risk.bootstrap import build_fetch_raw_data_service
from stock_risk.config.settings import load_settings

DEFAULT_START = date(2008, 1, 1)


def _cmd_fetch_raw(args: argparse.Namespace) -> int:
    service = build_fetch_raw_data_service(load_settings())
    report = service.run(start=date.fromisoformat(args.start), force=args.force)
    print(f"\n完成：成功 {len(report.succeeded)} 个，缺失 {len(report.failed)} 个。")
    print("详情见 data/raw/README.txt")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stock-risk")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-raw", help="下载探索期原始日线（截至 2019-12-31）")
    p.add_argument("--start", default=DEFAULT_START.isoformat(), help="开始日期，默认 2008-01-01")
    p.add_argument("--force", action="store_true", help="重新下载已完成的代码")
    p.set_defaults(func=_cmd_fetch_raw)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
