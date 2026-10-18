"""Massive 数据源试验命令（T08d-2a）：probe-massive。只解析参数与输出结果。

uv run stock-risk probe-massive [--out data/massive_probe] [--per-minute 5]
uv run stock-risk probe-massive --followup [--out data/massive_probe/followup] [--per-minute 5]
uv run stock-risk probe-massive-paid [--out data/massive_probe/paid] [--per-minute 100]
uv run stock-risk verify-massive-relay [--per-minute 100]   中转服务核验（T08d-2c，需输入 yes 确认）
uv run stock-risk d0b-inventory / d0b-calendar        D0b P1 盘点与日历文件（不联网；T12-D0b）
uv run stock-risk fetch-massive-d0b [--dry-run]      D0b 补取（经中转服务，需输入 yes；T12-D0b）
付费档的两个命令运行前打印将要连接的主机，输入 yes 才继续。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from stock_risk.bootstrap import (
    build_fetch_massive_d0b_service,
    build_probe_massive_followup_service,
    build_probe_massive_paid_service,
    build_probe_massive_service,
    build_verify_massive_relay_service,
    run_d0b_calendar,
    run_d0b_inventory,
)
from stock_risk.config.settings import PROJECT_ROOT, load_settings


def _cmd_probe_massive(args: argparse.Namespace) -> int:
    if args.followup:
        return _cmd_followup(args)
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


def _cmd_followup(args: argparse.Namespace) -> int:
    settings = load_settings()
    out = Path(args.out) if args.out else settings.massive_probe_dir / "followup"
    service = build_probe_massive_followup_service(settings, out, args.per_minute, progress=print)
    try:
        r = service.run()
    except KeyboardInterrupt:
        print("\n已中断：已完成的查询已保存进度；再次运行会跳过已完成项。")
        return 130
    s = r.summary
    for label, part in (("A", "a_type_queries"), ("B", "b_market_queries")):
        statuses = _count(v["status"] for v in s[part].values())
        print(f"{label} 清单 {len(s[part])} 种：{statuses}")
    print(f"C 案例 {len(s['case_status'])} 个：{_count(s['case_status'].values())}")
    for d, v in s["coverage_vs_tiingo"].items():
        print(f"D {d}：{v.get('counts', v['status'])}")
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


def _cmd_probe_massive_paid(args: argparse.Namespace) -> int:
    settings = load_settings()
    out = Path(args.out) if args.out else settings.massive_paid_dir
    service = build_probe_massive_paid_service(settings, out, args.per_minute, progress=print)
    print(f"将要连接的主机：{service.source_host}（密钥 MASSIVE_ADVANCED_API_KEY，以请求头发送）")
    print(f"输出目录：{out}")
    plan = service.planned
    print(
        f"计划：日线 {plan['prices']} 次；拆股与分红 {plan['actions']} 个代码；"
        f"清单快照 {plan['lists']} 个（可能翻页），另有抽样查询"
    )
    print("日线被 HTTP 401 或 403 拒绝时停止。")
    if not _confirmed():
        return 1
    try:
        r = service.run()
    except KeyboardInterrupt:
        print("\n已中断：已完成的项已保存进度；再次运行会跳过已完成项。")
        return 130
    s = r.summary
    prices = _count(v["status"] for p in s["prices"].values() for v in (p["raw"], p["adj"]))
    actions = _count(v["status"] for v in s["actions"].values())
    lists = _count(v["status"] for v in s["labels"]["lists"].values())
    print(f"\n日线 {prices}；拆股与分红 {actions}；清单 {lists}")
    print(f"封存：{s['seal']}")
    print(
        f"本次请求 {r.requests_now} 次；累计请求 {s['requests_total']} 次，"
        f"累计耗时 {s['elapsed_seconds_total']} 秒"
    )
    if r.stopped:
        print(f"停止原因：{r.stopped}（进度已保存）")
    if s["plan_rejected"]:
        rej = s["plan_rejected"]
        print(f"付费档拒绝历史价格：HTTP {rej.get('http_status')} {rej.get('message')}")
    if r.remaining:
        print(f"尚未完成 {len(r.remaining)} 项")
    for p in r.paths:
        print(f"  {p}")
    return 1 if r.stopped else 0


def _confirmed() -> bool:
    """付费档命令运行前的确认：只有输入 yes 才继续，否则不发任何请求。"""
    try:
        answer = input("确认连接上述主机，请输入 yes：")
    except EOFError:
        answer = ""
    if answer.strip() != "yes":
        print("未确认，未发出任何请求。")
        return False
    return True


def _cmd_verify_massive_relay(args: argparse.Namespace) -> int:
    settings = load_settings()
    service = build_verify_massive_relay_service(settings, args.per_minute, progress=print)
    print(f"将要连接的主机：{service.source_host}（密钥 MASSIVE_ADVANCED_API_KEY，以请求头发送）")
    print(f"输出目录：{settings.massive_paid_dir / 'verify'}")
    plan = service.planned
    print(
        f"计划：第 2 节试探 {plan['coverage']} 次；第 3.1 节价格 {plan['prices']} 只；"
        f"第 3.2 节清单查询 {plan['lists']} 种（可能翻页）"
    )
    print("遇到 HTTP 401 或 403 立即停止。")
    if not _confirmed():
        return 1
    try:
        r = service.run()
    except KeyboardInterrupt:
        print("\n已中断：已完成的项已保存进度；再次运行会跳过已完成项。")
        return 130
    s = r.summary
    print(f"\n不可用的接口：{s['unavailable'] or '无'}")
    print(f"封存：{s['seal']}")
    print(
        f"本次请求 {r.requests_now} 次；累计请求 {s['requests_total']} 次，"
        f"累计耗时 {s['elapsed_seconds_total']} 秒"
    )
    if r.stopped:
        print(f"停止原因：{r.stopped}（进度已保存）")
    if s["auth_rejected"]:
        rej = s["auth_rejected"]
        print(f"被拒绝：{rej.get('what')} HTTP {rej.get('http_status')} {rej.get('message')}")
    if r.remaining:
        print(f"尚未完成 {len(r.remaining)} 项")
    for p in r.paths:
        print(f"  {p}")
    return 1 if r.stopped else 0


_D0B_NAMES = {"raw": "原始日线", "adj": "复权日线（诊断）", "splits": "拆股", "dividends": "分红"}


def _cmd_d0b_inventory(args: argparse.Namespace) -> int:
    r = run_d0b_inventory(PROJECT_ROOT, Path(args.out))
    print(f"身份清单 {len(r['identity'])} 只；缺口 {len(r['specs'])} 项")
    diff = [c for c in r["compare"] if not c.same]
    print("映射比较（第 2.7 节）：" + ("全部一致" if not diff else f"{len(diff)} 只不一致"))
    for c in diff:
        print(f"  {c.symbol}：身份清单 {c.identity}；构建器 {c.builder}")
    for p in r["paths"]:
        print(f"  {p}")
    return 1 if diff else 0


def _cmd_d0b_calendar(args: argparse.Namespace) -> int:
    r = run_d0b_calendar(PROJECT_ROOT, Path(args.out), Path(args.raw_dir))
    print(f"{r['library']} XNYS：{r['days']} 个交易日，{r['first']} 至 {r['last']}")
    print(f"  {r['file']}  SHA-256 {r['sha256']}")
    print(f"与 SPY 日历代理（SHA-256 {r['spy_sha256']}）比较：")
    print(f"  只在 SPY：{len(r['only_in_spy'])} 天 {[d.isoformat() for d in r['only_in_spy']]}")
    print(f"  只在 XNYS：{len(r['only_in_xnys'])} 天 {[d.isoformat() for d in r['only_in_xnys']]}")
    return 0


def _cmd_fetch_massive_d0b(args: argparse.Namespace) -> int:
    service = build_fetch_massive_d0b_service(
        identity_path=Path(args.identity),
        gaps_path=Path(args.gaps),
        out_dir=Path(args.out),
        dry_run=args.dry_run,
        per_minute=args.per_minute,
        progress=print,
    )
    plan = service.plan()
    print(f"将要连接的主机：{plan.host}（密钥 MASSIVE_ADVANCED_API_KEY，以请求头发送）")
    print(f"输出目录：{args.out}")
    symbols = sorted({s.symbol for s in plan.pending})
    print(f"代码（{len(symbols)} 只）：{'、'.join(symbols) or '无'}")
    for cat, n in plan.counts().items():
        print(f"  {_D0B_NAMES[cat]}：{n} 次")
    print(f"已完成（跳过）：{len(plan.done)} 项；待请求：{len(plan.pending)} 项")
    print(f"预计请求总次数：{len(plan.pending)}（每项 1 页；翻页时另加）")
    print("请求止日均不晚于 2019-12-31；遇到 HTTP 401 或 403 立即停止。")
    if args.dry_run:
        print("--dry-run：只打印计划，未联网，未发出任何请求。")
        return 0
    if not plan.pending:
        print("没有待请求的项。")
        return 0
    if not _confirmed():
        return 1
    try:
        r = service.run()
    except KeyboardInterrupt:
        print("\n已中断：已完成的请求已记入账本；再次运行会跳过已完成项。")
        return 130
    print(
        f"\n本次请求 {r.requests_now} 次；完成 {len(r.completed)} 项，"
        f"失败 {len(r.failed)} 项，未完成 {len(r.incomplete)} 项"
    )
    for k in r.failed + r.incomplete:
        print(f"  {k}")
    if r.date_missing:
        print(f"有缺日期记录的请求 {len(r.date_missing)} 项（未解决的数据问题，原文见台账）：")
        for k in r.date_missing:
            print(f"  {k}")
    if r.stopped:
        print(f"停止原因：{r.stopped}（账本已保存）；尚未处理 {r.remaining} 项")
    return 1 if (r.stopped or r.failed or r.incomplete) else 0


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
    p.add_argument(
        "--followup",
        action="store_true",
        help="第 2.4 节追加探查（去掉类型、交易所过滤；补充案例；与 Tiingo 对照），"
        "默认输出 data/massive_probe/followup/",
    )
    p.set_defaults(func=_cmd_probe_massive)
    q = sub.add_parser(
        "probe-massive-paid",
        help="Massive 付费档验证：日线、拆股分红、无标签证券分类（不全量下载；T08d-2b）",
    )
    q.add_argument("--out", default="", help="输出目录，默认 data/massive_probe/paid/")
    q.add_argument(
        "--per-minute", type=int, default=None, help="每分钟请求上限，默认 100（付费档）"
    )
    q.set_defaults(func=_cmd_probe_massive_paid)
    v = sub.add_parser(
        "verify-massive-relay",
        help="中转服务接入核验：接口试探、价格对账、参考数据比对（T08d-2c；运行前需输入 yes）",
    )
    v.add_argument(
        "--per-minute", type=int, default=None, help="每分钟请求上限，默认 100（付费档）"
    )
    v.set_defaults(func=_cmd_verify_massive_relay)
    d = sub.add_parser(
        "d0b-inventory",
        help="D0b P1：证券身份、映射比较与缺口清单（只读本地文件，不联网；T12-D0b 2.1、2.2、2.7）",
    )
    d.add_argument("--out", default="output/d0b", help="输出目录，默认 output/d0b/")
    d.set_defaults(func=_cmd_d0b_inventory)
    c = sub.add_parser(
        "d0b-calendar",
        help="D0b P1：XNYS 交易日历文件并与 SPY 日历代理核对（不联网；T12-D0b 第 2.5 节）",
    )
    c.add_argument("--out", default="output/d0b/calendar_xnys.csv", help="日历文件路径")
    c.add_argument("--raw-dir", default="data/raw", help="SPY 日历代理所在目录")
    c.set_defaults(func=_cmd_d0b_calendar)
    f = sub.add_parser(
        "fetch-massive-d0b",
        help="D0b 补取：只请求缺口清单中的项（经中转服务；运行前需输入 yes；T12-D0b 第 2.3 节）",
    )
    f.add_argument("--identity", default="output/d0b/identity.csv", help="身份清单")
    f.add_argument("--gaps", default="output/d0b/gaps.csv", help="缺口清单")
    f.add_argument("--out", default="data/massive_d0b/download", help="输出目录")
    f.add_argument("--dry-run", action="store_true", help="只打印计划，不联网，不读密钥")
    f.add_argument("--per-minute", type=int, default=None, help="每分钟请求上限，默认 100")
    f.set_defaults(func=_cmd_fetch_massive_d0b)
