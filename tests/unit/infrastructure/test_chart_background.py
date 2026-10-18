"""T-RL 第 4 节：回放图的大盘参照说明（发起人裁决第 2 项；RL-07 正向约束）。

用现有的图表数据构造方式（build_chart_data → render_chart）生成一张图，从写出的 chart.html
读回 plotly 的曲线与布局，断言：说明文字一字不差；右轴（y3）恰好两条曲线，名称固定；
y 值逐点等于输入的 63 日回撤 × 100；单一虚线、无颜色数组、色标或填充；没有任何形状或注释
引用右轴。个股的状态色带、缺价标记等现有元素不在约束之内。合成数据，只写临时目录。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from stock_risk.application.replay import ReplayService, ReplayTarget
from stock_risk.application.replay_chart_data import build_chart_data, drawdown_63
from stock_risk.domain.lifecycle.params import EXPLORATION_ASSUMPTIONS, EngineParams, candidate_grid
from stock_risk.infrastructure.replay.chart import render_chart
from tests.unit.application.test_replay_run import service  # noqa: F401  (fixture)

NOTE = "SPY、QQQ 的 63 日回撤：参照显示，不是大盘状态，不进入信号"
NAMES = {"SPY": "SPY 63 日回撤（参照，%，右轴）", "QQQ": "QQQ 63 日回撤（参照，%，右轴）"}
P385 = EngineParams(dict(candidate_grid())[385], EXPLORATION_ASSUMPTIONS)


def _figure(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    html = path.read_text("utf-8")
    m = re.search(r'Plotly\.newPlot\(\s*"chart",\s*', html)
    assert m is not None
    dec = json.JSONDecoder()
    traces, end = dec.raw_decode(html, m.end())
    rest = re.match(r"\s*,\s*", html[end:])
    assert rest is not None
    layout, _ = dec.raw_decode(html, end + rest.end())
    return traces, layout


def _refs_right_axis(item: dict[str, Any]) -> bool:
    return any(str(item.get(k, "")).startswith("y3") for k in ("yref", "yaxis"))


def test_background_reference_only(service: ReplayService, tmp_path: Path) -> None:  # noqa: F811
    prepared = service.prepare(ReplayTarget("AAA"), 10)
    rep = service.replay(prepared, 385, P385)
    background = {s: drawdown_63(service.loader.load(s).history.bars) for s in ("SPY", "QQQ")}
    data = build_chart_data(rep, "run-x", background)
    path = tmp_path / "chart.html"
    render_chart(data, path, "plotly.min.js")
    traces, layout = _figure(path)

    # 说明文字一字不差，出现在图的注释中
    assert data.note == NOTE
    assert NOTE in [a.get("text") for a in layout.get("annotations", [])]

    # 右轴恰好两条曲线，名称固定，不多不少
    right = [t for t in traces if _refs_right_axis(t)]
    assert [t.get("name") for t in right] == [NAMES["SPY"], NAMES["QQQ"]]

    for sym, trace in zip(("SPY", "QQQ"), right, strict=True):
        # y 值逐点等于输入的 63 日回撤 × 100（缺失为空）
        expected = [None if (v := background[sym].get(d)) is None else v * 100 for d in data.days]
        assert trace["y"] == expected, sym
        assert trace["x"] == [d.isoformat() for d in data.days]
        # 单一线条：虚线；颜色为单一值或未设置；没有颜色数组、色标、填充
        line = trace.get("line", {})
        assert line.get("dash") == "dot"
        assert "color" not in line or isinstance(line["color"], str)
        assert trace.get("fill") in (None, "none")
        assert "colorscale" not in line and "colorscale" not in trace.get("marker", {})
        assert not isinstance(trace.get("marker", {}).get("color"), list)
        assert trace.get("mode") in (None, "lines")

    # 没有任何形状或注释引用右轴（例如阈值线、按回撤分段的色带或标注）
    assert not [s for s in layout.get("shapes", []) if _refs_right_axis(s)]
    assert not [a for a in layout.get("annotations", []) if _refs_right_axis(a)]
    assert not any("market-risk" in str(a.get("text", "")) for a in layout.get("annotations", []))

    # 个股的状态色带等现有元素照常存在（不在约束之内）
    assert data.bands and layout.get("shapes")
