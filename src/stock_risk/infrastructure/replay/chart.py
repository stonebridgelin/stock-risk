"""用 plotly 绘制回放图（实现 T04b chart.html），以及从写出的 HTML 读回事件标记（一致性检查用）。

plotly.js 以本地文件共用：每个 run 目录根下写一份 plotly.min.js，各图表用相对路径引用，
离线可看、不依赖网络，也不在每个文件里内嵌。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from stock_risk.application.replay_chart_data import ChartData
from stock_risk.application.replay_outputs import MISSING_NOTE

PLOTLY_JS = "plotly.min.js"
EVENT_TRACE = "事件"
BAND_COLORS = {
    "关注": "rgba(255, 193, 7, 0.18)",
    "警戒": "rgba(220, 53, 69, 0.18)",
    "数据中断": "rgba(108, 117, 125, 0.30)",
    "不可计算": "rgba(173, 181, 189, 0.15)",
}
EVENT_SYMBOLS = {
    "START": ("triangle-down", "#c0392b"),
    "ESCALATE": ("triangle-down", "#7b241c"),
    "DOWNGRADE": ("triangle-up", "#d68910"),
    "RELEASE": ("star", "#1e8449"),
    "DATA_BREAK": ("x", "#555555"),
    "DATA_ISSUE": ("x-thin", "#888888"),
    "STABILIZED": ("circle-open", "#2471a3"),
    "RECOVERY_OBS": ("circle", "#2471a3"),
    "RECOVERY_REVOKED": ("circle-x", "#884ea0"),
    "RECOVERY_FAIL": ("square-x", "#884ea0"),
}


def write_plotly_js(run_dir: Path) -> Path:
    path = run_dir / PLOTLY_JS
    if not path.exists():
        path.write_text(get_plotlyjs(), "utf-8")
    return path


def render_chart(data: ChartData, path: Path, plotly_js_rel: str) -> None:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.04
    )
    x = [d.isoformat() for d in data.days]
    fig.add_trace(
        go.Scatter(x=x, y=data.close, name="收盘（复权）", line={"color": "#212529", "width": 1.2}),
        row=1,
        col=1,
    )
    for name, ys, style in (
        ("L_ep", data.l_ep, {"color": "#c0392b", "dash": "dot", "width": 1}),
        ("确认线", data.confirm_line, {"color": "#2471a3", "dash": "dash", "width": 1}),
        ("强恢复门槛", data.strong_threshold, {"color": "#1e8449", "dash": "dashdot", "width": 1}),
    ):
        fig.add_trace(go.Scatter(x=x, y=ys, name=name, line=style, connectgaps=False), row=1, col=1)
    ev_x = [e.day.isoformat() for e in data.events]
    fig.add_trace(
        go.Scatter(
            x=ev_x,
            y=[e.y for e in data.events],
            mode="markers",
            name=EVENT_TRACE,
            customdata=[e.type for e in data.events],
            text=[e.text for e in data.events],
            hovertemplate="%{x}<br>%{text}<extra></extra>",
            marker={
                "symbol": [EVENT_SYMBOLS.get(e.type, ("circle", "#000"))[0] for e in data.events],
                "color": [EVENT_SYMBOLS.get(e.type, ("circle", "#000"))[1] for e in data.events],
                "size": 9,
            },
        ),
        row=1,
        col=1,
    )
    if data.missing_days:
        fig.add_trace(
            go.Scatter(
                x=[d.isoformat() for d in data.missing_days],
                y=[None] * len(data.missing_days),
                mode="markers",
                name=f"缺价日（{MISSING_NOTE}）",
                text=[MISSING_NOTE] * len(data.missing_days),
                hovertemplate="%{x}<br>%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        for d in data.missing_days:
            fig.add_vline(
                x=d.isoformat(),
                line={"color": "#6c757d", "width": 1, "dash": "dot"},
            )
    fig.add_trace(
        go.Scatter(x=x, y=data.d_t, name="D_T（无预警）", line={"color": "#6c757d", "width": 1}),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=data.d_ep, name="D_ep（有预警）", line={"color": "#c0392b", "width": 1}),
        row=2,
        col=1,
    )
    for sym, ys in data.background.items():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[None if v is None else v * 100 for v in ys],
                name=f"{sym} 63 日回撤（%，右轴）",
                yaxis="y3",
                line={"width": 1, "dash": "dot"},
            ),
        )
    shapes = [
        {
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": a.isoformat(),
            "x1": b.isoformat(),
            "y0": 0,
            "y1": 1,
            "fillcolor": BAND_COLORS.get(s, "rgba(0,0,0,0.05)"),
            "line": {"width": 0},
            "layer": "below",
        }
        for a, b, s in data.bands
    ]
    shapes += [
        {
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": a.isoformat(),
            "x1": b.isoformat(),
            "y0": 0.97,
            "y1": 1,
            "fillcolor": "rgba(52, 152, 219, 0.6)",
            "line": {"width": 0},
        }
        for a, b, _ in data.corrections
    ]
    annotations = [
        {
            "text": data.note,
            "xref": "paper",
            "yref": "paper",
            "x": 0,
            "y": -0.08,
            "showarrow": False,
            "font": {"size": 11, "color": "#6c757d"},
        },
        {
            "text": "色带：黄 = 关注，红 = 警戒，深灰 = 数据中断，浅灰 = 不可计算",
            "xref": "paper",
            "yref": "paper",
            "x": 1,
            "y": -0.08,
            "xanchor": "right",
            "showarrow": False,
            "font": {"size": 11, "color": "#6c757d"},
        },
    ]
    annotations += [
        {
            "text": label,
            "xref": "x",
            "yref": "paper",
            "x": a.isoformat(),
            "y": 1.0,
            "showarrow": False,
            "font": {"size": 10, "color": "#1f618d"},
            "xanchor": "left",
        }
        for a, _, label in data.corrections
    ]
    fig.update_layout(
        title={"text": data.title, "font": {"size": 14}},
        shapes=shapes,
        annotations=annotations,
        hovermode="x unified",
        height=820,
        legend={"orientation": "h", "y": 1.08},
        yaxis3={"overlaying": "y2", "side": "right", "title": "回撤 %", "showgrid": False},
        margin={"l": 60, "r": 60, "t": 110, "b": 70},
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="D", row=2, col=1)
    fig.write_html(path, include_plotlyjs=plotly_js_rel, full_html=True, div_id="chart")


def read_chart_events(path: Path) -> list[tuple[str, str]]:
    """从写出的 chart.html 读回事件标记（日期, 事件类型）。"""
    html = path.read_text("utf-8")
    m = re.search(r'Plotly\.newPlot\(\s*"chart",\s*', html)
    if m is None:
        raise ValueError(f"{path}: 找不到图表数据")
    traces, _ = json.JSONDecoder().raw_decode(html, m.end())
    ev = next((t for t in traces if t.get("name") == EVENT_TRACE), None)
    if ev is None:
        return []
    return list(zip(ev.get("x", []), ev.get("customdata", []), strict=True))
