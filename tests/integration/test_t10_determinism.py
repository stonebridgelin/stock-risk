"""T10-M0b 第 5 节：逐字节确定性（红线 5，M-08）。

入口为 ReplayRunService.run_reference（写出 daily、events、episodes 与图表四类目标文件；清单中没有
parallel_seconds）。时钟按测试给定的固定时间序列依次返回，不读取真实时间；提交信息固定。
输入为 data/raw 中的 AAPL、NFLX（不提交）；没有数据时跳过。只写临时目录。

适用范围：同一 Python 版本、同一平台、同一依赖锁定；跨平台的浮点末位差异不在本测试范围内。
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from stock_risk.application import replay_run as replay_run_module
from stock_risk.application.replay_run import GitState, ReplayRunService
from stock_risk.bootstrap import build_replay_service
from stock_risk.config.settings import load_settings
from stock_risk.domain.lifecycle.params import reference_candidates
from stock_risk.domain.lifecycle.semantic_digest import semantic_digest
from stock_risk.infrastructure.replay.environment import LocalRunEnvironment
from stock_risk.infrastructure.replay.files import ReplayFileSink

SETTINGS = load_settings()
pytestmark = pytest.mark.skipif(
    not (SETTINGS.raw_data_dir / "SPY.csv").exists(), reason="本机没有 data/raw"
)
SYMBOLS = ["AAPL", "NFLX"]
PLACEHOLDER = "<占位>"
CSV_TARGETS = ("daily.csv", "events.csv", "episodes.csv")
MANIFEST_META = ("run_id", "started_at", "finished_at", "elapsed_seconds")


class FixedGitEnv(LocalRunEnvironment):
    """真实的文档与配置哈希、版本信息；提交信息固定（不随工作区状态变化）。"""

    def git_state(self) -> GitState:
        return GitState("0123456789abcdef", False, (), None)


class SeqClock:
    """按调用次数依次返回固定时间序列中的值；用完即报错（调用次数与顺序必须一致）。"""

    def __init__(self, values: list[datetime]) -> None:
        self.values, self.calls = values, 0

    def __call__(self) -> datetime:
        value = self.values[self.calls]
        self.calls += 1
        return value


def _run(root: Path, started: datetime, ends: list[datetime], monkeypatch) -> tuple:  # type: ignore[no-untyped-def]
    digests: dict[tuple[str, int], str] = {}
    by_cand = {c: i for i, c in reference_candidates()}
    original = replay_run_module.replay_prepared

    def recording(prepared, params, calendar, upto=None):  # type: ignore[no-untyped-def]
        result = original(prepared, params, calendar, upto)
        cid = by_cand.get(params.candidate)
        if cid is not None and upto is None:
            digests[(prepared.target.label, cid)] = semantic_digest(result)
        return result

    monkeypatch.setattr(replay_run_module, "replay_prepared", recording)
    run = ReplayRunService(
        build_replay_service(SETTINGS),
        ReplayFileSink(root),
        FixedGitEnv(SETTINGS.project_root),
        progress=lambda _: None,
    )
    clock = SeqClock(ends)
    report = run.run_reference(started, clock, symbols=SYMBOLS)
    monkeypatch.setattr(replay_run_module, "replay_prepared", original)
    return report, digests, clock.calls


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _strip_root(data: bytes, root: Path) -> bytes:
    """唯一允许的处理：文件中嵌入的输出根目录路径换成同一个占位符。"""
    for form in {str(root), root.as_posix()}:
        data = data.replace(form.encode("utf-8"), b"<ROOT>")
    return data


T1 = datetime(2026, 10, 17, 9, 0, 0)
END1 = [datetime(2026, 10, 17, 9, 3, 0)]


def test_same_metadata_byte_identical(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    ra, da, ca = _run(a_root, T1, list(END1), monkeypatch)
    rb, db, cb = _run(b_root, T1, list(END1), monkeypatch)
    assert ra.run_id == rb.run_id and ca == cb
    fa, fb = _files(a_root), _files(b_root)
    assert list(fa) == list(fb)  # 文件清单相同
    for name in fa:
        assert _strip_root(fa[name], a_root) == _strip_root(fb[name], b_root), name
    assert da == db and len(da) == len(SYMBOLS) * len(reference_candidates())


# ---- 测试二：只改运行元数据 ----

T2 = datetime(2026, 10, 18, 21, 30, 15)
END2 = [datetime(2026, 10, 18, 21, 47, 0)]


def _norm_csv(data: bytes, run_id: str, name: str) -> list[list[str]]:
    """CSV 的 run_id 列：每行的值必须是本次 run_id，置为占位值；run_id 不得出现在其他位置。"""
    text = data.decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    col = rows[0].index("run_id")
    for r in rows[1:]:
        assert r[col] == run_id, name
        r[col] = PLACEHOLDER
    assert text.count(run_id) == len(rows) - 1, name  # 只在 run_id 列出现
    return rows


def _norm_chart(data: bytes, run_id: str, name: str) -> bytes:
    """图表标题中的 run_id 子串：只出现一次，且位于标题文本末尾。"""
    text = data.decode("utf-8")
    assert text.count(run_id) == 1, name
    pattern = re.compile(
        r'"title":\{"font":\{"size":14\},"text":"[^"]* run ' + re.escape(run_id) + r'"\}'
    )
    assert len(pattern.findall(text)) == 1, name
    return text.replace(run_id, PLACEHOLDER).encode("utf-8")


def _norm_index(data: bytes, run_id: str) -> bytes:
    """运行索引中的 run_id：<title> 与 <h1> 各一处。"""
    text = data.decode("utf-8")
    assert text.count(run_id) == 2
    assert f"<title>回放 {run_id}</title>" in text and f"<h1>回放 {run_id}（" in text
    return text.replace(run_id, PLACEHOLDER).encode("utf-8")


def _norm_manifest(data: bytes, run_id: str) -> dict:  # type: ignore[type-arg]
    """清单的 run_id、started_at、finished_at、elapsed_seconds 逐个置为占位值；其余字段原样比较。"""
    m = json.loads(data.decode("utf-8"))
    for key in MANIFEST_META:
        assert key in m, key
        m[key] = PLACEHOLDER
    assert run_id not in json.dumps(m, ensure_ascii=False)  # run_id 不在其他字段中
    return m


def _normalized(files: dict[str, bytes], run_id: str) -> Iterator[tuple[str, object]]:
    for name, data in files.items():
        key = name.replace(run_id, PLACEHOLDER, 1)  # 输出目录名
        assert key.startswith(f"{PLACEHOLDER}/"), name
        base = name.rsplit("/", 1)[-1]
        if base in CSV_TARGETS:
            yield key, _norm_csv(data, run_id, name)
        elif base == "chart.html":
            yield key, _norm_chart(data, run_id, name)
        elif base == "index.html":
            yield key, _norm_index(data, run_id)
        elif base == "run_manifest.json":
            yield key, _norm_manifest(data, run_id)
        else:
            assert run_id.encode() not in data, name
            yield key, data


def test_only_metadata_changes(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    ra, da, ca = _run(a_root, T1, list(END1), monkeypatch)
    rb, db, cb = _run(b_root, T2, list(END2), monkeypatch)
    assert ra.run_id != rb.run_id and ca == cb  # 调用次数与顺序不变
    assert da == db  # 每个（对象, 候选）的全语义摘要不变
    na = dict(_normalized(_files(a_root), ra.run_id))
    nb = dict(_normalized(_files(b_root), rb.run_id))
    assert list(na) == list(nb)
    for key in na:
        assert na[key] == nb[key], key
