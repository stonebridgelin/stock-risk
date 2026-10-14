"""运行环境信息（实现 application.replay_run.RunEnvironment）：git 提交号与工作区状态、文件哈希、
版本。

git 可执行文件：环境变量 GIT_EXECUTABLE 优先，否则在 PATH 中查找。找不到 git 时视为工作区状态未知，
运行标为 debug（I16：只有干净工作区的运行才是正式运行）。
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from stock_risk.application.replay_run import GitState

PACKAGES = ("stock-risk", "plotly", "exchange-calendars")

# 运行清单中记录哈希的文件（I16）：规格与夹具文档、计划、配置
TRACKED_FILES = {
    "spec_10": "docs/develop_docs/10-算法与评估规格_v0.7.md",
    "fixtures_12": "docs/develop_docs/12-验收夹具_v0.6.md",
    "fixtures_13": "docs/develop_docs/13-验收夹具补充_v0.7.1.md",
    "engine_14": "docs/develop_docs/14-生命周期引擎实现说明.md",
    "plan_03": "docs/develop_docs/03-探索期开发计划.md",
    "engineering_11": "docs/develop_docs/11-工程规格_v0.4.md",
    "listing_periods": "config/listing_periods.csv",
    "symbol_segments": "config/symbol_segments.csv",
    "price_corrections": "config/price_corrections.csv",
    "universe": "config/universe_exploration.csv",
    "data_manifest": "data/raw/MANIFEST.sha256",
    "calendar_spy": "data/raw/SPY.csv",
}


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


class LocalRunEnvironment:
    def __init__(self, project_root: Path) -> None:
        self._root = project_root

    def _git(self, *args: str) -> str:
        exe = os.environ.get("GIT_EXECUTABLE") or shutil.which("git")
        if not exe:
            raise FileNotFoundError("找不到 git（设置 GIT_EXECUTABLE 或加入 PATH）")
        out = subprocess.run(
            [exe, "-C", str(self._root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return out.stdout

    def git_state(self) -> GitState:
        try:
            commit = self._git("rev-parse", "HEAD").strip()
            changed = tuple(ln for ln in self._git("status", "--porcelain").splitlines() if ln)
        except (OSError, subprocess.CalledProcessError) as e:
            return GitState(commit=None, dirty=True, changed=(), error=str(e))
        return GitState(commit=commit, dirty=bool(changed), changed=changed, error=None)

    def file_hashes(self) -> dict[str, dict[str, str | None]]:
        return {k: {"path": p, "sha256": _sha256(self._root / p)} for k, p in TRACKED_FILES.items()}

    def versions(self) -> dict[str, str]:
        out = {"python": platform.python_version(), "platform": platform.platform()}
        for name in PACKAGES:
            try:
                out[name] = version(name)
            except PackageNotFoundError:
                out[name] = "未安装"
        return out
