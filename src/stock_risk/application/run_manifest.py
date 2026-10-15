"""回放与对照回放共用的运行清单构建（I16；T08a D-03）。

- run_id = 开始时间 + 提交号前 7 位；同名目录已存在时加序号，不覆盖；
- 工作区干净且提交号可得、并且调用方允许时为 formal，否则为 debug（P-26）；
- 清单的公共字段：运行编号与种类、模式、git 状态、开始时间、文档与配置哈希、版本、参数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from stock_risk.domain.lifecycle.params import EXPLORATION_ASSUMPTIONS, FIXED
from stock_risk.domain.numeric import DEFAULT_TOLERANCES


@dataclass(frozen=True, slots=True)
class GitState:
    commit: str | None
    dirty: bool
    changed: tuple[str, ...]
    error: str | None


class RunEnvironment(Protocol):
    def git_state(self) -> GitState: ...

    def file_hashes(self) -> dict[str, dict[str, str | None]]: ...

    def versions(self) -> dict[str, str]: ...


class RunDirSink(Protocol):
    def create_run_dir(self, run_id: str) -> Path: ...

    def run_dir_exists(self, run_id: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class RunStart:
    run_id: str
    run_kind: str
    run_dir: Path
    git: GitState


def make_run_id(started: datetime, commit: str | None) -> str:
    return f"{started:%Y%m%dT%H%M%S}-{(commit or 'nogit')[:7]}"


def start_run(
    sink: RunDirSink, env: RunEnvironment, started: datetime, formal_allowed: bool = True
) -> RunStart:
    """确定 run_id 与运行种类并建立运行目录。"""
    git = env.git_state()
    formal = formal_allowed and bool(git.commit) and not git.dirty
    run_id = base = make_run_id(started, git.commit)
    n = 2
    while sink.run_dir_exists(run_id):
        run_id = f"{base}-{n}"
        n += 1
    return RunStart(run_id, "formal" if formal else "debug", sink.create_run_dir(run_id), git)


def base_manifest(
    run: RunStart, env: RunEnvironment, mode: str, started: datetime
) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "run_kind": run.run_kind,
        "mode": mode,
        "git": asdict(run.git),
        "started_at": started.isoformat(),
        "documents_and_config": env.file_hashes(),
        "versions": env.versions(),
        "parameters": {
            "fixed_values": asdict(FIXED),
            "exploration_assumptions_I6a": {
                **asdict(EXPLORATION_ASSUMPTIONS),
                "note": "探索假设，非发起人批准值",
            },
            "numeric_tolerances_I6b": {**asdict(DEFAULT_TOLERANCES), "note": "待登记"},
        },
    }


def finish_manifest(manifest: dict[str, Any], finished: datetime) -> dict[str, Any]:
    manifest["finished_at"] = finished.isoformat()
    started = datetime.fromisoformat(manifest["started_at"])
    manifest["elapsed_seconds"] = round((finished - started).total_seconds(), 1)
    return manifest
