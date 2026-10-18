"""T10-M0b 第 4 节：领域层纯净的自动检查（红线 5）。

用 ast 扫描 src/stock_risk/domain 下全部 .py 文件：导入白名单、禁止通配导入，禁止读取时钟、
非确定的随机数、内置 I/O 与动态执行，禁止对时间与随机模块（或其类）动态取属性。

**说明**：这是静态护栏，只检查源码中的写法，不能证明任意 Python 代码都是纯净的。
例如以变量传入的种子是否合法（Random(seed) 中 seed 的取值），仍由调用方的入口契约保证。

导入白名单（标准库）要增加模块时，由产品经理批准，并写进下面 ALLOWED_STDLIB 的注释。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DOMAIN = Path(__file__).resolve().parents[2] / "src" / "stock_risk" / "domain"

# 产品经理批准的标准库白名单（T10-M0b 第 4 节第 1 项）
ALLOWED_STDLIB = frozenset(
    {
        "__future__",
        "bisect",
        "collections",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "hashlib",
        "itertools",
        "json",
        "math",
        "random",
        "re",
        "statistics",
        "typing",
    }
)
BANNED_BUILTINS = frozenset({"open", "print", "input", "eval", "exec", "__import__", "breakpoint"})
CLOCK_METHODS = frozenset({"now", "today", "utcnow"})
TIME_CLASSES = frozenset({"datetime.datetime", "datetime.date"})
UTC_NAMES = frozenset({"datetime.UTC", "datetime.timezone.utc"})
# getattr / hasattr 的第一个参数不得是这些（按别名解析后的原名）
NO_DYNAMIC_ATTR = frozenset(
    {"datetime", "datetime.datetime", "datetime.date", "time", "random", "random.Random"}
)


def _allowed_module(name: str) -> bool:
    return (
        name in ALLOWED_STDLIB
        or name == "stock_risk.domain"
        or name.startswith("stock_risk.domain.")
    )


class _Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}
        self.problems: list[str] = []

    # ---- 导入与别名 ----

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for a in node.names:
            if not _allowed_module(a.name):
                self._bad(node, f"导入不在白名单：{a.name}")
            if a.asname:
                self.aliases[a.asname] = a.name
            else:
                top = a.name.split(".")[0]
                self.aliases[top] = top

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        if node.level == 0 and not _allowed_module(module):
            self._bad(node, f"导入不在白名单：{module}")
        for a in node.names:
            if a.name == "*":
                self._bad(node, f"通配导入：from {module} import *")
                continue
            self.aliases[a.asname or a.name] = f"{module}.{a.name}" if module else a.name

    # ---- 名字解析 ----

    def resolve(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    def _bad(self, node: ast.AST, msg: str) -> None:
        self.problems.append(f"第 {getattr(node, 'lineno', '?')} 行：{msg}")

    # ---- 调用 ----

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if (
            isinstance(func, ast.Name)
            and func.id in BANNED_BUILTINS
            and func.id not in self.aliases
        ):
            self._bad(node, f"禁止的内置函数：{func.id}")
        name = self.resolve(func)
        if name:
            self._check_call(node, name)
        if isinstance(func, ast.Name) and func.id in ("getattr", "hasattr") and node.args:
            target = self.resolve(node.args[0])
            if target in NO_DYNAMIC_ATTR:
                self._bad(node, f"{func.id} 作用于时间或随机模块：{target}")
        self.generic_visit(node)

    def _check_call(self, node: ast.Call, name: str) -> None:
        base, _, method = name.rpartition(".")
        if base in TIME_CLASSES and method in CLOCK_METHODS:
            self._bad(node, f"读取当前时间：{name}")
        elif base in TIME_CLASSES and method == "utcfromtimestamp":
            self._bad(node, f"禁止 utcfromtimestamp：{name}")
        elif base in TIME_CLASSES and method == "fromtimestamp":
            self._check_fromtimestamp(node, name)
        elif name == "random.Random":
            self._check_random(node)
        elif name.startswith("random."):
            self._bad(node, f"random 模块级函数或 SystemRandom：{name}")

    def _check_fromtimestamp(self, node: ast.Call, name: str) -> None:
        tz: ast.expr | None = None
        if len(node.args) >= 2:
            tz = node.args[1]
        for kw in node.keywords:
            if kw.arg == "tz":
                tz = kw.value
        if tz is None:
            self._bad(node, f"{name} 没有时区参数")
        elif self.resolve(tz) not in UTC_NAMES:
            self._bad(node, f"{name} 的时区不是 UTC：{ast.unparse(tz)}")

    def _check_random(self, node: ast.Call) -> None:
        seed: ast.expr | None = node.args[0] if node.args else None
        for kw in node.keywords:
            if kw.arg in ("x", "seed"):
                seed = kw.value
        if seed is None:
            self._bad(node, "Random() 没有种子")
        elif isinstance(seed, ast.Constant) and seed.value is None:
            self._bad(node, "Random(None)：种子为 None")


def check_source(source: str) -> list[str]:
    checker = _Checker()
    checker.visit(ast.parse(source))
    return checker.problems


def _domain_files() -> list[Path]:
    return sorted(DOMAIN.rglob("*.py"))


def test_domain_sources_are_pure() -> None:
    files = _domain_files()
    assert len(files) > 30  # 扫描到了领域层
    problems = {
        str(p.relative_to(DOMAIN)): found
        for p in files
        if (found := check_source(p.read_text("utf-8")))
    }
    assert problems == {}


# ---- 自检：每条禁止规则至少一例 ----


@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "import time",
        "from pathlib import Path",
        "from random import *",
        "open('x')",
        "print(1)",
        "input()",
        "eval('1')",
        "exec('x = 1')",
        "__import__('os')",
        "breakpoint()",
        "from datetime import datetime\ndatetime.now()",
        "from datetime import date\ndate.today()",
        "import datetime\ndatetime.datetime.utcnow()",
        "from datetime import datetime as dt\ndt.now()",
        "from datetime import datetime\ndatetime.utcfromtimestamp(0)",
        "from datetime import datetime\ndatetime.fromtimestamp(0)",
        "from datetime import date\ndate.fromtimestamp(0)",
        "from datetime import datetime\ndatetime.fromtimestamp(0, tz=other_tz)",
        "from datetime import datetime, timezone, timedelta\n"
        "datetime.fromtimestamp(0, timezone(timedelta(hours=8)))",
        "import random as r\nr.random()",
        "import random\nrandom.choice([1])",
        "from random import choice as pick\npick(x)",
        "import random\nrandom.SystemRandom()",
        "import random\nrandom.Random()",
        "from random import Random\nRandom(None)",
        "from random import Random\nRandom(seed=None)",
        "from datetime import datetime\ngetattr(datetime, 'now')()",
        "import random\ngetattr(random, 'random')()",
        "import datetime as d\nhasattr(d, 'datetime')",
        "from random import Random\ngetattr(Random, 'seed')",
    ],
)
def test_checker_detects_violation(source: str) -> None:
    assert check_source(source), source


@pytest.mark.parametrize(
    "source",
    [
        "from datetime import UTC, datetime\ndatetime.fromtimestamp(0, tz=UTC)",
        "from datetime import datetime, timezone\ndatetime.fromtimestamp(0, tz=timezone.utc)",
        "from datetime import UTC, datetime\ndatetime.fromtimestamp(0, UTC)",
        "import datetime as d\nd.datetime.fromtimestamp(0, tz=d.UTC)",
        "class A:\n    def f(self, name):\n        return getattr(self, name)",
        "import random\nrandom.Random(seed)",
        "from stock_risk.domain.numeric import quantize_price",
    ],
)
def test_checker_allows(source: str) -> None:
    assert check_source(source) == [], source
