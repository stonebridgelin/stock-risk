"""T07：L 路径的"有无"对照（复核 T7-Q3）。产品经理分析用，非产品代码（I23）。

在 T06 的 624 套之外，临时增加 V3 参考取值但 long_path = OFF 的 12 套（结构组合 385–396），
用 `stock-risk compare` 的同一套代码运行；因为指定了套，运行自动标为 debug，只用于本复盘。
用法（仓库根目录）：uv run python docs/analysis/T07/run_l_off.py
"""
from dataclasses import replace
from datetime import datetime

import stock_risk.application.compare.service as svc
from stock_risk.application.compare import variants as V
from stock_risk.bootstrap import build_compare_service
from stock_risk.config.settings import load_settings
from stock_risk.domain.lifecycle.params import LongPath

_orig = V.variant_sets


def patched():
    base = _orig()
    grid = {s.candidate_id: s.candidate for s in base if s.n == 20}
    extra = tuple(V.VariantSet("V3", "V3", replace(V.V3_REFERENCE, long_path=LongPath.OFF), i, grid[i])
                  for i in V.STRUCTURE_N20)
    return base + extra


svc.variant_sets = patched
import sys
sel = [int(a) for a in sys.argv[1:]] or list(V.STRUCTURE_N20)
ids = [f"V3_x8_x15_OFF_r2off_N20_s{i}" for i in sel]
service = build_compare_service(load_settings())
m = service.run(datetime.now().astimezone(), lambda: datetime.now().astimezone(), workers=6, only_sets=ids)
print(m["run_id"], m["run_kind"], m["elapsed_seconds"], len(m["failed"]), m["invariant_violations"])
