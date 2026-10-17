"""T08a C-03：06 号第 3.4 节的补充检查。

产品经理分析用，非产品代码（I23）。对 s385 的 V1、V3、V4 参考取值（x₁ 8%、x₂ 15%、X1、r2 OFF），
L 路径启动、以 RELEASE 结束、启动日 ≥ 2010-01-01、且解除收盘 ≤ 启动收盘的预警，检查：
① 预警低点日期不早于启动日；② 解除收盘高于启动日起（含）至解除日的最低收盘。
收盘价取自 T04b 正式运行的 V0 daily.csv。用法：python c03_lowdate.py [run_id]。
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = sys.argv[1] if len(sys.argv) > 1 else '20261014T200012-0e0d45e'
CMP = ROOT / 'output/compare' / RUN
T04 = ROOT / 'output/replay/20261014T015654-dc5b14c'
SETS = {'V1': 'V1_x8_x15_X1_r2off_N20_s385', 'V3': 'V3_x8_x15_X1_r2off_N20_s385', 'V4': 'V4_x8_x15_X1_r2off_N20_s385'}
_cl = {}


def closes(t):
    if t not in _cl:
        d = pd.read_csv(T04 / 'c385' / t / 'daily.csv', usecols=['day', 'close'], dtype={'day': str})
        _cl[t] = d.assign(close=pd.to_numeric(d.close, errors='coerce')).dropna().set_index('day').close
    return _cl[t]


for name, sid in SETS.items():
    ep = pd.read_csv(CMP / 'sets' / sid / 'episodes.csv', dtype=str)
    ep = ep[(ep.entry_path == 'L') & (ep.end_reason == 'RELEASE') & (ep.start_date >= '2010-01-01')]
    ep = ep[pd.to_numeric(ep.end_close) <= pd.to_numeric(ep.start_close)]
    low_ok = (ep.low_date >= ep.start_date)
    rose = []
    for e in ep.itertuples():
        c = closes('SHLD_corrected' if e.target.startswith('SHLD') else e.target)
        w = c[(c.index >= e.start_date) & (c.index <= e.end_date)]
        rose.append(float(e.end_close) > w.min())
    print(name, '符合筛选', len(ep), '低点不早于启动日', int(low_ok.sum()), '解除收盘高于启动后最低收盘', int(sum(rose)))
