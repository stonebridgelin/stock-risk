"""T05 探索复盘脚本（产品经理分析用，非产品代码）。

用法：在仓库根目录依次运行 events_pct.py、coverage.py；需要 pandas（不是项目依赖）。
读取正式运行 20261014T015654-dc5b14c 的 reference12 输出（产品经理分析脚本，非产品代码）。"""
import json, os
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
RUN = str(ROOT / 'output/replay/20261014T015654-dc5b14c')
OUT = ROOT / 'output/t05'  # 中间结果（不入库）
CANDS = list(range(385, 397))
def targets():
    return sorted(os.listdir(f'{RUN}/c385'))
def episodes(c, t):
    e = pd.read_csv(f'{RUN}/c{c}/{t}/episodes.csv', dtype=str)
    ev = pd.read_csv(f'{RUN}/c{c}/{t}/events.csv', dtype=str)
    st = ev[ev.event_type == 'START'].copy()
    vals = st['values'].map(json.loads)
    st['p_ep'] = vals.map(lambda v: v.get('p_ep'))
    st['p_ep_day'] = vals.map(lambda v: v.get('p_ep_day'))
    st['a_start'] = vals.map(lambda v: v.get('a_start'))
    st['d_t'] = vals.map(lambda v: v.get('d_t'))
    st['r2_score'] = vals.map(lambda v: v.get('r2_score'))
    e = e.merge(st[['episode_id', 'p_ep', 'p_ep_day', 'a_start', 'd_t', 'r2_score']], on='episode_id', how='left')
    for col in ['start_close', 'low_close', 'end_close', 'p_ep', 'a_start', 'd_t', 'duration_trading_days',
                'observed_valid_days', 'n_recovery_obs', 'n_revoked', 'n_recovery_fail', 'n_escalate']:
        e[col] = pd.to_numeric(e[col], errors='coerce')
    e['dd_start'] = 1 - e.start_close / e.p_ep
    e['dd_max'] = 1 - e.low_close / e.p_ep
    e['atr_pct'] = e.a_start / e.p_ep
    e['dev'] = e.start_date >= '2010-01-01'
    return e
def all_episodes(cands=CANDS):
    out = []
    for c in cands:
        for t in targets():
            x = episodes(c, t); x['cand'] = c; x['target'] = t; out.append(x)
    return pd.concat(out, ignore_index=True)
def daily(c, t, cols=None):
    return pd.read_csv(f'{RUN}/c{c}/{t}/daily.csv', dtype=str, usecols=cols)
