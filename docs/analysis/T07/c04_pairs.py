"""T08a C-04：单开关配对按基准体系分层报告；N14 / N20 配对检查（12 种结构组合）。

产品经理分析用，非产品代码（I23）。先运行 event_metrics_chunk.py 生成 output/t07/event_metrics.csv。
用法：python c04_pairs.py [run_id]，结果写 output/t07/c04_pairs_by_system.csv 与 c04_n14.csv。
口径：
- 配对差 = b − a；表中为同一层内全部配对的中位数。
- S1_new：每套在示意事件（峰值日 ≥ 2010-01-01、未截尾，185 个，含 10 个上限事件）上的 s1_new 中位数（只计有新通知的事件）；s1_new_diff 是两套各自中位数之差。
- s1_paired_diff：同一事件两套都有新通知时的逐事件配对差，每对取中位数后再取层内中位数。
- 谷底前解除：每套中 [峰值日, 谷底日) 内有 RELEASE 的事件数。
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = sys.argv[1] if len(sys.argv) > 1 else '20261014T200012-0e0d45e'
CMP = ROOT / 'output/compare' / RUN
OUT = ROOT / 'output/t07'

em = pd.read_csv(OUT / 'event_metrics.csv', dtype={'peak': str})
em = em[(em.peak >= '2010-01-01') & em.censored.isna()]
per_set = em.groupby('set_id').agg(s1_new=('s1_new', 'median'), rel_before=('n_release_before', lambda x: int((x > 0).sum())))

ps = pd.read_csv(CMP / 'pairs_summary.csv')
ps['system'] = ps.set_a.str.split('_').str[0] + '→' + ps.set_b.str.split('_').str[0]
for k in ['s1_new', 'rel_before']:
    ps[k + '_diff'] = ps.set_b.map(per_set[k]) - ps.set_a.map(per_set[k])
cols = ['per_year_start_diff', 'per_year_start_short_restart_diff', 'per_year_escalate_diff', 'occupancy_pooled_diff',
        'below_l_ep_20_diff', 's1_new_diff', 'rel_before_diff']
# 逐事件配对差（Gemini 第二阶段审核 C-01）：同一事件两套都有新通知时的 s1_new 差，先取每对的中位数，再取层内中位数
ev = em.dropna(subset=['s1_new']).set_index(['set_id', 'target', 'peak']).s1_new
def paired(r):
    a = ev.xs(r.set_a, level='set_id') if r.set_a in ev.index.get_level_values(0) else None
    b = ev.xs(r.set_b, level='set_id') if r.set_b in ev.index.get_level_values(0) else None
    if a is None or b is None:
        return float('nan')
    j = pd.concat([a, b], axis=1, keys=['a', 'b']).dropna()
    return float((j.b - j.a).median()) if len(j) else float('nan')
ps['s1_paired_diff'] = [paired(r) for r in ps.itertuples()]
cols.append('s1_paired_diff')
t = ps.groupby(['pair_kind', 'system'])[cols].median().round(3)
t.insert(0, 'n_pairs', ps.groupby(['pair_kind', 'system']).size())
t.to_csv(OUT / 'c04_pairs_by_system.csv')
print(t.to_string())

# N14 / N20：同一结构组合下新通知日期是否相同
rows = []
nn = ps[ps.pair_kind == 'N']
for r in nn.itertuples():
    a = em[em.set_id == r.set_a].set_index(['target', 'peak']).notice_day
    b = em[em.set_id == r.set_b].set_index(['target', 'peak']).notice_day
    j = pd.concat([a, b], axis=1, keys=['a', 'b'])
    same = int(((j.a == j.b) | (j.a.isna() & j.b.isna())).sum())
    rows.append(dict(set_a=r.set_a, set_b=r.set_b, events=len(j), same_notice=same, occ_diff=r.occupancy_pooled_diff))
n = pd.DataFrame(rows)
n.to_csv(OUT / 'c04_n14.csv', index=False)
print(n.to_string())
