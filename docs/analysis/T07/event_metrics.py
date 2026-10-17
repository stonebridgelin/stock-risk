"""T07：在示意事件集（T05 的 events_pct20.csv，190 个）上，计算对照回放各套的覆盖、新通知、谷底在场与解除指标。

产品经理分析用，非产品代码（I23）。读取对照正式运行（默认 T08a 的 20261014T200012-0e0d45e，可用第一个参数指定 run_id）；收盘价取自 T04b 正式运行的 V0 daily.csv。
全部套数都计算（含 N14，C-04）；旧版只算 N20。
用法：仓库根目录先运行 docs/analysis/T05/events_pct.py，再运行本脚本；结果写到 output/t07/。
口径与 docs/analysis/T05/coverage.py 相同（新通知 = [峰值日, 谷底日) 内第一个 START 或升为警戒的 ESCALATE）。
"""
import json, sys, os
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
RUN = sys.argv[1] if len(sys.argv) > 1 else '20261014T200012-0e0d45e'
CMP = ROOT / 'output/compare' / RUN
T04 = ROOT / 'output/replay/20261014T015654-dc5b14c'
OUT = ROOT / 'output/t07'
EV = pd.read_csv(ROOT / 'output/t05/events_pct20.csv')

def closes():
    out = {}
    for t in EV.target.unique():
        d = pd.read_csv(T04 / 'c385' / t / 'daily.csv', usecols=['day', 'close'], dtype={'day': str})
        out[t] = dict(zip(d.day, pd.to_numeric(d.close, errors='coerce')))
    return out

CL = None

def init():
    global CL
    CL = closes()

def one(set_id):
    d = pd.read_csv(CMP / 'sets' / set_id / 'daily_compact.csv.gz', dtype=str)
    d['warn'] = d.level.isin(['1', '2', 'WATCH', 'ALERT'])
    ev = pd.read_csv(CMP / 'sets' / set_id / 'events.csv', dtype=str, usecols=['target', 'date', 'event_type', 'episode_id', 'level_after'])
    rows = []
    for t, dt in d.groupby('target'):
        tgt = 'SHLD_corrected' if t.startswith('SHLD') else t
        evs = EV[EV.target == tgt]
        if evs.empty:
            continue
        dt = dt.set_index('day'); e = ev[ev.target == t]
        notices = e[(e.event_type == 'START') | ((e.event_type == 'ESCALATE') & (e.level_after == 'ALERT'))]
        rels = e[e.event_type == 'RELEASE']
        days = dt.index.values; valid = (dt.data_status != 'MISSING').values
        for x in evs.itertuples():
            seg = dt.loc[(dt.index >= x.peak) & (dt.index < x.trough)]
            n = notices[(notices.date >= x.peak) & (notices.date < x.trough)].sort_values('date').head(1)
            depth = x.peak_close - x.trough_close
            r = dict(set_id=set_id, target=tgt, peak=x.peak, trough=x.trough, dd=x.dd, cap=x.cap, censored=x.censored,
                     covered=bool(seg.warn.any()), warn_at_trough=bool(dt.loc[x.trough, 'warn']) if x.trough in dt.index else False,
                     n_release_before=int(((rels.date >= x.peak) & (rels.date < x.trough)).sum()))
            if len(n):
                nd = n.iloc[0].date; r['notice_day'] = nd; r['notice_kind'] = n.iloc[0].event_type
                r['s1_new'] = (x.peak_close - CL[tgt][nd]) / depth
            if r['warn_at_trough']:
                epi = dt.loc[x.trough, 'episode_id']
                rr = rels[(rels.episode_id == epi) & (rels.date > x.trough)].sort_values('date').head(1)
                if len(rr):
                    rd = rr.iloc[0].date; r['release_day'] = rd
                    r['release_wait'] = int(((days > x.trough) & (days <= rd) & valid).sum())
                    r['release_recovered'] = (CL[tgt][rd] - x.trough_close) / depth
                else:
                    r['unreleased'] = True
            rows.append(r)
    return rows

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.load(open(CMP / 'run_manifest.json'))
    sets = [s['set_id'] if isinstance(s, dict) else s for s in m['variant_sets']]
    with Pool(6, initializer=init) as p:
        res = p.map(one, sets, chunksize=4)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(OUT / 'event_metrics.csv', index=False)
    print(len(sets), len(df))
