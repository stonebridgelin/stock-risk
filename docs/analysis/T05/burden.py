"""预警负担的补充描述（复核 R1 A-2、A-3、A-1、D-05 后新增）。

1. 每个有效证券年的启动次数，拆为首次启动（距上次解除 > 20 个有效交易日或此前无解除）、短期再启动（≤ 20 日）与加重次数；
2. RELEASE 按计数段每日证据分类：只有 a′、只有 b′、每日都是 a′+b′、交替混合；DIRECT 单列；
3. "最大跌幅阈值—预警次数 / 预警日数"曲线（不设门槛）：分子分母都只含以 RELEASE 结束的预警；
   截尾预警的最大跌幅只是截至观察终点的值，不计入（复核 R2-Q7 修订）；
4. 占用率：合并证券日口径与跨证券中位数口径（768 组）。
"""
import json
import pandas as pd, numpy as np
from load import RUN, targets, CANDS, OUT, daily, all_episodes, ROOT

def valid_days(c, t):
    d = daily(c, t, ['day', 'data_status', 'display'])
    return d[(d.data_status != 'MISSING')].day.tolist(), d[(d.day >= '2010-01-01') & (d.data_status != 'MISSING') & d.display.isin(['正常', '关注', '警戒'])].shape[0]

def frequency(cands=(385, 393)):
    E = pd.read_pickle(OUT / 'episodes_ref12.pkl')
    out = []
    for c in cands:
        for t in targets():
            if t == 'SHLD_uncorrected':
                continue
            vd, comp = valid_days(c, t); idx = {x: i for i, x in enumerate(vd)}
            e = E[(E.cand == c) & (E.target == t)].sort_values('start_date').reset_index(drop=True)
            first = quick = 0
            for k, r in e.iterrows():
                if r.start_date < '2010-01-01':
                    continue
                prev = e.iloc[k - 1] if k > 0 else None
                if prev is not None and prev.end_reason == 'RELEASE' and idx[r.start_date] - idx[prev.end_date] <= 20:
                    quick += 1
                else:
                    first += 1
            esc = int(e[e.start_date >= '2010-01-01'].n_escalate.sum())
            yrs = comp / 252
            out.append(dict(cand=c, target=t, years=yrs, first=first / yrs, quick=quick / yrs, escalate=esc / yrs, starts=(first + quick) / yrs))
    return pd.DataFrame(out)

def release_kinds(cands=CANDS):
    out = []
    for c in cands:
        for t in targets():
            if t == 'SHLD_uncorrected':
                continue
            ev = pd.read_csv(f'{RUN}/c{c}/{t}/events.csv', dtype=str)
            ev = ev[(ev.event_type == 'RELEASE') & (ev.date >= '2010-01-01')]
            for r in ev.itertuples():
                reasons = json.loads(r.reasons)
                if 'DIRECT_STRONG_RECOVERY' in reasons:
                    kind = 'DIRECT' if reasons == ['DIRECT_STRONG_RECOVERY'] else 'DIRECT+rc'
                else:
                    seg = [frozenset(s) for s in json.loads(r.segment)]
                    if all(s == {'a′'} for s in seg): kind = 'only_a'
                    elif all(s == {'b′'} for s in seg): kind = 'only_b'
                    elif all(s == {'a′', 'b′'} for s in seg): kind = 'both_daily'
                    else: kind = 'mixed'
                out.append(dict(cand=c, target=t, date=r.date, kind=kind))
    return pd.DataFrame(out)

def dd_curve(c=385, thresholds=(0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30)):
    E = pd.read_pickle(OUT / 'episodes_ref12.pkl')
    e = E[(E.cand == c) & E.dev & (E.target != 'SHLD_uncorrected')]
    done = e[e.end_reason == 'RELEASE']; cens = e[e.end_reason != 'RELEASE']
    rows = []
    for x in thresholds:
        m = done.dd_max < x
        rows.append(dict(threshold=x, episodes_below=int(m.sum()), share_episodes=m.mean(),
                         days_below=int(done[m].observed_valid_days.sum()),
                         share_days=done[m].observed_valid_days.sum() / done.observed_valid_days.sum()))
    return pd.DataFrame(rows), len(done), len(cens)

def occupancy_768():
    s = pd.read_csv(ROOT / 'output/replay/20261014T035713-dc5b14c/summary.csv')
    s = s[s.variant != 'uncorrected']
    med = s.groupby('candidate_id').occupancy.median()
    pooled = pd.read_csv(ROOT / 'output/replay/20261014T035713-dc5b14c/summary_by_candidate.csv').set_index('candidate_id').occupancy
    return pd.DataFrame(dict(pooled=pooled, median_across=med))

if __name__ == '__main__':
    pd.set_option('display.width', 220)
    f = frequency(); f.to_csv(OUT / 'frequency.csv', index=False)
    print(f.groupby('cand')[['first', 'quick', 'escalate', 'starts']].describe(percentiles=[.5]).round(2).T)
    k = release_kinds(); k.to_csv(OUT / 'release_kinds.csv', index=False)
    print(pd.crosstab(k.cand, k.kind, normalize='index').round(3))
    cur, nd, nc = dd_curve(); print(cur.round(3), 'released', nd, 'not released', nc)
    o = occupancy_768(); print(o.describe().round(3)); print(o.loc[[285, 291, 385, 393]].round(3))
