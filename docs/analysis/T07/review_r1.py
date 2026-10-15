"""T07 复核 R1 的补充统计（产品经理分析用，非产品代码，I23）。在仓库根目录运行；需要先运行 event_metrics.py。"""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]
CMP = ROOT / 'output/compare/20261014T064853-5de21a7'
T04 = ROOT / 'output/replay/20261014T015654-dc5b14c/c385'
A = 'V0_x0_x0_OFF_r2off_N20_s385'; B = 'V3_x8_x15_X1_r2off_N20_s385'; C4 = 'V4_x8_x15_X1_r2off_N20_s385'; B1 = 'V1_x8_x15_X1_r2off_N20_s385'
EM = pd.read_csv(ROOT / 'output/t07/event_metrics.csv')
EV = pd.read_csv(ROOT / 'output/t05/events_pct20.csv')
_cl = {}
def closes(t):
    if t not in _cl:
        d = pd.read_csv(T04 / t / 'daily.csv', usecols=['day', 'close', 'data_status'])
        _cl[t] = d[d.close.notna()].reset_index(drop=True)
    return _cl[t]
def events(s):
    return pd.read_csv(CMP / 'sets' / s / 'events.csv', dtype=str)
def levels(s):
    return pd.read_csv(CMP / 'sets' / s / 'daily_compact.csv.gz', dtype=str).set_index(['target', 'day']).level

def q1():
    lb = levels(B); ea = events(A); eb = events(B)
    a = EM[EM.set_id == A].set_index(['target', 'peak']); b = EM[EM.set_id == B].set_index(['target', 'peak'])
    j = a[['notice_day', 'notice_kind', 'cap', 'censored']].join(b[['notice_day', 'notice_kind']], rsuffix='_n').dropna(subset=['notice_day'])
    rows = []
    for (t, p), r in j.iterrows():
        if pd.isna(r.notice_day_n) or not (r.notice_day_n > r.notice_day) or lb.get((t, r.notice_day), '0') != '0':
            continue
        d = closes(t); i0 = d.index[d.day == r.notice_day][0]; i1 = d.index[d.day == r.notice_day_n][0]
        c0 = d.close[i0]
        rows.append(dict(target=t, peak=p, v0_day=r.notice_day, v0_kind=r.notice_kind, v3_day=r.notice_day_n, v3_kind=r.notice_kind_n,
                         delay=i1 - i0, end_drop=1 - d.close[i1] / c0, min_drop=1 - d.close[i0:i1 + 1].min() / c0,
                         complete_event=(not r.cap) and pd.isna(r.censored)))
    k = pd.DataFrame(rows); k.to_csv(ROOT / 'output/t07/q1_delayed.csv', index=False)
    print('Q1 n', len(k), 'kind pairs', k.groupby(['v0_kind', 'v3_kind']).size().to_dict())
    for name, x in [('all', k), ('complete', k[k.complete_event])]:
        print(f' {name} n={len(x)} delay q', x.delay.quantile([.25, .5, .75, .9]).tolist(), 'end_drop q', x.end_drop.quantile([.25, .5, .75, .9]).round(4).tolist(),
              'min_drop q', x.min_drop.quantile([.5, .9]).round(4).tolist(), 'max', round(x.end_drop.max(), 4), round(x.min_drop.max(), 4))
    cnt = EM[EM.set_id.isin([A, B])]; full = cnt[(~cnt.cap) & cnt.censored.isna()]
    for s in [A, B]:
        x = full[full.set_id == s]; print(' complete events', s[:2], len(x), 'covered', x.covered.sum(), 'new', x.notice_day.notna().sum(), 'trough', x.warn_at_trough.sum(), 's1 med', round(x.s1_new.median(), 3))

def q2():
    f = pd.read_csv(CMP / 'release_followups.csv')
    for s in [A, B1, B, C4]:
        x = f[f.set_id == s]
        c20 = x[x.complete_20 == True]; c60 = x[x.complete_60 == True]
        print('Q2', s[:2], 'releases', len(x), 'complete20', len(c20), 'incomplete20', len(x) - len(c20), 'complete60', len(c60),
              'drop20 q', c20.drop_20.quantile([.1, .25, .5]).round(3).tolist(), 'drop60 q', c60.drop_60.quantile([.1, .25, .5]).round(3).tolist(),
              'share drop60<=-10%', round((c60.drop_60 <= -0.10).mean(), 3))
    u = pd.read_csv(CMP / 'unreleased.csv'); print(' unreleased', u[u.set_id.isin([A, B])].groupby('set_id').size().to_dict())
    # 共同事件配对：同一示意事件，谷底后首次解除
    a = EM[EM.set_id == A].set_index(['target', 'peak']); b = EM[EM.set_id == B].set_index(['target', 'peak'])
    j = a[['trough', 'release_day', 'cap', 'censored']].join(b[['release_day']], rsuffix='_n')
    n_all = len(j); j = j.dropna(subset=['release_day', 'release_day_n'])
    rows = []
    for (t, p), r in j.iterrows():
        d = closes(t); tr = EV[(EV.target == t) & (EV.peak == p)].iloc[0].trough_close
        for tag, day in [('v0', r.release_day), ('v3', r.release_day_n)]:
            i = d.index[d.day == day][0]
            if i + 60 >= len(d):
                rows.append(dict(target=t, peak=p, who=tag, complete=False)); continue
            c = d.close[i]; w = d.close[i + 1:i + 61]
            rows.append(dict(target=t, peak=p, who=tag, complete=True, drop60=w.min() / c - 1, below_trough=bool((w < tr).any()), release_close=c))
    x = pd.DataFrame(rows)
    ok = x.groupby(['target', 'peak']).complete.all(); pairs = ok[ok].index
    y = x.set_index(['target', 'peak']).loc[pairs].reset_index()
    p = y.pivot_table(index=['target', 'peak'], columns='who', values=['drop60', 'below_trough', 'release_close'], aggfunc='first')
    print(' paired events', len(p), 'excluded (no release in one/both)', n_all - len(j), 'excluded (window incomplete)', len(j) - len(p))
    print('  below event trough within 60d: v0', int(p.below_trough.v0.sum()), 'v3', int(p.below_trough.v3.sum()))
    print('  drop60 median v0', round(p.drop60.v0.median(), 3), 'v3', round(p.drop60.v3.median(), 3), ' v3 release later/same/earlier:',
          int((y[y.who == 'v3'].set_index(['target', 'peak']).index.size)))
    same = (j.release_day == j.release_day_n).sum(); later = (j.release_day_n > j.release_day).sum(); earlier = (j.release_day_n < j.release_day).sum()
    print('  release day same/later/earlier', same, later, earlier)

def q2c():
    c = pd.read_csv(CMP / 'special_cancelled_escalations.csv'); x = c[c.set_id == B]
    print('Q2c cancelled rows', len(x), x.new_status.value_counts().to_dict(), 'no alert coverage', int((x.new_status != '警戒').sum()))

def q3():
    l = pd.read_csv(CMP / 'special_l_path_starts.csv'); x = l[l.set_id == B].copy()
    ea = events(A); starts_a = ea[ea.event_type == 'START']
    rows = []
    for r in x.itertuples():
        d = closes(r.target); i = d.index[d.day == r.date][0]; obs = len(d) - 1 - i
        w = d.close[i + 1:i + 61]; c = d.close[i]
        nxt = starts_a[(starts_a.target == r.target) & (starts_a.date > r.date)].date.min()
        lvl0 = r.v0_status
        rows.append(dict(target=r.target, date=r.date, v0_status=lvl0, observed=obs, complete60=obs >= 60,
                         min60=w.min() / c - 1, end60=(w.iloc[-1] / c - 1) if obs >= 60 else None, max60=w.max() / c - 1,
                         v0_next_start=nxt, v0_wait=(d.index[d.day == nxt][0] - i) if isinstance(nxt, str) else None))
    k = pd.DataFrame(rows); k.to_csv(ROOT / 'output/t07/q3_l_starts.csv', index=False)
    cpl = k[k.complete60]
    print('Q3 L starts', len(k), 'stocks', k.target.nunique(), 'complete60', len(cpl), 'incomplete', k[~k.complete60][['target', 'date', 'observed']].values.tolist())
    print(' complete: min60 q', cpl.min60.quantile([.25, .5, .75]).round(3).tolist(), 'end60 q', cpl.end60.quantile([.25, .5, .75]).round(3).tolist(),
          'max60 q', cpl.max60.quantile([.25, .5, .75]).round(3).tolist())
    z = k[k.v0_status == '正常']
    print(' V0 正常 cases', len(z), 'V0 next START wait q', z.v0_wait.quantile([.25, .5, .75]).tolist(), 'none', int(z.v0_wait.isna().sum()))
    print(' per stock', k.groupby('target').size().sort_values(ascending=False).to_dict())

def q4():
    for s in [B1, B, C4]:
        ep = pd.read_csv(CMP / 'sets' / s / 'episodes.csv', dtype=str); ev = events(s)
        ep = ep[(ep.entry_path == 'L') & (ep.end_reason == 'RELEASE') & (ep.start_date >= '2010-01-01')]
        ep = ep[pd.to_numeric(ep.end_close) <= pd.to_numeric(ep.start_close)]
        rel = ev[ev.event_type == 'RELEASE'].set_index('episode_id')
        kinds = []
        for e in ep.itertuples():
            r = rel.loc[e.episode_id]; reasons = json.loads(r.reasons); seg = [frozenset(x) for x in json.loads(r.segment)] if isinstance(r.segment, str) else []
            low = float(e.low_close); rose = float(e.end_close) > low * 1.0000001
            if 'DIRECT_STRONG_RECOVERY' in reasons: k = 'DIRECT'
            elif seg and all(x == {'a′'} for x in seg): k = '只有 a′'
            elif seg and any('b′' in x for x in seg): k = '含 b′'
            else: k = '其他'
            kinds.append((k, rose))
        kk = pd.Series([f'{a}|{"有回升" if b else "无回升"}' for a, b in kinds]).value_counts().to_dict()
        print('Q4', s[:2], len(ep), kk)

def q5():
    noti = ['START', 'ESCALATE', 'DOWNGRADE', 'RELEASE', 'STABILIZED', 'RECOVERY_OBS', 'RECOVERY_REVOKED', 'RECOVERY_FAIL']
    for s in [A, B]:
        ev = events(s); ev = ev[(ev.date >= '2010-01-01') & ev.event_type.isin(noti)]
        days = ev.groupby(['target', 'date']).size()
        main = ev[ev.event_type.isin(['START', 'ESCALATE', 'DOWNGRADE', 'RELEASE'])]
        prog = ev[~ev.event_type.isin(['START', 'ESCALATE', 'DOWNGRADE', 'RELEASE'])]
        pd_days = prog.groupby(['target', 'date']).size()
        print('Q5', s[:2], 'events', len(ev), 'stock-days with any', len(days), 'merge saves', len(ev) - len(days),
              '| main', len(main), 'progress', len(prog), 'progress stock-days', len(pd_days), '| progress days not sharing a main-event day',
              len(set(pd_days.index) - set(main.groupby(['target', 'date']).size().index)))

if __name__ == '__main__' and 'r2' not in __import__('sys').argv:
    q1(); q2(); q2c(); q3(); q4(); q5()


def r2_fixes():
    """复核 R2 的补充：配对日期按 60 日完整的 174 个统计；不利结果的配对四格；非上限、非截尾子集。"""
    a = EM[EM.set_id == A].set_index(['target', 'peak']); b = EM[EM.set_id == B].set_index(['target', 'peak'])
    j = a[['release_day', 'cap', 'censored']].join(b[['release_day']], rsuffix='_n').dropna(subset=['release_day', 'release_day_n'])
    rows = []
    for (t, p), r in j.iterrows():
        d = closes(t); tr = EV[(EV.target == t) & (EV.peak == p)].iloc[0].trough_close
        rec = dict(target=t, peak=p, same=r.release_day == r.release_day_n, later=r.release_day_n > r.release_day,
                   full=(not r.cap) and pd.isna(r.censored))
        ok = True
        for tag, day in [('v0', r.release_day), ('v3', r.release_day_n)]:
            i = d.index[d.day == day][0]
            if i + 60 >= len(d): ok = False; break
            w = d.close[i + 1:i + 61]; rec[tag] = bool((w < tr).any())
        if ok: rows.append(rec)
    x = pd.DataFrame(rows)
    for name, y in [('174', x), ('non-cap/censored', x[x.full])]:
        print('R2', name, len(y), 'same', int(y.same.sum()), 'v3 later', int(y.later.sum()), 'v3 earlier', int((~y.same & ~y.later).sum()),
              '| both adverse', int((y.v0 & y.v3).sum()), 'only v0', int((y.v0 & ~y.v3).sum()), 'only v3', int((~y.v0 & y.v3).sum()), 'neither', int((~y.v0 & ~y.v3).sum()))
    k = pd.read_csv(ROOT / 'output/t07/q3_l_starts.csv'); c = k[k.complete60]
    print('R2 L: complete', len(c), 'min60<0', int((c.min60 < 0).sum()), '>0', int((c.min60 > 0).sum()), '=0', int((c.min60 == 0).sum()))


if __name__ == '__main__' and 'r2' in __import__('sys').argv:
    r2_fixes()
