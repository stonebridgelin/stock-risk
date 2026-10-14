"""示意事件的覆盖、通知、谷底在场、解除与预警日分布（复核 R1 后修订，v2）。

修订：事件日只计可计算日（显示为正常、关注、警戒，且非缺价），与分母一致；
谷底后的解除只取谷底当日那次预警（episode_id 相同）的 RELEASE；未解除的记录其终止原因与已观察天数；
S1 同时给出"含无新通知记 1"与"只含有新通知"两种口径；新通知区分 START 与 ESCALATE。
"""
import pandas as pd, numpy as np
from load import daily, targets, CANDS, OUT

COMPUTABLE = ['正常', '关注', '警戒']
ev = pd.read_csv(OUT / 'events_pct20.csv')
rows, burden = [], []
for c in CANDS:
    for t in targets():
        if t == 'SHLD_uncorrected':
            continue
        d = daily(c, t, ['day', 'in_replay', 'data_status', 'display', 'level', 'episode_id', 'transition', 'close', 'end_reason'])
        d = d[d.in_replay == 'true'].reset_index(drop=True)
        d['close'] = pd.to_numeric(d.close, errors='coerce')
        d['warn'] = d.level.isin(['WATCH', 'ALERT'])
        d['valid'] = d.data_status != 'MISSING'
        dv = d[(d.day >= '2010-01-01') & d.valid & d.display.isin(COMPUTABLE)]
        inev = np.zeros(len(dv), bool)
        for e in ev[ev.target == t].itertuples():
            inev |= ((dv.day >= e.peak) & (dv.day <= e.end)).values
            seg = d[(d.day >= e.peak) & (d.day < e.trough)]
            pk, depth = e.peak_close, e.peak_close - e.trough_close
            st = seg[seg.transition.isin(['T6', 'T7', 'T9', 'T10', 'T11'])]
            notice = st.iloc[0] if len(st) else None
            trow = d[d.day == e.trough].iloc[0]
            r = dict(cand=c, target=t, peak=e.peak, trough=e.trough, dd=e.dd, end_why=e.end_why, cap=e.cap,
                     censored=e.censored, computable_at_peak=bool(d.loc[d.day == e.peak, 'display'].isin(COMPUTABLE).any()),
                     covered=bool(seg.warn.any()), warn_at_peak=bool(d.loc[d.day == e.peak, 'warn'].any()),
                     notice_day=None if notice is None else notice.day,
                     notice_kind=None if notice is None else ('START' if notice.transition in ('T6', 'T7') else 'ESCALATE'),
                     s1_new=None if notice is None else (pk - notice.close) / depth,
                     warn_at_trough=bool(trow.warn), n_release_before_trough=int(seg.transition.isin(['T12', 'T14']).sum()))
            r['s1_all'] = 1.0 if notice is None else r['s1_new']
            if trow.warn:
                after = d[(d.day > e.trough) & (d.episode_id == trow.episode_id)]
                rel = after[after.transition.isin(['T12', 'T14'])].head(1)
                if len(rel):
                    rr = rel.iloc[0]
                    r['release_day'] = rr.day
                    r['release_days_after_trough'] = int(((d.day > e.trough) & (d.day <= rr.day) & d.valid).sum())
                    r['release_recovered'] = (rr.close - e.trough_close) / depth
                else:
                    last = after.iloc[-1] if len(after) else trow
                    r['unreleased_end_reason'] = last.end_reason if isinstance(last.end_reason, str) else 'OPEN'
                    r['unreleased_observed_days'] = int(((d.day > e.trough) & (d.day <= last.day) & d.valid).sum())
            rows.append(r)
        burden.append(dict(cand=c, target=t, warn_days=int(dv.warn.sum()), warn_days_outside=int((dv.warn.values & ~inev).sum()),
                           days=len(dv), event_days=int(inev.sum())))
pd.DataFrame(rows).to_csv(OUT / 'coverage_pct20.csv', index=False)
pd.DataFrame(burden).to_csv(OUT / 'burden_pct20.csv', index=False)
