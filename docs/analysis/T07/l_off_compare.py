"""T07：L 路径有无的对照（V3 参考取值，long_path X1 与 OFF）。读取 run_l_off.py 产生的调试运行。"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import event_metrics as em
ROOT = em.ROOT
OFF_RUN = ROOT / 'output/compare/20261015T002211-aa5a5c1'
def metrics(run, s):
    em.CMP = run; return pd.DataFrame(em.one(s))
if __name__ == '__main__' and 'four' not in sys.argv:
    em.init()
    bon = pd.read_csv(em.CMP / 'burden_by_set.csv').set_index('set_id'); boff = pd.read_csv(OFF_RUN / 'burden_by_set.csv').set_index('set_id')
    rows = []
    base = em.CMP
    for i in range(385, 397):
        on = f'V3_x8_x15_X1_r2off_N20_s{i}'; off = f'V3_x8_x15_OFF_r2off_N20_s{i}'
        mon = metrics(base, on); moff = metrics(OFF_RUN, off); em.CMP = base
        j = mon.set_index(['target', 'peak'])[['notice_day', 's1_new']].join(moff.set_index(['target', 'peak'])[['notice_day', 's1_new']], rsuffix='_off')
        rows.append(dict(s=i, start_on=bon.loc[on, 'per_year_start'], start_off=boff.loc[off, 'per_year_start'],
                         occ_on=bon.loc[on, 'occupancy_pooled'], occ_off=boff.loc[off, 'occupancy_pooled'],
                         new_on=mon.notice_day.notna().sum(), new_off=moff.notice_day.notna().sum(),
                         s1_on=mon.s1_new.median(), s1_off=moff.s1_new.median(),
                         earlier_with_L=int((j.notice_day < j.notice_day_off).sum()), later_with_L=int((j.notice_day > j.notice_day_off).sum()),
                         covered_on=mon.covered.sum(), covered_off=moff.covered.sum(), trough_on=mon.warn_at_trough.sum(), trough_off=moff.warn_at_trough.sum()))
    r = pd.DataFrame(rows); r.to_csv(ROOT / 'output/t07/l_off_compare.csv', index=False)
    pd.set_option('display.width', 250); print(r.round(3).to_string())


def four_class():
    """复核 R2：新通知的四类（双方都有 / 仅启用 L 有 / 仅关闭 L 有 / 都没有），以及提前事件缩短的有效交易日。"""
    em.init(); base = em.CMP; out = []
    for i in range(385, 397):
        on = metrics(base, f'V3_x8_x15_X1_r2off_N20_s{i}').set_index(['target', 'peak'])
        off = metrics(OFF_RUN, f'V3_x8_x15_OFF_r2off_N20_s{i}').set_index(['target', 'peak']); em.CMP = base
        j = on[['notice_day']].join(off[['notice_day']], rsuffix='_off')
        both = j.notice_day.notna() & j.notice_day_off.notna()
        gains = []
        for (t, p), r in j[both & (j.notice_day < j.notice_day_off)].iterrows():
            d = em.CL[t]; days = sorted(k for k, v in d.items() if v == v)
            gains.append(sum(1 for k in days if r.notice_day < k <= r.notice_day_off))
        lost = j[j.notice_day.isna() & j.notice_day_off.notna()].index.tolist()
        out.append(dict(s=i, both=int(both.sum()), only_on=int((j.notice_day.notna() & j.notice_day_off.isna()).sum()),
                        only_off=len(lost), neither=int((j.notice_day.isna() & j.notice_day_off.isna()).sum()),
                        earlier=len(gains), gain_median=pd.Series(gains).median() if gains else None, lost=lost))
    r = pd.DataFrame(out); pd.set_option('display.width', 250); print(r.to_string())


if __name__ == '__main__' and 'four' in sys.argv:
    four_class()
