"""示意用的百分比下跌事件（阈值为示意值，不是登记值）。

规则取自 10 号第 5.3 节 E_pct，以下几处为简化或执行口径（复核 R1 A-1 后修订，v2）：
- 度量按有效日线计算（与 I18 一致）：DD = 1 - C / 最近 W 条有效日线的最高收盘。
- 结束：(a) 收盘 ≥ 峰值收盘；(b) DD < X/2 连续 10 个有效交易日（第 10 天为结束日）；
  (c) 自最初开始日满 252 个有效交易日（上限，标 CAP，谷底为暂定值）。判断顺序 a、c、b。
- 合并：只对 (b)。(b) 成立后向后看 20 个有效交易日，若再次满足开始条件，则不结束，
  安静计数清零，从下一日起照常逐日判断（不跳过中间各日，252 上限不重置）。
- 截尾：结束日之后不足 20 个有效交易日的数据（观察终点），或事件区间 [峰值日, 结束日 + 20] 内有缺价日，
  标 censored，原因 LISTING_END / EXPLORATION_END / MISSING。
  缺价日不停止事件：事件照常按有效日线继续计算，只加截尾标记；跨缺价事件的谷底与结束日不视为已确定。
- 以 (a)、(c) 结束后再次满足开始条件的，记为新事件（5.3 节未规定，此为简化）。
"""
import pandas as pd
from load import daily

LISTING_END = {'SHLD_corrected', 'SHLD_uncorrected', 'WFT', 'GMCR'}


def series(t):
    d = daily(385, t, ['day', 'data_status', 'close'])
    d['close'] = pd.to_numeric(d.close, errors='coerce')
    missing = set(d.day[d.data_status == 'MISSING'])
    v = d[d.close.notna()].reset_index(drop=True)
    return v.day.values, v.close.values, missing


def pct_events(t, X=0.20, W=63, QUIET=10, MERGE=20, CAP=252, CONFIRM=20):
    days, c, missing = series(t)
    n = len(c)
    hi = pd.Series(c).rolling(W, min_periods=W).max().values
    dd = lambda k: 1 - c[k] / hi[k]
    evs = []
    i = W - 1
    while i < n:
        if not dd(i) >= X - 1e-12:
            i += 1
            continue
        mx = c[i - W + 1:i + 1].max()
        p = max(k for k in range(i - W + 1, i + 1) if c[k] == mx)
        j, below, end, why = i, 0, None, None
        while True:
            j += 1
            if j >= n:
                end, why = n - 1, 'DATA_END'; break
            if c[j] >= c[p]:
                end, why = j, 'a'; break
            if j - i >= CAP - 1:
                end, why = j, 'c'; break
            below = below + 1 if dd(j) < X / 2 else 0
            if below == QUIET:
                k = next((k for k in range(j + 1, min(j + 1 + MERGE, n)) if dd(k) >= X - 1e-12), None)
                if k is not None:
                    below = 0
                    continue
                end, why = j, 'b'; break
        lo = p + int(pd.Series(c[p:end + 1]).idxmin())
        cens = None
        if why == 'DATA_END' or end + CONFIRM > n - 1:
            cens = 'LISTING_END' if t in LISTING_END else 'EXPLORATION_END'
        last = days[min(end + CONFIRM, n - 1)]
        if any(days[p] <= m <= last for m in missing):
            cens = 'MISSING'
        evs.append(dict(target=t, peak=days[p], peak_close=c[p], start=days[i], trough=days[lo],
                        trough_close=c[lo], dd=1 - c[lo] / c[p], end=days[end], end_why=why,
                        cap=(why == 'c'), censored=cens))
        i = end + 1
    return pd.DataFrame(evs)


if __name__ == '__main__':
    from load import targets, OUT
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.concat([pct_events(t, 0.20) for t in targets() if t != 'SHLD_uncorrected'], ignore_index=True)
    ev = ev[ev.peak >= '2010-01-01']
    ev.to_csv(OUT / 'events_pct20.csv', index=False)
    print(len(ev), ev.end_why.value_counts().to_dict(), ev.censored.value_counts(dropna=False).to_dict())
