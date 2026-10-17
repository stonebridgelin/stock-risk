"""T08a C-04：分段运行 event_metrics（设备上单次运行时间受限）。

用法：python event_metrics_chunk.py <run_id> <起> <止>，结果写 output/t07/em_parts/<起>.csv；
全部完成后用 event_metrics_chunk.py <run_id> merge 合并为 output/t07/event_metrics.csv。
"""
import json, sys
import pandas as pd
import event_metrics as em

run = sys.argv[1]
em.CMP = em.ROOT / 'output/compare' / run
parts = em.OUT / 'em_parts'
if sys.argv[2] == 'merge':
    df = pd.concat([pd.read_csv(f) for f in sorted(parts.glob('*.csv'))], ignore_index=True)
    df.to_csv(em.OUT / 'event_metrics.csv', index=False)
    print(df.set_id.nunique(), len(df))
else:
    a, b = int(sys.argv[2]), int(sys.argv[3])
    m = json.load(open(em.CMP / 'run_manifest.json'))
    sets = [s['set_id'] if isinstance(s, dict) else s for s in m['variant_sets']][a:b]
    em.init()
    parts.mkdir(parents=True, exist_ok=True)
    rows = [r for s in sets for r in em.one(s)]
    pd.DataFrame(rows).to_csv(parts / f'{a:04d}.csv', index=False)
    print(a, b, len(sets), len(rows))
