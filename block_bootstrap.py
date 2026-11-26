"""Moving-block bootstrap confidence intervals for differences in total absolute error.
Errors are summed over states at each forecast origin (keeping cross-state dependence); consecutive
origins are resampled in blocks so that overlapping forecast/outage windows stay together.
  routine forecasts: weekly origins, horizons <= 4 weeks  -> block length 4 origins
  simulated outages: origins every 4 weeks, 8-week windows -> block length 3 origins
Usage: python block_bootstrap.py results"""
import sys, numpy as np, pandas as pd
R = sys.argv[1]; B = 5000; rng = np.random.default_rng(2026)

def mbb_ci(series, block):
    x = series.values; n = len(x); nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

rows = []
F = pd.read_csv(f'{R}/forecasts.csv', parse_dates=['origin']); F['ae'] = (F.pred - F.obs).abs()
for (p, h), g in F.groupby(['pathogen', 'h']):
    tot = g.groupby(['origin', 'model']).ae.sum().unstack().sort_index()
    for a, b in [('sir_ml_ww', 'pure_ml_ww'), ('sir_ml_ww', 'sir_ml'), ('sir_ml_ww', 'naive')]:
        m, lo, hi = mbb_ci(tot[a] - tot[b], 4)
        rows.append(dict(experiment='routine', pathogen=p, h=h, comparison=f'{a} - {b}', mean_diff_per_origin=m,
                         ci_lo=lo, ci_hi=hi, pct_of_b=100 * m / tot[b].mean(), significant=(hi < 0) or (lo > 0)))
O = pd.read_csv(f'{R}/outage_forecasts.csv', parse_dates=['origin']); O['ae'] = (O.pred - O.obs).abs()
for p, g in O.groupby('pathogen'):
    tot = g.groupby(['origin', 'model']).ae.sum().unstack().sort_index()
    for a, b in [('sir_ww', 'sir_ml'), ('sir_ww', 'ww_ratio'), ('ww_ratio', 'sir_ml'), ('sir_ww', 'locf')]:
        m, lo, hi = mbb_ci(tot[a] - tot[b], 3)
        rows.append(dict(experiment='outage', pathogen=p, h=np.nan, comparison=f'{a} - {b}', mean_diff_per_origin=m,
                         ci_lo=lo, ci_hi=hi, pct_of_b=100 * m / tot[b].mean(), significant=(hi < 0) or (lo > 0)))
T = pd.DataFrame(rows); T.round(2).to_csv(f'{R}/block_bootstrap.csv', index=False)
pd.set_option('display.width', 200); print(T.round(1).to_string(index=False))
