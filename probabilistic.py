"""Probabilistic (interval) forecasts by rolling empirical residual calibration of log errors
(in the spirit of split-conformal prediction; no finite-sample coverage guarantee is claimed).

For each pathogen, model and horizon h, the predictive quantiles at origin T are
    q_tau = expm1( log1p(point forecast) + Q_tau(E_T) ),
where E_T is the set of past log errors log1p(obs) - log1p(pred) of the same model and horizon,
pooled over states, whose target weeks are observed by T (origin + h <= T). Only past information is used.
Intervals: 50%, 80%, 95% central intervals plus the median (tau = 0.5).
Scored with the weighted interval score (WIS; Bracher et al., 2021) and empirical coverage.
Evaluation starts once at least MIN_ORIGINS past origins are available for calibration.

Usage: python probabilistic.py results
"""
import sys, numpy as np, pandas as pd
R = sys.argv[1]
ALPHAS = [0.05, 0.2, 0.5]           # 95%, 80%, 50% intervals
MIN_ORIGINS = 8
F = pd.read_csv(f'{R}/forecasts.csv', parse_dates=['origin'])
F['err'] = np.log1p(F.obs) - np.log1p(F.pred)
F['target'] = F.origin + pd.to_timedelta(F.h * 7, unit='D')
out = []
for (p, m, h), g in F.groupby(['pathogen', 'model', 'h']):
    origins = sorted(g.origin.unique())
    for T in origins:
        past = g[g.target <= T]
        if past.origin.nunique() < MIN_ORIGINS: continue
        e = past.err.values
        cur = g[g.origin == T]
        lp = np.log1p(cur.pred.values)
        row = {'median': np.expm1(lp + np.quantile(e, 0.5))}
        for a in ALPHAS:
            row[f'lo{a}'] = np.expm1(lp + np.quantile(e, a / 2))
            row[f'hi{a}'] = np.expm1(lp + np.quantile(e, 1 - a / 2))
        for i, (_, r) in enumerate(cur.iterrows()):
            d = dict(pathogen=p, model=m, h=h, origin=T, state=r.state, obs=r.obs)
            for k, v in row.items(): d[k] = max(v[i], 0.0)
            out.append(d)
P = pd.DataFrame(out)

def wis(df):
    y = df.obs; s = 0.5 * (y - df['median']).abs()
    for a in ALPHAS:
        l, u = df[f'lo{a}'], df[f'hi{a}']
        IS = (u - l) + (2 / a) * (l - y).clip(lower=0) + (2 / a) * (y - u).clip(lower=0)
        s = s + (a / 2) * IS
    return s / (len(ALPHAS) + 0.5)
P['wis'] = wis(P)
for a in ALPHAS:
    P[f'cov{int(round((1 - a) * 100))}'] = (P.obs >= P[f'lo{a}']) & (P.obs <= P[f'hi{a}'])
P.to_csv(f'{R}/probabilistic_forecasts.csv', index=False)

# relative WIS vs naive: per state, then geometric mean over states
mw = P.groupby(['pathogen', 'h', 'state', 'model']).wis.mean().unstack('model')
rel = np.exp(np.log(mw.div(mw['naive'], axis=0)).groupby(['pathogen', 'h']).mean())
cov = P.groupby(['pathogen', 'model'])[['cov50', 'cov80', 'cov95']].mean()
rel.round(3).to_csv(f'{R}/probabilistic_relative_WIS.csv')
cov.round(3).to_csv(f'{R}/probabilistic_coverage.csv')

# moving-block bootstrap (blocks of 4 origins) for WIS differences, summed over states per origin
rng = np.random.default_rng(2026); B = 5000; boot = []
for (p, h), g in P.groupby(['pathogen', 'h']):
    tot = g.groupby(['origin', 'model']).wis.sum().unstack().sort_index()
    for a_, b_ in [('sir_ml_ww', 'pure_ml_ww'), ('sir_ml_ww', 'sir_ml'), ('sir_ml_ww', 'naive')]:
        x = (tot[a_] - tot[b_]).values; n = len(x); blk = 4; nb = int(np.ceil(n / blk))
        st = rng.integers(0, n - blk + 1, size=(B, nb))
        idx = (st[:, :, None] + np.arange(blk)).reshape(B, -1)[:, :n]
        mm = x[idx].mean(1); lo, hi = np.percentile(mm, [2.5, 97.5])
        boot.append(dict(pathogen=p, h=h, comparison=f'{a_} - {b_}', mean=x.mean(), pct=100 * x.mean() / tot[b_].mean(),
                         lo=lo, hi=hi, significant=(hi < 0) or (lo > 0)))
Bt = pd.DataFrame(boot); Bt.round(2).to_csv(f'{R}/probabilistic_bootstrap.csv', index=False)
pd.set_option('display.width', 200)
print('evaluation origins:', P.groupby('pathogen').origin.nunique().to_dict(), 'from', P.origin.min().date())
print('\nRelative WIS vs naive\n', rel.round(3)); print('\nCoverage\n', cov.round(3)); print('\n', Bt.round(1).to_string(index=False))
