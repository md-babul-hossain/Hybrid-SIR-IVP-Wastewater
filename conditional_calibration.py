"""Regime-conditional interval calibration.

Unconditional calibration (probabilistic.py) pools all past log errors, so intervals are too narrow when
admissions rise and too wide otherwise (PIT diagnostics). Here the calibration set E_T is restricted to past
forecasts made in the same *forecast regime*, which is known at the origin:
    regime = 'up'   if the model's point forecast exceeds current admissions by more than 5% (pred > 1.05 * now)
             'flat' if 0.95 * now <= pred <= 1.05 * now
             'down' otherwise.
If a regime has fewer than MIN_N past errors, the pooled set is used. Everything else (quantile levels,
scoring, evaluation origins) is identical to the unconditional method.

Usage: python conditional_calibration.py results
"""
import sys, numpy as np, pandas as pd
R = sys.argv[1]; ALPHAS = [0.05, 0.2, 0.5]; MIN_ORIGINS = 8; MIN_N = 40
F = pd.read_csv(f'{R}/forecasts.csv', parse_dates=['origin'])
F = F[F.model.isin(['naive', 'pure_ml_ww', 'sir_ml', 'sir_ml_ww'])].copy()
F['err'] = np.log1p(F.obs) - np.log1p(F.pred); F['target'] = F.origin + pd.to_timedelta(F.h * 7, unit='D')
F['regime'] = np.where(F.pred > 1.05 * F.now, 'up', np.where(F.pred < 0.95 * F.now, 'down', 'flat'))
F['rise'] = F.obs >= 1.25 * F.now
out = []
for (p, m, h), g in F.groupby(['pathogen', 'model', 'h']):
    for T in sorted(g.origin.unique()):
        past = g[g.target <= T]
        if past.origin.nunique() < MIN_ORIGINS: continue
        for _, r in g[g.origin == T].iterrows():
            for method in ['pooled', 'conditional']:
                e = past.err.values
                if method == 'conditional':
                    ec = past[past.regime == r.regime].err.values
                    if len(ec) >= MIN_N: e = ec
                lp = np.log1p(r.pred)
                d = dict(pathogen=p, model=m, h=h, origin=T, state=r.state, obs=r.obs, method=method, regime=r.regime, rise=r.rise,
                         median=max(np.expm1(lp + np.quantile(e, 0.5)), 0))
                for a in ALPHAS:
                    d[f'lo{a}'] = max(np.expm1(lp + np.quantile(e, a / 2)), 0); d[f'hi{a}'] = max(np.expm1(lp + np.quantile(e, 1 - a / 2)), 0)
                out.append(d)
P = pd.DataFrame(out)
y = P.obs; s = 0.5 * (y - P['median']).abs()
for a in ALPHAS:
    l, u = P[f'lo{a}'], P[f'hi{a}']; s = s + (a / 2) * ((u - l) + (2 / a) * (l - y).clip(lower=0) + (2 / a) * (y - u).clip(lower=0))
    P[f'cov{int(round((1 - a) * 100))}'] = (y >= l) & (y <= u)
P['wis'] = s / (len(ALPHAS) + 0.5)
P.to_csv(f'{R}/conditional_calibration_forecasts.csv', index=False)
mw = P.groupby(['pathogen', 'method', 'h', 'state', 'model']).wis.mean().unstack('model')
rel = np.exp(np.log(mw.div(mw['naive'], axis=0)).groupby(['pathogen', 'method', 'h']).mean())
cov = P.groupby(['pathogen', 'model', 'method'])[['cov50', 'cov80', 'cov95']].mean()
covr = P.groupby(['pathogen', 'model', 'method', 'rise'])[['cov50', 'cov95']].mean().unstack('rise')
# WIS change conditional vs pooled for the same model (paired block bootstrap over origins)
rng = np.random.default_rng(7); bt = []
for (p, m), g in P.groupby(['pathogen', 'model']):
    t = g.groupby(['origin', 'method']).wis.sum().unstack().sort_index(); x = (t.conditional - t.pooled).values
    k = len(x); nb = int(np.ceil(k / 4)); st = rng.integers(0, k - 3, size=(5000, nb))
    idx = (st[:, :, None] + np.arange(4)).reshape(5000, -1)[:, :k]; mm = x[idx].mean(1)
    bt.append(dict(pathogen=p, model=m, pct=100 * x.mean() / t.pooled.mean(), lo=np.percentile(mm, 2.5), hi=np.percentile(mm, 97.5)))
rel.round(3).to_csv(f'{R}/conditional_relative_WIS.csv'); cov.round(3).to_csv(f'{R}/conditional_coverage.csv')
covr.round(3).to_csv(f'{R}/conditional_coverage_by_rise.csv'); pd.DataFrame(bt).round(2).to_csv(f'{R}/conditional_bootstrap.csv', index=False)
pd.set_option('display.width', 220)
print(rel.round(3)); print(cov.round(3)); print(covr.round(3)); print(pd.DataFrame(bt).round(2))
