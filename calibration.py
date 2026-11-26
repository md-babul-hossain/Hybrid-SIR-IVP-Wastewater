"""Calibration diagnostics for the interval forecasts.

1. Probability integral transform (PIT): for each forecast, PIT = share of the calibration errors E_T (same
   construction as probabilistic.py) below the realised log error (mid-rank for ties). Uniform PIT = calibrated.
   Reported as 10-bin histograms and, separately, for forecasts made during rises (obs >= 1.25 * current).
2. WIS decomposition (Bracher et al. 2021): WIS = dispersion + overprediction + underprediction,
   for the main evaluation and for the common state-weeks with the CDC ensembles.
3. Relative WIS against naive persistence by state.

Usage: python calibration.py results
"""
import sys, numpy as np, pandas as pd
R = sys.argv[1]; ALPHAS = [0.05, 0.2, 0.5]; MIN_ORIGINS = 8
F = pd.read_csv(f'{R}/forecasts.csv', parse_dates=['origin'])
F = F[F.model.isin(['naive', 'pure_ml_ww', 'sir_ml', 'sir_ml_ww'])]
F['err'] = np.log1p(F.obs) - np.log1p(F.pred); F['target'] = F.origin + pd.to_timedelta(F.h * 7, unit='D')
F['rise'] = F.obs >= 1.25 * F.now
rows = []
for (p, m, h), g in F.groupby(['pathogen', 'model', 'h']):
    for T in sorted(g.origin.unique()):
        past = g[g.target <= T]
        if past.origin.nunique() < MIN_ORIGINS: continue
        e = np.sort(past.err.values); cur = g[g.origin == T]
        lo = np.searchsorted(e, cur.err.values, 'left'); hi = np.searchsorted(e, cur.err.values, 'right')
        for (_, r), a, b in zip(cur.iterrows(), lo, hi):
            rows.append(dict(pathogen=p, model=m, h=h, origin=T, state=r.state, rise=r.rise, pit=(a + b) / (2 * len(e))))
PIT = pd.DataFrame(rows); PIT.to_csv(f'{R}/pit_values.csv', index=False)
bins = np.linspace(0, 1, 11)
hist = []
for (p, m, sub), g in [((p, m, s), g) for (p, m), G in PIT.groupby(['pathogen', 'model']) for s, g in
                       [('all', G), ('rising', G[G.rise]), ('not rising', G[~G.rise])]]:
    c, _ = np.histogram(g.pit, bins); f = c / c.sum()
    hist.append(dict(pathogen=p, model=m, subset=sub, n=len(g), **{f'b{i}': f[i] for i in range(10)},
                     ks=np.max(np.abs(np.sort(g.pit) - (np.arange(1, len(g) + 1) - 0.5) / len(g))),
                     below_2_5=(g.pit < 0.025).mean(), above_97_5=(g.pit > 0.975).mean()))
Hh = pd.DataFrame(hist); Hh.round(4).to_csv(f'{R}/pit_histograms.csv', index=False)

def decomp(df):
    y = df.obs; med = df['median']
    disp = 0; over = 0.5 * (med - y).clip(lower=0); under = 0.5 * (y - med).clip(lower=0)
    for a in ALPHAS:
        l, u = df[f'lo{a}'], df[f'hi{a}']
        disp = disp + (a / 2) * (u - l)
        over = over + (l - y).clip(lower=0); under = under + (y - u).clip(lower=0)
    K = len(ALPHAS) + 0.5
    return pd.DataFrame({'dispersion': disp / K, 'overprediction': over / K, 'underprediction': under / K})
P = pd.read_csv(f'{R}/probabilistic_forecasts.csv', parse_dates=['origin'])
P = pd.concat([P, decomp(P)], axis=1)
D1 = P.groupby(['pathogen', 'model'])[['dispersion', 'overprediction', 'underprediction', 'wis']].mean()
D1.round(2).to_csv(f'{R}/wis_decomposition.csv')
Hs = pd.read_csv(f'{R}/hub_comparison_scores.csv', parse_dates=['origin'])
Hs = pd.concat([Hs.reset_index(drop=True), decomp(Hs.reset_index(drop=True))], axis=1)
D2 = Hs.groupby(['pathogen', 'h', 'model'])[['dispersion', 'overprediction', 'underprediction', 'wis']].mean()
D2.round(2).to_csv(f'{R}/wis_decomposition_hub.csv')
D2a = Hs.groupby(['pathogen', 'model'])[['dispersion', 'overprediction', 'underprediction', 'wis']].mean()
D2a.round(2).to_csv(f'{R}/wis_decomposition_hub_overall.csv')
mw = P.groupby(['pathogen', 'state', 'model']).wis.mean().unstack('model')
rs = mw.div(mw['naive'], axis=0); rs.round(3).to_csv(f'{R}/relative_WIS_by_state.csv')
pd.set_option('display.width', 220)
print(Hh[Hh.model.isin(['sir_ml_ww', 'naive'])].round(3).to_string())
print(D1.round(1)); print(D2a.round(1)); print(rs.round(2))
