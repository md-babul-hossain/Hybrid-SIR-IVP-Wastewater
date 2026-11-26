"""Matched ablation: does forecasting the transmission rate and integrating the SIR-IVP beat forecasting
admissions directly, when everything else is held fixed?

Both arms use the same gradient-boosting learner and hyperparameters, the same forecast origins,
the same retraining schedule and training windows, and the same predictors (reconstructed log-beta lags,
current log admissions, season and, optionally, wastewater). They differ only in the target:
  hybrid : log beta_{T+k}, converted to admissions by integrating the SIR-IVP from I(T+1)
  direct : log C_{T+k} (admissions + pseudo-count), predicted directly
Point forecasts are scored by MAE relative to persistence; interval forecasts by WIS after the same rolling
empirical residual calibration as probabilistic.py. Paired moving-block bootstrap over origins (blocks of 4).

Usage: python matched_ablation.py us_state_week_panel.csv results
"""
import sys, numpy as np, pandas as pd
panel_path, R = sys.argv[1], sys.argv[2]
sys.argv = ['hybrid_sir_ivp_experiment.py', panel_path, R]
import hybrid_sir_ivp_experiment as E

ARMS = {'hybrid_ww': (E.F_WW, 'hybrid'), 'direct_ww': (E.F_WW, 'direct'),
        'hybrid': (E.F_BASE, 'hybrid'), 'direct': (E.F_BASE, 'direct')}

def run(path):
    D = E.build(path); g = E.GAMMA[path]
    origins = sorted(D.loc[D.week_end >= E.TEST_START, 'week_end'].unique())
    out = []; models = {}
    for i, T in enumerate(origins):
        if i % E.RETRAIN_EVERY == 0:
            models = {}
            for k in range(1, E.H + 1):
                tr = D[D.week_end + pd.Timedelta(weeks=k) <= T].dropna(subset=[f'y_lb_{k}', f'y_C_{k}'])
                for arm, (F, kind) in ARMS.items():
                    y = tr[f'y_lb_{k}'] if kind == 'hybrid' else np.log(tr[f'y_C_{k}'])
                    models[(arm, k)] = E.gbm().fit(tr[F], y)
        for _, r in D[D.week_end == T].iterrows():
            if pd.isna(r[f'y_C_{E.H}']): continue
            xr = r.to_frame().T
            fc = {'naive': np.full(E.H, r.C)}
            for arm, (F, kind) in ARMS.items():
                p = np.exp([models[(arm, k)].predict(xr[F].astype(float))[0] for k in range(1, E.H + 1)])
                fc[arm] = E.sir_ivp_forecast(r.I_next, np.clip(p, 1e-6, 50), g) if kind == 'hybrid' else p
            for m, v in fc.items():
                for k in range(1, E.H + 1):
                    out.append(dict(pathogen=path, state=r.state, origin=T, h=k, model=m,
                                    obs=r[f'y_C_{k}'] - E.PSEUDO, pred=max(v[k - 1] - E.PSEUDO, 0.0)))
    return pd.DataFrame(out)

ALPHAS = [0.05, 0.2, 0.5]; MIN_ORIGINS = 8
def calibrate(F):
    F = F.copy(); F['err'] = np.log1p(F.obs) - np.log1p(F.pred)
    F['target'] = F.origin + pd.to_timedelta(F.h * 7, unit='D'); out = []
    for (p, m, h), g in F.groupby(['pathogen', 'model', 'h']):
        for T in sorted(g.origin.unique()):
            past = g[g.target <= T]
            if past.origin.nunique() < MIN_ORIGINS: continue
            e = past.err.values; cur = g[g.origin == T]; lp = np.log1p(cur.pred.values)
            q = {'median': np.expm1(lp + np.quantile(e, 0.5))}
            for a in ALPHAS:
                q[f'lo{a}'] = np.expm1(lp + np.quantile(e, a / 2)); q[f'hi{a}'] = np.expm1(lp + np.quantile(e, 1 - a / 2))
            for i, (_, r) in enumerate(cur.iterrows()):
                d = dict(pathogen=p, model=m, h=h, origin=T, state=r.state, obs=r.obs)
                for k_, v in q.items(): d[k_] = max(v[i], 0.0)
                out.append(d)
    P = pd.DataFrame(out); y = P.obs; s = 0.5 * (y - P['median']).abs()
    for a in ALPHAS:
        l, u = P[f'lo{a}'], P[f'hi{a}']
        s = s + (a / 2) * ((u - l) + (2 / a) * (l - y).clip(lower=0) + (2 / a) * (y - u).clip(lower=0))
    P['wis'] = s / (len(ALPHAS) + 0.5); return P

def rel_geo(df, col):
    m = df.groupby(['pathogen', 'h', 'state', 'model'])[col].mean().unstack('model')
    return np.exp(np.log(m.div(m['naive'], axis=0)).groupby(['pathogen', 'h']).mean())

rng = np.random.default_rng(2026)
def boot(df, col, a, b, B=5000, blk=4):
    rows = []
    for (p, h), g in df.groupby(['pathogen', 'h']):
        tot = g.groupby(['origin', 'model'])[col].sum().unstack().sort_index()
        x = (tot[a] - tot[b]).values; n = len(x); nb = int(np.ceil(n / blk))
        st = rng.integers(0, n - blk + 1, size=(B, nb)); idx = (st[:, :, None] + np.arange(blk)).reshape(B, -1)[:, :n]
        lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
        rows.append(dict(pathogen=p, h=h, score=col, comparison=f'{a} - {b}', pct=100 * x.mean() / tot[b].mean(),
                         lo=lo, hi=hi, significant=(hi < 0) or (lo > 0)))
    return rows

if __name__ == '__main__':
    F = pd.concat([run(p) for p in ['covid', 'flu']]); F.to_csv(f'{R}/ablation_forecasts.csv', index=False)
    F['ae'] = (F.pred - F.obs).abs()
    P = calibrate(F)
    relM = rel_geo(F, 'ae'); relW = rel_geo(P, 'wis')
    relM.round(3).to_csv(f'{R}/ablation_relative_MAE.csv'); relW.round(3).to_csv(f'{R}/ablation_relative_WIS.csv')
    rows = []
    for a, b in [('hybrid_ww', 'direct_ww'), ('hybrid', 'direct')]:
        rows += boot(F, 'ae', a, b) + boot(P, 'wis', a, b)
    Bt = pd.DataFrame(rows); Bt.round(3).to_csv(f'{R}/ablation_bootstrap.csv', index=False)
    pd.set_option('display.width', 200)
    print('origins (MAE):', F.groupby('pathogen').origin.nunique().to_dict(), ' (WIS):', P.groupby('pathogen').origin.nunique().to_dict())
    print('\nRelative MAE\n', relM.round(3)); print('\nRelative WIS\n', relW.round(3)); print(Bt.round(2).to_string(index=False))
