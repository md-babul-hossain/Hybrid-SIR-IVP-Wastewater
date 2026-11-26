"""Sensitivity analyses for the hybrid SIR-IVP model (routine 1-4-week forecasts, Jan 2025-Oct 2026).

  gamma      : recovery rate gamma varied (COVID-19 0.7, 1.0, 1.5, 2.0; influenza 1.0, 1.5, 2.0, 3.0 per week)
  hyper      : gradient-boosting hyperparameters varied one at a time around the default
  ablation   : groups of predictors removed (beta lags, current admissions, season, wastewater changes)
  importance : permutation importance of each predictor for log-beta forecasts (model trained on data before 2025)

Forecasts are propagated with the exact weekly closed form of the SIR-IVP (Lemma 2), which equals the
RK45 solution used in the main analysis to integration tolerance. All variants use the same origins,
retraining schedule and evaluation as the main analysis; results are relative MAE against naive persistence
(geometric mean over states).

Usage: python sensitivity.py us_state_week_panel.csv results {gamma|hyper|ablation|importance} [pathogen]
"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
PANEL, OUT, WHAT = sys.argv[1:4]
PATHS = [sys.argv[4]] if len(sys.argv) > 4 else ['covid', 'flu']
sys.argv = [sys.argv[0], PANEL, OUT]
import hybrid_sir_ivp_experiment as E
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

BASE = dict(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=20, l2_regularization=1.0)

def closed_form(I0, betas, g):
    out = []; I = I0
    for b in betas:
        r = b - g; out.append(b * I * (np.expm1(r) / r if abs(r) > 1e-9 else 1.0)); I *= np.exp(r)
    return np.array(out)

def run_variant(path, gamma=None, feats=None, params=None, D=None):
    g = gamma or E.GAMMA[path]
    if D is None:
        E.GAMMA[path] = g; D = E.build(path)
    feats = feats or E.F_WW; params = {**BASE, **(params or {})}
    origins = sorted(D.loc[D.week_end >= E.TEST_START, 'week_end'].unique())
    rows = []; models = {}
    for i, T in enumerate(origins):
        if i % E.RETRAIN_EVERY == 0:
            models = {}
            for k in range(1, E.H + 1):
                tr = D[D.week_end + pd.Timedelta(weeks=k) <= T].dropna(subset=[f'y_lb_{k}', f'y_C_{k}'])
                models[k] = HistGradientBoostingRegressor(random_state=0, **params).fit(tr[feats], tr[f'y_lb_{k}'])
        X = D[(D.week_end == T)].dropna(subset=[f'y_C_{E.H}'])
        if X.empty: continue
        bp = np.column_stack([np.exp(models[k].predict(X[feats].astype(float))) for k in range(1, E.H + 1)])
        for j, (_, r) in enumerate(X.iterrows()):
            fc = closed_form(r.I_next, np.clip(bp[j], 1e-6, 50), g)
            for k in range(1, E.H + 1):
                obs = r[f'y_C_{k}'] - E.PSEUDO
                rows.append(dict(state=r.state, origin=T, h=k, obs=obs, pred=max(fc[k - 1] - E.PSEUDO, 0),
                                 naive=r.C - E.PSEUDO))
    F = pd.DataFrame(rows)
    F['ae'] = (F.pred - F.obs).abs(); F['ae_n'] = (F.naive - F.obs).abs()
    m = F.groupby(['state', 'h'])[['ae', 'ae_n']].mean()
    rel = np.exp(np.log(m.ae / m.ae_n).groupby('h').mean())
    return rel

res = []
for path in PATHS:
    if WHAT == 'gamma':
        grid = {'covid': [0.7, 1.0, 1.5, 2.0], 'flu': [1.0, 1.5, 2.0, 3.0]}[path]
        for g in grid:
            for name, feats in [('with wastewater', E.F_WW), ('hospital data only', E.F_BASE)]:
                E.GAMMA[path] = g
                rel = run_variant(path, gamma=g, feats=feats)
                res.append(dict(pathogen=path, analysis='gamma', variant=f'gamma={g}', model=name, **{f'h{k}': rel[k] for k in rel.index}))
                print(res[-1], flush=True)
    elif WHAT == 'hyper':
        D = E.build(path)
        grid = [('default', {}), ('max_depth=2', dict(max_depth=2)), ('max_depth=5', dict(max_depth=5)),
                ('learning_rate=0.1', dict(learning_rate=0.1)), ('max_iter=150', dict(max_iter=150)),
                ('max_iter=600', dict(max_iter=600)), ('min_samples_leaf=10', dict(min_samples_leaf=10)),
                ('min_samples_leaf=40', dict(min_samples_leaf=40))]
        for name, p in grid:
            rel = run_variant(path, params=p, D=D)
            res.append(dict(pathogen=path, analysis='hyper', variant=name, model='with wastewater', **{f'h{k}': rel[k] for k in rel.index}))
            print(res[-1], flush=True)
    elif WHAT == 'ablation':
        D = E.build(path)
        grid = [('all predictors', E.F_WW),
                ('without beta lags', [f for f in E.F_WW if f not in ('lb_l1', 'lb_l2', 'lb_l3')]),
                ('without current admissions', [f for f in E.F_WW if f != 'lC']),
                ('without season', [f for f in E.F_WW if f not in ('sin', 'cos')]),
                ('without wastewater changes', [f for f in E.F_WW if f not in ('dww1', 'dww2')]),
                ('without wastewater', E.F_BASE),
                ('current beta only', ['lb'])]
        for name, feats in grid:
            rel = run_variant(path, feats=feats, D=D)
            res.append(dict(pathogen=path, analysis='ablation', variant=name, model='', **{f'h{k}': rel[k] for k in rel.index}))
            print(res[-1], flush=True)
    elif WHAT == 'importance':
        D = E.build(path)
        tr_all = D[D.week_end < E.TEST_START]; te = D[D.week_end >= E.TEST_START]
        for k in range(1, E.H + 1):
            tr = tr_all[tr_all.week_end + pd.Timedelta(weeks=k) < E.TEST_START].dropna(subset=[f'y_lb_{k}'])
            t = te.dropna(subset=[f'y_lb_{k}'] + E.F_WW)
            m = HistGradientBoostingRegressor(random_state=0, **BASE).fit(tr[E.F_WW], tr[f'y_lb_{k}'])
            pi = permutation_importance(m, t[E.F_WW], t[f'y_lb_{k}'], scoring='neg_mean_absolute_error', n_repeats=20, random_state=0)
            base = np.mean(np.abs(m.predict(t[E.F_WW]) - t[f'y_lb_{k}']))
            for f, v, s in zip(E.F_WW, pi.importances_mean, pi.importances_std):
                res.append(dict(pathogen=path, analysis='importance', variant=f, model=f'h{k}', importance=v,
                                sd=s, rel_increase=v / base))
        print(pd.DataFrame([r for r in res if r['pathogen'] == path]).round(4).to_string(), flush=True)
pd.DataFrame(res).to_csv(f'{OUT}/sensitivity_{WHAT}_{"_".join(PATHS)}.csv', index=False)
