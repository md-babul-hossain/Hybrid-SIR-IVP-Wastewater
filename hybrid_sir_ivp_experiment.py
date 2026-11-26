"""
Wastewater-informed hybrid SIR-IVP + machine learning forecasts of weekly hospital admissions
(first experiment; 10 U.S. states; COVID-19 and influenza).

Pipeline
  1. Reconstruct an effective transmission rate beta(t) week by week from admissions, by inverting
     the SIR infection equations   dI/dt = (beta - gamma) I,   dC/dt = beta I   over each week
     (beta piecewise constant; susceptible depletion absorbed into beta_eff over 1-4 week horizons).
  2. Forecast log beta for weeks T+1..T+4 with: persistence, linear AR, gradient boosting (ML) without
     wastewater, and ML WITH wastewater features (proposed hybrid).
  3. Solve the SIR initial value problem forward from I(T+1) with the forecast beta path (scipy solve_ivp)
     to obtain admissions forecasts.
  4. Compare with naive persistence and a pure-ML model (no SIR) in a rolling-origin, out-of-sample test.

Usage: python hybrid_sir_ivp_experiment.py us_state_week_panel.csv results_dir
"""
import sys, os, numpy as np, pandas as pd, warnings
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')

PANEL, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
STATES = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'MI', 'WA']
H = 4                                   # forecast horizon (weeks)
TEST_START = pd.Timestamp('2025-01-04')
RETRAIN_EVERY = 4                       # weeks
GAMMA = {'covid': 1.0, 'flu': 1.5, 'rsv': 1.0}      # recovery rate per week (~7 and ~4.7 day infectious periods)
SEGMENTS = [(pd.Timestamp('2022-01-01'), pd.Timestamp('2024-05-04')),   # mandatory reporting
            (pd.Timestamp('2024-11-09'), pd.Timestamp('2099-01-01'))]   # mandatory again (voluntary gap excluded)
WARMUP = 4
PSEUDO = 1.0                            # pseudo-count so zero-admission weeks do not collapse I(t)

panel = pd.read_csv(PANEL, parse_dates=['week_end'])
panel = panel[panel.state.isin(STATES) & (panel.preliminary == 0)]

# ---------------- SIR pieces ----------------
def weekly_incidence(beta, I0, g):
    r = beta - g
    return beta * I0 * (np.expm1(r) / r if abs(r) > 1e-9 else 1.0)

def reconstruct(C, g):
    """Return beta_t and I_t (I at start of week t) such that the SIR week-t incidence equals C_t."""
    n = len(C); beta = np.full(n, np.nan); I = np.full(n + 1, np.nan)
    I[0] = C[0] / weekly_incidence(g, 1.0, g)        # start near equilibrium (beta = gamma)
    for t in range(n):
        f = lambda b: weekly_incidence(b, I[t], g) - C[t]
        # adaptive bracket: by Theorem 1 the root is positive and unique; widen until the sign changes
        lo, hi = 1e-6, 50.0
        while f(lo) > 0: lo /= 10.0
        while f(hi) < 0: hi *= 2.0
        beta[t] = brentq(f, lo, hi, xtol=1e-12)
        I[t + 1] = I[t] * np.exp(beta[t] - g)
    return beta, I

def sir_ivp_forecast(I_start, betas, g):
    """Solve dI/dt=(beta(t)-g)I, dC/dt=beta(t)I with piecewise-constant weekly beta; return weekly C."""
    def rhs(t, y):
        b = betas[min(int(t), len(betas) - 1)]
        return [(b - g) * y[0], b * y[0]]
    sol = solve_ivp(rhs, (0, len(betas)), [I_start, 0.0], t_eval=np.arange(len(betas) + 1),
                    method='RK45', rtol=1e-8, atol=1e-10, max_step=0.25)
    return np.diff(sol.y[1])

# ---------------- build per-state series with features ----------------
def build(path):
    g = GAMMA[path]; rows = []
    adm = f'{path}_adm'
    ww = {'covid': 'ww_covid_log10conc_rel', 'flu': 'ww_flu_wval_median', 'rsv': 'ww_rsv_log10conc_rel'}[path]
    for st in STATES:
        d0 = panel[panel.state == st].sort_values('week_end')
        for a, b in (SEGMENTS[1:] if path == 'rsv' else SEGMENTS):   # RSV admissions reported only from late 2024
            d = d0[(d0.week_end >= a) & (d0.week_end <= b)].copy().reset_index(drop=True)
            if len(d) < 20: continue
            d['C'] = d[adm].ffill(limit=2).fillna(0) + PSEUDO        # past-only gap filling
            beta, I = reconstruct(d.C.values, g)
            d['beta'] = beta; d['I_start'] = I[:-1]; d['I_next'] = I[1:]
            d['lb'] = np.log(d.beta)
            w = d[ww].ffill(limit=3)                                       # past-only gap filling
            d['ww'] = np.log(w + 0.1) if path == 'flu' else w
            for k in range(1, 4): d[f'lb_l{k}'] = d.lb.shift(k)
            d['lC'] = np.log(d.C); d['lC_l1'] = d.lC.shift(1); d['lC_l2'] = d.lC.shift(2)
            d['ww_l1'] = d.ww.shift(1); d['dww1'] = d.ww - d.ww.shift(1); d['dww2'] = d.ww - d.ww.shift(2)
            wk = d.week_end.dt.isocalendar().week.astype(float)
            d['sin'] = np.sin(2 * np.pi * wk / 52.18); d['cos'] = np.cos(2 * np.pi * wk / 52.18)
            for k in range(1, H + 1):
                d[f'y_lb_{k}'] = d.lb.shift(-k); d[f'y_C_{k}'] = d.C.shift(-k)
            d['state'] = st; d['seg_pos'] = np.arange(len(d))
            rows.append(d[d.seg_pos >= WARMUP])
    return pd.concat(rows, ignore_index=True)

F_BASE = ['lb', 'lb_l1', 'lb_l2', 'lb_l3', 'lC', 'sin', 'cos']
F_WW = F_BASE + ['ww', 'ww_l1', 'dww1', 'dww2']
F_AR = ['lb', 'lb_l1', 'lb_l2']
F_PURE = ['lC', 'lC_l1', 'lC_l2', 'ww', 'ww_l1', 'dww1', 'dww2', 'sin', 'cos']

def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=0)

def run(path, test_start=None):
    D = build(path); g = GAMMA[path]
    origins = sorted(D.loc[D.week_end >= (test_start or TEST_START), 'week_end'].unique())
    out = []; models = {}
    for i, T in enumerate(origins):
        if i % RETRAIN_EVERY == 0:      # retrain on examples whose targets are observed by T
            models = {}
            for k in range(1, H + 1):
                tr = D[D.week_end + pd.Timedelta(weeks=k) <= T].dropna(subset=[f'y_lb_{k}', f'y_C_{k}'])
                y = tr[f'y_lb_{k}']
                models[('ar', k)] = Ridge(alpha=1.0).fit(tr[F_AR].dropna(), y[tr[F_AR].notna().all(1)])
                models[('ml', k)] = gbm().fit(tr[F_BASE], y)
                models[('mlww', k)] = gbm().fit(tr[F_WW], y)
                models[('pure', k)] = gbm().fit(tr[F_PURE], np.log(tr[f'y_C_{k}']))
        X = D[D.week_end == T]
        for _, r in X.iterrows():
            if pd.isna(r[f'y_C_{H}']): continue
            xr = r.to_frame().T
            fc = {'naive': np.full(H, r.C)}
            paths = {'sir_const': np.full(H, r.beta)}
            if xr[F_AR].notna().all(1).iloc[0]:
                paths['sir_ar'] = np.exp([models[('ar', k)].predict(xr[F_AR].astype(float))[0] for k in range(1, H + 1)])
            paths['sir_ml'] = np.exp([models[('ml', k)].predict(xr[F_BASE].astype(float))[0] for k in range(1, H + 1)])
            paths['sir_ml_ww'] = np.exp([models[('mlww', k)].predict(xr[F_WW].astype(float))[0] for k in range(1, H + 1)])
            for m, bp in paths.items():
                fc[m] = sir_ivp_forecast(r.I_next, np.clip(bp, 1e-6, 50), g)
            fc['pure_ml_ww'] = np.exp([models[('pure', k)].predict(xr[F_PURE].astype(float))[0] for k in range(1, H + 1)])
            for m, v in fc.items():
                for k in range(1, H + 1):
                    out.append(dict(pathogen=path, state=r.state, origin=T, h=k, model=m,
                                    obs=r[f'y_C_{k}'] - PSEUDO, pred=max(v[k - 1] - PSEUDO, 0.0), now=r.C - PSEUDO))
    return pd.DataFrame(out), D

def evaluate(F):
    F = F.copy(); F['ae'] = (F.pred - F.obs).abs()
    F['ale'] = (np.log1p(F.pred) - np.log1p(F.obs)).abs()
    mae = F.groupby(['pathogen', 'state', 'h', 'model']).ae.mean().unstack('model')
    rel = mae.div(mae['naive'], axis=0)
    rel_g = np.exp(np.log(rel).groupby(['pathogen', 'h']).mean())          # geometric mean over states
    pooled = F.groupby(['pathogen', 'h', 'model']).ae.mean().unstack('model')
    # surge warning: admissions rise >= 25% over the horizon
    F['ev_obs'] = F.obs >= 1.25 * F.now; F['ev_pred'] = F.pred >= 1.25 * F.now
    def prf(x):
        tp = (x.ev_obs & x.ev_pred).sum(); fp = (~x.ev_obs & x.ev_pred).sum(); fn = (x.ev_obs & ~x.ev_pred).sum()
        p = tp / (tp + fp) if tp + fp else np.nan; rc = tp / (tp + fn) if tp + fn else np.nan
        return pd.Series(dict(events=int(x.ev_obs.sum()), precision=p, recall=rc,
                              F1=2 * p * rc / (p + rc) if p and rc else np.nan))
    surge = F[F.model != 'naive'].groupby(['pathogen', 'h', 'model']).apply(prf)
    return rel_g, pooled, surge, rel

if __name__ == '__main__':
    allF, recon = [], []
    for p in ['covid', 'flu']:
        F, D = run(p); allF.append(F)
        recon.append(D[['state', 'week_end', 'C', 'beta']].assign(pathogen=p))
        print(p, 'done', len(F))
    F = pd.concat(allF); F.to_csv(f'{OUTDIR}/forecasts.csv', index=False)
    pd.concat(recon).to_csv(f'{OUTDIR}/reconstructed_beta.csv', index=False)
    rel_g, pooled, surge, rel = evaluate(F)
    rel_g.round(3).to_csv(f'{OUTDIR}/relative_MAE_vs_naive.csv')
    pooled.round(1).to_csv(f'{OUTDIR}/MAE_admissions.csv')
    surge.round(3).to_csv(f'{OUTDIR}/surge_warning.csv')
    rel.round(3).to_csv(f'{OUTDIR}/relative_MAE_by_state.csv')
    pd.set_option('display.width', 200)
    print('\nRelative MAE vs naive (geometric mean over 10 states; <1 = better than naive)\n', rel_g.round(3))
    print('\nMAE (admissions/week, pooled)\n', pooled.round(1))
    print('\nSurge warning (>=25% rise)\n', surge.round(3))
