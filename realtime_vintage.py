"""Real-time (data-vintage) evaluation of the hybrid SIR-IVP model.

At each forecast origin T the model sees hospital admissions exactly as first published: we use the earliest
archived data vintage that contains week T (COVID-19: weekly snapshots of target-data/covid-hospital-admissions.csv
in the git history of the CDC COVID-19 Forecast Hub; influenza: the as_of column of target-data/time-series.csv
in the FluSight hub). This is the same information the Hub teams had when they submitted. The transmission-rate
reconstruction, training examples, forecasts and interval calibration all use only that vintage. Weeks not
contained in a vintage (before the archive starts) are taken from the revised panel. Wastewater is not archived,
so the revised NWSS values are used, lagged by one week (forecasts at T use wastewater through T-1) to allow for
publication delay. Forecasts are scored against the same revised admissions as in the main analysis.

Usage: python realtime_vintage.py us_state_week_panel.csv results vintages_10states.csv
"""
import sys, numpy as np, pandas as pd, warnings
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
PANEL, R, VINT = sys.argv[1:4]
STATES = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'MI', 'WA']
H = 4; GAMMA = {'covid': 1.0, 'flu': 1.5}; WARMUP = 4; PSEUDO = 1.0; RETRAIN_EVERY = 4; MIN_ORIGINS = 8
SEGMENTS = [(pd.Timestamp('2022-01-01'), pd.Timestamp('2024-05-04')), (pd.Timestamp('2024-11-09'), pd.Timestamp('2099-01-01'))]
ALPHAS = [0.05, 0.2, 0.5]
panel = pd.read_csv(PANEL, parse_dates=['week_end'])
panel = panel[panel.state.isin(STATES) & (panel.preliminary == 0)]
V = pd.read_csv(VINT, parse_dates=['as_of', 'week_end']).drop_duplicates(['pathogen', 'as_of', 'state', 'week_end'], keep='last')

def winc(b, I0, g):
    r = b - g
    return b * I0 * (np.expm1(r) / r if abs(r) > 1e-9 else 1.0)

def reconstruct(C, g):
    n = len(C); beta = np.full(n, np.nan); I = np.full(n + 1, np.nan); I[0] = C[0] / winc(g, 1.0, g)
    for t in range(n):
        f = lambda b: winc(b, I[t], g) - C[t]
        lo, hi = 1e-6, 50.0
        while f(lo) > 0: lo /= 10.0
        while f(hi) < 0: hi *= 2.0
        beta[t] = brentq(f, lo, hi, xtol=1e-12); I[t + 1] = I[t] * np.exp(beta[t] - g)
    return beta, I

def ivp(I0, betas, g):
    rhs = lambda t, y: [(betas[min(int(t), len(betas) - 1)] - g) * y[0], betas[min(int(t), len(betas) - 1)] * y[0]]
    s = solve_ivp(rhs, (0, len(betas)), [I0, 0.0], t_eval=np.arange(len(betas) + 1), method='RK45', rtol=1e-8, atol=1e-10, max_step=0.25)
    return np.diff(s.y[1])

def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0)

F_BASE = ['lb', 'lb_l1', 'lb_l2', 'lb_l3', 'lC', 'sin', 'cos']
F_WW = F_BASE + ['ww', 'ww_l1', 'dww1', 'dww2']

def build(path, T, vint):
    """Feature table using admissions as published in `vint` (week_end -> value per state), truncated at T."""
    g = GAMMA[path]; adm = f'{path}_adm'; rows = []
    ww = {'covid': 'ww_covid_log10conc_rel', 'flu': 'ww_flu_wval_median'}[path]
    for st in STATES:
        d0 = panel[(panel.state == st) & (panel.week_end <= T)].sort_values('week_end').copy()
        v = vint.get(st)
        if v is not None:
            m = d0.week_end.map(v); d0[adm] = m.where(m.notna(), d0[adm])
        for a, b in SEGMENTS:
            d = d0[(d0.week_end >= a) & (d0.week_end <= b)].copy().reset_index(drop=True)
            if len(d) < WARMUP + 2: continue
            d['C'] = d[adm].ffill(limit=2).fillna(0) + PSEUDO
            beta, I = reconstruct(d.C.values, g)
            d['beta'] = beta; d['I_next'] = I[1:]; d['lb'] = np.log(beta)
            w = d[ww].ffill(limit=3).shift(1)                       # wastewater through the previous week
            d['ww'] = np.log(w + 0.1) if path == 'flu' else w
            for k in range(1, 4): d[f'lb_l{k}'] = d.lb.shift(k)
            d['lC'] = np.log(d.C); d['ww_l1'] = d.ww.shift(1)
            d['dww1'] = d.ww - d.ww.shift(1); d['dww2'] = d.ww - d.ww.shift(2)
            wk = d.week_end.dt.isocalendar().week.astype(float)
            d['sin'] = np.sin(2 * np.pi * wk / 52.18); d['cos'] = np.cos(2 * np.pi * wk / 52.18)
            for k in range(1, H + 1): d[f'y_lb_{k}'] = d.lb.shift(-k)
            d['state'] = st; d['seg_pos'] = np.arange(len(d))
            rows.append(d[d.seg_pos >= WARMUP])
    return pd.concat(rows, ignore_index=True)

out_pts = []
for path in ['covid', 'flu']:
    g = GAMMA[path]; Vp = V[V.pathogen == path]
    first = Vp.groupby('as_of').week_end.max()
    truth = panel.set_index(['state', 'week_end'])[f'{path}_adm']
    origins = [T for T in sorted(panel.week_end.unique()) if pd.Timestamp('2025-01-04') <= T <= pd.Timestamp('2026-10-03')]
    models = {}; n_fit = 0; hist = []
    for T in origins:
        cand = first[first >= T]
        if cand.empty: continue
        a = cand.index.min()
        if (a - T).days > 13: continue                        # no vintage released for this week
        vv = Vp[(Vp.as_of == a) & (Vp.week_end <= T)]
        vint = {st: x.set_index('week_end').value for st, x in vv.groupby('state')}
        D = build(path, T, vint)
        if n_fit % RETRAIN_EVERY == 0:
            models = {}
            for k in range(1, H + 1):
                tr = D[D.week_end + pd.Timedelta(weeks=k) <= T].dropna(subset=[f'y_lb_{k}'])
                models[('sir_ml', k)] = gbm().fit(tr[F_BASE], tr[f'y_lb_{k}'])
                models[('sir_ml_ww', k)] = gbm().fit(tr[F_WW], tr[f'y_lb_{k}'])
        n_fit += 1
        X = D[D.week_end == T]
        cur = {(r.state): r for _, r in X.iterrows()}
        # point forecasts
        for st, r in cur.items():
            xr = r.to_frame().T
            fc = {'naive_rt': np.full(H, r.C)}
            for m, cols in [('sir_ml', F_BASE), ('sir_ml_ww', F_WW)]:
                bp = np.exp([models[(m, k)].predict(xr[cols].astype(float))[0] for k in range(1, H + 1)])
                fc[m + '_rt'] = ivp(r.I_next, np.clip(bp, 1e-6, 50), g)
            for m, v in fc.items():
                for k in range(1, H + 1):
                    tgt = T + pd.Timedelta(weeks=k)
                    hist.append(dict(pathogen=path, state=st, origin=T, h=k, model=m, target=tgt,
                                     pred=max(v[k - 1] - PSEUDO, 0.0), obs=truth.get((st, tgt), np.nan), asof=a))
        # interval calibration from earlier real-time forecasts, errors measured against this vintage
        Hh = pd.DataFrame(hist)
        known = D.set_index(['state', 'week_end']).C - PSEUDO
        past = Hh[Hh.target <= T].copy()
        past['obs_T'] = [known.get((s, t), np.nan) for s, t in zip(past.state, past.target)]
        past = past.dropna(subset=['obs_T']); past['err'] = np.log1p(past.obs_T) - np.log1p(past.pred)
        now = Hh[Hh.origin == T]
        for (m, k), cur_g in now.groupby(['model', 'h']):
            e = past[(past.model == m) & (past.h == k)]
            if e.origin.nunique() < MIN_ORIGINS: continue
            e = e.err.values; lp = np.log1p(cur_g.pred.values)
            q = {'median': np.quantile(e, 0.5)}
            for al in ALPHAS:
                q[f'lo{al}'] = np.quantile(e, al / 2); q[f'hi{al}'] = np.quantile(e, 1 - al / 2)
            for i, (_, r) in enumerate(cur_g.iterrows()):
                d = dict(pathogen=path, state=r.state, origin=T, h=k, model=m, obs=r.obs, point=r.pred, asof=a)
                for kk, vq in q.items(): d[kk] = max(np.expm1(lp[i] + vq), 0.0)
                out_pts.append(d)
    print(path, 'origins with a vintage:', n_fit, flush=True)
P = pd.DataFrame(out_pts).dropna(subset=['obs'])
P.to_csv(f'{R}/realtime_probabilistic_forecasts.csv', index=False)
print(P.groupby(['pathogen', 'model']).origin.agg(['nunique', 'min', 'max']))
