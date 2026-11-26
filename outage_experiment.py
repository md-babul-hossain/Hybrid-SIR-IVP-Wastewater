"""
Reporting-outage experiment: how well can hospital admissions be tracked when hospital reporting
stops but wastewater surveillance continues?

(A) Simulated outages: at each origin T (every 4 weeks, Jan 2025 - Aug 2026) hospital data after T are
    hidden for L = 8 weeks; wastewater remains available. Methods:
      locf        last observed admissions carried forward
      sir_const   SIR-IVP with beta held at its last reconstructed value
      sir_ml      SIR-IVP with ML-forecast beta (no wastewater)          [hospital-only hybrid]
      ww_ratio    ML scaling of last admissions by the change in wastewater (no SIR)
      sir_ww      SIR-IVP with beta inferred each week from concurrent wastewater (proposed)
(B) Real outage: May-Oct 2024 voluntary-reporting period, models trained on data up to 4 May 2024.

Usage: python outage_experiment.py us_state_week_panel.csv results_dir
"""
import sys, os, numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
PANEL, OUTDIR = sys.argv[1], sys.argv[2]
_argv = sys.argv; sys.argv = ['x', PANEL, OUTDIR]
import hybrid_sir_ivp_experiment as E          # reuse reconstruction + SIR-IVP solver
sys.argv = _argv

L = 8
STATES = E.STATES
panel = pd.read_csv(PANEL, parse_dates=['week_end'])
panel = panel[panel.state.isin(STATES) & (panel.preliminary == 0)]
GAP0, GAP1 = pd.Timestamp('2024-05-11'), pd.Timestamp('2024-11-02')

def series(path):
    """Per state: full weekly wastewater series + beta reconstruction on reliable segments."""
    out = []
    ww_col = 'ww_covid_log10conc_rel' if path == 'covid' else 'ww_flu_wval_median'
    for st in STATES:
        d = panel[panel.state == st].sort_values('week_end').reset_index(drop=True)
        w = d[ww_col].ffill(limit=3)
        d['ww'] = w if path == 'covid' else np.log(w + 0.1)
        d['C'] = np.nan; d['beta'] = np.nan; d['I_next'] = np.nan; d['seg'] = -1
        for s, (a, b) in enumerate(E.SEGMENTS):
            m = (d.week_end >= a) & (d.week_end <= b)
            C = d.loc[m, f'{path}_adm'].ffill(limit=2).fillna(0).values + E.PSEUDO
            beta, I = E.reconstruct(C, E.GAMMA[path])
            d.loc[m, 'C'] = C; d.loc[m, 'beta'] = beta; d.loc[m, 'I_next'] = I[1:]; d.loc[m, 'seg'] = s
            idx = d.index[m]; d.loc[idx[:E.WARMUP], 'seg'] = -1          # drop warm-up weeks
        d['lb'] = np.log(d.beta); d['lC'] = np.log(d.C)
        for k in range(1, 4): d[f'lb_l{k}'] = d.lb.shift(k)
        wk = d.week_end.dt.isocalendar().week.astype(float)
        d['sin'] = np.sin(2 * np.pi * wk / 52.18); d['cos'] = np.cos(2 * np.pi * wk / 52.18)
        d['state'] = st
        out.append(d)
    return out

def examples(S, T_max, kmax):
    """Training pairs (origin t, horizon k) with origin and target both in the same reliable segment and target <= T_max."""
    rows = []
    for d in S:
        n = len(d)
        for t in range(n):
            if d.seg[t] < 0 or d.week_end[t] + pd.Timedelta(weeks=1) > T_max: continue
            for k in range(1, kmax + 1):
                j = t + k
                if j >= n or d.seg[j] != d.seg[t] or d.week_end[j] > T_max: break
                rows.append(feat(d, t, k) | dict(y_lb=d.lb[j], y_dlC=d.lC[j] - d.lC[t]))
    return pd.DataFrame(rows).dropna()

def feat(d, t, k):
    j = t + k
    return dict(k=k, lb=d.lb[t], lb_l1=d.lb_l1[t], lb_l2=d.lb_l2[t], lb_l3=d.lb_l3[t], lC=d.lC[t],
                sin=d.sin[j], cos=d.cos[j], ww_t=d.ww[t], ww_j=d.ww[j], dww=d.ww[j] - d.ww[t],
                dww_step=d.ww[j] - d.ww[j - 1])

F_NO = ['k', 'lb', 'lb_l1', 'lb_l2', 'lb_l3', 'lC', 'sin', 'cos']
F_WW = F_NO + ['ww_t', 'ww_j', 'dww', 'dww_step']
F_RATIO = ['k', 'ww_t', 'ww_j', 'dww', 'dww_step', 'sin', 'cos']

def fit(tr):
    m = {}
    m['no'] = E.gbm().fit(tr[F_NO], tr.y_lb)
    m['ww'] = E.gbm().fit(tr[F_WW], tr.y_lb)
    m['ratio'] = E.gbm().fit(tr[F_RATIO], tr.y_dlC)
    return m

def predict_outage(d, t, Lw, m, path):
    g = E.GAMMA[path]
    X = pd.DataFrame([feat(d, t, k) for k in range(1, Lw + 1)])
    ok = X[F_WW].notna().all(1).all()
    res = {'locf': np.full(Lw, d.C[t]),
           'sir_const': E.sir_ivp_forecast(d.I_next[t], np.full(Lw, d.beta[t]), g),
           'sir_ml': E.sir_ivp_forecast(d.I_next[t], np.clip(np.exp(m['no'].predict(X[F_NO])), 1e-6, 50), g)}
    if ok:
        res['ww_ratio'] = d.C[t] * np.exp(m['ratio'].predict(X[F_RATIO]))
        res['sir_ww'] = E.sir_ivp_forecast(d.I_next[t], np.clip(np.exp(m['ww'].predict(X[F_WW])), 1e-6, 50), g)
    return {k: np.maximum(v - E.PSEUDO, 0) for k, v in res.items()}

def simulated(path):
    S = series(path); out = []
    origins = pd.date_range('2025-01-04', panel.week_end.max() - pd.Timedelta(weeks=L), freq='4W-SAT')
    for T in origins:
        m = fit(examples(S, T, L))
        for d in S:
            t = d.index[d.week_end == T]
            if not len(t): continue
            t = t[0]
            if t + L >= len(d) or d.seg[t] < 0 or (d.seg[t:t + L + 1] != d.seg[t]).any(): continue
            P = predict_outage(d, t, L, m, path)
            for meth, v in P.items():
                for k in range(1, L + 1):
                    out.append(dict(pathogen=path, state=d.state[t], origin=T, k=k, model=meth,
                                    obs=d.C[t + k] - E.PSEUDO, pred=v[k - 1]))
    return pd.DataFrame(out)

def real_gap(path):
    S = series(path); T = pd.Timestamp('2024-05-04'); Lw = int((GAP1 - T).days / 7)
    m = fit(examples(S, T, 8))
    out = []
    for d in S:
        t = d.index[d.week_end == T][0]
        X = pd.DataFrame([feat(d, t, min(k, 8)) | dict(ww_j=d.ww[t + k], dww=d.ww[t + k] - d.ww[t],
                          dww_step=d.ww[t + k] - d.ww[t + k - 1], sin=d.sin[t + k], cos=d.cos[t + k]) for k in range(1, Lw + 1)])
        X['ww_j'] = X.ww_j.ffill().fillna(d.ww[t]); X['dww'] = X.ww_j - d.ww[t]
        X['dww_step'] = X.ww_j.diff().fillna(X.dww.iloc[0])
        g = E.GAMMA[path]
        pw = E.sir_ivp_forecast(d.I_next[t], np.clip(np.exp(m['ww'].predict(X[F_WW])), 1e-6, 50), g) - E.PSEUDO
        pn = E.sir_ivp_forecast(d.I_next[t], np.clip(np.exp(m['no'].predict(X[F_NO])), 1e-6, 50), g) - E.PSEUDO
        for k in range(1, Lw + 1):
            r = d.iloc[t + k]
            out.append(dict(pathogen=path, state=d.state[t], week_end=r.week_end, k=k,
                            reported=r[f'{path}_adm'], reporting_frac=r.reporting_frac,
                            sir_ww=max(pw[k - 1], 0), sir_ml=max(pn[k - 1], 0), locf=d.C[t] - E.PSEUDO))
    return pd.DataFrame(out)

# ---------- decision-time (lead-2) forecasts for allocation during outages ----------
# A reserve for week T+k is committed at week T+k-2, when wastewater is known only up to week T+m, m = k-2.
# Target week T+j therefore uses wastewater from week T+min(j, m) (lag_j = max(0, j-m)).
def feat_lag(d, t, k, lag):
    a = t + k - lag                                    # latest wastewater week available for this target
    f = feat(d, t, k)
    f.update(ww_j=d.ww[a], dww=d.ww[a] - d.ww[t], dww_step=d.ww[a] - d.ww[a - 1], lag=lag)
    return f
F_WWL = F_WW + ['lag']; F_RATIOL = F_RATIO + ['lag']
def examples_lag(S, T_max, kmax, maxlag=2):
    rows = []
    for d in S:
        n = len(d)
        for t in range(n):
            if d.seg[t] < 0 or d.week_end[t] + pd.Timedelta(weeks=1) > T_max: continue
            for k in range(1, kmax + 1):
                j = t + k
                if j >= n or d.seg[j] != d.seg[t] or d.week_end[j] > T_max: break
                for lag in range(0, min(maxlag, k + maxlag - 2) + 1):   # extra=0 reproduces lags 0..min(2,k)
                    if t + k - lag - 1 < 0: continue
                    rows.append(feat_lag(d, t, k, lag) | dict(y_lb=d.lb[j], y_dlC=d.lC[j] - d.lC[t]))
    return pd.DataFrame(rows).dropna()
def alloc_forecasts(path, extra=0):
    """extra: additional publication lag (weeks) of wastewater beyond the 2-week decision lead."""
    S = series(path); g = E.GAMMA[path]; out = []
    origins = pd.date_range('2025-01-04', panel.week_end.max() - pd.Timedelta(weeks=L), freq='4W-SAT')
    for T in origins:
        tr = examples_lag(S, T, L, 2 + extra); tr0 = examples(S, T, L)
        m_ww = E.gbm().fit(tr[F_WWL], tr.y_lb); m_r = E.gbm().fit(tr[F_RATIOL], tr.y_dlC)
        m_no = E.gbm().fit(tr0[F_NO], tr0.y_lb)
        for d in S:
            t = d.index[d.week_end == T]
            if not len(t): continue
            t = t[0]
            if t + L >= len(d) or d.seg[t] < 0 or (d.seg[t:t + L + 1] != d.seg[t]).any(): continue
            for k in range(2, L + 1):
                mm = k - 2 - extra                       # latest wastewater week = T + mm (may precede T)
                X = pd.DataFrame([feat_lag(d, t, j, max(0, j - mm)) for j in range(1, k + 1)])
                if X[F_WWL].isna().any().any(): continue
                Xn = pd.DataFrame([feat(d, t, j) for j in range(1, k + 1)])
                P_ = {'locf': d.C[t],
                      'sir_const': E.sir_ivp_forecast(d.I_next[t], np.full(k, d.beta[t]), g)[-1],
                      'sir_ml': E.sir_ivp_forecast(d.I_next[t], np.clip(np.exp(m_no.predict(Xn[F_NO])), 1e-6, 50), g)[-1],
                      'ww_ratio': d.C[t] * np.exp(m_r.predict(X[F_RATIOL].iloc[[-1]]))[0],
                      'sir_ww': E.sir_ivp_forecast(d.I_next[t], np.clip(np.exp(m_ww.predict(X[F_WWL])), 1e-6, 50), g)[-1]}
                for meth, v in P_.items():
                    out.append(dict(pathogen=path, state=d.state[t], origin=T, k=k, model=meth,
                                    obs=d.C[t + k] - E.PSEUDO, pred=max(v - E.PSEUDO, 0)))
    return pd.DataFrame(out)

if __name__ == '__main__':
    if len(sys.argv) > 3 and sys.argv[3] == 'alloc':
        extra = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        Z = pd.concat([alloc_forecasts(p, extra) for p in ['covid', 'flu']])
        name = 'outage_alloc_forecasts.csv' if extra == 0 else f'outage_alloc_forecasts_lag{extra}.csv'
        Z.to_csv(f'{OUTDIR}/{name}', index=False); print('alloc forecasts', len(Z)); sys.exit()
    os.makedirs(OUTDIR, exist_ok=True)
    A = pd.concat([simulated(p) for p in ['covid', 'flu']]); A.to_csv(f'{OUTDIR}/outage_forecasts.csv', index=False)
    B = pd.concat([real_gap(p) for p in ['covid', 'flu']]); B.to_csv(f'{OUTDIR}/real_gap_2024.csv', index=False)
    A['ae'] = (A.pred - A.obs).abs()
    common = A.groupby(['pathogen', 'state', 'origin']).model.transform('nunique') == A.model.nunique()
    A = A[common]
    mae = A.groupby(['pathogen', 'state', 'k', 'model']).ae.mean().unstack('model')
    rel = np.exp(np.log(mae.div(mae.locf, axis=0)).groupby(['pathogen', 'k']).mean())
    rel.round(3).to_csv(f'{OUTDIR}/outage_relMAE_vs_locf.csv')
    cum = A.groupby(['pathogen', 'state', 'origin', 'model']).agg(o=('obs', 'sum'), p=('pred', 'sum')).reset_index()
    cum['ape'] = (cum.p - cum.o).abs() / cum.o.clip(lower=1) * 100
    cumt = cum.groupby(['pathogen', 'model']).ape.median().unstack('model')
    cumt.round(1).to_csv(f'{OUTDIR}/outage_cumulative_APE.csv')
    pd.set_option('display.width', 200)
    print('Relative MAE vs LOCF by week into outage\n', rel.round(3))
    print('\nMedian abs % error of total 8-week admissions\n', cumt.round(1))
    print('\nn outages:', A.groupby('pathogen').apply(lambda x: x[['state', 'origin']].drop_duplicates().shape[0]).to_dict())
