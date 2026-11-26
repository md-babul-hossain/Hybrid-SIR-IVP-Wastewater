"""Sensitivity analysis of prob_allocation.py: COVID-19 and influenza log errors are drawn as PAIRS from the
same past state, origin and horizon (preserving their dependence) instead of independently.
Original description follows.

Surge-capacity allocation with predictive distributions (Theorem 5) versus point forecasts (Theorem 4).

Routine setting: decision at origin T for target week T+2 (h = 2), reserve R = frac * sum_s B_s.
Predictive distribution of combined COVID-19 + influenza admissions in state s:
    D_s = expm1(log1p(pred_cov) + e_cov) + expm1(log1p(pred_flu) + e_flu),
with e drawn from the rolling sets of past log errors of the same model and horizon (pooled over states,
targets observed by T; at least 8 past origins), the same calibration as the interval forecasts.
Optimal allocation (Theorem 5): x_s = max(0, Q_s(q*) - B_s), with the common quantile level q* chosen so
that sum_s x_s = R (water-filling), computed on 4,000 Monte Carlo draws.
Point policy: the LP of the main text with the point forecast. Both are evaluated by realised unmet demand.

Usage: python prob_allocation_paired.py us_state_week_panel.csv results
"""
import sys, numpy as np, pandas as pd
P, R_ = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], P, R_]
import allocation as A
F = pd.read_csv(f'{R_}/forecasts.csv', parse_dates=['origin']); F = F[F.h == 2]
F['target'] = F.origin + pd.Timedelta(weeks=2)
F['err'] = np.log1p(F.obs) - np.log1p(F.pred)
MODELS = ['naive', 'sir_ml', 'sir_ml_ww']
MIN_ORIGINS, NS = 8, 4000
rng = np.random.default_rng(2026)
states = sorted(F.state.unique()); B = A.capacity(states).reindex(states).values

def waterfill(S, B, R):
    """S: (n_states, n_samples) demand draws. Return x with sum R minimising mean sum (S - B - x)^+."""
    Q = lambda q: np.quantile(S, q, axis=1)
    tot = lambda q: np.maximum(Q(q) - B, 0).sum()
    if tot(1.0) <= R:                                       # reserve exceeds every plausible surge
        x = np.maximum(Q(1.0) - B, 0); return x + (R - x.sum()) * B / B.sum(), 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tot(mid) < R else (lo, mid)
    x = np.maximum(Q(hi) - B, 0)
    return x * R / x.sum(), hi

rows, qlev = [], []
origins = sorted(F.origin.unique())
for T in origins:
    g = F[F.origin == T]
    obs = g[g.model == 'naive'].groupby('state').obs.sum().reindex(states).values
    if np.isnan(obs).any() or g.groupby(['state', 'model']).size().min() < 2: continue
    past = F[F.target <= T]
    ok = all(past[(past.model == m) & (past.pathogen == p)].origin.nunique() >= MIN_ORIGINS for m in MODELS for p in ['covid', 'flu'])
    if not ok: continue
    for frac in [0.05, 0.10, 0.20]:
        R = frac * B.sum()
        pol = {'no_reserve': np.zeros(len(states)), 'capacity_proportional': R * B / B.sum(), 'oracle': A.allocate(obs, B, R)}
        for m in MODELS:
            pt = g[g.model == m].groupby('state').pred.sum().reindex(states).values
            pol[f'{m} (point)'] = A.allocate(pt, B, R)
            pm = past[past.model == m].pivot_table(index=['origin', 'state'], columns='pathogen', values='err').dropna()
            E2 = pm[['covid', 'flu']].values                      # paired (e_cov, e_flu) from the same state-origin
            idx = rng.integers(0, len(E2), size=(len(states), NS))
            S = np.zeros((len(states), NS))
            for j, p in enumerate(['covid', 'flu']):
                pr = g[(g.model == m) & (g.pathogen == p)].set_index('state').pred.reindex(states).values
                S += np.expm1(np.log1p(pr)[:, None] + E2[idx, j])
            x, q = waterfill(S, B, R); pol[f'{m} (distribution)'] = x
            qlev.append(dict(origin=T, frac=frac, model=m, q_star=q))
        for name, x in pol.items():
            rows.append(dict(origin=T, frac=frac, policy=name, unmet=np.maximum(obs - B - x, 0).sum()))
U = pd.DataFrame(rows)
out = []
for frac, u in U.groupby('frac'):
    tot = u.groupby('policy').unmet.sum(); gain = tot['no_reserve'] - tot['oracle']
    s = pd.DataFrame({'total_unmet': tot.round(0), 'pct_of_oracle_reduction': (100 * (tot['no_reserve'] - tot) / gain).round(1)})
    s['reserve_frac'] = frac; s['decisions'] = u.origin.nunique(); out.append(s.sort_values('total_unmet'))
S = pd.concat(out); S.to_csv(f'{R_}/prob_allocation_paired_summary.csv')
U.to_csv(f'{R_}/prob_allocation_paired_weekly.csv', index=False)
Q = pd.DataFrame(qlev); Q.to_csv(f'{R_}/prob_allocation_paired_qstar.csv', index=False)
# paired block bootstrap: distribution policy minus point policy, per decision week
boot = []; rng2 = np.random.default_rng(1)
for frac in [0.05, 0.10, 0.20]:
    u = U[U.frac == frac].pivot(index='origin', columns='policy', values='unmet').sort_index()
    for m in MODELS:
        d = (u[f'{m} (distribution)'] - u[f'{m} (point)']).values; k = len(d); blk = 4; nb = int(np.ceil(k / blk))
        st = rng2.integers(0, k - blk + 1, size=(5000, nb)); idx = (st[:, :, None] + np.arange(blk)).reshape(5000, -1)[:, :k]
        mm = d[idx].mean(1); lo, hi = np.percentile(mm, [2.5, 97.5])
        boot.append(dict(reserve_frac=frac, model=m, mean_diff=d.mean(), total_diff=d.sum(), lo=lo, hi=hi))
Bt = pd.DataFrame(boot); Bt.to_csv(f'{R_}/prob_allocation_paired_bootstrap.csv', index=False)
pd.set_option('display.width', 200)
print(S.to_string()); print(Bt.round(1).to_string()); print(Q.groupby(['frac', 'model']).q_star.describe().round(3))
