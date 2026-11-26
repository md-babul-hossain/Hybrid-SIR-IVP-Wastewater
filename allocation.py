"""Surge-capacity allocation across states driven by admission forecasts.

Each week a national flexible reserve R (staffed-bed capacity, in weekly-admission units) is allocated across
the 10 states two weeks ahead, by solving the linear program
    min_x,u  sum_s u_s - eps * sum_s w_s x_s
    s.t.     u_s >= Dhat_s - B_s - x_s,  u_s >= 0,  x_s >= 0,  sum_s x_s = R
where Dhat_s = forecast COVID-19 + influenza admissions, B_s = routine respiratory capacity
(1.2 x the state's median weekly combined admissions in 2023-2024), and the small eps-term spreads any
slack toward states with larger forecast demand (w_s = Dhat_s / sum Dhat). The realised unmet surge load is
    sum_s max(0, D_s - B_s - x_s)
with D_s the observed admissions. Policies differ only in the forecast used for Dhat.
Scenarios: (1) routine, decision at origin T for week T+2 (results/forecasts.csv, h=2);
           (2) reporting outage, decision at week T+k-2 for week T+k, k=2..8, using only wastewater available at the
           decision time (results/outage_alloc_forecasts.csv).
Usage: python allocation.py us_state_week_panel.csv results
"""
import sys, numpy as np, pandas as pd
from scipy.optimize import linprog
P, R_ = sys.argv[1], sys.argv[2]
panel = pd.read_csv(P, parse_dates=['week_end'])
EPS = 1e-4

def capacity(states):
    d = panel[panel.state.isin(states) & (panel.use_in_model == 1) &
              (panel.week_end >= '2023-01-01') & (panel.week_end < '2025-01-01')]
    return 1.2 * (d.covid_adm + d.flu_adm).groupby(d.state).median()

def allocate(Dhat, B, R):
    n = len(Dhat); w = Dhat / max(Dhat.sum(), 1e-9)
    c = np.r_[-EPS * w, np.ones(n)]                               # variables [x, u]
    A_ub = np.hstack([-np.eye(n), -np.eye(n)]); b_ub = -(Dhat - B)  # -x - u <= -(Dhat - B)
    A_eq = np.r_[np.ones(n), np.zeros(n)][None, :]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[R], bounds=[(0, None)] * (2 * n), method='highs')
    return res.x[:n]

def evaluate(T, key, models, frac):
    """T: long table with columns key + ['state','model','pred','obs'] (combined covid+flu)."""
    states = sorted(T.state.unique()); B = capacity(states).reindex(states).values
    R = frac * B.sum(); out = []
    for k, g in T.groupby(key):
        obs = g[g.model == models[0]].set_index('state').obs.reindex(states).values
        if np.isnan(obs).any(): continue
        pol = {'no_reserve': np.zeros(len(states)),
               'capacity_proportional': R * B / B.sum(),
               'oracle': allocate(obs, B, R)}
        for m in models:
            p = g[g.model == m].set_index('state').pred.reindex(states).values
            if np.isnan(p).any(): break
            pol[m] = allocate(p, B, R)
        else:
            for name, x in pol.items():
                out.append(dict(zip(key if isinstance(key, list) else [key], k if isinstance(k, tuple) else [k]),
                                policy=name, unmet=np.maximum(obs - B - x, 0).sum()))
    return pd.DataFrame(out)

def combine(F, idx):
    g = F.groupby(idx + ['state', 'model'])[['pred', 'obs']].sum(min_count=2).reset_index()
    return g.dropna()

def summarise(U, label):
    tot = U.groupby('policy').unmet.sum()
    s = pd.DataFrame({'total_unmet': tot.round(0),
                      'pct_of_no_reserve': (100 * tot / tot['no_reserve']).round(1)})
    gain = tot['no_reserve'] - tot['oracle']
    s['pct_of_oracle_reduction'] = (100 * (tot['no_reserve'] - tot) / gain).round(1)
    s['scenario'] = label
    return s.sort_values('total_unmet')

if __name__ == '__main__':
    F = pd.read_csv(f'{R_}/forecasts.csv'); F = F[F.h == 2]
    C = combine(F, ['origin'])
    rmods = ['naive', 'sir_const', 'sir_ar', 'pure_ml_ww', 'sir_ml', 'sir_ml_ww']
    O = pd.read_csv(f'{R_}/outage_alloc_forecasts.csv')   # decision-time (lead-2) forecasts
    CO = combine(O, ['origin', 'k'])
    omods = ['locf', 'sir_const', 'sir_ml', 'ww_ratio', 'sir_ww']
    res = []
    for frac in [0.05, 0.10, 0.20]:
        res.append(summarise(evaluate(C, ['origin'], rmods, frac), 'routine').assign(reserve_frac=frac))
        res.append(summarise(evaluate(CO, ['origin', 'k'], omods, frac), 'outage').assign(reserve_frac=frac))
    import os
    if os.path.exists(f'{R_}/outage_alloc_forecasts_lag1.csv'):     # sensitivity: extra 1-week publication lag
        O1 = pd.read_csv(f'{R_}/outage_alloc_forecasts_lag1.csv'); CO1 = combine(O1, ['origin', 'k'])
        for frac in [0.05, 0.10, 0.20]:
            res.append(summarise(evaluate(CO1, ['origin', 'k'], omods, frac), 'outage_extra_lag1').assign(reserve_frac=frac))
    S = pd.concat(res).reset_index().rename(columns={'index': 'policy'})
    S.to_csv(f'{R_}/allocation_summary.csv', index=False)
    pd.set_option('display.width', 200)
    print(S.to_string(index=False))
    st = sorted(C.state.unique()); B = capacity(st)
    print('\nRoutine capacity B_s:', B.round(0).to_dict(), ' sum', round(B.sum()))
