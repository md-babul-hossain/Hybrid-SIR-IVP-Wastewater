"""Supplementary RSV analysis with the same hybrid pipeline. RSV admissions are reported from late 2024,
so models are trained on Nov 2024 onward and tested from 6 Sep 2025.
Usage: python rsv_supplement.py us_state_week_panel.csv results"""
import sys, pandas as pd
P, R = sys.argv[1], sys.argv[2]
sys.argv = ['x', P, R]
import hybrid_sir_ivp_experiment as E
F, D = E.run('rsv', pd.Timestamp('2025-09-06'))
F.to_csv(f'{R}/rsv_forecasts.csv', index=False)
rel_g, pooled, surge, rel = E.evaluate(F)
rel_g.round(3).to_csv(f'{R}/rsv_relative_MAE_vs_naive.csv')
print(rel_g.round(3)); print(pooled.round(1)); print('origins', F.origin.nunique(), 'first', F.origin.min(), 'last', F.origin.max())
