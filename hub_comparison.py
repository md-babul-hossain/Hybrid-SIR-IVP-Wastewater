"""Compare the hybrid SIR-IVP interval forecasts with the CDC COVID-19 Forecast Hub and FluSight ensembles.

Alignment: a Hub forecast with reference date R is made with admissions observed through week R-7.
Our forecast origin T is the last observed week, so Hub file R = T + 7 days and Hub horizon = h - 1
(Hub horizons 0..3 correspond to our 1..4-week horizons; target week T + 7h in both).
All methods are scored with the same quantile levels (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975) and the
same observed (revised) admissions from the NHSN panel. Only state-weeks where every method has a
forecast are compared.

Usage: python hub_comparison.py us_state_week_panel.csv results <covid_hub_dir> <flusight_hub_dir>
"""
import sys, glob, os, numpy as np, pandas as pd
PANEL, R, CH, FH = sys.argv[1:5]
FIPS = {'CA': '06', 'TX': '48', 'FL': '12', 'NY': '36', 'PA': '42', 'IL': '17', 'OH': '39', 'GA': '13', 'MI': '26', 'WA': '53'}
INV = {v: k for k, v in FIPS.items()}
ALPHAS = [0.05, 0.2, 0.5]
Q = {0.025: 'lo0.05', 0.975: 'hi0.05', 0.1: 'lo0.2', 0.9: 'hi0.2', 0.25: 'lo0.5', 0.75: 'hi0.5', 0.5: 'median'}

def load_hub(folder, model, target):
    out = []
    for f in sorted(glob.glob(os.path.join(folder, 'model-output', model, '*.csv'))):
        d = pd.read_csv(f, dtype={'location': str})
        d = d[(d.target == target) & (d.output_type == 'quantile') & d.location.isin(INV) & d.horizon.between(0, 3)]
        d['output_type_id'] = d.output_type_id.astype(float).round(3)
        d = d[d.output_type_id.isin(list(Q))]
        out.append(d)
    d = pd.concat(out)
    d['state'] = d.location.map(INV); d['h'] = d.horizon + 1
    d['origin'] = pd.to_datetime(d.reference_date) - pd.Timedelta(days=7)
    w = d.pivot_table(index=['state', 'origin', 'h'], columns='output_type_id', values='value').rename(columns=Q).reset_index()
    w['model'] = model
    return w.dropna()

def wis(df):
    y = df.obs; s = 0.5 * (y - df['median']).abs()
    for a in ALPHAS:
        l, u = df[f'lo{a}'], df[f'hi{a}']
        s = s + (a / 2) * ((u - l) + (2 / a) * (l - y).clip(lower=0) + (2 / a) * (y - u).clip(lower=0))
    return s / (len(ALPHAS) + 0.5)

panel = pd.read_csv(PANEL, parse_dates=['week_end'])
P = pd.read_csv(f'{R}/probabilistic_forecasts.csv', parse_dates=['origin'])
P = P[P.model.isin(['naive', 'sir_ml', 'sir_ml_ww', 'pure_ml_ww'])]
RT = f'{R}/realtime_probabilistic_forecasts.csv'
if os.path.exists(RT):                      # real-time (data-vintage) forecasts from realtime_vintage.py
    P = pd.concat([P, pd.read_csv(RT, parse_dates=['origin']).drop(columns=['point', 'asof'])])
rows, cov_rows = [], []
for path, hubdir, ens, base, tgt in [('covid', CH, 'CovidHub-ensemble', 'CovidHub-baseline', 'wk inc covid hosp'),
                                     ('flu', FH, 'FluSight-ensemble', 'FluSight-baseline', 'wk inc flu hosp')]:
    H = pd.concat([load_hub(hubdir, ens, tgt), load_hub(hubdir, base, tgt)])
    H['model'] = H.model.map({ens: 'hub_ensemble', base: 'hub_baseline'})
    ours = P[P.pathogen == path][['state', 'origin', 'h', 'model', 'obs'] + list(Q.values())]
    obs = ours[['state', 'origin', 'h', 'obs']].drop_duplicates()
    H = H.merge(obs, on=['state', 'origin', 'h'], how='inner')
    A = pd.concat([ours, H[ours.columns]])
    n_mod = A.model.nunique()
    keep = A.groupby(['state', 'origin', 'h']).model.transform('nunique') == n_mod
    A = A[keep].copy(); A['wis'] = wis(A); A['pathogen'] = path
    rows.append(A)
D = pd.concat(rows)
D.to_csv(f'{R}/hub_comparison_scores.csv', index=False)

# relative WIS vs Hub ensemble (per state mean, geometric mean across states)
mw = D.groupby(['pathogen', 'h', 'state', 'model']).wis.mean().unstack('model')
rel = np.exp(np.log(mw.div(mw['hub_ensemble'], axis=0)).groupby(['pathogen', 'h']).mean())
relall = np.exp(np.log(D.groupby(['pathogen', 'state', 'model']).wis.mean().unstack('model')
                       .pipe(lambda m: m.div(m['hub_ensemble'], axis=0))).groupby('pathogen').mean())
for a in ALPHAS:
    D[f'cov{int(round((1 - a) * 100))}'] = (D.obs >= D[f'lo{a}']) & (D.obs <= D[f'hi{a}'])
cov = D.groupby(['pathogen', 'model'])[['cov50', 'cov80', 'cov95']].mean()
n = D.groupby('pathogen').apply(lambda x: pd.Series({'origins': x.origin.nunique(), 'first': x.origin.min().date(),
                                                     'last': x.origin.max().date(), 'state_weeks': len(x) // x.model.nunique()}))
# block bootstrap: ours - ensemble, total WIS per origin (all horizons, states)
rng = np.random.default_rng(2026); boot = []
for p, g in D.groupby('pathogen'):
    tot = g.groupby(['origin', 'model']).wis.sum().unstack().sort_index()
    for m in [x for x in ['sir_ml_ww', 'sir_ml', 'sir_ml_ww_rt', 'sir_ml_rt', 'naive_rt', 'hub_baseline', 'naive'] if x in tot]:
        x = (tot[m] - tot['hub_ensemble']).values; k = len(x); blk = 4; nb = int(np.ceil(k / blk))
        st = rng.integers(0, k - blk + 1, size=(5000, nb)); idx = (st[:, :, None] + np.arange(blk)).reshape(5000, -1)[:, :k]
        mm = x[idx].mean(1); lo, hi = np.percentile(mm, [2.5, 97.5])
        boot.append(dict(pathogen=p, comparison=f'{m} - hub_ensemble', mean=x.mean(), pct=100 * x.mean() / tot['hub_ensemble'].mean(),
                         lo=lo, hi=hi, significant=(hi < 0) or (lo > 0)))
B = pd.DataFrame(boot)
# absolute error of the predictive median, relative to the Hub ensemble (isolates data revision from interval method)
D['ae_med'] = (D['median'] - D.obs).abs()
mae = D.groupby(['pathogen', 'h', 'model']).ae_med.mean().unstack('model')
mae_rel = mae.div(mae['hub_ensemble'], axis=0)
mae_all = D.groupby(['pathogen', 'model']).ae_med.mean().unstack('model')
mae_rel.round(3).to_csv(f'{R}/hub_median_MAE_relative.csv'); mae_all.round(1).to_csv(f'{R}/hub_median_MAE.csv')
rel.round(3).to_csv(f'{R}/hub_relative_WIS_by_horizon.csv'); relall.round(3).to_csv(f'{R}/hub_relative_WIS_overall.csv')
cov.round(3).to_csv(f'{R}/hub_coverage.csv'); B.round(2).to_csv(f'{R}/hub_bootstrap.csv', index=False); n.to_csv(f'{R}/hub_sample.csv')
pd.set_option('display.width', 200)
print(n); print('\nRelative WIS vs Hub ensemble (by horizon)\n', rel.round(3)); print('\nOverall\n', relall.round(3))
print('\nCoverage\n', cov.round(3)); print('\nMedian MAE rel. to ensemble\n', mae_rel.round(3)); print(mae_all.round(1)); print('\n', B.round(1).to_string(index=False))
