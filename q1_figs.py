"""Publication figures (main text Figs 2-8; supplementary Fig. S2). Run from the folder that contains results/.
Usage: python q1_figs.py results figs
"""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
R, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)

# ---- style: validated categorical slots (blue, orange, aqua), neutral grey for baselines -------------
C = dict(ww='#2a78d6', hosp='#1baf7a', pure='#eb6834', naive='#8a8984', ens='#e34948', const='#6250d6', ratio='#eda100')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'Inter', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.axisbelow': True, 'legend.frameon': False, 'lines.linewidth': 2,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
def tag(ax, s): ax.text(-0.12, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight='bold', color=INK, va='bottom')
PN = {'covid': 'COVID-19', 'flu': 'Influenza'}
GAMMA = {'covid': 1.0, 'flu': 1.5}
STATES = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'MI', 'WA']

# ---- Fig 2: reconstructed effective reproduction number heatmap ------------------------------------
rb = pd.read_csv(f'{R}/reconstructed_beta.csv', parse_dates=['week_end'])
cmap = LinearSegmentedColormap.from_list('div', ['#104281', '#3987e5', '#f0efec', '#e66767', '#a52a2a'])
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
weeks = pd.date_range('2022-01-29', rb.week_end.max(), freq='7D')
for ax, p in zip(axes, ['covid', 'flu']):
    d = rb[rb.pathogen == p].assign(R=lambda x: x.beta / GAMMA[p])
    M = d.pivot(index='state', columns='week_end', values='R').reindex(index=STATES, columns=weeks)
    M = np.exp(np.log(M).T.rolling(3, center=True, min_periods=2).mean().T)   # display only: 3-week geometric mean
    xe = list(weeks - pd.Timedelta(days=3.5)) + [weeks[-1] + pd.Timedelta(days=3.5)]
    im = ax.pcolormesh(xe, np.arange(len(STATES) + 1), np.log2(M.values), cmap=cmap,
                       norm=TwoSlopeNorm(0, -1.2, 1.2), shading='flat', rasterized=True)
    ax.axvspan(pd.Timestamp('2024-05-05'), pd.Timestamp('2024-11-30'), color='#d9d8d3', zorder=0)
    ax.text(pd.Timestamp('2024-08-15'), 5, 'voluntary\nreporting\n(excluded)', ha='center', va='center', fontsize=7, color=INK2)
    ax.set_yticks(np.arange(len(STATES)) + 0.5); ax.set_yticklabels(STATES); ax.invert_yaxis(); ax.grid(False)
    ax.set_title(f'{PN[p]} ($\\gamma={GAMMA[p]}$ per week)', loc='left'); tag(ax, 'ab'[axes.tolist().index(ax)])
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator()); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.015, ticks=np.log2([0.5, 0.71, 1, 1.41, 2]))
cb.ax.set_yticklabels(['0.5', '0.7', '1', '1.4', '2']); cb.set_label('$\\mathcal{R}_w=\\beta_w/\\gamma$ (log scale)'); cb.outline.set_visible(False)
fig.savefig(f'{OUT}/fig2_reproduction_heatmap.png'); plt.close(fig)

# ---- Fig 3: accuracy against persistence and against the CDC ensembles ------------------------------
pw = pd.read_csv(f'{R}/probabilistic_relative_WIS.csv'); hb = pd.read_csv(f'{R}/hub_relative_WIS_by_horizon.csv')
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
for j, p in enumerate(['covid', 'flu']):
    ax = axes[0, j]; d = pw[pw.pathogen == p]
    for m, lab, c, ls in [('sir_ml_ww', 'Hybrid SIR-IVP, with wastewater', C['ww'], '-'), ('sir_ml', 'Hybrid SIR-IVP, hospital data only', C['hosp'], '-'),
                          ('pure_ml_ww', 'Pure ML + wastewater', C['pure'], '-'), ('sir_const', 'SIR-IVP, constant $\\beta$', C['const'], ':')]:
        ax.plot(d.h, d[m], ls, color=c, marker='o', ms=4, label=lab)
    ax.axhline(1, color=C['naive'], lw=1.2, ls='--'); ax.text(4.05, 1.0, 'persistence', fontsize=7, color=INK2, va='center')
    ax.set_ylim(0.5, 1.6 if p == 'covid' else 1.6); ax.set_title(PN[p], loc='left'); tag(ax, 'ab'[j])
    if j == 0: ax.set_ylabel('Relative WIS vs persistence')
    ax = axes[1, j]; d = hb[hb.pathogen == p]
    for m, lab, c, ls in [('sir_ml_ww_rt', 'Hybrid, with wastewater (real time)', C['ww'], '-'), ('sir_ml_rt', 'Hybrid, hospital only (real time)', C['hosp'], '-'),
                          ('sir_ml_ww', 'Hybrid, with wastewater (revised data)', C['ww'], ':'), ('hub_baseline', 'Hub baseline', C['naive'], '-')]:
        ax.plot(d.h, d[m], ls, color=c, marker='o', ms=4, label=lab)
    ax.axhline(1, color=C['ens'], lw=1.4); ax.text(4.05, 1.0, 'CDC\nensemble', fontsize=7, color=C['ens'], va='center')
    ax.set_ylim(0.7, 1.75); ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel('Forecast horizon (weeks)'); tag(ax, 'cd'[j])
    if j == 0: ax.set_ylabel('Relative WIS vs CDC ensemble')
axes[0, 0].legend(loc='upper left', fontsize=6.5); axes[1, 1].legend(loc='upper right', fontsize=6.5, bbox_to_anchor=(1.0, 0.92))
axes[0, 1].annotate('off scale: 2.8 and 5.6\nat 3 and 4 weeks', xy=(2.05, 1.58), xytext=(2.4, 1.3), fontsize=6.5, color=C['const'], arrowprops=dict(arrowstyle='->', color=C['const'], lw=0.8))
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_accuracy.png'); plt.close(fig)

# ---- Fig 4: calibration -----------------------------------------------------------------------------
ph = pd.read_csv(f'{R}/pit_histograms.csv'); dec = pd.read_csv(f'{R}/wis_decomposition_hub_overall.csv')
fig = plt.figure(figsize=(7.2, 4.8)); gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5], hspace=0.6, wspace=0.65)
for i, p in enumerate(['covid', 'flu']):
    for j, sub in enumerate(['not rising', 'rising']):
        ax = fig.add_subplot(gs[i, j]); r = ph[(ph.pathogen == p) & (ph.model == 'sir_ml_ww') & (ph.subset == sub)].iloc[0]
        f = [r[f'b{k}'] for k in range(10)]
        ax.bar(np.arange(10) / 10 + 0.05, np.array(f) * 10, width=0.096, color=C['ww'] if sub == 'not rising' else C['pure'])
        ax.axhline(1, color=INK2, lw=1, ls='--'); ax.set_ylim(0, 5); ax.set_xlim(0, 1)
        ax.set_title(f'{PN[p]}: {sub} (n={int(r.n)})', loc='left', fontsize=8)
        if j == 0: ax.set_ylabel('PIT density')
        if i == 1: ax.set_xlabel('PIT')
        if i == 0 and j == 0: tag(ax, 'a')
for i, p in enumerate(['covid', 'flu']):
    ax = fig.add_subplot(gs[i, 2]); d = dec[dec.pathogen == p].set_index('model')
    order = [('hub_ensemble', 'CDC ensemble'), ('sir_ml_ww_rt', 'Hybrid (real time)'), ('sir_ml_ww', 'Hybrid (revised)'), ('hub_baseline', 'Hub baseline')]
    y = np.arange(len(order)); left = np.zeros(len(order))
    for comp, col in [('dispersion', '#86b6ef'), ('underprediction', C['pure']), ('overprediction', C['const'])]:
        v = np.array([d.loc[m, comp] for m, _ in order]); ax.barh(y, v, left=left, color=col, edgecolor='white', lw=1, label=comp); left += v
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in order], fontsize=7); ax.invert_yaxis(); ax.grid(axis='y', visible=False)
    ax.set_title(f'{PN[p]}: WIS components', loc='left', fontsize=8)
    if i == 1: ax.set_xlabel('Mean WIS (admissions per state-week)')
    if i == 0: tag(ax, 'b'); ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.45, -0.18), ncol=3, columnspacing=0.8, handlelength=1)
fig.savefig(f'{OUT}/fig4_calibration.png'); plt.close(fig)

# ---- Fig 5: probabilistic forecast examples (h = 2) -------------------------------------------------
P = pd.read_csv(f'{R}/probabilistic_forecasts.csv', parse_dates=['origin']); P = P[(P.model == 'sir_ml_ww') & (P.h == 2)]
P['target'] = P.origin + pd.Timedelta(weeks=2)
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
for i, p in enumerate(['covid', 'flu']):
    for j, st in enumerate(['CA', 'NY', 'TX']):
        ax = axes[i, j]; d = P[(P.pathogen == p) & (P.state == st)].sort_values('target')
        ax.fill_between(d.target, d['lo0.05'], d['hi0.05'], color='#b7d3f6', lw=0, label='95% interval')
        ax.fill_between(d.target, d['lo0.5'], d['hi0.5'], color='#6da7ec', lw=0, label='50% interval')
        ax.plot(d.target, d['median'], color=C['ww'], lw=1.4, label='Median forecast')
        ax.plot(d.target, d.obs, color=INK, lw=0, marker='o', ms=1.8, label='Observed')
        ax.set_title(f'{PN[p]}, {st}', loc='left', fontsize=8); ax.tick_params(axis='x', labelsize=7, rotation=0)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b\n%Y')); ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator([1, 7]))
        if j == 0: ax.set_ylabel('Weekly admissions')
axes[0, 0].legend(fontsize=6.5, loc='upper left'); fig.tight_layout(); fig.savefig(f'{OUT}/fig5_probabilistic.png'); plt.close(fig)

# ---- Fig 6: reporting outages ------------------------------------------------------------------------
om = pd.read_csv(f'{R}/outage_relMAE_vs_locf.csv'); of = pd.read_csv(f'{R}/outage_forecasts.csv', parse_dates=['origin'])
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
for j, p in enumerate(['covid', 'flu']):
    ax = axes[j]; d = om[om.pathogen == p]
    for m, lab, c in [('sir_ww', 'Hybrid, with wastewater', C['ww']), ('sir_ml', 'Hybrid, hospital only', C['hosp']),
                      ('ww_ratio', 'Wastewater ratio scaling', C['ratio'])]:
        ax.plot(d.k, d[m], color=c, marker='o', ms=3.5, label=lab)
    ax.axhline(1, color=C['naive'], ls='--', lw=1.2); ax.set_ylim(0.25, 1.15); ax.set_xticks(range(1, 9))
    ax.set_title(PN[p], loc='left'); ax.set_xlabel('Week of outage'); tag(ax, 'ab'[j])
    if j == 0: ax.set_ylabel('Relative MAE vs last value'); ax.legend(fontsize=6.5, loc='lower left')
# example: COVID-19 outage with the largest wave inside the window
e = of[of.pathogen == 'covid']; piv = e.pivot_table(index=['state', 'origin', 'k'], columns='model', values=['pred', 'obs'])
gain = (e[e.model == 'locf'].groupby(['state', 'origin']).obs.agg(lambda x: x.max() / x.iloc[0]))
st, org = gain.idxmax(); ex = e[(e.state == st) & (e.origin == org)]
ax = axes[2]; full = pd.read_csv(f'{R}/reconstructed_beta.csv', parse_dates=['week_end'])
hist = full[(full.pathogen == 'covid') & (full.state == st) & (full.week_end > org - pd.Timedelta(weeks=8)) & (full.week_end <= org + pd.Timedelta(weeks=8))]
ax.plot(hist.week_end, hist.C - 1, color=INK, lw=1.4, label='Observed (hidden after origin)')
for m, lab, c in [('locf', 'Last value', C['naive']), ('sir_ml', 'Hybrid, hospital only', C['hosp']), ('sir_ww', 'Hybrid, with wastewater', C['ww'])]:
    d = ex[ex.model == m].sort_values('k'); ax.plot(org + pd.to_timedelta(d.k * 7, unit='D'), d.pred, color=c, lw=1.6, label=lab)
ax.axvline(org, color=INK2, lw=0.8, ls=':'); ax.set_title(f'COVID-19, {st}: outage from {org:%d %b %Y}', loc='left', fontsize=8)
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator([9, 11, 1])); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b\n%Y')); ax.set_ylabel('Weekly admissions'); ax.legend(fontsize=6, loc='upper left'); tag(ax, 'c')
fig.tight_layout(); fig.savefig(f'{OUT}/fig6_outage.png'); plt.close(fig)

# ---- Fig 7: 2024 voluntary-reporting period ---------------------------------------------------------
g = pd.read_csv(f'{R}/real_gap_2024.csv', parse_dates=['week_end']); g = g[g.pathogen == 'covid']
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), sharex=True)
for ax, st in zip(axes, ['NY', 'IL', 'OH', 'MI']):
    d = g[g.state == st]; ax.plot(d.week_end, d.reported / d.reporting_frac, color=INK, lw=1.3, label='Proxy (reported / fraction reporting)')
    ax.plot(d.week_end, d.sir_ww, color=C['ww'], label='Hybrid, with wastewater'); ax.plot(d.week_end, d.sir_ml, color=C['hosp'], label='Hybrid, hospital only')
    ax.plot(d.week_end, d.locf, color=C['naive'], ls='--', lw=1.2, label='Last value')
    ax.set_title(st, loc='left'); ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b')); ax.tick_params(axis='x', labelsize=7)
axes[0].set_ylabel('Weekly COVID-19 admissions'); fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(); fig.savefig(f'{OUT}/fig7_gap2024.png'); plt.close(fig)

# ---- Fig 8: allocation -------------------------------------------------------------------------------
al = pd.read_csv(f'{R}/allocation_summary.csv'); pa = pd.read_csv(f'{R}/prob_allocation_summary.csv')
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axes[0]; x = np.arange(3); w = 0.26
for i, (m, lab) in enumerate([('naive', 'Persistence'), ('sir_ml', 'Hybrid, hospital only'), ('sir_ml_ww', 'Hybrid, with wastewater')]):
    for k, (kind, hatch) in enumerate([('point', None), ('distribution', None)]):
        v = [pa[(pa.policy == f'{m} ({kind})') & (pa.reserve_frac == f)].pct_of_oracle_reduction.iloc[0] for f in [0.05, 0.1, 0.2]]
        col = [C['naive'], C['hosp'], C['ww']][i]
        ax.bar(x + (i - 1) * w + (k - 0.5) * w / 2, v, width=w / 2 - 0.01, color=col, alpha=0.45 if kind == 'point' else 1.0)
cp = [pa[(pa.policy == 'capacity_proportional') & (pa.reserve_frac == f)].pct_of_oracle_reduction.iloc[0] for f in [0.05, 0.1, 0.2]]
for xi, v in zip(x, cp): ax.plot([xi - 0.42, xi + 0.42], [v, v], color=INK2, ls='--', lw=1)
ax.set_xticks(x); ax.set_xticklabels(['5%', '10%', '20%']); ax.set_xlabel('Reserve (share of routine capacity)'); ax.set_ylim(0, 135); ax.set_yticks(range(0, 101, 20))
ax.set_ylabel('% of oracle reduction in unmet demand'); ax.set_title('Routine reporting: point vs distribution', loc='left'); tag(ax, 'a')
ax.legend(handles=[Patch(color=C['naive'], label='Persistence'), Patch(color=C['hosp'], label='Hybrid, hospital only'), Patch(color=C['ww'], label='Hybrid, with wastewater'),
                   Patch(color='#bbbbbb', alpha=0.45, label='point forecast (LP)'), Patch(color='#555555', label='predictive distribution (Thm 5)'),
                   Line2D([], [], color=INK2, ls='--', lw=1, label='capacity-proportional')], fontsize=6, loc='upper center', ncol=2, columnspacing=0.8)
ax = axes[1]; o = al[al.scenario == 'outage']
pol = [('locf', 'Last value', C['naive']), ('sir_ml', 'Hybrid, hospital only', C['hosp']), ('capacity_proportional', 'Capacity-proportional', '#bbbab5'),
       ('ww_ratio', 'Wastewater ratio', C['ratio']), ('sir_ww', 'Hybrid, with wastewater', C['ww'])]
for i, (m, lab, col) in enumerate(pol):
    v = [o[(o.policy == m) & (o.reserve_frac == f)].pct_of_oracle_reduction.iloc[0] for f in [0.05, 0.1, 0.2]]
    ax.bar(x + (i - 2) * 0.16, v, width=0.15, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels(['5%', '10%', '20%']); ax.set_xlabel('Reserve (share of routine capacity)'); ax.set_ylim(0, 135); ax.set_yticks(range(0, 101, 20))
ax.set_title('Reporting outages (point forecasts)', loc='left'); ax.legend(fontsize=6, loc='upper center', ncol=2, columnspacing=0.8); tag(ax, 'b')
fig.tight_layout(); fig.savefig(f'{OUT}/fig8_allocation.png'); plt.close(fig)

# ---- Fig S2: sensitivity ------------------------------------------------------------------------------
fs = [f for f in os.listdir(R) if f.startswith('sensitivity_')]
if len(fs) >= 8:
    S = pd.concat([pd.read_csv(f'{R}/{f}') for f in fs])
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 5.4), gridspec_kw=dict(width_ratios=[0.9, 1, 1, 1]))
    for i, p in enumerate(['covid', 'flu']):
        d = S[(S.pathogen == p) & (S.analysis == 'gamma')]; ax = axes[i, 0]
        for model, c in [('with wastewater', C['ww']), ('hospital data only', C['hosp'])]:
            q = d[d.model == model]; gv = q.variant.str.replace('gamma=', '').astype(float)
            ax.plot(gv, q.h2, color=c, marker='o', ms=3.5, label=f'{model}, 2 wk'); ax.plot(gv, q.h4, color=c, marker='s', ms=3.5, ls=':', label=f'{model}, 4 wk')
        ax.axvline(GAMMA[p], color=INK2, lw=0.8, ls=':'); ax.set_xlabel('$\\gamma$ (per week)'); ax.set_ylim(0.5, 1.0); ax.set_title(f'{PN[p]}: recovery rate', loc='left', fontsize=8)
        ax.set_ylabel('Relative MAE vs persistence')
        for jj, an, ttl in [(1, 'hyper', 'hyperparameters'), (2, 'ablation', 'predictor groups')]:
            d = S[(S.pathogen == p) & (S.analysis == an)]; ax = axes[i, jj]; y = np.arange(len(d))
            ax.scatter(d.h2, y, color=C['ww'], s=14, label='2 weeks', zorder=3); ax.scatter(d.h4, y, color=C['ww'], marker='s', facecolor='white', s=14, label='4 weeks', zorder=3)
            ax.set_yticks(y); ax.set_yticklabels(d.variant, fontsize=6); ax.invert_yaxis(); ax.set_xlim(0.55, 1.05)
            ax.axvline(d.h2.iloc[0], color=INK2, lw=0.7, ls=':'); ax.set_title(f'{PN[p]}: {ttl}', loc='left', fontsize=8); ax.grid(axis='y', visible=False)
            if i == 1: ax.set_xlabel('Relative MAE')
        d = S[(S.pathogen == p) & (S.analysis == 'importance') & (S.model == 'h2')].sort_values('importance', ascending=False); ax = axes[i, 3]
        names = {'lb': 'log $\\beta_T$', 'lb_l1': 'log $\\beta_{T-1}$', 'lb_l2': 'log $\\beta_{T-2}$', 'lb_l3': 'log $\\beta_{T-3}$', 'lC': 'log admissions', 'sin': 'season (sin)',
                 'cos': 'season (cos)', 'ww': 'wastewater', 'ww_l1': 'wastewater (lag 1)', 'dww1': '$\\Delta$ wastewater (1 wk)', 'dww2': '$\\Delta$ wastewater (2 wk)'}
        ax.barh(np.arange(len(d)), 100 * d.rel_increase, xerr=100 * d.sd / (d.importance / d.rel_increase), color=C['ww'], error_kw=dict(lw=0.7, ecolor=INK2))
        ax.set_yticks(np.arange(len(d))); ax.set_yticklabels([names[v] for v in d.variant], fontsize=6); ax.invert_yaxis(); ax.grid(axis='y', visible=False)
        ax.set_title(f'{PN[p]}: importance (2 wk)', loc='left', fontsize=8)
        if i == 1: ax.set_xlabel('% increase in log-$\\beta$ MAE')
    h, l = axes[0, 0].get_legend_handles_labels(); h2, l2 = axes[0, 1].get_legend_handles_labels()
    fig.legend(h + h2, [x.replace('with wastewater', 'with ww').replace('hospital data only', 'hospital only') for x in l] + ['2 weeks ahead (hyperparameters, predictors)', '4 weeks ahead (hyperparameters, predictors)'], loc='lower center', ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f'{OUT}/figS2_sensitivity.png'); plt.close(fig)
print('figures written to', OUT)
