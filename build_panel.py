"""Build a state-week panel: NHSN hospital data + CDC NWSS wastewater.
Usage: python build_panel.py <input_dir> <output_csv>
Inputs (CDC data.cdc.gov downloads):
  Weekly_Hospital_Respiratory_Data__HRD__Metrics_by_Jurisdiction__*.csv
  CDC_Wastewater_Viral_Activity_Level_for_SARS-CoV-2__Influenza_A_and_RSV.csv
  export.csv                          (CDC Wastewater Data for SARS-CoV-2)
  CDC_Wastewater_Data_for_RSV.csv
"""
import sys, glob, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
IN, OUT = sys.argv[1], sys.argv[2]
START = '2022-01-01'
num = lambda s: pd.to_numeric(s.astype(str).str.replace(',', '').str.strip(), errors='coerce')
to_sat = lambda d: d + pd.to_timedelta((5 - d.dt.weekday) % 7, unit='D')   # week ending Saturday (CDC MMWR)

STATES = {'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO','Connecticut':'CT',
'Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN',
'Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI',
'Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
'New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA',
'Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'}
CODES = set(STATES.values())

# ---------- 1. Hospital data (NHSN HRD) ----------
hcols = {'Week Ending Date':'week_end','Geographic aggregation':'state',
 'Total COVID-19 Admissions':'covid_adm','Total Influenza Admissions':'flu_adm','Total RSV Admissions':'rsv_adm',
 'Total Patients Hospitalized with COVID-19':'covid_occ','Total Patients Hospitalized with Influenza':'flu_occ',
 'Total Patients Hospitalized with RSV':'rsv_occ','Number of Inpatient Beds':'inpatient_beds',
 'Number of Inpatient Beds Occupied':'inpatient_beds_occ','Number of ICU Beds':'icu_beds','Number of ICU Beds Occupied':'icu_beds_occ',
 'Number Hospitals Reporting Number of Inpatient Beds':'n_hosp_reporting'}
hf = glob.glob(os.path.join(IN, 'Weekly_Hospital_Respiratory_Data*.csv'))[0]
h = pd.read_csv(hf, usecols=list(hcols), dtype=str).rename(columns=hcols)
h = h[h.state.isin(CODES)].copy()
h['week_end'] = pd.to_datetime(h.week_end)
for c in hcols.values():
    if c not in ('week_end','state'): h[c] = num(h[c])
# reporting completeness relative to the state's typical number of reporting hospitals (Jan-Apr 2024 mandatory period)
ref = h[(h.week_end >= '2024-01-01') & (h.week_end < '2024-05-01')].groupby('state').n_hosp_reporting.median()
h['reporting_frac'] = (h.n_hosp_reporting / h.state.map(ref)).round(3)
h['voluntary_period'] = ((h.week_end >= '2024-05-11') & (h.week_end <= '2024-11-02')).astype(int)
h['low_reporting'] = ((h.reporting_frac < 0.8) | h.voluntary_period.eq(1)).astype(int)

# ---------- 2. WVAL (standardised wastewater activity, 3 pathogens) ----------
v = pd.read_csv(os.path.join(IN, 'CDC_Wastewater_Viral_Activity_Level_for_SARS-CoV-2__Influenza_A_and_RSV.csv'), dtype=str)
v['state'] = v['State/Territory'].map(STATES); v = v.dropna(subset=['state'])
v['week_end'] = to_sat(pd.to_datetime(v.Week_End)); v['wval'] = num(v.Site_WVAL)
v['pop'] = num(v.Population_Served)
v['p'] = v.Pathogen_Target.map({'SARS-CoV-2':'covid','Influenza A virus':'flu','RSV':'rsv'})
v = v.dropna(subset=['wval','p']).drop_duplicates(['Site','week_end','p'])
def wagg(g):
    w = g['pop'].fillna(g['pop'].median()).clip(lower=1)
    return pd.Series({'wval_median': g.wval.median(), 'wval_popwt': np.average(g.wval, weights=w),
                      'n_sites': g.Site.nunique(), 'pop_served': g['pop'].sum()})
va = v.groupby(['state','week_end','p']).apply(wagg).reset_index()
va = va.pivot(index=['state','week_end'], columns='p').sort_index(axis=1)
va.columns = [f'ww_{p}_{m}' for m, p in va.columns]
va = va.reset_index()

# ---------- 3. Raw concentrations (SARS-CoV-2, RSV): copies/L only, per-site normalised ----------
def raw(fname, tag):
    cols = ['site','state_territory','sample_collect_date','pcr_target_avg_conc_lin','pcr_target_units','population_served','sample_id']
    w = pd.read_csv(os.path.join(IN, fname), usecols=cols, dtype=str)
    w = w[w.pcr_target_units.str.strip().str.lower() == 'copies/l wastewater'].copy()   # drop sludge & log10 units
    w['state'] = w.state_territory.str.upper(); w = w[w.state.isin(CODES)]
    w['d'] = pd.to_datetime(w.sample_collect_date, errors='coerce'); w = w[w.d >= START]
    w['c'] = num(w.pcr_target_avg_conc_lin); w['pop'] = num(w.population_served)
    w = w.dropna(subset=['c','d']); w = w[w.c >= 0]
    w['week_end'] = to_sat(w.d)
    w['logc'] = np.log10(w.c + 1)
    # one value per site-week (mean of replicates/samples)
    sw = w.groupby(['state','site','week_end']).agg(logc=('logc','mean'), pop=('pop','max')).reset_index()
    # per-site normalisation (past-only, no look-ahead) -> comparable across sites/labs
    sw = sw.sort_values(['site', 'week_end'])
    # past-only normalisation: expanding median of the site's own history up to and including the current week
    sw['rel'] = sw.logc - sw.groupby('site').logc.transform(lambda x: x.expanding(min_periods=4).median())
    sw = sw.dropna(subset=['rel'])
    sw['pop'] = sw['pop'].fillna(sw['pop'].median()).clip(lower=1)
    out = sw.groupby(['state','week_end']).apply(lambda g: pd.Series({
        f'ww_{tag}_log10conc_rel': np.average(g.rel, weights=g['pop']),
        f'ww_{tag}_raw_n_sites': g.site.nunique()})).reset_index()
    return out
rc = raw('export.csv', 'covid')
rr = raw('CDC_Wastewater_Data_for_RSV.csv', 'rsv')

# ---------- 4. Merge ----------
p = h[h.week_end >= START].merge(va, on=['state','week_end'], how='left') \
      .merge(rc, on=['state','week_end'], how='left').merge(rr, on=['state','week_end'], how='left')
last = p.week_end.max()
p['preliminary'] = (p.week_end > last - pd.Timedelta(weeks=2)).astype(int)
p['use_in_model'] = ((p.low_reporting == 0) & (p.preliminary == 0)).astype(int)
p = p.sort_values(['state','week_end']).reset_index(drop=True)
order = ['state','week_end','covid_adm','flu_adm','rsv_adm','covid_occ','flu_occ','rsv_occ',
         'inpatient_beds','inpatient_beds_occ','icu_beds','icu_beds_occ','n_hosp_reporting','reporting_frac']
order += sorted([c for c in p.columns if c.startswith('ww_')])
order += ['voluntary_period','low_reporting','preliminary','use_in_model']
p = p[order]
for c in p.columns:
    if c.startswith('ww_') and ('wval' in c or 'log10' in c): p[c] = p[c].round(4)
p['week_end'] = p.week_end.dt.date
p.to_csv(OUT, index=False)
print(f'wrote {OUT}: {len(p):,} rows, {p.state.nunique()} states, {p.week_end.min()} to {p.week_end.max()}')
