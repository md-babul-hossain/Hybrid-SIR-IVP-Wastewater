"""Collect archived (real-time) data vintages of weekly hospital admissions for the ten study states.

COVID-19: every committed version of target-data/covid-hospital-admissions.csv in the git history of
          https://github.com/CDCgov/covid19-forecast-hub (commit date = as_of date).
Influenza: target-data/time-series.csv of https://github.com/cdcepi/FluSight-forecast-hub (as_of column).

Usage: python build_vintages.py <covid19-forecast-hub clone (full history)> <FluSight-forecast-hub clone> vintages_10states.csv
"""
import sys, subprocess, io, pandas as pd
CH, FH, OUT = sys.argv[1:4]
FIPS = {'06': 'CA', '48': 'TX', '12': 'FL', '36': 'NY', '42': 'PA', '17': 'IL', '39': 'OH', '13': 'GA', '26': 'MI', '53': 'WA'}
PATH = 'target-data/covid-hospital-admissions.csv'
log = subprocess.run(['git', '-C', CH, 'log', '--format=%h %ad', '--date=short', '--', PATH],
                     capture_output=True, text=True, check=True).stdout.split('\n')
rows, seen = [], set()
for line in log:                                   # newest first: keep the last commit of each day
    if not line.strip(): continue
    h, day = line.split()
    if day in seen: continue
    r = subprocess.run(['git', '-C', CH, 'show', f'{h}:{PATH}'], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip(): continue      # commit that removed or moved the file
    seen.add(day)
    d = pd.read_csv(io.StringIO(r.stdout), dtype={'location': str})
    dc = 'date' if 'date' in d else 'target_end_date'
    d = d[d.location.isin(FIPS)]
    rows.append(pd.DataFrame(dict(as_of=day, week_end=d[dc], state=d.location.map(FIPS), value=d.value, pathogen='covid')))
F = pd.read_csv(f'{FH}/target-data/time-series.csv', dtype={'location': str})
F = F[(F.target == 'wk inc flu hosp') & F.location.isin(FIPS)]
rows.append(pd.DataFrame(dict(as_of=F.as_of, week_end=F.target_end_date, state=F.location.map(FIPS),
                              value=F.observation, pathogen='flu')))
V = pd.concat(rows).drop_duplicates(['pathogen', 'as_of', 'state', 'week_end'], keep='last')
V.to_csv(OUT, index=False)
print(V.groupby('pathogen').as_of.nunique())
