"""Surge-onset early warning from the 1-4 week forecasts (results/forecasts.csv).
A wave onset is the first week w in which admissions reach 1.5x the minimum of the previous 8 weeks
(and exceed it by >= 25 admissions), after at least 4 consecutive weeks without meeting that condition.
At origin T the alarm level is A_T = max(1.5 * min(adm[T-7..T]), min + 25). An alarm is raised at T if the
current week is below A_T and any 1-4 week forecast reaches A_T.
An onset is detected if an alarm was raised at an origin within the 4 weeks before it; lead time = weeks from
the first such alarm to the onset. An alarm is false if no onset follows within 4 weeks.
Usage: python early_warning.py us_state_week_panel.csv results"""
import sys, pandas as pd, numpy as np
P, R = sys.argv[1], sys.argv[2]
REL, ABS, BASEW, QUIET = 1.5, 25, 8, 4
panel = pd.read_csv(P, parse_dates=['week_end'])
F = pd.read_csv(f'{R}/forecasts.csv', parse_dates=['origin'])
rows, summ = [], []
for path in ['covid', 'flu']:
    for st in F.state.unique():
        y = panel[panel.state == st].set_index('week_end')[f'{path}_adm']
        base = y.rolling(BASEW, min_periods=BASEW).min()
        level = np.maximum(REL * base, base + ABS)                     # alarm level using weeks up to and including T
        cond = y >= level.shift(1)                                     # week w vs. baseline of weeks w-8..w-1
        test_weeks = y[(y.index > F.origin.min()) & (y.index <= F.origin.max() + pd.Timedelta(weeks=4))].index
        onsets = [w for w in test_weeks if cond[w] and not cond.loc[w - pd.Timedelta(weeks=QUIET):w - pd.Timedelta(weeks=1)].any()]
        f = F[(F.pathogen == path) & (F.state == st)]
        for m, g in f.groupby('model'):
            if m == 'naive': continue
            gw = g.pivot(index='origin', columns='h', values='pred'); now = g.groupby('origin').now.first()
            A = level.reindex(gw.index)
            alarm = (gw.max(axis=1) >= A) & (now < A)
            aw = alarm[alarm].index
            for o in onsets:
                win = [a for a in aw if o - pd.Timedelta(weeks=4) <= a < o]
                rows.append(dict(pathogen=path, state=st, model=m, onset=o.date(), detected=bool(win),
                                 lead_weeks=(o - min(win)).days / 7 if win else np.nan))
            false = sum(1 for a in aw if not any(a < o <= a + pd.Timedelta(weeks=4) for o in onsets))
            summ.append(dict(pathogen=path, state=st, model=m, n_alarms=len(aw), false_alarms=false))
E = pd.DataFrame(rows); S = pd.DataFrame(summ)
E.to_csv(f'{R}/early_warning_onsets.csv', index=False)
t = E.groupby(['pathogen', 'model']).agg(onsets=('detected', 'size'), detected=('detected', 'sum'),
                                         mean_lead_weeks=('lead_weeks', 'mean'))
t['hit_rate'] = t.detected / t.onsets
t = t.join(S.groupby(['pathogen', 'model'])[['n_alarms', 'false_alarms']].sum())
t['false_alarm_ratio'] = t.false_alarms / t.n_alarms
t = t[['onsets', 'detected', 'hit_rate', 'mean_lead_weeks', 'n_alarms', 'false_alarms', 'false_alarm_ratio']].round(3)
t.to_csv(f'{R}/early_warning_summary.csv')
print(t.to_string())
