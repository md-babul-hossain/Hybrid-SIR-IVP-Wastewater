"""Empirical check of Theorem 3 and Corollary 2 (error propagation) on the real forecasts.

For every test forecast of the hybrid SIR-IVP with wastewater (same models, origins and retraining as the main
analysis) we compute the predicted and the reconstructed (true) weekly rates beta_{T+1..T+k}, the rate errors
e_j = betahat_j - beta_j, and the log admission error. Theorem 3 gives the exact identity
    log Chat_{T+k} - log C_{T+k} = sum_{j<k} e_j + lambda_k e_k,   1/b2 < lambda_k < 1/b1 + 1,
with b1, b2 the smaller and larger of betahat_k and beta_k. We verify the identity and the bounds for every
forecast and report how much of the log error comes from the accumulated term sum_{j<k} e_j.

Usage: python verify_error_propagation.py us_state_week_panel.csv results
"""
import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
PANEL, OUT = sys.argv[1:3]
sys.argv = [sys.argv[0], PANEL, OUT]
import hybrid_sir_ivp_experiment as E

def g_log(b, gam):
    r = b - gam
    return np.log(b * (np.expm1(r) / r if abs(r) > 1e-12 else 1.0))

rows = []
for path in ['covid', 'flu']:
    gam = E.GAMMA[path]; D = E.build(path)
    origins = sorted(D.loc[D.week_end >= E.TEST_START, 'week_end'].unique()); models = {}
    for i, T in enumerate(origins):
        if i % E.RETRAIN_EVERY == 0:
            for k in range(1, E.H + 1):
                tr = D[D.week_end + pd.Timedelta(weeks=k) <= T].dropna(subset=[f'y_lb_{k}', f'y_C_{k}'])
                models[k] = E.gbm().fit(tr[E.F_WW], tr[f'y_lb_{k}'])
        X = D[D.week_end == T].dropna(subset=[f'y_lb_{E.H}'])
        if X.empty: continue
        bhat = np.column_stack([np.exp(models[k].predict(X[E.F_WW].astype(float))) for k in range(1, E.H + 1)])
        btrue = np.exp(X[[f'y_lb_{k}' for k in range(1, E.H + 1)]].values.astype(float))
        for j, (_, r) in enumerate(X.iterrows()):
            e = bhat[j] - btrue[j]
            for k in range(1, E.H + 1):
                cum = e[:k - 1].sum()
                lhat = np.log(r.I_next) + (bhat[j, :k - 1] - gam).sum() + g_log(bhat[j, k - 1], gam)
                ltrue = np.log(r.I_next) + (btrue[j, :k - 1] - gam).sum() + g_log(btrue[j, k - 1], gam)
                err = lhat - ltrue; last = err - cum
                lam = last / e[k - 1] if abs(e[k - 1]) > 1e-12 else np.nan
                b1, b2 = min(bhat[j, k - 1], btrue[j, k - 1]), max(bhat[j, k - 1], btrue[j, k - 1])
                obs_err = np.log(r[f'y_C_{k}']) - ltrue                      # reconstruction reproduces data
                rows.append(dict(pathogen=path, state=r.state, origin=T, h=k, log_err=err, cum=cum, last=last,
                                 lam=lam, lam_ok=(np.isnan(lam) or (1 / b2 - 1e-9 <= lam <= 1 / b1 + 1 + 1e-9)),
                                 bound=np.abs(e[:k - 1]).sum() + (1 / b1 + 1) * abs(e[k - 1]),
                                 abs_e_k=abs(e[k - 1]), abs_dlogb_k=abs(np.log(bhat[j, k - 1]) - np.log(btrue[j, k - 1])),
                                 recon_check=abs(obs_err)))
    print(path, 'done', flush=True)
V = pd.DataFrame(rows); V.to_csv(f'{OUT}/error_propagation_check.csv', index=False)
S = V.groupby(['pathogen', 'h'])[[c for c in V.columns if c not in ('pathogen', 'h')]].apply(lambda x: pd.Series(dict(
    n=len(x), mean_abs_log_err=x.log_err.abs().mean(), mean_bound=x.bound.mean(),
    bound_holds=(x.log_err.abs() <= x.bound + 1e-9).mean(), lambda_in_range=x.lam_ok.mean(),
    share_cumulative=(x.cum.abs() / (x.cum.abs() + x.last.abs())).mean(),
    mae_log_beta=x.abs_dlogb_k.mean(), max_recon_err=x.recon_check.max())))
S.round(4).to_csv(f'{OUT}/error_propagation_summary.csv')
pd.set_option('display.width', 200); print(S.round(4))
