"""Synthetic validation of the theoretical results (Section 4 of the paper).
Usage: python synthetic_validation.py <results_dir> <panel_csv>"""
import sys, os, numpy as np, pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
R, PANEL = sys.argv[1], sys.argv[2]
G = 1.0; W = 156
phi = lambda r: np.expm1(r) / r if abs(r) > 1e-12 else 1.0
def psi(r):
    if abs(r) < 1e-8: return 0.5
    return 1 / (1 - np.exp(-r)) - 1 / r          # phi'/phi in closed form
a_fun = lambda b, g: b / (1 + b * psi(b - g))

def exact_recon(C, I0, g):
    I = I0; out = []; Is = []
    for c in C:
        b = brentq(lambda x: x * I * phi(x - g) - c, 1e-9, 60); out.append(b); Is.append(I); I = I * np.exp(b - g)
    return np.array(out), np.array(Is)
def euler_recon(C, I0, g):
    I = I0; out = []
    for c in C:
        b = c / I; out.append(b); I = I + c - g * I
        I = max(I, 1e-9)
    return np.array(out)

def simulate(beta_fn, I0=1000.0, g=G):
    def rhs(t, y): b = beta_fn(t); return [(b - g) * y[0], b * y[0]]
    sol = solve_ivp(rhs, (0, W), [I0, 0], t_eval=np.arange(W + 1), rtol=1e-10, atol=1e-10, max_step=0.05)
    return np.diff(sol.y[1])

rng = np.random.default_rng(1)
smooth = lambda t: G * (1 + 0.45 * np.sin(2 * np.pi * t / 30) + 0.25 * np.sin(2 * np.pi * t / 11 + 1))
pw_vals = smooth(np.arange(W) + 0.5)
pw = lambda t: pw_vals[min(int(t), W - 1)]
avg = np.array([np.mean(smooth(np.linspace(w, w + 1, 201))) for w in range(W)])
rows = []
for name, fn, truth in [('piecewise-constant beta', pw, pw_vals), ('smooth beta', smooth, avg)]:
    C = simulate(fn)
    be, _ = exact_recon(C, 1000.0, G); bu = euler_recon(C, 1000.0, G)
    rows.append(dict(case=name, method='exact (this paper)', max_abs_err=np.abs(be - truth).max(), rmse=np.sqrt(np.mean((be - truth) ** 2))))
    rows.append(dict(case=name, method='forward Euler', max_abs_err=np.abs(bu - truth).max(), rmse=np.sqrt(np.mean((bu - truth) ** 2))))
T1 = pd.DataFrame(rows)

# noise stability + forgetting of initial condition (piecewise-constant data)
C = simulate(pw); b_true, I_true = exact_recon(C, 1000.0, G)
b_w, I_w = exact_recon(C, 5000.0, G)                                    # wrong initial condition (x5)
eps = 0.10; noisy = []
for rep in range(200):
    e = rng.uniform(-eps, eps, W); noisy.append((e,) + exact_recon(C * np.exp(e), 1000.0, G))
# Theorem 2 needs K to contain every rate involved (true, wrongly initialised and noisy reconstructions)
K = (min(b_true.min(), b_w.min(), min(x[1].min() for x in noisy)), max(b_true.max(), b_w.max(), max(x[1].max() for x in noisy)))
bb = np.linspace(*K, 4000); aa = np.array([a_fun(b, G) for b in bb]); kappa = np.abs(1 - aa).max(); A = aa.max()
D0 = np.log(5); dlog = np.abs(np.log(I_w) - np.log(I_true))
ok_init = np.all(dlog <= kappa ** np.arange(W) * D0 + 1e-9)
worst = []; ok_noise = True
for e, b_n, I_n in noisy:
    D = np.abs(np.log(I_n) - np.log(I_true)); bound_I = A * eps / (1 - kappa)
    berr = np.abs(b_n - b_true); bound_b = A * (eps + bound_I)
    ok_noise &= (D.max() <= bound_I + 1e-9) and (berr.max() <= bound_b + 1e-9)
    worst.append((D.max(), berr.max()))
worst = np.array(worst)
T2 = dict(beta_range=K, kappa=kappa, A=A, init_halflife_weeks=np.log(0.5) / np.log(kappa), init_bound_holds=bool(ok_init),
          init_err_week4=dlog[4], init_err_week10=dlog[10], noise_eps=eps, bound_logI=A * eps / (1 - kappa),
          worst_logI=worst[:, 0].max(), bound_beta=A * (eps + A * eps / (1 - kappa)), worst_beta=worst[:, 1].max(),
          rms_beta_noise=None, noise_bound_holds=bool(ok_noise))

# kappa on the real reconstructions
rb = pd.read_csv(f'{R}/reconstructed_beta.csv')
real = []
for p, g in [('covid', 1.0), ('flu', 1.5)]:
    b = rb[rb.pathogen == p].beta
    lo, hi = b.quantile(0.01), b.quantile(0.99)
    xs = np.linspace(lo, hi, 2000); av = np.array([a_fun(x, g) for x in xs])
    k = np.abs(1 - av).max()
    real.append(dict(pathogen=p, gamma=g, beta_p01=lo, beta_p99=hi, kappa=k, A=av.max(), halflife_weeks=np.log(0.5) / np.log(k),
                     weeks_to_1pct=np.log(0.01) / np.log(k)))
T3 = pd.DataFrame(real)

pd.set_option('display.width', 200)
print(T1.round(5).to_string(index=False)); print(pd.Series(T2)); print(T3.round(3).to_string(index=False))
T1.to_csv(f'{R}/synthetic_reconstruction.csv', index=False)
pd.Series(T2).to_csv(f'{R}/synthetic_stability.csv'); T3.to_csv(f'{R}/real_kappa.csv', index=False)
