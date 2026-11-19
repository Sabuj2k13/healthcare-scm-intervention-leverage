"""Supplementary code (S2 add-on): intervention leverage, break-even ratios and
exact (budget-constrained) bundle enumeration for the DEMATEL risk network.

Usage:
    python leverage_enumeration.py                 # reproduces Section 5.6 results
    python leverage_enumeration.py costs.csv B     # budget-constrained optimum
    python leverage_enumeration.py --robustness    # submodularity + decision robustness (slow, ~15 min)
    python leverage_enumeration.py --validation    # Section 5.7: closed-form verification + cascade simulation (~1 min)
costs.csv: one line with 16 comma-separated costs (F1..F16), any common unit.
"""
import sys, itertools
import numpy as np

# Aggregated direct-relation matrix A (Table 5 of the manuscript).
A = np.array([
    [0.000, 0.514, 0.032, 0.010, 0.013, 0.651, 0.390, 0.622, 0.010, 0.028, 0.419, 0.010, 0.680, 0.390, 0.390, 0.017],
    [0.055, 0.000, 0.010, 0.680, 1.000, 0.680, 0.477, 0.680, 0.225, 0.680, 0.680, 0.390, 0.680, 1.000, 0.680, 0.028],
    [0.011, 0.015, 0.000, 0.028, 0.090, 0.225, 0.373, 0.028, 0.680, 0.390, 0.010, 0.680, 0.390, 0.225, 0.651, 0.390],
    [0.021, 0.028, 0.075, 0.000, 0.028, 0.390, 0.033, 0.390, 0.225, 0.225, 0.419, 0.030, 0.680, 0.390, 0.225, 0.010],
    [0.087, 0.390, 0.225, 0.680, 0.000, 0.680, 0.390, 0.419, 0.680, 0.651, 0.651, 0.680, 1.000, 0.680, 0.680, 0.680],
    [0.419, 0.183, 0.680, 0.390, 0.390, 0.000, 0.680, 0.028, 0.225, 0.622, 0.680, 0.622, 0.680, 0.680, 0.390, 0.390],
    [0.010, 0.225, 0.013, 0.390, 0.010, 0.028, 0.000, 0.047, 0.390, 0.419, 0.010, 0.225, 0.225, 0.390, 0.210, 0.107],
    [0.047, 0.075, 0.047, 0.028, 0.651, 0.419, 0.390, 0.000, 0.028, 1.000, 0.028, 0.028, 0.390, 0.680, 0.032, 0.047],
    [0.096, 0.535, 0.390, 0.680, 0.622, 0.419, 0.047, 0.075, 0.000, 0.075, 0.390, 1.000, 0.680, 0.075, 0.075, 0.390],
    [0.390, 0.373, 0.010, 0.680, 0.225, 1.000, 0.225, 0.419, 0.047, 0.000, 0.010, 0.028, 0.680, 0.680, 0.225, 0.225],
    [0.047, 0.025, 0.680, 0.419, 0.028, 0.651, 0.047, 0.390, 0.047, 0.028, 0.000, 0.032, 0.390, 0.028, 0.047, 0.047],
    [0.075, 0.390, 0.166, 0.535, 0.390, 0.651, 1.000, 0.680, 0.680, 0.075, 0.028, 0.000, 0.680, 0.680, 0.419, 0.680],
    [0.010, 0.010, 0.225, 0.477, 0.225, 0.010, 0.010, 0.010, 0.390, 0.028, 0.010, 0.680, 0.000, 0.680, 0.067, 0.010],
    [0.654, 1.000, 0.680, 0.390, 0.651, 0.680, 0.419, 0.680, 0.680, 0.390, 0.225, 0.390, 1.000, 0.000, 0.680, 0.448],
    [0.712, 0.680, 1.000, 0.680, 0.680, 0.390, 0.419, 0.680, 0.390, 0.680, 0.419, 0.448, 0.680, 0.390, 0.000, 0.126],
    [0.105, 0.651, 0.390, 0.622, 0.680, 0.210, 0.680, 0.225, 0.390, 0.225, 0.010, 0.448, 0.225, 0.390, 0.010, 0.000]
])
LAM = 0.5
N = A.shape[0]
X = A / A.sum(axis=1).max()          # Eq. (9)
I = np.eye(N)
T = X @ np.linalg.inv(I - X)          # Eq. (10)
TOTAL = T.sum()

def reduction(bundle, lam=LAM):
    """Exact % reduction in total influence when rows in `bundle` are scaled by (1 - lam), Eq. (13)."""
    d = np.ones(N); d[list(bundle)] = 1 - lam
    X2 = d[:, None] * X
    return 100 * (TOTAL - (X2 @ np.linalg.inv(I - X2)).sum()) / TOTAL

def closed_form(i, lam=LAM):
    """Eq. (14): lam * D_i (1 + R_i) / (1 + lam t_ii), as % of total influence."""
    D, R = T.sum(1), T.sum(0)
    return 100 * lam * D[i] * (1 + R[i]) / (1 + lam * T[i, i]) / TOTAL

def best_by_size():
    out = {}
    for k in range(1, N + 1):
        out[k] = max(itertools.combinations(range(N), k), key=reduction)
    return out

def budget_optimum(costs, budget):
    """Exact solution of max V(S) s.t. sum(costs[S]) <= budget over all 2^N bundles."""
    best, best_v = (), 0.0
    for k in range(1, N + 1):
        for S in itertools.combinations(range(N), k):
            if sum(costs[i] for i in S) <= budget:
                v = reduction(S)
                if v > best_v:
                    best, best_v = S, v
    return best, best_v

name = lambda S: "{" + ", ".join(f"F{i+1}" for i in sorted(S)) + "}"

def all_bundle_values(Xm=None, lam=LAM, chunk=4096):
    """V(S) for all 2^N bundles (index = bitmask), computed in batches."""
    Xm = X if Xm is None else Xm
    Tm = Xm @ np.linalg.inv(I - Xm); tot = Tm.sum()
    masks = np.arange(1 << N); bits = ((masks[:, None] >> np.arange(N)) & 1).astype(bool)
    f = np.empty(1 << N)
    for s in range(0, 1 << N, chunk):
        d = np.where(bits[s:s + chunk], 1 - lam, 1.0); X2 = d[:, :, None] * Xm[None]
        f[s:s + chunk] = 100 * (tot - (X2 @ np.linalg.inv(I[None] - X2)).sum((1, 2))) / tot
    return f

def submodularity_check(f):
    """Monotonicity and diminishing returns f(S+i)+f(S+j) >= f(S+i+j)+f(S) over the full lattice."""
    M = np.arange(1 << N)
    min_gain = min((f[M | (1 << i)] - f[M])[(M >> i) & 1 == 0].min() for i in range(N))
    checks, violations, min_slack = 0, 0, np.inf
    for i in range(N):
        for j in range(i + 1, N):
            S = M[((M >> i) & 1 == 0) & ((M >> j) & 1 == 0)]
            g = f[S | (1 << i)] + f[S | (1 << j)] - f[S | (1 << i) | (1 << j)] - f[S]
            checks += len(S); violations += int((g < -1e-9).sum()); min_slack = min(min_slack, g.min())
    return min_gain, checks, violations, min_slack

def decision_robustness(delta=0.2, runs=5000, kmax=5, seed=2024):
    """Re-solve the size-k bundle problem in perturbed worlds (Section 3.4 perturbation);
    returns, per k, the share of runs in which the baseline optimum stays optimal and the regret (pp)."""
    rng = np.random.default_rng(seed); off = ~np.eye(N, dtype=bool)
    combos = {k: np.array(list(itertools.combinations(range(N), k))) for k in range(1, kmax + 1)}
    def values(Am, k):
        Xm = Am / Am.sum(1).max(); Tm = Xm @ np.linalg.inv(I - Xm); tot = Tm.sum()
        d = np.ones((len(combos[k]), N)); np.put_along_axis(d, combos[k], 1 - LAM, axis=1)
        X2 = d[:, :, None] * Xm[None]
        return 100 * (tot - (X2 @ np.linalg.inv(I[None] - X2)).sum((1, 2))) / tot
    base = {k: int(np.argmax(values(A, k))) for k in combos}
    stay = {k: 0 for k in combos}; regret = {k: [] for k in combos}
    for _ in range(runs):
        Ap = np.where(off, np.minimum(1, A * rng.uniform(1 - delta, 1 + delta, (N, N))), A)
        for k in combos:
            v = values(Ap, k); o = int(np.argmax(v))
            stay[k] += (o == base[k]); regret[k].append(v[o] - v[base[k]])
    return {k: (100 * stay[k] / runs, float(np.mean(regret[k])), float(np.percentile(regret[k], 95)),
                float(np.max(regret[k]))) for k in combos}

def cascade_size(P, U):
    """Mean number of OTHER factors reached, over all 16 starting factors and all runs.
    P: N x N transmission probabilities; U: runs x N x N uniform draws (common random numbers)."""
    reach = (U < P[None]).astype(np.uint8) | np.eye(N, dtype=np.uint8)[None]
    for _ in range(5):                       # 2^5 > N: transitive closure by repeated squaring
        reach = (np.einsum('rij,rjk->rik', reach, reach) > 0).astype(np.uint8)
    return (reach.sum(2) - 1).mean()

def cascade_validation(kappas=(0.1, 0.2, 0.35, 0.5), runs=20000, seed=7):
    """Independent cascade model on the direct-relation judgments: p_ij = kappa * a_ij."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    lev = np.array([reduction([i]) for i in range(N)])
    D, R = T.sum(1), T.sum(0)
    causes = [i for i in range(N) if D[i] - R[i] > 0]
    best8 = max(itertools.combinations(range(N), 8), key=reduction)
    for kap in kappas:
        P = np.clip(kap * A, 0, 1); U = rng.random((runs, N, N)); base = cascade_size(P, U)
        def red(b):
            d = np.ones(N); d[list(b)] = 1 - LAM
            return 100 * (base - cascade_size(P * d[:, None], U)) / base
        cr = np.array([red([i]) for i in range(N)])
        print(f"kappa={kap}: mean cascade {base:.2f}; top F{np.argmax(cr)+1}; "
              f"rho(leverage)={spearmanr(cr, lev)[0]:.3f} rho(D-R)={spearmanr(cr, D-R)[0]:.3f} "
              f"rho(D+R)={spearmanr(cr, D+R)[0]:.3f}; F15 rank {int(np.argsort(-cr).tolist().index(14))+1}; best8 - causes = {red(best8) - red(causes):.1f} pp")

if __name__ == "__main__" and "--validation" in sys.argv:
    err = max(abs(reduction([i]) - closed_form(i)) for i in range(N))
    print(f"Closed-form verification: max abs error {err:.2e} pp")
    cascade_validation()
    sys.exit(0)

if __name__ == "__main__" and "--robustness" in sys.argv:
    f = all_bundle_values()
    print("Monotone/submodular check (min gain, checks, violations, min slack):", submodularity_check(f))
    for delta in (0.1, 0.2, 0.3):
        print(f"delta = {delta}")
        for k, (stay, mean_r, p95_r, max_r) in decision_robustness(delta).items():
            print(f"  k={k}: baseline optimal in {stay:.1f}% of runs; regret mean {mean_r:.3f}, "
                  f"p95 {p95_r:.3f}, max {max_r:.3f} pp")
    sys.exit(0)

if __name__ == "__main__":
    single = [reduction([i]) for i in range(N)]
    assert max(abs(single[i] - closed_form(i)) for i in range(N)) < 1e-9
    top = int(np.argmax(single))
    print("Single-factor leverage and break-even ratio vs top factor:")
    for i in np.argsort(single)[::-1]:
        print(f"  F{i+1:<3} dSigma = {single[i]:6.2f}%   tau = {single[i]/single[top]:.2f}")
    if len(sys.argv) == 3:
        costs = [float(c) for c in open(sys.argv[1]).read().strip().split(",")]
        S, v = budget_optimum(costs, float(sys.argv[2]))
        print(f"Budget {sys.argv[2]}: optimal bundle {name(S)} removes {v:.2f}%")
    else:
        for k, S in best_by_size().items():
            print(f"  best {k:2d}-factor bundle {name(S):<60} {reduction(S):6.2f}%")
