"""Task 2 - the SMT baseline the paper is measured against.

dReal is the right comparison because SMT-based neural Lyapunov verification
(Chang, Roohi, Gao, NeurIPS 2019) is Prof. Cui's own reference [14], and her paper
dismisses it as tractable only for small systems, so beating it is the argument her
citation already sets up. We encode the SAME Lyapunov condition (4b) that CROWN
certifies and hand it to dReal:

  property   : for all x in region R, Lie(x) < 0
  we check   : is there an x in R with Lie(x) >= 0 ?  (the negation)
  dReal says : delta-unsat  -> no such x, property certified
               delta-sat    -> here is a counterexample

The Lie derivative is grad V . f(x, u_droop(x)), built symbolically with dReal's
sin, cos, exp. ELU is written with the same relu/exp identity used everywhere else
so the encoded network is bit-for-bit the trained one. V and its gradient are the
analytic forms from src/graphs.py, so this and the CROWN path certify the identical
function.

ENVIRONMENT: dReal ships no Windows wheel and its source build needs Bazel and
IBEX, so it does not run on the Windows host this repo was developed on (recorded
in NOTES.md, same class as the dead IBM CROWN repo). It runs on Colab and Linux,
where `pip install dreal` works. On a host without dReal this script writes a
result recording that, so the pipeline never fabricates a number it did not get.

Run on Colab: python experiments/dreal_baseline.py --rung 1
Writes results/dreal_seed{seed}_rung{rung}.json.
"""
import os, sys, json, time, argparse
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from data import load_system
from lyapunov import LyapunovOnState

try:
    from dreal import Variable, sin as dsin, cos as dcos, exp as dexp, Max as dMax, \
        logical_and, CheckSatisfiability, Config
    HAVE_DREAL = True
except Exception:
    HAVE_DREAL = False


def drelu(a):
    return dMax(a, 0.0)


def delu(a):
    # ELU(a) = relu(a) - relu(1 - exp(-relu(-a))), the identity used in torch too
    return drelu(a) - drelu(1.0 - dexp(-drelu(-a)))


def delu_prime(a):
    # d/da ELU(a) = exp(-relu(-a))
    return dexp(-drelu(-a))


def build_symbolic(system, V, xstar, free_idx, lo, hi):
    """Return (constraints, lie_expr, xvars) for the region with only free_idx
    varying. Everything is a dReal expression of the free variables."""
    n = system["n"]
    R = system["minus_ref"]                       # (2n, 2(n-1))
    W1 = V.net.dense1.weight.detach().numpy(); b1 = V.net.dense1.bias.detach().numpy()
    W2 = V.net.dense2.weight.detach().numpy()[0]  # (H,)
    om = float(system["omega_scale"])
    M = system["M"]; Pm = system["Pm"]; F = system["F"]; G = system["G"]
    coff = system["linear_coff"]; maxa = system["max_action"]

    xs = xstar.numpy().ravel()
    x = [float(xs[i]) for i in range(2 * n)]      # constants by default
    xvars = {}
    cons = []
    for k, i in enumerate(free_idx):
        v = Variable(f"x{i}")
        x[i] = v; xvars[i] = v
        cons += [v >= float(lo[k]), v <= float(hi[k])]

    # reduced z = x @ R  (list of 2(n-1) expressions)
    z = [sum(x[i] * float(R[i, j]) for i in range(2 * n)) for j in range(R.shape[1])]
    # a = z @ W1.T + b1  (H expressions)
    H = W1.shape[0]
    a = [sum(z[j] * float(W1[h, j]) for j in range(len(z))) + float(b1[h]) for h in range(H)]
    # V = sum_h W2_h * ELU(a_h) + b2   (b2 cancels in V - V*, not needed for Lie)
    # grad_z V = (W2 * ELU'(a)) @ W1   -> length 2(n-1)
    g = [float(W2[h]) * delu_prime(a[h]) for h in range(H)]
    grad_z = [sum(g[h] * float(W1[h, j]) for h in range(H)) for j in range(len(z))]
    # grad_x = grad_z @ R.T   -> length 2n
    grad_x = [sum(grad_z[j] * float(R[i, j]) for j in range(len(z))) for i in range(2 * n)]

    # dynamics xdot
    delta = [x[i] for i in range(n)]; omega = [x[n + i] for i in range(n)]
    u = [ (lambda val: dMax(-float(maxa[i]), dMax(val, -1e9)) if False else
           float(maxa[i]) - drelu(float(maxa[i]) - val) + drelu(-float(maxa[i]) - val))(float(coff[i]) * omega[i])
          for i in range(n) ]
    xdot = [om * omega[i] for i in range(n)]
    for i in range(n):
        sin_term = sum(float(F[i, j]) * dsin(delta[i] - delta[j]) for j in range(n))
        cos_term = sum(float(G[i, j]) * (dcos(delta[i] - delta[j]) - (1.0 if i == j else 0.0)) for j in range(n))
        xdot.append((float(Pm[i]) - u[i] - sin_term - cos_term) / float(M[i]))
    lie = sum(grad_x[i] * xdot[i] for i in range(2 * n))
    return cons, lie


def run(seed=0, rung=1, delta=1e-3, budget=600):
    out = dict(seed=seed, rung=rung, condition="4b", solver="dReal", delta=delta,
               time_budget_s=budget)
    if not HAVE_DREAL:
        out["result"] = "dreal_unavailable"
        out["note"] = ("dReal has no Windows wheel and its source build needs Bazel+IBEX; "
                       "run this on Colab or Linux. No number fabricated.")
        path = os.path.join(ROOT, "results", f"dreal_seed{seed}_rung{rung}.json")
        json.dump(out, open(path, "w"), indent=2); print(json.dumps(out, indent=2)); return out

    INNER = 0.05
    if rung == 1:
        from e3_rung1_twobus import build_twobus_system, equilibrium, GI2
        s = build_twobus_system(); xstar, _ = equilibrium(s)
        V = LyapunovOnState(s, hidden=32)
        V.load_state_dict(torch.load(os.path.join(ROOT, "results", f"lyap_twobus_seed{seed}.pt")))
        free = [GI2, GI2 + 2]; rho = 1.2
    else:
        s = load_system(); n = s["n"]
        xstar = torch.tensor(np.load(os.path.join(ROOT, "results", f"equilibrium_seed{seed}.npy")),
                             dtype=torch.float32).reshape(1, -1)
        V = LyapunovOnState(s)
        V.load_state_dict(torch.load(os.path.join(ROOT, "results", f"lyap_cegis_seed{seed}.pt")))
        free = [4, 4 + n]; rho = 1.5
    V.eval()
    xs = xstar.numpy().ravel()
    lo = [xs[i] + INNER for i in free]; hi = [xs[i] + INNER + rho for i in free]
    cons, lie = build_symbolic(s, V, xstar, free, lo, hi)

    cfg = Config(); cfg.precision = delta
    formula = logical_and(*cons, lie >= 0.0)      # negation of (4b) over the region
    t0 = time.time()
    res = CheckSatisfiability(formula, delta)
    dt = time.time() - t0
    out["seconds"] = round(dt, 2)
    if res is None:
        out["result"] = "certified"; out["note"] = "delta-unsat: no state with Lie>=0 in region"
    else:
        out["result"] = "counterexample"; out["note"] = "delta-sat"; out["model"] = str(res)
    if dt >= budget:
        out["result"] = "timeout"
    path = os.path.join(ROOT, "results", f"dreal_seed{seed}_rung{rung}.json")
    json.dump(out, open(path, "w"), indent=2); print(json.dumps(out, indent=2)); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--budget", type=int, default=600)
    a = ap.parse_args(); run(a.seed, a.rung, budget=a.budget)
