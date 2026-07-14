"""E2 - certify condition (4a), positive definiteness of V.

She named (4a) in her email, so we certify it: V(x) > V(x*) for x away from the
equilibrium. This never touches the swing dynamics, it is a property of V alone,
so if any condition were going to certify easily on the as-trained network this
is the one. It does not, and that is the finding.

On the as-trained network CROWN finds a genuine counterexample to (4a) on the
gen-5 annulus, a state with V < V*, because her sampling-based training reported
V > V* on 100 percent of its training samples while a dense or directed check
near the equilibrium shows V is not actually positive definite there. That is the
same sampling gap the whole project is about, showing up in the condition that
was supposed to be trivial. The CEGIS-hardened network from E4 certifies (4a) on
the same annulus with a nonnegative margin, so closing the loop fixes (4a) as a
by-product of fixing (4b).

We also report the bound tightness ladder IBP vs CROWN vs CROWN-Optimized on a
fixed box, so the paper can quantify what branch-and-bound-grade bounds buy over
interval propagation. The annulus starts at an inner offset INNER because
V(x*) - V* = 0 at the equilibrium itself, the same equilibrium handling as (4b).

Writes results/e2_seed{seed}.json.
"""
import os, sys, json, argparse
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import load_system
from lyapunov import LyapunovOnState
from controller import LinearDroop
from graphs import LyapunovCondition
from verify import certify_box, audit_verified, bound_ladder

GI = 4          # gen 5, 0-indexed
INNER = 0.05    # inner corner offset from equilibrium


def slice_box(xstar, n, dlo, dhi, wlo, whi):
    xL = xstar.clone(); xU = xstar.clone()
    xL[0, GI] = xstar[0, GI] + dlo; xU[0, GI] = xstar[0, GI] + dhi
    xL[0, GI + n] = xstar[0, GI + n] + wlo; xU[0, GI + n] = xstar[0, GI + n] + whi
    return xL, xU


def certify_at(cond, xstar, n, rho, seed):
    xL, xU = slice_box(xstar, n, INNER, INNER + rho, INNER, INNER + rho)
    r = certify_box(cond, xL, xU, eps=0.0, method="CROWN",
                    min_width=0.03, time_budget=40, seed=seed)
    audit = audit_verified(cond, xL, xU, eps=0.0, seed=seed + 7)
    return r, float(audit)


def max_certifiable_rho(cond, xstar, n, rho_grid, seed):
    rho_max, detail = 0.0, None
    for rho in rho_grid:
        r, audit = certify_at(cond, xstar, n, rho, seed)
        if not r["certified"] or audit < -1e-3:
            break
        rho_max = rho
        detail = dict(rho=rho, certified_lower_bound_F=r.get("certified_lower_bound_F"),
                      audit_min_F=round(audit, 4), subdomains=r.get("subdomains"))
    return rho_max, detail


def dense_min_F(cond, xstar, n, rho, k=301):
    g = torch.linspace(INNER, INNER + rho, k)
    D, W = torch.meshgrid(g, g, indexing="ij")
    X = xstar.repeat(D.numel(), 1).clone()
    X[:, GI] = xstar[0, GI] + D.reshape(-1)
    X[:, GI + n] = xstar[0, GI + n] + W.reshape(-1)
    with torch.no_grad():
        F = cond(X).squeeze(-1)
    return float(F.min()), float((F < 0).float().mean())


def eval_net(s, xstar, n, ckpt, seed, rho_grid):
    V = LyapunovOnState(s)
    V.load_state_dict(torch.load(ckpt)); V.eval()
    cond = LyapunovCondition(s, V, LinearDroop(s), mode="4a", x_star=xstar)
    rho_max, detail = max_certifiable_rho(cond, xstar, n, rho_grid, seed)
    # verdict on the full 1.5 annulus, the region E4 certifies (4b) on
    r_full, audit_full = certify_at(cond, xstar, n, 1.5, seed)
    dmin, dfrac = dense_min_F(cond, xstar, n, 1.5)
    return dict(certified_rho=rho_max, certified_detail=detail,
                full_annulus_verdict=r_full["verdict"],
                full_annulus_certified_lb=r_full.get("certified_lower_bound_F"),
                full_annulus_F_at_ce=r_full.get("F_at_ce"),
                full_annulus_subdomains=r_full["subdomains"],
                dense_min_F=round(dmin, 5), dense_frac_below_0=round(dfrac, 5)), cond


def run(seed=0):
    s = load_system(); n = s["n"]
    xstar = torch.tensor(np.load(os.path.join(ROOT, "results", f"equilibrium_seed{seed}.npy")),
                         dtype=torch.float32).reshape(1, -1)
    rho_grid = [round(0.25 * k, 2) for k in range(1, 7)]      # 0.25 .. 1.5

    as_trained, cond_at = eval_net(s, xstar, n,
                                   os.path.join(ROOT, "results", f"lyap_seed{seed}.pt"),
                                   seed, rho_grid)

    cegis_path = os.path.join(ROOT, "results", f"lyap_cegis_seed{seed}.pt")
    cegis = eval_net(s, xstar, n, cegis_path, seed, rho_grid)[0] if os.path.exists(cegis_path) else None

    # tightness ladder on a fixed gen-5 slice box (as-trained network)
    xL, xU = slice_box(xstar, n, INNER, INNER + 1.0, INNER, INNER + 1.0)
    ladder = bound_ladder(cond_at, xL, xU)
    ibp = ladder["IBP"]["lower_bound"]; cro = ladder["CROWN-Optimized"]["lower_bound"]
    tighten = (abs(ibp) / abs(cro)) if cro != 0 else float("inf")

    out = dict(seed=seed, condition="4a",
               region="gen-5 slice annulus, inner offset %.2f" % INNER,
               as_trained=as_trained, cegis=cegis,
               bound_ladder_gen5_box=ladder,
               crown_opt_over_ibp_tightness=round(tighten, 3))
    with open(os.path.join(ROOT, "results", f"e2_seed{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args().seed)
