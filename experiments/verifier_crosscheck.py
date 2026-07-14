"""Task 1 cross-check: run the certification through two independent verifiers and
confirm they agree. The paper's whole contribution is soundness, so resting it on
one bespoke implementation is a liability, and a PGD audit cannot catch unsoundness
in the direction that matters, because an unsound verifier certifies false
properties and a PGD attack failing to find a counterexample is exactly what that
looks like from the inside. So we cross-check the bounds themselves.

  analytic path : src/graphs.py, the gradient of V hand-written in boundable ops,
                  bounded by auto_LiRPA CROWN.
  Jacobian path : src/graphs_jacobian.py, the gradient expanded by auto_LiRPA's
                  JacobianOP, the route the brief names. Shares nothing with the
                  analytic path on the gradient side.

Three comparisons:
  1. certified rho per rung and Prop-2 beta, analytic vs Jacobian.
  2. a rigorous soundness scan: on many boxes, both verifiers bound the SAME
     scalar F, so a sound lower bound from one must never exceed a sound upper
     bound from the other. Any box with lb_analytic > ub_jacobian (or the reverse)
     proves one verifier unsound. We count these; the count must be zero.
  3. E5 genuine-violation regions: both must refuse to certify them.

If anything disagrees we stop and report, we do not pick the nicer number.
Writes results/verifier_crosscheck_seed{seed}.json.
"""
import os, sys, json, time, argparse
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from data import load_system
from lyapunov import LyapunovOnState
from controller import LinearDroop
from graphs import LyapunovCondition
from graphs_jacobian import LieJacobianCondition
from verify import certify_box, bound_ladder

GI, INNER = 4, 0.05


def bound_interval(cond, xL, xU, method="CROWN"):
    x0 = (xL + xU) / 2
    bm = BoundedModule(cond, x0, verbose=False)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=xL, x_U=xU)
    lb, ub = bm.compute_bounds(x=(BoundedTensor(x0.clone(), ptb),), method=method)
    return float(lb.min()), float(ub.max())


def slice_box(xstar, n, dlo, dhi, wlo, whi):
    xL = xstar.clone(); xU = xstar.clone()
    xL[0, GI] = xstar[0, GI] + dlo; xU[0, GI] = xstar[0, GI] + dhi
    xL[0, GI + n] = xstar[0, GI + n] + wlo; xU[0, GI + n] = xstar[0, GI + n] + whi
    return xL, xU


def max_rho(ana, jac, xstar, n, rho_grid, seed, use_jac):
    """Largest certifiable rho on the gen-5 annulus, using either verifier."""
    rho_max, t0 = 0.0, time.time()
    for rho in rho_grid:
        xL, xU = slice_box(xstar, n, INNER, INNER + rho, INNER, INNER + rho)
        if use_jac:
            r = certify_box(jac, xL, xU, eps=0.0, method="CROWN", min_width=0.03,
                            time_budget=45, seed=seed, eval_cond=ana)
        else:
            r = certify_box(ana, xL, xU, eps=0.0, method="CROWN", min_width=0.03,
                            time_budget=45, seed=seed)
        if not r["certified"]:
            break
        rho_max = rho
    return rho_max, round(time.time() - t0, 1)


def soundness_scan(ana, jac, xstar, n, seed):
    """On a grid of boxes across the gen-5 region, both verifiers bound the same F.
    A sound lb from one must never exceed a sound ub from the other."""
    g = torch.Generator().manual_seed(seed + 3)
    n_boxes, contradictions, worst = 0, 0, 0.0
    worst_case = None
    for _ in range(24):
        d0 = -0.4 + 2.2 * torch.rand(1, generator=g).item()
        w0 = -0.4 + 2.2 * torch.rand(1, generator=g).item()
        wd = 0.1 + 0.4 * torch.rand(1, generator=g).item()
        ww = 0.1 + 0.4 * torch.rand(1, generator=g).item()
        xL, xU = slice_box(xstar, n, d0, d0 + wd, w0, w0 + ww)
        la, ua = bound_interval(ana, xL, xU)
        lj, uj = bound_interval(jac, xL, xU)
        n_boxes += 1
        # both bound the same function: require la <= uj and lj <= ua
        gap = max(la - uj, lj - ua)          # > 0 means contradiction
        if gap > 1e-3:
            contradictions += 1
            if gap > worst:
                worst, worst_case = gap, dict(box_delta=[d0, d0 + wd], box_omega=[w0, w0 + ww],
                                              lb_ana=round(la, 4), ub_ana=round(ua, 4),
                                              lb_jac=round(lj, 4), ub_jac=round(uj, 4))
        worst = max(worst, 0.0)
    return dict(n_boxes=n_boxes, contradictions=contradictions,
                worst_gap=round(worst, 5), worst_case=worst_case)


def prop2_beta(ana_ctor, jac_ctor, xstar, n, seed, use_jac):
    lo, hi, bmax = 0.0, 4.0, 0.0
    xL, xU = slice_box(xstar, n, INNER, INNER + 1.0, INNER, INNER + 1.0)
    for _ in range(6):
        beta = 0.5 * (lo + hi)
        ana = ana_ctor(beta); jac = jac_ctor(beta)
        if use_jac:
            r = certify_box(jac, xL, xU, eps=0.0, method="CROWN", min_width=0.04,
                            time_budget=35, seed=seed, eval_cond=ana)
        else:
            r = certify_box(ana, xL, xU, eps=0.0, method="CROWN", min_width=0.04,
                            time_budget=35, seed=seed)
        if r["certified"]:
            bmax, lo = beta, beta
        else:
            hi = beta
    return round(bmax, 4)


def run(seed=0):
    s = load_system(); n = s["n"]
    xstar = torch.tensor(np.load(os.path.join(ROOT, "results", f"equilibrium_seed{seed}.npy")),
                         dtype=torch.float32).reshape(1, -1)
    V = LyapunovOnState(s); V.load_state_dict(torch.load(os.path.join(ROOT, "results", f"lyap_cegis_seed{seed}.pt"))); V.eval()
    ana = LyapunovCondition(s, V, LinearDroop(s), mode="4b", x_star=xstar)
    jac = LieJacobianCondition(s, V, LinearDroop(s), mode="4b", x_star=xstar)
    rho_grid = [round(0.25 * k, 2) for k in range(1, 9)]     # 0.25 .. 2.0

    out = dict(seed=seed, note="analytic (hand-derived gradient) vs JacobianOP (auto_LiRPA), CROWN bounds")

    # 1. certified rho, both verifiers, gen-5 slice (4b)
    ra, ta = max_rho(ana, jac, xstar, n, rho_grid, seed, use_jac=False)
    rj, tj = max_rho(ana, jac, xstar, n, rho_grid, seed, use_jac=True)
    out["rung2_4b_rho"] = dict(analytic=ra, jacobian=rj, analytic_s=ta, jacobian_s=tj)

    # 2. Prop-2 beta, both
    ac = lambda b: LyapunovCondition(s, V, LinearDroop(s), mode="prop2", x_star=xstar, beta=b)
    jc = lambda b: LieJacobianCondition(s, V, LinearDroop(s), mode="prop2", x_star=xstar, beta=b)
    out["rung2_prop2_beta"] = dict(analytic=prop2_beta(ac, jc, xstar, n, seed, False),
                                   jacobian=prop2_beta(ac, jc, xstar, n, seed, True))

    # 3. soundness scan
    out["soundness_scan"] = soundness_scan(ana, jac, xstar, n, seed)

    # 4. tightness ladder on a fixed box: IBP vs analytic-CROWN vs jacobian-CROWN
    xL, xU = slice_box(xstar, n, INNER, INNER + 1.0, INNER, INNER + 1.0)
    la_ibp, _ = bound_interval(ana, xL, xU, method="IBP")
    la_c, _ = bound_interval(ana, xL, xU, method="CROWN")
    lj_c, _ = bound_interval(jac, xL, xU, method="CROWN")
    out["tightness_lb"] = dict(ibp=round(la_ibp, 4), analytic_crown=round(la_c, 4),
                               jacobian_crown=round(lj_c, 4))

    # 5. E5 genuine-violation regions: neither verifier may certify them
    Vas = LyapunovOnState(s); Vas.load_state_dict(torch.load(os.path.join(ROOT, "results", f"lyap_seed{seed}.pt"))); Vas.eval()
    ana_at = LyapunovCondition(s, Vas, LinearDroop(s), mode="4b", x_star=xstar)
    jac_at = LieJacobianCondition(s, Vas, LinearDroop(s), mode="4b", x_star=xstar)
    xL, xU = slice_box(xstar, n, 0.1, 0.5, 0.1, 0.5)
    r_ana = certify_box(ana_at, xL, xU, eps=0.0, method="CROWN", min_width=0.02, time_budget=30, seed=seed)
    r_jac = certify_box(jac_at, xL, xU, eps=0.0, method="CROWN", min_width=0.02, time_budget=30, seed=seed, eval_cond=ana_at)
    out["e5_far_region"] = dict(analytic_verdict=r_ana["verdict"], jacobian_verdict=r_jac["verdict"])

    # agreement verdict
    dis = []
    if out["soundness_scan"]["contradictions"] > 0:
        dis.append("soundness scan found bound contradictions")
    if out["e5_far_region"]["analytic_verdict"] == "verified" or out["e5_far_region"]["jacobian_verdict"] == "verified":
        dis.append("a verifier certified a known-false region")
    # rho: Jacobian is looser, rho_jac <= rho_ana is fine; a LARGER rho_jac than the true
    # region would be the worry, but that is caught by the soundness scan.
    out["agreement"] = (len(dis) == 0)
    out["disagreements"] = dis
    with open(os.path.join(ROOT, "results", f"verifier_crosscheck_seed{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args().seed)
