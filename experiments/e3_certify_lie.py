"""E3 - certify the Lie-derivative condition (4b) and its Proposition-2
exponential strengthening, on a staircase of increasingly hard regions.

This is where the nonlinearity lives, because the Lie derivative
  Lie(x) = grad_x V(x) . f(x, u_droop(x))
runs the sin and cos swing dynamics through the gradient of the ELU network, so
unlike (4a) it cannot be certified without bounding the closed-loop dynamics.

The staircase, each rung reported honestly whether it wins or loses:

  Rung 2, gen-5 projected slice. Exactly her Fig. 2 geometry, every coordinate
    pinned at equilibrium except (delta_5, omega_5). On the far region her
    as-trained network has a genuine (4b) counterexample with Lie > 0, which no
    amount of sampling would have certified away. The CEGIS network certifies
    (4b) on the near annulus out to rho, and we bisect the largest exponential
    rate beta for which Proposition 2, Lie <= -beta (V - V*), certifies.

  Rung 3, small box in the full 20-D state. Certify (4b) with an explicit
    tolerance eps on x* +/- r across every coordinate, push r outward until it
    stops certifying, and report the boundary radius and the subdomain count,
    because branch-and-bound cost is what limits the full-state result.

  Rung 4, a larger full 20-D box. Attempt it under a fixed time budget and report
    the verdict, the subdomain count, and the wall-clock, without dressing up a
    timeout as anything other than a timeout.

Writes results/e3_seed{seed}.json. The committed rung-2 far-region result that E4
builds on is regenerated here so it traces to a script.
"""
import os, sys, json, time, argparse
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import load_system
from lyapunov import LyapunovOnState
from controller import LinearDroop
from graphs import LyapunovCondition
from verify import certify_box, audit_verified, bound_ladder

GI = 4          # gen 5, 0-indexed
INNER = 0.05


def slice_box(xstar, n, dlo, dhi, wlo, whi):
    xL = xstar.clone(); xU = xstar.clone()
    xL[0, GI] = xstar[0, GI] + dlo; xU[0, GI] = xstar[0, GI] + dhi
    xL[0, GI + n] = xstar[0, GI + n] + wlo; xU[0, GI + n] = xstar[0, GI + n] + whi
    return xL, xU


def full_box(xstar, r):
    return xstar - r, xstar + r


def load_cond(s, xstar, ckpt, mode, beta=None):
    V = LyapunovOnState(s)
    V.load_state_dict(torch.load(ckpt)); V.eval()
    return LyapunovCondition(s, V, LinearDroop(s), mode=mode, x_star=xstar, beta=beta)


def rung2(s, xstar, n, seed, as_trained_ckpt, cegis_ckpt):
    out = {}
    # as-trained, far region [0.1, 0.5]^2 on the gen-5 slice: the genuine violation
    cond_at = load_cond(s, xstar, as_trained_ckpt, "4b")
    xL, xU = slice_box(xstar, n, 0.1, 0.5, 0.1, 0.5)
    r = certify_box(cond_at, xL, xU, eps=0.0, method="CROWN",
                    min_width=0.01, time_budget=30, seed=seed)
    ce = r.get("counterexample")
    out["as_trained_far_region"] = dict(
        region={"gen": 5, "delta_offset": [0.1, 0.5], "omega_offset": [0.1, 0.5]},
        verdict=r["verdict"], F_at_ce=r.get("F_at_ce"),
        lie_at_ce=(-r["F_at_ce"] if r.get("F_at_ce") is not None else None),
        counterexample_gen5=(dict(delta=round(ce[GI], 4), omega=round(ce[GI + n], 4))
                             if ce is not None else None),
        subdomains=r["subdomains"], seconds=r["seconds"], method=r["method"])
    # tightness ladder on the +/-0.5 box, as-trained
    xLb, xUb = slice_box(xstar, n, -0.5, 0.5, -0.5, 0.5)
    out["bound_ladder_pm0p5_box"] = {k: v["lower_bound"] for k, v in bound_ladder(cond_at, xLb, xUb).items()}

    # CEGIS, near annulus: max certifiable rho for (4b)
    cond_ce = load_cond(s, xstar, cegis_ckpt, "4b")
    rho_max, detail = 0.0, None
    for rho in [round(0.25 * k, 2) for k in range(1, 7)]:      # 0.25 .. 1.5
        xL, xU = slice_box(xstar, n, INNER, INNER + rho, INNER, INNER + rho)
        rr = certify_box(cond_ce, xL, xU, eps=0.0, method="CROWN",
                         min_width=0.03, time_budget=40, seed=seed)
        if not rr["certified"]:
            break
        audit = audit_verified(cond_ce, xL, xU, eps=0.0, seed=seed + 7)
        if audit < -1e-3:
            break
        rho_max = rho
        detail = dict(rho=rho, certified_lower_bound_F=rr.get("certified_lower_bound_F"),
                      audit_min_F=round(float(audit), 4), subdomains=rr["subdomains"])
    out["cegis_4b_annulus"] = dict(certified_rho=rho_max, detail=detail)

    # Proposition 2: largest beta certifiable on the rho=1.0 annulus, CEGIS network
    beta_lo, beta_hi, beta_max = 0.0, 4.0, 0.0
    xL, xU = slice_box(xstar, n, INNER, INNER + 1.0, INNER, INNER + 1.0)
    for _ in range(7):                       # bisection on beta
        beta = 0.5 * (beta_lo + beta_hi)
        cond_p2 = load_cond(s, xstar, cegis_ckpt, "prop2", beta=beta)
        rr = certify_box(cond_p2, xL, xU, eps=0.0, method="CROWN",
                         min_width=0.04, time_budget=30, seed=seed)
        ok = rr["certified"] and audit_verified(cond_p2, xL, xU, 0.0, seed=seed + 7) >= -1e-3
        if ok:
            beta_max = beta; beta_lo = beta
        else:
            beta_hi = beta
    out["cegis_prop2_beta_max"] = round(beta_max, 4)
    return out


def rung3(s, xstar, seed, cegis_ckpt, eps=0.05):
    """Full 20-D box x* +/- r, certify (4b) with tolerance eps, push r outward."""
    cond = load_cond(s, xstar, cegis_ckpt, "4b")
    r_max, detail, boundary = 0.0, None, None
    for r in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]:
        xL, xU = full_box(xstar, r)
        rr = certify_box(cond, xL, xU, eps=eps, method="CROWN",
                         min_width=0.01, time_budget=45, seed=seed)
        if rr["certified"]:
            r_max = r
            detail = dict(r=r, subdomains=rr["subdomains"], seconds=rr["seconds"],
                          certified_lower_bound_F=rr.get("certified_lower_bound_F"))
        else:
            boundary = dict(r=r, verdict=rr["verdict"], reason=rr.get("reason"),
                            F_at_ce=rr.get("F_at_ce"), subdomains=rr["subdomains"],
                            seconds=rr["seconds"])
            break
    return dict(eps=eps, certified_r_max=r_max, certified_detail=detail,
                first_failure=boundary)


def rung4(s, xstar, seed, cegis_ckpt, r=0.5, eps=0.05, time_budget=60):
    cond = load_cond(s, xstar, cegis_ckpt, "4b")
    xL, xU = full_box(xstar, r)
    t0 = time.time()
    rr = certify_box(cond, xL, xU, eps=eps, method="CROWN",
                     min_width=0.01, time_budget=time_budget, seed=seed)
    return dict(r=r, eps=eps, verdict=rr["verdict"], reason=rr.get("reason"),
                certified=rr["certified"], subdomains=rr["subdomains"],
                seconds=round(time.time() - t0, 1),
                F_at_ce=rr.get("F_at_ce"))


def run(seed=0):
    s = load_system(); n = s["n"]
    xstar = torch.tensor(np.load(os.path.join(ROOT, "results", f"equilibrium_seed{seed}.npy")),
                         dtype=torch.float32).reshape(1, -1)
    at = os.path.join(ROOT, "results", f"lyap_seed{seed}.pt")
    ce = os.path.join(ROOT, "results", f"lyap_cegis_seed{seed}.pt")
    have_ce = os.path.exists(ce)

    out = dict(seed=seed, condition="4b + prop2",
               rung2=rung2(s, xstar, n, seed, at, ce if have_ce else at))
    if have_ce:
        out["rung3_full20D"] = rung3(s, xstar, seed, ce)
        out["rung4_full20D_scale"] = rung4(s, xstar, seed, ce)
    with open(os.path.join(ROOT, "results", f"e3_seed{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args().seed)
