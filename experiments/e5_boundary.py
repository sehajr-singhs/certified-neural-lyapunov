"""E5 - where it breaks, and why. Grow the region until certification fails, and
for every failure say which of two entirely different things happened:

  genuine violation : a real counterexample x with Lie(x) > 0 was found, so the
                      property is FALSE on that region, no verifier could certify it
  incompleteness    : no counterexample was found but the bound stayed loose, so
                      the verifier could not prove a property that may still hold

Conflating these is the worst error available in this project, so certify_box
already returns "violation" (with the counterexample) or "unknown" (with the
reason), and E5 just sweeps the region size and records which one appears at the
boundary, per rung.

Writes results/e5_seed{seed}.json.
"""
import os, sys, json, argparse
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import load_system
from lyapunov import LyapunovOnState
from controller import LinearDroop
from graphs import LyapunovCondition
from verify import certify_box

GI = 4; INNER = 0.05


def slice_box(xstar, n, dlo, dhi, wlo, whi):
    xL = xstar.clone(); xU = xstar.clone()
    xL[0, GI] = xstar[0, GI] + dlo; xU[0, GI] = xstar[0, GI] + dhi
    xL[0, GI + n] = xstar[0, GI + n] + wlo; xU[0, GI + n] = xstar[0, GI + n] + whi
    return xL, xU


def classify(r):
    if r["certified"]:
        return "certified"
    if r["verdict"] == "violation":
        return "genuine_violation"
    return "incompleteness"          # unknown: no CE, loose bound / budget


def load_cond(s, xstar, ckpt):
    V = LyapunovOnState(s); V.load_state_dict(torch.load(ckpt)); V.eval()
    return LyapunovCondition(s, V, LinearDroop(s), mode="4b", x_star=xstar)


def sweep_slice(cond, xstar, n, rhos, seed):
    row = []
    for rho in rhos:
        xL, xU = slice_box(xstar, n, INNER, INNER + rho, INNER, INNER + rho)
        r = certify_box(cond, xL, xU, eps=0.0, method="CROWN",
                        min_width=0.03, time_budget=35, seed=seed)
        row.append(dict(rho=rho, outcome=classify(r), verdict=r["verdict"],
                        F_at_ce=r.get("F_at_ce"), reason=r.get("reason"),
                        subdomains=r["subdomains"]))
        if not r["certified"]:
            break
    return row


def sweep_full(cond, xstar, rs, seed, eps=0.05):
    row = []
    for rr in rs:
        xL, xU = xstar - rr, xstar + rr
        r = certify_box(cond, xL, xU, eps=eps, method="CROWN",
                        min_width=0.01, time_budget=40, seed=seed)
        row.append(dict(r=rr, outcome=classify(r), verdict=r["verdict"],
                        F_at_ce=r.get("F_at_ce"), reason=r.get("reason"),
                        subdomains=r["subdomains"]))
        if not r["certified"]:
            break
    return row


def boundary(row, key):
    cert = [d[key] for d in row if d["outcome"] == "certified"]
    fail = next((d for d in row if d["outcome"] != "certified"), None)
    return (max(cert) if cert else 0.0), fail


def run(seed=0):
    s = load_system(); n = s["n"]
    xstar = torch.tensor(np.load(os.path.join(ROOT, "results", f"equilibrium_seed{seed}.npy")),
                         dtype=torch.float32).reshape(1, -1)
    at = os.path.join(ROOT, "results", f"lyap_seed{seed}.pt")
    ce = os.path.join(ROOT, "results", f"lyap_cegis_seed{seed}.pt")

    out = dict(seed=seed, eps_full=0.05)

    # gen-5 slice, as-trained: fails immediately and genuinely
    sweep_at = sweep_slice(load_cond(s, xstar, at), xstar, n, [0.25, 0.5], seed)
    b_at, f_at = boundary(sweep_at, "rho")
    out["slice_as_trained"] = dict(certified_rho=b_at, first_failure=f_at, sweep=sweep_at)

    if os.path.exists(ce):
        cond_ce = load_cond(s, xstar, ce)
        # gen-5 slice, CEGIS: push past rho=1.5 to find the true boundary
        rhos = [round(0.5 * k, 2) for k in range(1, 9)]        # 0.5 .. 4.0
        sw = sweep_slice(cond_ce, xstar, n, rhos, seed)
        b, f = boundary(sw, "rho")
        out["slice_cegis"] = dict(certified_rho=b, first_failure=f, sweep=sw)
        # full 20-D, CEGIS: find the radius where it first breaks and classify
        rs = [0.005, 0.01, 0.02, 0.04, 0.08]
        swf = sweep_full(cond_ce, xstar, rs, seed)
        bf, ff = boundary(swf, "r")
        out["full20D_cegis"] = dict(certified_r=bf, first_failure=ff, sweep=swf)

    with open(os.path.join(ROOT, "results", f"e5_seed{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args().seed)
