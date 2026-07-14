"""E3 rung 1 - the correctness gate, a two-machine lossy swing system, 4-D state.

The spec calls this the gate: on a system small enough that branch-and-bound is
never the bottleneck, the whole pipeline (dynamics, ELU Lyapunov net, analytic
gradient graph, CROWN, branch-and-bound, PGD audit) must certify a real region.
If it cannot certify here, nothing downstream means anything.

We build a genuine two-bus lossy network with a conductance coupling
G_ij cos(delta_i - delta_j), pick a target equilibrium and back-solve the power
injections Pm so that state is an equilibrium, train a small ELU V against
condition (4b) with the same loss family as E0, then certify (4b) on an annulus
around the equilibrium. Success here is expected, and getting it is the gate.

Writes results/e3_rung1_seed{seed}.json and results/lyap_twobus_seed{seed}.pt.
"""
import os, sys, json, argparse
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dynamics import SwingDynamics
from controller import LinearDroop
from lyapunov import LyapunovOnState
from graphs import LyapunovCondition
from equilibrium import find_equilibrium
from verify import certify_box, audit_verified
from data import _minus_ref_one


def build_twobus_system(dstar=(0.0, 0.35)):
    n = 2
    M = np.array([1.0, 1.2])
    K = np.array([[0.0, 2.0], [2.0, 0.0]])
    gamma = np.array([[0.0, 0.15], [0.15, 0.0]])       # nonzero loss angle => lossy
    F = K * np.cos(gamma)
    G = -K * np.sin(gamma)
    d = np.asarray(dstar)
    dij = d[:, None] - d[None, :]
    sin_term = (F * np.sin(dij)).sum(1)
    cos_term = (G * (np.cos(dij) - np.eye(n))).sum(1)
    Pm = sin_term + cos_term                            # makes dstar an equilibrium
    return dict(n=n, omega_R=1.0, omega_scale=1.0, M=M, D=np.zeros(n),
                F=F, G=G, K=K, gamma=gamma, Pm=Pm,
                linear_coff=np.array([3.0, 3.0]), max_action=np.array([5.0, 5.0]),
                minus_ref=_minus_ref_one(n))


def train_V(system, xstar, seed=0, iters=3000):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    V = LyapunovOnState(system, hidden=32)
    cond = LyapunovCondition(system, V, LinearDroop(system), mode="4b", x_star=xstar)
    opt = torch.optim.Adam(V.parameters(), lr=0.02)
    Vstar = lambda: V(xstar).reshape(())
    for it in range(iters):
        d = torch.randn(400, 2, generator=gen) * 0.6
        w = torch.randn(400, 2, generator=gen) * 0.6
        x = xstar + torch.cat([d, w], 1)
        x = torch.cat([xstar, x], 0)
        lie = cond.lie(x).squeeze(-1)
        Vb = V(x).squeeze(-1)
        dist = (x - xstar).norm(dim=1)
        lie_eq = cond.lie(xstar).squeeze(-1)
        loss = ((torch.tanh(lie) * torch.exp(-dist)).mean()
                + 0.5 * torch.relu(-Vb + Vstar()).mean()
                + 20.0 * lie_eq.pow(2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    return V


def run(seed=0):
    system = build_twobus_system()
    xstar, res = find_equilibrium(system, seed=seed)
    V = train_V(system, xstar, seed=seed)
    torch.save(V.state_dict(), os.path.join(ROOT, "results", f"lyap_twobus_seed{seed}.pt"))
    cond = LyapunovCondition(system, V, LinearDroop(system), mode="4b", x_star=xstar)

    # certify (4b) on a 4-D annulus: every coordinate in [x*+INNER, x*+INNER+rho],
    # bisect the largest rho, audit each certificate with PGD.
    INNER = 0.05
    rho_max, detail = 0.0, None
    for rho in [round(0.1 * k, 2) for k in range(1, 13)]:      # 0.1 .. 1.2
        xL = xstar + INNER; xU = xstar + (INNER + rho)
        r = certify_box(cond, xL, xU, eps=0.0, method="CROWN",
                        min_width=0.02, time_budget=40, seed=seed)
        if not r["certified"]:
            break
        audit = audit_verified(cond, xL, xU, eps=0.0, seed=seed + 7)
        if audit < -1e-3:
            break
        rho_max = rho
        detail = dict(rho=rho, certified_lower_bound_F=r.get("certified_lower_bound_F"),
                      audit_min_F=round(float(audit), 4), subdomains=r["subdomains"],
                      seconds=r["seconds"])

    out = dict(seed=seed, rung="1 (two-bus, 4-D state)", condition="4b",
               equilibrium_residual=res, gate_passed=bool(rho_max > 0),
               certified_rho=rho_max, certified_detail=detail,
               note=("gate passed, pipeline certifies a real region on the two-bus system"
                     if rho_max > 0 else "gate FAILED, pipeline cannot certify the two-bus system"))
    with open(os.path.join(ROOT, "results", f"e3_rung1_seed{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args().seed)
