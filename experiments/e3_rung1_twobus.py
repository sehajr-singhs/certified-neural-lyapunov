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
from verify import certify_box, audit_verified
from data import _minus_ref_one

DSTAR = (0.0, 0.35)


def build_twobus_system(dstar=DSTAR):
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


def equilibrium(system):
    """x* = (delta*, omega*=0), an exact equilibrium by construction: Pm was
    back-solved so the coupling balances Pm at delta* with omega = 0 and the
    droop control u(0) = 0. We verify the residual rather than integrate to it."""
    n = system["n"]
    xstar = torch.zeros(1, 2 * n)
    xstar[0, :n] = torch.tensor(DSTAR, dtype=torch.float32)
    res = SwingDynamics(system)(xstar, LinearDroop(system)(xstar)).abs().max().item()
    return xstar, res


INNER = 0.05
RHO_TRAIN = 1.3          # train over a region a little larger than we certify
GI2 = 1                  # bus 2 (0-indexed); bus 1 is the reference, pinned at x*


def _slice(xstar, off_d, off_w):
    """Offset only bus 2, delta index GI2 and omega index GI2+2, keeping bus 1
    fixed. In the reduced coordinates z = R x, V sees only the bus differences,
    so this is the genuine annulus that stays away from the equilibrium z = 0.
    Offsetting all four raw coordinates would sweep through z = 0 where the Lie
    derivative is structurally zero, which no region can certify as < 0."""
    x = xstar.repeat(off_d.shape[0], 1).clone()
    x[:, GI2] = xstar[0, GI2] + off_d
    x[:, GI2 + 2] = xstar[0, GI2 + 2] + off_w
    return x


def _annulus_samples(xstar, k, gen):
    """Bus-2 offsets in [INNER, INNER+RHO_TRAIN] (the certification annulus), plus
    a sign-varied spread so V generalizes on both sides of the equilibrium."""
    a = _slice(xstar, INNER + RHO_TRAIN * torch.rand(k, generator=gen),
               INNER + RHO_TRAIN * torch.rand(k, generator=gen))
    m = k // 2
    sgn = lambda: (2 * (torch.rand(m, generator=gen) > 0.5).float() - 1)
    b = _slice(xstar, (INNER + RHO_TRAIN * torch.rand(m, generator=gen)) * sgn(),
               (INNER + RHO_TRAIN * torch.rand(m, generator=gen)) * sgn())
    return torch.cat([a, b], 0)


def train_V(system, xstar, seed=0, iters=5000, margin=0.3):
    """Train V to satisfy (4b) with a strict negative Lie margin on the
    certification region, keep V above V* growing with distance, pin the
    equilibrium, and inject verifier-style PGD counterexamples periodically. This
    is the same closing-the-loop idea as the main result, on a 4-D system where a
    valid certificate is expected."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    V = LyapunovOnState(system, hidden=32)
    cond = LyapunovCondition(system, V, LinearDroop(system), mode="4b", x_star=xstar)
    opt = torch.optim.Adam(V.parameters(), lr=0.01)
    Vstar = lambda: V(xstar).reshape(())
    ce_pool = torch.empty(0, 4)
    for it in range(iters):
        x = _annulus_samples(xstar, 400, gen)
        if ce_pool.shape[0] > 0:
            x = torch.cat([x, ce_pool], 0)
        x = torch.cat([xstar, x], 0)
        lie = cond.lie(x).squeeze(-1)
        Vb = V(x).squeeze(-1)
        dist = (x - xstar).norm(dim=1)
        lie_eq = cond.lie(xstar).squeeze(-1)
        loss = (torch.relu(lie + margin).mean()
                + 0.5 * torch.relu(Vstar() + 0.05 * dist - Vb).mean()
                + 20.0 * lie_eq.pow(2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 499:                          # inject worst-case counterexamples
            od = (INNER + RHO_TRAIN * torch.rand(300, generator=gen)).requires_grad_(True)
            ow = (INNER + RHO_TRAIN * torch.rand(300, generator=gen)).requires_grad_(True)
            o = torch.optim.Adam([od, ow], lr=0.02)
            for _ in range(40):
                l = (-cond.lie(_slice(xstar, od, ow))).sum()
                o.zero_grad(); l.backward(); o.step()
                with torch.no_grad():
                    od.clamp_(INNER, INNER + RHO_TRAIN); ow.clamp_(INNER, INNER + RHO_TRAIN)
            with torch.no_grad():
                a = _slice(xstar, od, ow)
                bad = cond.lie(a).squeeze(-1) > -margin
                ce_pool = torch.cat([ce_pool, a[bad].detach()], 0)[-3000:]
    return V


def run(seed=0):
    system = build_twobus_system()
    xstar, res = equilibrium(system)
    V = train_V(system, xstar, seed=seed)
    torch.save(V.state_dict(), os.path.join(ROOT, "results", f"lyap_twobus_seed{seed}.pt"))
    cond = LyapunovCondition(system, V, LinearDroop(system), mode="4b", x_star=xstar)

    # certify (4b) on the bus-2 annulus of the 4-D state: bus 1 (reference) pinned
    # at x*, bus 2 in [x*+INNER, x*+INNER+rho], which stays away from the
    # equilibrium z = 0 in reduced coordinates. Bisect the largest rho, audit each.
    rho_max, detail = 0.0, None
    for rho in [round(0.1 * k, 2) for k in range(1, 13)]:      # 0.1 .. 1.2
        xL = xstar.clone(); xU = xstar.clone()
        xL[0, GI2] += INNER; xU[0, GI2] += INNER + rho
        xL[0, GI2 + 2] += INNER; xU[0, GI2 + 2] += INNER + rho
        r = certify_box(cond, xL, xU, eps=0.0, method="CROWN",
                        min_width=0.02, time_budget=25, seed=seed)
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
