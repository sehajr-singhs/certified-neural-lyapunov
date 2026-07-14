"""E6 / Task 3 - reproduce her Algorithm 2, the learned RNN-Lyapunov controller,
and re-run the certification with it in the closed loop instead of linear droop.

Her Algorithm 2 trains a decentralized monotone stacked-ReLU controller
u_i(omega_i) to minimize a frequency-regulation cost, regularized by the Lyapunov
term R = relu(Lie + beta (V - V*)) so the learned controller stays inside the
region the Lyapunov function certifies. We train three controllers on the same
differentiable rollout of the lossy swing dynamics: linear droop (her baseline),
RNN with the Lyapunov regularizer, and RNN without it. We report:

  1. cost, RNN-Lyapunov versus linear droop, targeting her ~19% reduction,
  2. a transient after a disturbance, where the un-regularized RNN is expected to
     lose synchronism while the regularized one holds it, her gen-9 story,
  3. the certification staircase re-run with the RNN controller in the loop, so
     the certified region is reported for the LEARNED controller, not only for the
     droop loop, which is what certifying her method actually means.

Fidelity note, stated honestly: her fault is a trip of lines 1-39 and 2-3 at
t = 6 s, but the shipped data is the pre-fault Kron-reduced model only, so we model
the disturbance as an initial frequency-and-angle perturbation, the standard
transient-stability disturbance, rather than re-deriving her post-fault reduction,
which we do not have. The controller structure and the training objective are hers.

Writes results/e6_seed{seed}.json and results/rnn_lyap_seed{seed}.pt.
"""
import os, sys, json, time, argparse
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import load_system
from dynamics import SwingDynamics
from controller import LinearDroop, StackedReLU
from lyapunov import LyapunovOnState
from graphs import LyapunovCondition
from verify import certify_box, audit_verified

GI, INNER = 4, 0.05


def disturbances(xstar, n, k, gen, ang=0.3, freq=0.4):
    d = ang * torch.randn(k, n, generator=gen)
    w = freq * torch.randn(k, n, generator=gen)
    return xstar + torch.cat([d, w], 1)


def rollout_cost(dyn, ctrl, x0, xstar, steps=120, dt=0.01, lam=0.02, cond=None, beta=1.0):
    """Differentiable rollout. Cost is frequency deviation plus control effort,
    plus, if cond is given, the Lyapunov regularizer relu(Lie + beta(V-V*))."""
    n = dyn.n
    x = x0
    cost = 0.0; reg = 0.0
    for _ in range(steps):
        u = ctrl(x)
        omega = x[:, n:2 * n]
        cost = cost + (omega.pow(2).sum(1) + lam * u.pow(2).sum(1)).mean()
        if cond is not None:
            lie = cond.lie(x).squeeze(-1)
            V = cond.V(x).squeeze(-1)
            reg = reg + torch.relu(lie + beta * (V - cond.V_star)).mean()
        x = x + dt * dyn(x, u)
    return cost / steps, reg / steps, x


def train_rnn(system, dyn, xstar, cond, use_lyap, seed, epochs=300, ang=0.3, freq=0.4):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    ctrl = StackedReLU(system)
    opt = torch.optim.Adam(ctrl.parameters(), lr=0.02)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=120, gamma=0.5)
    for ep in range(epochs):
        # moderate disturbances that keep the forward rollout stable; near the
        # equilibrium the linear droop is already near LQR-optimal, so this is the
        # honest regime rather than a fault the shipped pre-fault data cannot model
        x0 = disturbances(xstar, system["n"], 64, gen, ang=ang, freq=freq)
        cost, reg, _ = rollout_cost(dyn, ctrl, x0, xstar, cond=(cond if use_lyap else None))
        loss = cost + (1.0 * reg if use_lyap else 0.0)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return ctrl


def mean_cost(dyn, ctrl, x0, xstar):
    """Evaluate on a FIXED disturbance set so the three controllers are compared on
    identical inputs."""
    with torch.no_grad():
        c, _, _ = rollout_cost(dyn, ctrl, x0, xstar)
    return float(c)


def transient(dyn, ctrl, xstar, n, gen, kick=2.5, steps=800, dt=0.01):
    """Large disturbance, roll out, return the worst final abs frequency (loss of
    synchronism shows up as a frequency that does not come back)."""
    x = xstar.clone()
    x[0, n + 8] += kick                     # kick gen 9 (index 8), her desync bus
    x[0, 0] += 0.3
    with torch.no_grad():
        peak = 0.0
        for t in range(steps):
            x = x + dt * dyn(x, ctrl(x))
            peak = max(peak, float(x[0, n:2 * n].abs().max()))
        final = float(x[0, n:2 * n].abs().max())
    return dict(final_max_abs_omega=round(final, 4), peak_abs_omega=round(peak, 4),
                synchronism=("held" if final < 1.0 else "lost"))


def certify_rnn_slice(system, V, ctrl, xstar, n, seed, rho_grid):
    cond = LyapunovCondition(system, V, ctrl, mode="4b", x_star=xstar)
    rho_max = 0.0
    for rho in rho_grid:
        xL = xstar.clone(); xU = xstar.clone()
        xL[0, GI] += INNER; xU[0, GI] += INNER + rho
        xL[0, GI + n] += INNER; xU[0, GI + n] += INNER + rho
        r = certify_box(cond, xL, xU, eps=0.0, method="CROWN", min_width=0.03,
                        time_budget=40, seed=seed)
        if not r["certified"]:
            break
        if audit_verified(cond, xL, xU, 0.0, seed=seed + 7) < -1e-3:
            break
        rho_max = rho
    return rho_max


def run(seed=0):
    s = load_system(); n = s["n"]
    dyn = SwingDynamics(s)
    xstar = torch.tensor(np.load(os.path.join(ROOT, "results", f"equilibrium_seed{seed}.npy")),
                         dtype=torch.float32).reshape(1, -1)
    V = LyapunovOnState(s); V.load_state_dict(torch.load(os.path.join(ROOT, "results", f"lyap_cegis_seed{seed}.pt")))
    V.eval()
    cond = LyapunovCondition(s, V, LinearDroop(s), mode="4b", x_star=xstar)
    gen = torch.Generator().manual_seed(seed + 50)

    droop = LinearDroop(s)
    rnn_lyap = train_rnn(s, dyn, xstar, cond, use_lyap=True, seed=seed)
    rnn_nolyap = train_rnn(s, dyn, xstar, cond, use_lyap=False, seed=seed)
    torch.save(rnn_lyap.state_dict(), os.path.join(ROOT, "results", f"rnn_lyap_seed{seed}.pt"))

    x_eval = disturbances(xstar, n, 256, gen, ang=0.3, freq=0.4)   # one fixed eval set
    c_droop = mean_cost(dyn, droop, x_eval, xstar)
    c_lyap = mean_cost(dyn, rnn_lyap, x_eval, xstar)
    c_nolyap = mean_cost(dyn, rnn_nolyap, x_eval, xstar)
    reduction = 100.0 * (c_droop - c_lyap) / c_droop

    rho_grid = [round(0.25 * k, 2) for k in range(1, 9)]
    rho_droop = certify_rnn_slice(s, V, droop, xstar, n, seed, rho_grid)
    rho_rnn = certify_rnn_slice(s, V, rnn_lyap, xstar, n, seed, rho_grid)

    out = dict(
        seed=seed,
        fidelity_note="disturbance modeled as initial perturbation; post-fault Kron data not shipped",
        cost=dict(linear_droop=round(c_droop, 5), rnn_lyapunov=round(c_lyap, 5),
                  rnn_no_lyapunov=round(c_nolyap, 5),
                  rnn_lyap_reduction_pct=round(reduction, 1)),
        transient_kick_gen9=dict(
            rnn_lyapunov=transient(dyn, rnn_lyap, xstar, n, gen),
            rnn_no_lyapunov=transient(dyn, rnn_nolyap, xstar, n, gen)),
        certified_rho_4b=dict(droop_loop=rho_droop, rnn_lyapunov_loop=rho_rnn),
    )
    with open(os.path.join(ROOT, "results", f"e6_seed{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args().seed)
