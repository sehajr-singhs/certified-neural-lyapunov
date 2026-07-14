"""E0 - reproduce her Algorithm 1 (neural Lyapunov training) in PyTorch.

Ports Lyapunov_Train_ref: truncated-normal sampling around the equilibrium, the
loss that penalises a positive Lie derivative weighted by proximity to x*, the
V > V* penalty, the equilibrium pinning terms, and the active-sampling step that
re-injects previously violating states. Reports her headline quantity: the
fraction of sampled states with Lie derivative <= 0 (her rho, target ~99.9%) and
the fraction with V > V*.

Writes results/e0_seed{seed}.json and results/lyap_seed{seed}.pt.

Run: KMP_DUPLICATE_LIB_OK=TRUE python experiments/e0_reproduce.py --seed 0
"""
import os, sys, json, time, argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import load_system                         # noqa: E402
from lyapunov import LyapunovOnState                 # noqa: E402
from controller import LinearDroop                   # noqa: E402
from dynamics import SwingDynamics                   # noqa: E402
from graphs import LyapunovCondition                 # noqa: E402
from equilibrium import find_equilibrium             # noqa: E402

# her cell-11 hyperparameters
DELTA_STD = 2.0
OMEGA_STD = 10.0
DELTA_STD_SMALL = 0.2
OMEGA_STD_SMALL = 0.5
NORM_CONTROL = 10.0
BATCH = 500


def truncated_normal(shape, std, gen, trunc=2.0):
    x = torch.zeros(shape)
    while True:
        bad = (x == 0)
        if not bad.any():
            break
        x[bad] = torch.randn(int(bad.sum()), generator=gen) * std
        x[x.abs() > trunc * std] = 0
    return x


def sample(system, xstar, n, batch, gen, small_ratio=0.0):
    d = system["n"]
    nb = batch - 1
    n_small = int(np.floor(small_ratio * nb))
    parts_d, parts_w = [], []
    if n_small > 0:
        parts_d.append(truncated_normal((n_small, d), DELTA_STD_SMALL, gen))
        parts_w.append(truncated_normal((n_small, d), OMEGA_STD_SMALL, gen))
    parts_d.append(truncated_normal((nb - n_small, d), DELTA_STD, gen))
    parts_w.append(truncated_normal((nb - n_small, d), OMEGA_STD, gen))
    dd = torch.cat(parts_d, 0)
    ww = torch.cat(parts_w, 0)
    noise = torch.cat([dd, ww], 1)
    x = xstar + noise
    return torch.cat([xstar, x], 0)     # first row is exactly the equilibrium


def lie_and_V(cond, x):
    lie = cond.lie(x)
    V = cond.V(x)
    return lie.squeeze(-1), V.squeeze(-1)


def train(seed=0, iters=4000, iters_first=200, iters_active=2000, log=True):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    system = load_system()
    xstar, res = find_equilibrium(system)
    V = LyapunovOnState(system)
    cond = LyapunovCondition(system, V, LinearDroop(system), mode="4b", x_star=xstar)
    opt = torch.optim.Adam(V.parameters(), lr=0.05)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.9)

    Vstar = V(xstar).reshape(())
    outbound = None
    t0 = time.time()
    loss_hist = []
    for it in range(iters):
        ratio_small = 0.5 if it >= iters_active else 0.0
        x = sample(system, xstar, system["n"], BATCH, gen, small_ratio=ratio_small)
        # active sampling: append previously violating states once the ratio is high
        if outbound is not None and outbound.shape[0] > 0:
            x = torch.cat([x, outbound], 0)
        lie, Vval = lie_and_V(cond, x)
        Vstar = V(xstar).reshape(())
        lie_eq = lie[0]
        dist = (x - xstar).norm(dim=1)
        N = x.shape[0]
        loss = ((15.0 * (torch.tanh(lie) * torch.exp(-dist / NORM_CONTROL)).sum()
                 + 10.0 * torch.relu(-Vval + Vstar).sum()) / N
                + 60.0 * lie_eq ** 2 + 100.0 * torch.relu(lie_eq)) / N
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        loss_hist.append(float(loss))

        with torch.no_grad():
            viol = lie > 0
            ratio_ok = float((~viol).float().mean())
            if ratio_ok >= 0.95:
                idx = torch.where(viol)[0]
                outbound = x[idx].detach() if idx.numel() > 0 else None
        if log and it % 500 == 0:
            print(f"seed{seed} it{it:4d} loss={float(loss):.4f} lie<=0 frac={ratio_ok:.4f}")

    # final evaluation on a fresh large sample (her Ratio_detivative / Ratio_Lya)
    with torch.no_grad():
        xe = sample(system, xstar, system["n"], 5000, gen)
        lie, Vval = lie_and_V(cond, xe)
        Vstar = V(xstar).reshape(())
        frac_lie = float((lie <= 0).float().mean())
        frac_V = float((Vval > Vstar).float().mean())
        # her "smaller region" test: delta std 0.2, omega std 0.5
        xs = sample(system, xstar, system["n"], 5000, gen, small_ratio=1.0)
        lies, Vs = lie_and_V(cond, xs)
        frac_lie_small = float((lies <= 0).float().mean())
    dur = time.time() - t0

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    torch.save(V.state_dict(), os.path.join(ROOT, "results", f"lyap_seed{seed}.pt"))
    np.save(os.path.join(ROOT, "results", f"equilibrium_seed{seed}.npy"), xstar.numpy().ravel())
    out = dict(
        seed=seed, iters=iters, train_seconds=round(dur, 1),
        equilibrium_residual=res,
        frac_lie_le_0_large_region=frac_lie,
        frac_V_gt_Vstar_large_region=frac_V,
        frac_lie_le_0_small_region=frac_lie_small,
        region={"delta_std": DELTA_STD, "omega_std": OMEGA_STD},
        final_loss=loss_hist[-1],
    )
    with open(os.path.join(ROOT, "results", f"e0_seed{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()
    train(seed=args.seed, iters=args.iters)
