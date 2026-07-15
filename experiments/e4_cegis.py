"""E4 - close the loop (CEGIS). The contribution.

Target: certify condition (4b), Lie < 0, on the gen-5 projected slice over a box
that sits away from the equilibrium (delta and omega offsets in [0.1, 0.5]), so
there is no structural equilibrium trap inside it. The as-trained V fails there
(E3, genuine counterexample with Lie = +5.22). Each CEGIS iteration:
  1. attack the target region with PGD to collect violating states,
  2. retrain V for a short burst driving Lie negative on those states plus a grid
     over the region, keeping V > V* and the equilibrium pinned,
  3. re-certify the region with branch-and-bound,
and records per iteration: counterexamples found, max attack Lie, the verifier's
worst lower bound on F = -Lie over the region, the verdict, and wall-clock.

The headline is whether the verdict flips from violation to verified and the
worst lower bound rises to >= 0. Writes results/e4_seed{seed}.json.
"""
import os, sys, json, time, argparse, copy
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import load_system
from lyapunov import LyapunovOnState
from controller import LinearDroop
from graphs import LyapunovCondition
from verify import certify_box, audit_verified

GI = 4  # gen 5


def slice_box(xstar, n, dlo, dhi, wlo, whi):
    xL = xstar.clone(); xU = xstar.clone()
    xL[0, GI] = xstar[0, GI] + dlo; xU[0, GI] = xstar[0, GI] + dhi
    xL[0, GI + n] = xstar[0, GI + n] + wlo; xU[0, GI + n] = xstar[0, GI + n] + whi
    return xL, xU


def sample_region(xstar, n, box, k, gen):
    xL, xU = box
    x = xstar.repeat(k, 1).clone()
    x[:, GI] = xL[0, GI] + (xU[0, GI] - xL[0, GI]) * torch.rand(k, generator=gen)
    x[:, GI + n] = xL[0, GI + n] + (xU[0, GI + n] - xL[0, GI + n]) * torch.rand(k, generator=gen)
    return x


def attack_region(cond, xstar, n, box, k, gen, steps=150, lr=0.02):
    """Attack ONLY the two slice coordinates; every other coordinate stays pinned
    at the equilibrium so the search never leaves the projected slice."""
    xL, xU = box
    dl, du = xL[0, GI].item(), xU[0, GI].item()
    wl, wu = xL[0, GI + n].item(), xU[0, GI + n].item()
    d = (dl + (du - dl) * torch.rand(k, generator=gen)).clone().requires_grad_(True)
    w = (wl + (wu - wl) * torch.rand(k, generator=gen)).clone().requires_grad_(True)
    opt = torch.optim.Adam([d, w], lr=lr)

    def build(dv, wv):
        x = xstar.repeat(k, 1)
        x = x.index_copy(1, torch.tensor([GI]), dv.unsqueeze(1))
        x = x.index_copy(1, torch.tensor([GI + n]), wv.unsqueeze(1))
        return x

    for _ in range(steps):
        x = build(d, w)
        loss = (-cond.lie(x)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            d.clamp_(dl, du); w.clamp_(wl, wu)
    with torch.no_grad():
        x = build(d, w)
        lie = cond.lie(x).squeeze(-1)
        viol = lie > 0
    return x.detach(), lie.detach(), viol


INNER = 0.05        # fixed inner corner offset from equilibrium (avoids the trap)


def max_certifiable_rho(cond, xstar, n, rho_grid, seed):
    """Largest rho such that the gen-5 slice box [x*+INNER, x*+INNER+rho]^2 in
    (delta, omega) certifies (4b) AND passes the independent PGD audit. Returns
    (rho_max, detail_of_last_certified)."""
    rho_max = 0.0
    detail = None
    for rho in rho_grid:
        box = slice_box(xstar, n, INNER, INNER + rho, INNER, INNER + rho)
        r = certify_box(cond, box[0], box[1], eps=0.0, method="CROWN",
                        min_width=0.03, time_budget=30, seed=seed)
        if not r["certified"]:
            break
        audit = audit_verified(cond, box[0], box[1], eps=0.0, seed=seed + 7)
        if audit < -1e-3:          # audit rejects an unsound certificate
            break
        rho_max = rho
        detail = dict(rho=rho, certified_lower_bound_F=r.get("certified_lower_bound_F"),
                      audit_min_F=round(float(audit), 4), subdomains=r.get("subdomains"))
    return rho_max, detail


def run(seed=0, iters=6):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 100)
    s = load_system(); n = s["n"]
    xstar = torch.tensor(np.load(os.path.join(ROOT, "results", f"equilibrium_seed{seed}.npy")),
                         dtype=torch.float32).reshape(1, -1)
    V = LyapunovOnState(s)
    V.load_state_dict(torch.load(os.path.join(ROOT, "results", f"lyap_seed{seed}.pt")))
    ctrl = LinearDroop(s)
    cond = LyapunovCondition(s, V, ctrl, mode="4b", x_star=xstar)
    opt = torch.optim.Adam(V.parameters(), lr=0.008)
    Vstar = lambda: V(xstar).reshape(())
    rho_grid = [round(0.5 * k, 2) for k in range(1, 6)]      # 0.5 .. 2.5, matches E5

    history = []
    ce_pool = torch.empty(0, 2 * n)
    for it in range(iters):
        t0 = time.time()
        # measure the current certifiable region (baseline at it==0, then after retrain)
        rho_max, detail = max_certifiable_rho(cond, xstar, n, rho_grid, seed)
        # attack the frontier region (just past what certified) for counterexamples
        frontier = slice_box(xstar, n, INNER, INNER + min(rho_max + 0.3, 1.5),
                             INNER, INNER + min(rho_max + 0.3, 1.5))
        xatk, lie_atk, viol = attack_region(cond, xstar, n, frontier, 512, gen)
        n_ce = int(viol.sum())
        max_lie = float(lie_atk.max())
        if n_ce > 0:
            ce_pool = torch.cat([ce_pool, xatk[viol]], 0)[-6000:]
        rec = dict(iter=it, certified_rho=rho_max, certified_area=round(rho_max ** 2, 4),
                   frontier_counterexamples=n_ce, frontier_max_lie=round(max_lie, 4),
                   certified_detail=detail, iter_seconds=round(time.time() - t0, 1))
        history.append(rec)
        print(json.dumps({k: rec[k] for k in ["iter", "certified_rho", "certified_area",
              "frontier_counterexamples", "frontier_max_lie", "iter_seconds"]}))
        if it == iters - 1:
            break
        # counterexample-guided retrain burst
        for _ in range(500):
            xr = sample_region(xstar, n, frontier, 400, gen)
            batch = xr if ce_pool.shape[0] == 0 else torch.cat([xr, ce_pool], 0)
            lie = cond.lie(batch).squeeze(-1)
            Vb = V(batch).squeeze(-1)
            lie_eq = cond.lie(xstar).squeeze(-1)
            loss = (torch.relu(lie + 0.3).mean()
                    + 0.1 * torch.relu(-Vb + Vstar()).mean()
                    + 10.0 * lie_eq.pow(2).mean())
            opt.zero_grad(); loss.backward(); opt.step()

    torch.save(V.state_dict(), os.path.join(ROOT, "results", f"lyap_cegis_seed{seed}.pt"))
    out = dict(seed=seed, condition="4b", gen=5, inner_offset=INNER,
               region="gen-5 slice box [x*+INNER, x*+INNER+rho]^2 in (delta,omega)",
               baseline_certified_rho=history[0]["certified_rho"],
               final_certified_rho=history[-1]["certified_rho"],
               iterations=history)
    with open(os.path.join(ROOT, "results", f"e4_seed{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nbaseline rho:", history[0]["certified_rho"], "-> final rho:", history[-1]["certified_rho"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=6)
    run(ap.parse_args().seed, ap.parse_args().iters)
