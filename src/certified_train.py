"""Certified training (CT-BaB) for the neural Lyapunov function.

This replaces the post-hoc CEGIS repair loop with a training objective that puts
the size of the certified region into the loss, by differentiating through the
CROWN bound, following Shi, Li, Hsieh, Zhang (arXiv:2411.18235). CEGIS trains,
verifies, and retrains on the finite counterexamples the verifier returns, which
is repair against a measure-zero sample; certified training instead makes the
CROWN certified margin over an entire region a differentiable term in the loss, so
the same architecture is shaped to be verifiable while it trains.

The three mechanics of CT-BaB, ported to the continuous-time swing system:

  1. Differentiable certified bound in the loss. For a region B we ask auto_LiRPA
     for the CROWN lower bound lb_F(B) of the Lyapunov condition F, computed with
     gradients flowing to V's weights (no torch.no_grad). Every linear-relaxation
     and bound-propagation step is differentiable, so d lb_F / d weights exists and
     we descend on the hinge relu(-lb_F + eps), pushing the certified margin above
     zero over the whole region rather than at sampled points.

  2. Training-time branch-and-bound. We keep a dynamic dataset D of non-overlapping
     subregions tiling the target box, bound them all in one batched compute_bounds
     call, and after every few steps split only the HARD subregions (those whose
     certified lower bound is still negative) into two along the active dimension
     that most raises the children's bound (Shi et al. Eq. 6). Splitting effort
     concentrates on the hard pieces, which is what keeps BaB tractable in the loop.

  3. Region size in the objective. The decrease hinge is evaluated over the full
     target box of side rho_target, so driving it non-negative certifies a region
     of that size; the positive-definiteness hinge (V - V* >= margin over the box)
     is the anti-collapse term, the continuous-time analogue of Shi et al.'s ROA
     term, because without it the decrease condition is met trivially by a flat V.

Deviations from Shi et al., stated plainly and logged in NOTES.md:
  - They CO-TRAIN the controller u and V. We train V ONLY, against the fixed
    saturated linear droop law Cui and Zhang train against, because the question
    here is whether V can be made verifiable-by-construction, not controller design.
  - They train with a discrete-time decrease V(x_{t+1}) - V(x_t) <= -kappa V(x_t).
    Ours is the continuous-time Lie derivative Lie = grad V . f, using the same
    LyapunovCondition graph the verifier uses, so training and verification bound
    the identical function.

Soundness is unchanged: this module only produces V. The certified radius is then
measured by the same certify_box branch-and-bound bisection CEGIS used, audited by
an independent PGD attack, and cross-checked against the JacobianOP verifier. A
number this module prints during training is a training diagnostic, never a
certificate; the certificate always comes from src/verify.py.

Written to scale: nothing here mentions dimension 2. The slice is a SliceSpec, the
box and the initial grid are built programmatically from slice_dim, and raising the
dimension is a config change. See SCALING.md.
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from lyapunov import LyapunovOnState
from controller import LinearDroop
from graphs import LyapunovCondition
from slice import SliceSpec


def _batched_lb(bm, xL, xU, method="CROWN"):
    """CROWN lower bound of the graph over each row's box. xL,xU: (N,2n). Returns
    (N,) lower bounds carrying grad to the model weights."""
    x0 = ((xL + xU) / 2)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=xL, x_U=xU)
    bx = BoundedTensor(x0, ptb)
    lb, ub = bm.compute_bounds(x=(bx,), method=method)
    return lb.reshape(-1)


def _pgd_worst_lie(cond, sp, xstar, box, k, generator, steps=60, lr=0.03):
    """Empirical worst-case Lie over the box, attacking only the active slice
    coordinates (the search never leaves the slice). Returns (worst_hinge_input
    states, lie values). Used for the PGD hinge term of the loss."""
    xstar = xstar.reshape(1, -1)
    xL, xU = box
    lo = torch.tensor([xL[0, d].item() for d in sp.active_dims])
    hi = torch.tensor([xU[0, d].item() for d in sp.active_dims])
    c = (lo + (hi - lo) * torch.rand(k, sp.slice_dim, generator=generator)).clone().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)
    for _ in range(steps):
        x = sp.embed(xstar, c)
        loss = cond.lie(x).sum()      # maximize Lie == minimize -Lie
        opt.zero_grad(); (-loss).backward(); opt.step()
        with torch.no_grad():
            c.clamp_(lo, hi)
    x = sp.embed(xstar, c.detach())
    return x, cond.lie(x).squeeze(-1).detach()


def _split_hard(bm4b, sp, cells, margin, max_cells):
    """Training-time BaB split. cells is a list of (xL,xU). Split every cell whose
    certified lower bound of F(4b) is < margin, along the active dimension that
    maximizes the smaller child bound (Shi et al. Eq. 6), until max_cells reached.
    Returns the new cell list. All bounds here are detached, this only reshapes D."""
    if len(cells) >= max_cells:
        return cells
    with torch.no_grad():
        xL = torch.cat([c[0] for c in cells], 0)
        xU = torch.cat([c[1] for c in cells], 0)
        lb = _batched_lb(bm4b, xL, xU)
    hard = [i for i in range(len(cells)) if lb[i].item() < margin]
    if not hard:
        return cells
    # rank hardest first, respect the cell budget
    hard.sort(key=lambda i: lb[i].item())
    keep = [cells[i] for i in range(len(cells)) if i not in set(hard)]
    budget = max_cells - len(keep)
    new = []
    for i in hard:
        if len(new) + 2 > budget:
            new.append(cells[i])          # no room to split, keep whole
            continue
        cL, cU = cells[i]
        best_dim, best_score = None, -1e30
        with torch.no_grad():
            for d in sp.active_dims:
                if (cU[0, d] - cL[0, d]).item() <= 1e-6:
                    continue
                mid = (cL[0, d] + cU[0, d]).item() / 2
                aL, aU = cL.clone(), cU.clone(); aU[0, d] = mid
                bL, bU = cL.clone(), cU.clone(); bL[0, d] = mid
                lbc = _batched_lb(bm4b, torch.cat([aL, bL], 0), torch.cat([aU, bU], 0))
                score = float(lbc.min())   # dim giving the best (highest) worst child
                if score > best_score:
                    best_score, best_dim = score, d
        if best_dim is None:
            new.append(cells[i]); continue
        mid = (cL[0, best_dim] + cU[0, best_dim]).item() / 2
        aL, aU = cL.clone(), cU.clone(); aU[0, best_dim] = mid
        bL, bU = cL.clone(), cU.clone(); bL[0, best_dim] = mid
        new.append((aL, aU)); new.append((bL, bU))
    return keep + new


def _warmstart_V(system, warmstart, require_warmstart):
    """Build V and load the as-trained weights. This is the controlled-comparison
    guarantee Prof. Cui asked for: certified training must REFINE the identical
    network CEGIS started from (the lyap_seed{k}.pt checkpoint), never spin up a
    fresh random net. So a warm-start checkpoint is MANDATORY by default, loaded
    with strict=True (the architecture must match exactly, or load_state_dict
    raises), and we return a weight fingerprint that goes into the result JSON so
    the run proves on its face that V was warm-started, not reinitialized.

    require_warmstart=False is an escape hatch for future from-scratch scaling
    experiments only; it is never used by the 2-D comparison e7 runs.
    """
    V = LyapunovOnState(system)
    if warmstart is None or not os.path.exists(warmstart):
        if require_warmstart:
            raise FileNotFoundError(
                f"certified training requires a warm-start checkpoint, got {warmstart!r}. "
                "Pass the as-trained lyap_seed{k}.pt so V is refined from the identical "
                "network CEGIS started from. Refusing to train a fresh random net, which "
                "would break the controlled comparison. (require_warmstart=False overrides.)")
        return V, dict(warmstart=warmstart, warmstart_loaded=False,
                       init_param_l2=None, note="FRESH RANDOM NET (require_warmstart=False)")
    V.load_state_dict(torch.load(warmstart), strict=True)   # same architecture or it raises
    with torch.no_grad():
        fp = float(torch.cat([p.reshape(-1) for p in V.parameters()]).norm())
    return V, dict(warmstart=os.path.abspath(warmstart), warmstart_loaded=True,
                   init_param_l2=round(fp, 6))


def train(system, xstar, active_buses, seed=0, warmstart=None, require_warmstart=True,
          inner=0.05, rho_target=2.5, init_max_side=0.7, max_cells=256,
          steps=1200, lr=6e-3, split_every=40, eps=0.15, margin_4a=0.05,
          w_4a=0.5, w_pgd=0.5, method="CROWN", verbose=True):
    """Certified-train V on the projected slice. Trains V only; the controller is
    the fixed LinearDroop. V is warm-started from the as-trained checkpoint and
    REFINED, never reinitialized (see _warmstart_V). Returns (V, history).
    Deterministic given seed."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 100)
    n = system["n"]
    xstar = torch.as_tensor(xstar, dtype=torch.float32).reshape(1, -1)
    sp = SliceSpec(system, active_buses)

    V, ws_info = _warmstart_V(system, warmstart, require_warmstart)
    ctrl = LinearDroop(system)
    cond4b = LyapunovCondition(system, V, ctrl, mode="4b", x_star=xstar)
    cond4a = LyapunovCondition(system, V, ctrl, mode="4a", x_star=xstar)

    ex = xstar.clone()
    bm4b = BoundedModule(cond4b, ex, verbose=False)
    bm4a = BoundedModule(cond4a, ex, verbose=False)

    cells = sp.grid(xstar, inner, inner + rho_target, init_max_side)
    init_cells = len(cells)
    opt = torch.optim.Adam(V.parameters(), lr=lr)
    for p in V.parameters():
        p.requires_grad_(True)

    t0 = time.time()
    history = []
    box_full = sp.box(xstar, inner, inner + rho_target)
    for step in range(steps):
        xL = torch.cat([c[0] for c in cells], 0)
        xU = torch.cat([c[1] for c in cells], 0)
        lb4b = _batched_lb(bm4b, xL, xU, method)
        lb4a = _batched_lb(bm4a, xL, xU, method)
        loss_4b = torch.relu(-lb4b + eps).mean()
        loss_4a = torch.relu(margin_4a - lb4a).mean()      # V - V* >= margin_4a, anti-collapse
        # PGD hinge: drive the empirical worst-case Lie negative fast
        xatk, lie_atk = _pgd_worst_lie(cond4b, sp, xstar, box_full, 256, gen)
        loss_pgd = torch.relu(cond4b.lie(xatk).squeeze(-1) + eps).mean()
        loss = loss_4b + w_4a * loss_4a + w_pgd * loss_pgd
        opt.zero_grad(); loss.backward(); opt.step()

        if (step + 1) % split_every == 0:
            cells = _split_hard(bm4b, sp, cells, margin=0.0, max_cells=max_cells)

        if verbose and (step % 100 == 0 or step == steps - 1):
            with torch.no_grad():
                frac = float((lb4b >= 0).float().mean())
                rec = dict(step=step, cells=len(cells), frac_4b_verified=round(frac, 3),
                           min_lb_4b=round(float(lb4b.min()), 3),
                           min_lb_4a=round(float(lb4a.min()), 3),
                           worst_pgd_lie=round(float(lie_atk.max()), 3),
                           loss=round(float(loss), 4))
            history.append(rec)
            print(rec)

    seconds = round(time.time() - t0, 1)
    return V, dict(seed=seed, slice_dim=sp.slice_dim, active_buses=sp.active_buses,
                   active_dims=sp.active_dims, inner=inner, rho_target=rho_target,
                   init_cells=init_cells, final_cells=len(cells), steps=steps,
                   train_seconds=seconds, method=method,
                   warmstart=ws_info, history=history)


if __name__ == "__main__":
    from data import load_system
    s = load_system()
    xstar = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "results", "equilibrium_seed0.npy"))
    V, h = train(s, xstar, active_buses=[5], seed=0, steps=300,
                 warmstart=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "results", "lyap_seed0.pt"))
    print("final cells", h["final_cells"], "seconds", h["train_seconds"])
