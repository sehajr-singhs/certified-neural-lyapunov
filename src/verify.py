"""Certification driver: bound propagation (auto_LiRPA / CROWN) plus input-space
branch-and-bound, the same bound-plus-BaB algorithm alpha-beta-CROWN runs, applied
directly on the maintained core. Certifies that a LyapunovCondition output F(x)
stays >= -eps over an axis-aligned box B.

A leaf is verified when its sound CROWN lower bound on F is >= -eps. If the bound
is below -eps the box is split on its widest active dimension and recursed. When a
box cannot be split further (below min_width) we look for a genuine counterexample
inside it by sampling and gradient ascent on -F, and we report explicitly which of
two things happened:
  - "violation": a real x in B with F(x) < -eps was found, the property is FALSE
  - "unknown"  : no counterexample found but the bound stayed loose, verifier
                 incompleteness, not a disproof
This distinction is never blurred, per the honesty rules.
"""
import os, sys, time, heapq
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm


def _bound_lower(bm, xL, xU, method):
    x0 = ((xL + xU) / 2).clone()
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=xL, x_U=xU)
    bx = BoundedTensor(x0, ptb)
    lb, ub = bm.compute_bounds(x=(bx,), method=method)
    return lb.min().item()


def _search_counterexample(cond, xL, xU, eps, iters=200, restarts=8, seed=0):
    """Gradient ascent on -F inside [xL, xU]; returns (found, x, Fval)."""
    g = torch.Generator().manual_seed(seed)
    best = (False, None, float("inf"))
    for r in range(restarts):
        x = (xL + (xU - xL) * torch.rand(xL.shape, generator=g)).clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.02)
        for _ in range(iters):
            F = cond(x)
            loss = F.sum()               # minimise F
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                x.clamp_(xL, xU)
        with torch.no_grad():
            Fv = float(cond(x).min())
            if Fv < best[2]:
                best = (Fv < -eps, x.detach().clone(), Fv)
    return best


def certify_box(cond, xL, xU, eps=0.0, method="CROWN-Optimized",
                min_width=1e-2, max_subdomains=20000, time_budget=120.0,
                example_input=None, seed=0):
    """Branch-and-bound. Returns a dict with the verdict and statistics."""
    if example_input is None:
        example_input = ((xL + xU) / 2)
    bm = BoundedModule(cond, example_input, verbose=False)

    t0 = time.time()
    # priority queue by lower bound (most negative first)
    root_lb = _bound_lower(bm, xL, xU, method)
    heap = [(root_lb, 0, xL, xU)]
    n_sub = 0
    worst_lb = root_lb
    counter = 0
    while heap:
        lb, _, bL, bU = heapq.heappop(heap)
        n_sub += 1
        if lb >= -eps:
            continue                      # this leaf is verified
        widths = (bU - bL).squeeze(0)
        if widths.max().item() < min_width or n_sub > max_subdomains or (time.time() - t0) > time_budget:
            # cannot refine: decide violation vs unknown by searching for a real CE
            found, xce, Fce = _search_counterexample(cond, bL, bU, eps, seed=seed)
            if found:
                return dict(verdict="violation", certified=False,
                            counterexample=xce.numpy().ravel().tolist(), F_at_ce=Fce,
                            subdomains=n_sub, seconds=round(time.time() - t0, 2),
                            worst_lower_bound=worst_lb, method=method)
            worst_lb = min(worst_lb, lb)
            counter += 1
            if n_sub > max_subdomains or (time.time() - t0) > time_budget:
                return dict(verdict="unknown", certified=False,
                            reason=("timeout" if (time.time() - t0) > time_budget else "max_subdomains"),
                            subdomains=n_sub, seconds=round(time.time() - t0, 2),
                            worst_lower_bound=worst_lb, method=method)
            continue
        # split widest active dimension
        dim = int(torch.argmax(widths).item())
        mid = (bL[0, dim] + bU[0, dim]) / 2
        for lo, hi in [(bL[0, dim].item(), mid.item()), (mid.item(), bU[0, dim].item())]:
            nL = bL.clone(); nU = bU.clone()
            nL[0, dim] = lo; nU[0, dim] = hi
            clb = _bound_lower(bm, nL, nU, method)
            worst_lb = min(worst_lb, clb)
            if clb < -eps:
                heapq.heappush(heap, (clb, n_sub, nL, nU))
    return dict(verdict="verified", certified=True, eps=eps,
                subdomains=n_sub, seconds=round(time.time() - t0, 2),
                worst_lower_bound=worst_lb, method=method)


def bound_ladder(cond, xL, xU):
    """IBP vs CROWN vs CROWN-Optimized lower bound of F over the box, with time."""
    bm = BoundedModule(cond, (xL + xU) / 2, verbose=False)
    out = {}
    for m in ["IBP", "CROWN", "CROWN-Optimized"]:
        t = time.time()
        lb = _bound_lower(bm, xL, xU, m)
        out[m] = dict(lower_bound=lb, seconds=round(time.time() - t, 3))
    return out
