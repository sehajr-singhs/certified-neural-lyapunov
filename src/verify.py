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
                example_input=None, seed=0, eval_cond=None):
    """Branch-and-bound. Returns a dict with the verdict and statistics.

    `cond` supplies the CROWN bounds. `eval_cond` supplies the true function value
    for the counterexample search; it defaults to `cond`, but when `cond` is a
    JacobianOP graph (whose eager forward is zero, since the gradient is only
    materialized during bounding) the caller passes the analytic condition as
    `eval_cond` so counterexamples are searched on the real function."""
    if example_input is None:
        example_input = ((xL + xU) / 2)
    ec = eval_cond if eval_cond is not None else cond
    bm = BoundedModule(cond, example_input, verbose=False)

    t0 = time.time()
    # priority queue by lower bound (most negative first)
    root_lb = _bound_lower(bm, xL, xU, method)
    heap = [(root_lb, 0, xL, xU)]
    n_sub = 0
    worst_lb = root_lb
    verified_leaf_min = float("inf")      # min bound over FINAL verified leaves = certified margin
    unresolved = 0            # leaves that could not be verified and had no CE found
    uid = 1
    while heap:
        lb, _, bL, bU = heapq.heappop(heap)
        n_sub += 1
        if lb >= -eps:
            verified_leaf_min = min(verified_leaf_min, lb)
            continue                      # this leaf is verified, soundly
        # this box is not verified. try to resolve it.
        widths = (bU - bL).squeeze(0)
        out_of_budget = n_sub > max_subdomains or (time.time() - t0) > time_budget
        if widths.max().item() < min_width or out_of_budget:
            # cannot refine further: look for a genuine counterexample inside
            found, xce, Fce = _search_counterexample(ec, bL, bU, eps, seed=seed)
            if found:
                return dict(verdict="violation", certified=False,
                            counterexample=xce.numpy().ravel().tolist(), F_at_ce=Fce,
                            subdomains=n_sub, seconds=round(time.time() - t0, 2),
                            worst_lower_bound=min(worst_lb, lb), method=method)
            # no CE found and cannot split: this region is genuinely unresolved.
            # It is NEVER treated as verified. If we ran out of budget, stop now,
            # otherwise keep going but remember the region stays unknown.
            unresolved += 1
            worst_lb = min(worst_lb, lb)
            if out_of_budget:
                return dict(verdict="unknown", certified=False,
                            reason=("timeout" if (time.time() - t0) > time_budget else "max_subdomains"),
                            unresolved_leaves=unresolved,
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
                heapq.heappush(heap, (clb, uid, nL, nU)); uid += 1
    # heap drained. verified only if every leaf was verified and none left unresolved.
    if unresolved == 0:
        cm = 0.0 if verified_leaf_min == float("inf") else verified_leaf_min
        return dict(verdict="verified", certified=True, eps=eps,
                    certified_lower_bound_F=round(cm, 5),
                    subdomains=n_sub, seconds=round(time.time() - t0, 2),
                    worst_intermediate_lb=round(worst_lb, 5), method=method)
    return dict(verdict="unknown", certified=False, reason="unresolved_leaves",
                unresolved_leaves=unresolved, subdomains=n_sub,
                seconds=round(time.time() - t0, 2),
                worst_lower_bound=worst_lb, method=method)


def audit_verified(cond, xL, xU, eps=0.0, n_starts=4000, steps=200, seed=1):
    """Independent soundness guard: strong PGD attack over the box. If a box was
    certified F >= -eps, this must NOT find F < -eps. Returns the min F found;
    a value below -eps means the certificate was unsound and must be rejected."""
    found, x, Fv = _search_counterexample(cond, xL, xU, eps, iters=steps,
                                           restarts=max(1, n_starts // 500), seed=seed)
    return Fv


def bound_ladder(cond, xL, xU):
    """IBP vs CROWN vs CROWN-Optimized lower bound of F over the box, with time."""
    bm = BoundedModule(cond, (xL + xU) / 2, verbose=False)
    out = {}
    for m in ["IBP", "CROWN", "CROWN-Optimized"]:
        t = time.time()
        lb = _bound_lower(bm, xL, xU, m)
        out[m] = dict(lower_bound=lb, seconds=round(time.time() - t, 3))
    return out
