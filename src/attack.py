"""PGD counterexample search in state space, used by E1 (sampling gap) and by the
CEGIS loop (E4). Maximises the Lie derivative to drive it positive, i.e. finds
states where the trained V violates condition (4b).
"""
import torch


def pgd_find_violations(cond, xstar, delta_r, omega_r, n_starts=2000,
                        steps=100, lr=0.03, seed=0):
    """Maximise Lie = -F(4b) inside the box x* +/- (delta_r, omega_r).
    Returns (frac_violating, best_lie, violating_states)."""
    n = cond.n
    g = torch.Generator().manual_seed(seed)
    lo = xstar.clone(); hi = xstar.clone()
    lo[0, :n] -= delta_r; hi[0, :n] += delta_r
    lo[0, n:] -= omega_r; hi[0, n:] += omega_r
    x = lo + (hi - lo) * torch.rand((n_starts, 2 * n), generator=g)
    x = x.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(steps):
        lie = cond.lie(x)              # want lie > 0
        loss = (-lie).sum()            # maximise lie
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            x.clamp_(lo, hi)
    with torch.no_grad():
        lie = cond.lie(x).squeeze(-1)
        viol = lie > 0
        frac = float(viol.float().mean())
        return frac, float(lie.max()), x[viol].detach()


def random_violation_rate(cond, xstar, delta_r, omega_r, n=200000, seed=0):
    """Uniform random sampling violation rate, the baseline her sampling uses."""
    d = cond.n
    g = torch.Generator().manual_seed(seed)
    lo = xstar.clone(); hi = xstar.clone()
    lo[0, :d] -= delta_r; hi[0, :d] += delta_r
    lo[0, d:] -= omega_r; hi[0, d:] += omega_r
    with torch.no_grad():
        x = lo + (hi - lo) * torch.rand((n, 2 * d), generator=g)
        lie = cond.lie(x).squeeze(-1)
        return float((lie > 0).float().mean()), float(lie.max())
