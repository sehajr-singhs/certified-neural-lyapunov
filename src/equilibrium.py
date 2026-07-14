"""Find the closed-loop equilibrium x* = (delta*, omega*) of the droop-controlled
lossy swing model by integrating the continuous dynamics to steady state, which
reproduces the way her notebook obtains equilibrium_init (simulate, take the last
state). At x* we require omega* = 0 and the angle equation balanced, so the Lie
derivative is structurally zero there.
"""
import numpy as np
import torch

from dynamics import SwingDynamics
from controller import LinearDroop


def find_equilibrium(system, dt=0.01, steps=200000, tol=1e-9, seed=0):
    dyn = SwingDynamics(system)
    ctrl = LinearDroop(system)
    n = system["n"]
    # her cell-6 initial condition: small positive angle spread, near-zero omega
    rng = np.random.default_rng(seed)
    delta0 = rng.uniform(0.0, 0.3, n)
    omega0 = np.zeros(n)
    x = torch.tensor(np.concatenate([delta0, omega0])[None], dtype=torch.float32)
    with torch.no_grad():
        for i in range(steps):
            u = ctrl(x)
            k1 = dyn(x, u)
            k2 = dyn(x + 0.5 * dt * k1, ctrl(x + 0.5 * dt * k1))
            k3 = dyn(x + 0.5 * dt * k2, ctrl(x + 0.5 * dt * k2))
            k4 = dyn(x + dt * k3, ctrl(x + dt * k3))
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if i % 500 == 0:
                res = dyn(x, ctrl(x)).abs().max().item()
                if res < tol:
                    break
    res = dyn(x, ctrl(x)).abs().max().item()
    return x.detach(), res


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load_system
    s = load_system()
    xstar, res = find_equilibrium(s)
    x = xstar.numpy().ravel()
    n = s["n"]
    print("residual |xdot| at equilibrium:", res)
    print("max |omega*|:", np.abs(x[n:]).max())
    print("delta* :", np.round(x[:n], 4))
    print("omega* :", np.round(x[n:], 6))
    np.save(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "equilibrium.npy"), x)
