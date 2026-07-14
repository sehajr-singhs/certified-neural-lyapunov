"""Neural Lyapunov function V_phi.

Matches Cui and Zhang's Lyapunov_model: a single ELU hidden layer of width 50
followed by a linear read-out to a scalar. ELU is chosen (in her paper and here)
because the Lie derivative needs a differentiable activation, and ELU has a
bounded, smooth derivative that GenBaB can bound.

V is a function of the reference-subtracted coordinates z = R x, where
z = [delta_i - delta_1, omega_i - omega_1] has dimension 2(n-1) = 18 for n = 10.
Working in relative angles removes the rotational symmetry of the swing model
(shifting every angle equally leaves the dynamics unchanged), so the equilibrium
is a proper isolated point in z.
"""
import numpy as np
import torch
import torch.nn as nn


class LyapunovNet(nn.Module):
    def __init__(self, in_dim=18, hidden=50):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, hidden)
        self.act = nn.ELU()
        self.dense2 = nn.Linear(hidden, 1)

    def forward(self, z):
        return self.dense2(self.act(self.dense1(z)))


class ReferenceReduce(nn.Module):
    """z = x @ minus_ref, mapping raw state (..., 2n) to reduced (..., 2(n-1))."""
    def __init__(self, system):
        super().__init__()
        self.register_buffer("R", torch.as_tensor(system["minus_ref"], dtype=torch.float32))

    def forward(self, x):
        return x @ self.R


class LyapunovOnState(nn.Module):
    """V as a function of the raw 20-D state, V(x) = V_phi(R x)."""
    def __init__(self, system, net=None, hidden=50):
        super().__init__()
        self.reduce = ReferenceReduce(system)
        n = system["n"]
        self.net = net if net is not None else LyapunovNet(2 * (n - 1), hidden)

    def forward(self, x):
        return self.net(self.reduce(x))


if __name__ == "__main__":
    from data import load_system
    s = load_system()
    V = LyapunovOnState(s)
    x = torch.zeros(5, 20)
    print("V(x) shape:", V(x).shape, "V at origin-state:", float(V(torch.zeros(1, 20))))
