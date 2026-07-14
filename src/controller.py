"""Controllers for the frequency-control problem.

LinearDroop is the saturated linear droop law that Cui and Zhang use when
training the Lyapunov function (Lyapunov_Train_ref uses self.linear_coff), so it
is the controller inside the closed loop we certify. It is piecewise linear, so
CROWN handles it exactly.

StackedReLU is her decentralised monotone control network (Algorithm 2),
u_i(omega_i) = sum_k s_ik relu(omega_i + b_ik) + z_ik relu(-omega_i + c_ik),
engineered monotone increasing through the origin. It is her learned controller;
we port it for E0's cost comparison, not for the Lyapunov certificate.

Both saturate to +/- max_action via the same relu identity she uses, which keeps
the whole controller ReLU-representable.
"""
import numpy as np
import torch
import torch.nn as nn


def _saturate(a, max_action):
    """clip(a, -max_action, +max_action) written as her relu identity."""
    return max_action - torch.relu(max_action - a) + torch.relu(-max_action - a)


class LinearDroop(nn.Module):
    def __init__(self, system):
        super().__init__()
        n = system["n"]
        self.n = n
        self.register_buffer("coff", torch.as_tensor(system["linear_coff"], dtype=torch.float32))
        self.register_buffer("max_action", torch.as_tensor(system["max_action"], dtype=torch.float32))

    def forward(self, x):
        omega = x[..., self.n:2 * self.n]
        return _saturate(self.coff * omega, self.max_action)


class StackedReLU(nn.Module):
    """Decentralised monotone controller. Parameterised so it is monotone
    increasing through the origin by construction, matching her MinimalRNNCell:
    weights are squared (nonnegative) and combined through her band recover
    matrices. Here we expose it as a clean stacked-ReLU with the same shape."""
    def __init__(self, system, internal_units=20):
        super().__init__()
        n = system["n"]
        self.n = n
        self.m = internal_units
        self.register_buffer("max_action", torch.as_tensor(system["max_action"], dtype=torch.float32))
        # her band-recover matrices make the stacked relus a monotone piecewise line
        w_recover = torch.tril(-torch.ones(self.m, self.m), diagonal=1) + 2 * torch.eye(self.m)
        b_recover = torch.triu(torch.ones(self.m, self.m), diagonal=0) - torch.eye(self.m)
        self.register_buffer("w_recover", w_recover)
        self.register_buffer("b_recover", b_recover)
        self.w_plus0 = nn.Parameter(0.1 * torch.rand(n, self.m))
        self.b_plus0 = nn.Parameter(0.1 * torch.rand(n, self.m))
        self.w_minus0 = nn.Parameter(0.1 * torch.rand(n, self.m))
        self.b_minus0 = nn.Parameter(0.1 * torch.rand(n, self.m))

    def forward(self, x):
        omega = x[..., self.n:2 * self.n]              # (..., n)
        w_plus = torch.square(self.w_plus0) @ self.w_recover
        b_plus = (-torch.square(self.b_plus0)) @ self.b_recover
        w_minus = (-torch.square(self.w_minus0)) @ self.w_recover
        b_minus = (-torch.square(self.b_minus0)) @ self.b_recover
        o = omega.unsqueeze(-1)                          # (..., n, 1)
        nl_plus = (torch.relu(o + b_plus) * w_plus).sum(dim=-1)
        nl_minus = (torch.relu(-o + b_minus) * w_minus).sum(dim=-1)
        return _saturate(nl_plus + nl_minus, self.max_action)


if __name__ == "__main__":
    from data import load_system
    s = load_system()
    x = torch.zeros(3, 20)
    x[:, 10:] = torch.linspace(-0.5, 0.5, 10)
    print("droop u:", LinearDroop(s)(x)[0])
    print("stackedReLU u:", StackedReLU(s)(x)[0])
