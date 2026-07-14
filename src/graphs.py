"""Verification graphs: modules whose scalar output F(x) encodes a Lyapunov
condition as "F(x) > 0 over the region B".

  mode "4a"    : F = V(x) - V(x*)                 positive definiteness of V
  mode "4b"    : F = -Lie(x)                       Lie derivative negative
  mode "prop2" : F = -(Lie(x) + beta (V(x)-V*))    exponential decrease, Prop 2

Lie(x) = grad_x V(x) . f(x, u_droop(x)) is the time derivative of V along the
closed-loop lossy swing dynamics.

For her network V(z) = W2 ELU(W1 z + b1) + b2 with z = R x (reduced coordinates),
the gradient is analytic:
    a = W1 z + b1
    ELU'(a) = exp(-relu(-a))          (1 for a>0, exp(a) for a<=0)
    grad_z V = (W2 * ELU'(a)) @ W1
    grad_x V = R @ grad_z V
Every operation here (linear, relu, exp, sin, cos, elementwise mul) has an
auto_LiRPA bound class, so CROWN bounds F(x) soundly without JacobianOP graph
expansion. JacobianOP stays available for deeper nets where no closed form is
convenient; for this one-hidden-layer ELU net the analytic graph is exact and
tighter. See NOTES.md.
"""
import numpy as np
import torch
import torch.nn as nn

from dynamics import SwingDynamics
from controller import LinearDroop
from lyapunov import LyapunovOnState


class LyapunovCondition(nn.Module):
    def __init__(self, system, lyap_on_state, controller=None, mode="4a",
                 x_star=None, beta=None):
        super().__init__()
        assert mode in ("4a", "4b", "prop2")
        self.mode = mode
        self.n = system["n"]
        self.dynamics = SwingDynamics(system)
        self.controller = controller if controller is not None else LinearDroop(system)
        self.V = lyap_on_state                      # LyapunovOnState
        self.register_buffer("R", torch.as_tensor(system["minus_ref"], dtype=torch.float32))
        if x_star is None:
            x_star = torch.zeros(1, 2 * self.n)
        self.register_buffer("x_star", torch.as_tensor(x_star, dtype=torch.float32).reshape(1, -1))
        with torch.no_grad():
            self.register_buffer("V_star", self.V(self.x_star).reshape(()))
        self.beta = beta

    def grad_x_V(self, x):
        """Analytic grad_x V(x), exact for the one-hidden-layer ELU net."""
        net = self.V.net
        W1 = net.dense1.weight            # (H, 18)
        b1 = net.dense1.bias              # (H,)
        W2 = net.dense2.weight            # (1, H)
        z = x @ self.R                    # (..., 18)
        a = z @ W1.t() + b1               # (..., H)
        elu_prime = torch.exp(-torch.relu(-a))            # (..., H)
        g = (W2 * elu_prime)              # (..., H)  (W2 is (1,H), broadcasts)
        grad_z = g @ W1                   # (..., 18)
        grad_x = grad_z @ self.R.t()      # (..., 20)
        return grad_x

    def lie(self, x):
        u = self.controller(x)
        xdot = self.dynamics(x, u)
        return (self.grad_x_V(x) * xdot).sum(dim=-1, keepdim=True)

    def forward(self, x):
        if self.mode == "4a":
            return self.V(x) - self.V_star
        lie = self.lie(x)
        if self.mode == "4b":
            return -lie
        # prop2
        assert self.beta is not None
        return -(lie + self.beta * (self.V(x) - self.V_star))


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load_system
    torch.manual_seed(0)
    s = load_system()
    V = LyapunovOnState(s)
    cond = LyapunovCondition(s, V, mode="4b")
    x = torch.randn(5, 20) * 0.2

    # validate analytic grad_x V against autograd
    xa = x.clone().requires_grad_(True)
    Va = V(xa).sum()
    ga = torch.autograd.grad(Va, xa)[0]
    gm = cond.grad_x_V(x)
    print("analytic vs autograd grad_x V max diff:", (ga - gm).abs().max().item())

    # validate Lie against autograd Lie
    def lie_autograd(xin):
        xin = xin.clone().requires_grad_(True)
        v = V(xin)
        g = torch.autograd.grad(v.sum(), xin, create_graph=False)[0]
        u = LinearDroop(s)(xin)
        xd = SwingDynamics(s)(xin, u)
        return (g * xd).sum(-1, keepdim=True)
    la = lie_autograd(x)
    lm = cond.lie(x)
    print("analytic vs autograd Lie max diff:", (la - lm).abs().max().item())
