"""Independent Lie-derivative graph using auto_LiRPA's JacobianOP, the route
Prof. Cui's brief names. Where src/graphs.py hand-derives the gradient of the
ELU network analytically and writes it in boundable ops, this module instead asks
auto_LiRPA to expand the gradient nodes itself via JacobianOP, so the two paths
share nothing on the gradient side and agreement between their certified regions
is a real soundness cross-check rather than a self-consistency check.

Important: JacobianOP.apply returns zeros in eager mode, because it is a symbolic
placeholder that auto_LiRPA expands into actual gradient-computation nodes only
during bound propagation. So this module is used ONLY for CROWN bounding. Function
values (PGD counterexample search, dense checks) always come from the analytic
LyapunovCondition in graphs.py, whose forward is validated against autograd to 1e-6
in that file's self-test. The cross-check compares the two verifiers' bounds and
certified regions, not their eager forwards.
"""
import os, sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_LiRPA.jacobian import JacobianOP
from dynamics import SwingDynamics
from controller import LinearDroop
from lyapunov import LyapunovOnState


class LieJacobianCondition(nn.Module):
    """F(x) for a Lyapunov condition, with the Lie derivative built through
    JacobianOP so auto_LiRPA bounds the gradient itself.

    mode "4a"    : F = V(x) - V*                       (no Jacobian, same as analytic)
    mode "4b"    : F = -(grad V . xdot)
    mode "prop2" : F = -(grad V . xdot + beta (V - V*))
    """
    def __init__(self, system, lyap_on_state, controller=None, mode="4b",
                 x_star=None, beta=None):
        super().__init__()
        assert mode in ("4a", "4b", "prop2")
        self.mode = mode
        self.n = system["n"]
        self.dynamics = SwingDynamics(system)
        self.controller = controller if controller is not None else LinearDroop(system)
        self.V = lyap_on_state
        if x_star is None:
            x_star = torch.zeros(1, 2 * self.n)
        self.register_buffer("x_star", torch.as_tensor(x_star, dtype=torch.float32).reshape(1, -1))
        with torch.no_grad():
            self.register_buffer("V_star", self.V(self.x_star).reshape(()))
        self.beta = beta

    def forward(self, x):
        V = self.V(x)                                   # (B, 1)
        if self.mode == "4a":
            return V - self.V_star
        JV = JacobianOP.apply(V, x)                      # (B, 1, 2n) gradient of V wrt x
        JV = JV.reshape(x.shape[0], -1)                 # (B, 2n)
        u = self.controller(x)
        xdot = self.dynamics(x, u)                       # (B, 2n), sin/cos live here
        lie = (JV * xdot).sum(dim=-1, keepdim=True)      # (B, 1)
        if self.mode == "4b":
            return -lie
        assert self.beta is not None
        return -(lie + self.beta * (V - self.V_star))


if __name__ == "__main__":
    # smoke test: can auto_LiRPA build and bound this graph, and does the bound
    # bracket the true Lie derivative computed by the analytic path?
    import numpy as np
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    from data import load_system
    from graphs import LyapunovCondition
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = load_system()
    xstar = torch.tensor(np.load(os.path.join(ROOT, "results", "equilibrium_seed0.npy")),
                         dtype=torch.float32).reshape(1, -1)
    V = LyapunovOnState(s); V.load_state_dict(torch.load(os.path.join(ROOT, "results", "lyap_cegis_seed0.pt")))
    V.eval()
    ana = LyapunovCondition(s, V, LinearDroop(s), mode="4b", x_star=xstar)
    jac = LieJacobianCondition(s, V, LinearDroop(s), mode="4b", x_star=xstar)

    GI, n = 4, s["n"]
    xL = xstar.clone(); xU = xstar.clone()
    xL[0, GI] += 0.05; xU[0, GI] += 0.35; xL[0, GI + n] += 0.05; xU[0, GI + n] += 0.35
    x0 = (xL + xU) / 2

    # true Lie at box center, analytic path (validated vs autograd in graphs.py)
    print("analytic F at center:", float(ana(x0)))

    bm = BoundedModule(jac, x0, verbose=False)
    ptb = PerturbationLpNorm(norm=float("inf"), x_L=xL, x_U=xU)
    bx = BoundedTensor(x0.clone(), ptb)
    lb, ub = bm.compute_bounds(x=(bx,), method="CROWN")
    print("JacobianOP CROWN bound on F over box: [%.4f, %.4f]" % (float(lb), float(ub)))

    # analytic path bound for comparison
    bm2 = BoundedModule(ana, x0, verbose=False)
    lb2, ub2 = bm2.compute_bounds(x=(BoundedTensor(x0.clone(), ptb),), method="CROWN")
    print("analytic    CROWN bound on F over box: [%.4f, %.4f]" % (float(lb2), float(ub2)))
