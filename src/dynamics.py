"""Lossy Kron-reduced NE39 swing dynamics as a torch.nn.Module.

Faithful to the continuous-time dynamics in Cui and Zhang's notebook
(the state_transfer*_continuous path). For a state x = [delta(n), omega(n)]:

    delta_dot_i = omega_scale * omega_i                       (omega_scale = 2*pi)
    M_i omega_dot_i = Pm_i - u_i
                      - sum_j F_ij sin(delta_i - delta_j)
                      - sum_j G_ij (cos(delta_i - delta_j) - [i==j])

with F = K cos(gamma) and G = -K sin(gamma). The G cos(.) term is the lossy
conductance coupling that prevents an analytic energy function (Chiang 1989),
which is the whole reason the frequency case needs a learned, certified V.

The module is written so auto_LiRPA can trace it: only elementwise ops, matmuls,
sin, and cos, no python branching on tensor values, no in-place writes.
"""
import numpy as np
import torch
import torch.nn as nn


class SwingDynamics(nn.Module):
    def __init__(self, system):
        super().__init__()
        n = system["n"]
        self.n = n
        self.omega_scale = float(system["omega_scale"])
        self.register_buffer("M", torch.as_tensor(system["M"], dtype=torch.float32))
        self.register_buffer("Pm", torch.as_tensor(system["Pm"], dtype=torch.float32))
        self.register_buffer("F", torch.as_tensor(system["F"], dtype=torch.float32))
        self.register_buffer("G", torch.as_tensor(system["G"], dtype=torch.float32))
        # (cos - I): zero out the i==i self term of the cos coupling
        self.register_buffer("Gdiag_mask", torch.eye(n, dtype=torch.float32))

    def forward(self, x, u):
        """x: (..., 2n) = [delta, omega]; u: (..., n) control. Returns xdot (..., 2n)."""
        n = self.n
        delta = x[..., :n]
        omega = x[..., n:2 * n]
        # pairwise angle differences d[...,i,j] = delta_i - delta_j
        dij = delta.unsqueeze(-1) - delta.unsqueeze(-2)
        sin_term = (self.F * torch.sin(dij)).sum(dim=-1)
        cos_term = (self.G * (torch.cos(dij) - self.Gdiag_mask)).sum(dim=-1)
        omega_dot = (self.Pm - u - sin_term - cos_term) / self.M
        delta_dot = self.omega_scale * omega
        return torch.cat([delta_dot, omega_dot], dim=-1)


# --- independent numpy ground truth, transcribed from her notebook transfer matrices ---
def reference_xdot_numpy(system, x, u):
    """Reimplements her continuous D_f_t via the exact transfer-matrix algebra
    from Lyapunov_Train_ref.Compuate_Dervative_Lya, used only to validate the
    torch module. x: (B, 2n), u: (B, n)."""
    n = system["n"]
    M = system["M"]; Pm = system["Pm"].reshape(1, -1)
    F = system["F"]; G = system["G"]; om = system["omega_scale"]
    B = x.shape[0]

    # her state_transfer matrices (continuous)
    st1 = np.vstack((
        np.hstack((np.zeros((n, n)), np.zeros((n, n)))),
        np.hstack((om * np.eye(n), np.zeros((n, n)))),
    ))
    stF = -((1.0 / M).reshape(n, 1) @ np.ones((1, n))) * F
    stG = -((1.0 / M).reshape(n, 1) @ np.ones((1, n))) * G
    st2 = np.hstack((np.zeros((n, n)), np.eye(n)))
    st3 = np.hstack((np.zeros((1, n)), Pm * (1.0 / M)))
    st4 = np.hstack((np.zeros((n, n)), -np.diag(1.0 / M)))
    sel_delta = np.vstack((np.eye(n), np.zeros((n, n))))

    delta = x @ sel_delta                       # (B, n)
    ones = np.ones((n, n))
    eye = np.eye(n)
    # d[b,i,j] = delta_i - delta_j
    dij = delta[:, :, None] * ones[None] - ones[None] * delta[:, None, :]
    sin_sum = np.sum(np.sin(dij) * stF[None], axis=2)              # (B, n)
    cos_sum = np.sum((np.cos(dij) - eye[None]) * stG[None], axis=2)  # (B, n)
    xdot = x @ st1 + (sin_sum + cos_sum) @ st2 + st3 + u @ st4
    return xdot


if __name__ == "__main__":
    from data import load_system
    sys_ = load_system()
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, size=(4, 20)).astype(np.float64)
    u = rng.uniform(-1, 1, size=(4, 10)).astype(np.float64)
    dyn = SwingDynamics(sys_)
    xt = dyn(torch.tensor(x, dtype=torch.float32), torch.tensor(u, dtype=torch.float32)).detach().numpy()
    xr = reference_xdot_numpy(sys_, x, u)
    print("max abs diff torch vs numpy-reference:", np.max(np.abs(xt - xr)))
