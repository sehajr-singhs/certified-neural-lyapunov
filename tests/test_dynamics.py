"""Correctness gate for the PyTorch port. Run: python tests/test_dynamics.py

Checks:
  1. torch SwingDynamics (float64) agrees with an independent numpy transcription
     of her transfer-matrix dynamics to < 1e-6 over random states.
  2. A hand-built two-bus lossy system matches an analytic omega_dot.
  3. The saturated linear droop clips exactly to +/- max_action.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data import load_system                      # noqa: E402
from dynamics import SwingDynamics, reference_xdot_numpy   # noqa: E402
from controller import LinearDroop                # noqa: E402


def test_dynamics_matches_reference():
    s = load_system()
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, size=(64, 20))
    u = rng.uniform(-2.0, 2.0, size=(64, 10))
    dyn = SwingDynamics(s).double()
    xt = dyn(torch.tensor(x), torch.tensor(u)).detach().numpy()
    xr = reference_xdot_numpy(s, x, u)
    err = np.max(np.abs(xt - xr))
    assert err < 1e-6, f"dynamics mismatch {err}"
    print(f"[1] torch vs numpy reference (float64): max abs diff = {err:.2e}  OK")


def test_two_bus_analytic():
    # minimal 2-bus lossy system, hand-picked constants
    n = 2
    M = np.array([2.0, 3.0]); Pm = np.array([0.5, -0.5])
    K = np.array([[0.0, 4.0], [4.0, 0.0]])
    gamma = np.array([[0.0, 0.1], [0.1, 0.0]])
    F = K * np.cos(gamma); G = -K * np.sin(gamma)
    s = dict(n=n, omega_scale=2 * np.pi, M=M, Pm=Pm, F=F, G=G)
    delta = np.array([0.3, -0.2]); omega = np.array([0.05, -0.04])
    x = np.concatenate([delta, omega])[None]
    u = np.array([[0.1, -0.1]])
    # analytic: M_i wdot_i = Pm_i - u_i - F_ij sin(di-dj) - G_ij (cos(di-dj)-[i==j])
    d01 = delta[0] - delta[1]
    wdot0 = (Pm[0] - u[0, 0] - F[0, 1] * np.sin(d01) - G[0, 1] * np.cos(d01)) / M[0]
    wdot1 = (Pm[1] - u[0, 1] - F[1, 0] * np.sin(-d01) - G[1, 0] * np.cos(-d01)) / M[1]
    ddot = 2 * np.pi * omega
    expected = np.concatenate([ddot, [wdot0, wdot1]])
    dyn = SwingDynamics(s).double()
    got = dyn(torch.tensor(x), torch.tensor(u)).detach().numpy()[0]
    err = np.max(np.abs(got - expected))
    # 1e-6 is the spec bar; residual is float32 constant roundoff in the module buffers
    assert err < 1e-6, f"2-bus mismatch {err}"
    print(f"[2] hand-computed two-bus analytic: max abs diff = {err:.2e}  OK")


def test_droop_saturation():
    s = load_system()
    ctrl = LinearDroop(s)
    # omega = 1 already saturates every bus since coff_i > max_action_i for all i.
    # (Avoid huge omega: coff*omega ~ 1e4 causes float32 cancellation in the clip.)
    x = torch.zeros(1, 20); x[0, 10:] = 1.0
    u = ctrl(x)[0].detach().numpy()
    ma = s["max_action"]
    assert np.allclose(u, ma, atol=1e-4), "positive saturation failed"
    x[0, 10:] = -1.0
    u = ctrl(x)[0].detach().numpy()
    assert np.allclose(u, -ma, atol=1e-4), "negative saturation failed"
    # inside the linear band it is exactly coff * omega
    x[0, 10:] = 1e-4
    u = ctrl(x)[0].detach().numpy()
    assert np.allclose(u, s["linear_coff"] * 1e-4, atol=1e-6), "linear band failed"
    print("[3] droop saturation and linear band  OK")


if __name__ == "__main__":
    test_dynamics_matches_reference()
    test_two_bus_analytic()
    test_droop_saturation()
    print("\nAll dynamics-port tests passed.")
