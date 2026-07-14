"""Load the IEEE NE39 Kron-reduced lossy-network data and assemble the swing
model constants exactly as in Cui and Zhang's notebook.

Provenance: IEEE_39bus_Kron.mat and Sol_lossy_std_0221.mat are unmodified from
Wenqi-Cui/Lyapunov-Regularized-RL. The Kron reduction to 10 generator buses
comes from the pg-sync-models MATLAB toolbox and the NE39 data from Chow's power
system toolbox (paper reference [25]). The linear droop coefficients in
Sol_lossy_std_0221.mat were obtained by MATLAB fmincon. We do not regenerate any
of these, we only read them.

The one place the code and the paper differ, logged in NOTES.md:
  - Her code integrates delta_dot = 2*pi*omega (omega in Hz), while paper eq (1a)
    writes delta_dot = omega. We keep her convention so E0 reproduces.
  - Her continuous training dynamics drop the explicit -D_i omega_i damping term
    (the controller u supplies damping). D is loaded for completeness but the
    swing dynamics used for Lyapunov certification match her code and omit it.
"""
import os
import numpy as np
import scipy.io as sio

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "upstream")

# per her notebook, cell "RNN for Control": the trained stacked-ReLU saturation
MAX_ACTION = np.array(
    [1.1643361, 5.072242, 4.765774, 4.142611, 4.124881, 4.7717752,
     4.528778, 4.201154, 6.0463214, 1.4594682], dtype=np.float64)

OMEGA_SCALE = 2.0 * np.pi  # her omega_scale, delta_dot = omega_scale * omega


def load_system(data_dir=DATA_DIR):
    """Return a dict of numpy constants for the 10-generator NE39 lossy model."""
    d = sio.loadmat(os.path.join(data_dir, "IEEE_39bus_Kron.mat"))
    kron = d["Kron_39bus"][0, 0]
    omega_R = float(np.asarray(kron["omega_R"]).ravel()[0])
    H = np.asarray(kron["H"], dtype=np.float64).reshape(-1)      # (10,) inertia const
    Damp = np.asarray(kron["D"], dtype=np.float64).reshape(-1)   # (10,) damping
    A = np.asarray(kron["A"], dtype=np.float64).reshape(-1)      # (10,) power injection
    K = np.asarray(kron["K"], dtype=np.float64)                  # (10,10) coupling magnitude
    gamma = np.asarray(kron["gamma"], dtype=np.float64)          # (10,10) loss phase shift

    n = H.shape[0]
    # her transforms, cell "Environment Setup"
    M = H * 2.0 / omega_R * 2.0 * np.pi          # (10,) M_i = H_i/30 for these numbers
    D = Damp / omega_R * 2.0 * np.pi             # (10,) loaded, unused in training dynamics
    F = K * np.cos(gamma)                        # sin-term coefficient  (B_ij-like)
    G = -K * np.sin(gamma)                       # cos-term coefficient  (conductance/loss)
    Pm = A.copy()                                # (10,) mechanical power

    sol = sio.loadmat(os.path.join(data_dir, "Sol_lossy_std_0221.mat"))
    linear_coff = np.asarray(sol["Sol"], dtype=np.float64).reshape(-1)  # (10,) droop gains

    # reference-bus reduction matrix minus_ref_one: state (20,) -> reduced (18,)
    # reduced = [delta_{i}-delta_1 for i=2..10 ; omega_i-omega_1 for i=2..10]
    minus_ref = _minus_ref_one(n)

    return dict(
        n=n, omega_R=omega_R, omega_scale=OMEGA_SCALE,
        M=M, D=D, F=F, G=G, K=K, gamma=gamma, Pm=Pm,
        linear_coff=linear_coff, max_action=MAX_ACTION.copy(),
        minus_ref=minus_ref,
    )


def _minus_ref_one(n):
    """Reproduce env.minus_ref_one: (2n, 2(n-1)) reference-subtraction operator,
    bus 1 (index 0) is the reference. state @ minus_ref -> reduced coordinates."""
    top = np.vstack((
        np.hstack((-np.ones((1, n - 1)), np.zeros((1, n - 1)))),
        np.hstack((np.eye(n - 1), np.zeros((n - 1, n - 1)))),
    ))
    bot = np.vstack((
        np.hstack((np.zeros((1, n - 1)), -np.ones((1, n - 1)))),
        np.hstack((np.zeros((n - 1, n - 1)), np.eye(n - 1))),
    ))
    return np.vstack((top, bot)).astype(np.float64)


if __name__ == "__main__":
    s = load_system()
    for k, v in s.items():
        v = np.asarray(v)
        print(f"{k:12s} shape={v.shape} " + (str(v) if v.size <= 12 else ""))
