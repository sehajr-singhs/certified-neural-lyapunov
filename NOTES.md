# NOTES — running log of discrepancies, pins, and dead ends

Kept for the meeting. Everything here is something the code or the tooling forced,
not something the plan predicted. Dead ends are logged as carefully as successes.

## Environment (verified 2026-07-13)

- Host: Windows 11, CPU only, no CUDA GPU. All verification timings in this repo
  are CPU wall-clock and will be reported as such, because the spec says Colab and
  the reader may run CPU-only.
- Python: Anaconda 3.13.9. The bare `python` alias on this machine is the broken
  Windows Store shim, so the interpreter is invoked by full path
  `C:/Users/sehaj/anaconda3/python.exe`. The Makefile uses `python` and expects a
  real interpreter on PATH.
- torch: 2.13.0+cpu installed from the pytorch.org cpu index. numpy 2.3.5,
  scipy 1.16.3, matplotlib 3.10.6 were already present.
- OpenMP clash: importing torch alongside Anaconda's MKL raises
  `OMP: Error #15: libiomp5md.dll already initialized`. Every entry point sets
  `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch. This is the documented
  Anaconda+torch workaround on Windows, it does not affect numerics here because
  we never run heavy multithreaded BLAS inside the verifier loop.

## Verifier tooling

- PyPI `auto_LiRPA` is pinned at 0.3 (only 0.2 and 0.3 exist on the index). The
  `JacobianOP` operator and GenBaB non-ReLU (sin, cos, ELU) branching we depend on
  landed in the 12/2025 GitHub release and are NOT on PyPI, so both `auto_LiRPA`
  and `alpha-beta-CROWN` must be installed from GitHub `main`, not pip. Installing
  the PyPI wheel would silently give a verifier that cannot bound the Lie
  derivative at all. Recorded here so nobody "fixes" requirements.txt back to pip.
- IBM CROWN repo attempt: <pending, will record the exact error class>.

## Discrepancies between her paper and her code (paper is ground truth for notation)

1. Angle dynamics scaling. Paper eq (1a) writes `delta_dot_i = omega_i`. Her code
   integrates `delta_dot_i = 2*pi*omega_i` (her `omega_scale = 2*pi`), because her
   omega is a per-unit frequency in Hz and the angle is in rad, so the 2*pi is the
   Hz-to-rad/s conversion. We keep her convention so E0 reproduces her numbers, and
   the paper will state the 2*pi explicitly.

2. Damping term. Paper eq (1b) has `- D_i omega_i` in the swing equation. Her
   continuous training dynamics (`state_transfer1_continuous` bottom-right block is
   zero) drop the explicit damping and let the controller u supply all damping. We
   match her code for the certified closed loop and load D only for reference. The
   paper's limitations section will note the certificate is for the un-damped-plus-
   controller model she actually trains against.

3. Controller inside the certified loop. V_phi is trained with the saturated LINEAR
   droop law (`Lyapunov_Train_ref` uses `self.linear_coff`), not the stacked-ReLU
   RNN controller (that is Algorithm 2, a separate learned controller regularised by
   the same V). So the Lie-derivative certificate is for the closed loop
   `f(x, u_droop(x))`. The droop law is piecewise linear, which CROWN handles
   exactly. Logged because it would be wrong to certify V against the RNN loop it
   was not trained on.

4. Reduced coordinates. V is a function of the reference-subtracted 18-D state
   `z = [delta_i - delta_1, omega_i - omega_1]` (her `minus_ref_one`), not the raw
   20-D state. Working in relative angles removes the swing model's rotational
   symmetry so the equilibrium is isolated. The verification graph reduces the raw
   box to z before V, and the Lie derivative uses the reduced dynamics `xdot @ R`.

## Data provenance (unmodified, under upstream/)

- `IEEE_39bus_Kron.mat`, `Sol_lossy_std_0221.mat` from
  Wenqi-Cui/Lyapunov-Regularized-RL. Kron reduction to 10 generator buses via the
  pg-sync-models MATLAB toolbox, NE39 data from Chow's power system toolbox
  (paper ref [25]), droop coefficients from MATLAB `fmincon`. We only read them.
- Decoded constants: n=10, omega_R=376.991 rad/s (2*pi*60), M_i = H_i/30,
  D_i = 5/6 (unused in training dynamics), Pm = A field, F = K cos(gamma),
  G = -K sin(gamma). The lossy conductance coupling is the G cos(.) term.

## Port validation (tests/test_dynamics.py, all pass)

- torch SwingDynamics vs an independent numpy transcription of her transfer-matrix
  dynamics: max abs diff 6.4e-7 over 64 random states, at the float32 constant
  precision of the module buffers (spec bar is 1e-6).
- hand-computed two-bus lossy system: max abs diff 1.6e-8.
- saturated droop clips to +/- max_action and is exactly coff*omega in the band.
  Note: driving omega to ~100 breaks the clip identity in float32 by catastrophic
  cancellation (coff*omega ~ 1e4), so tests saturate at omega=1, which is already
  past saturation since coff_i > max_action_i for every bus.
