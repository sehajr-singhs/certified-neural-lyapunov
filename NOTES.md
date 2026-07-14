# NOTES, running log of discrepancies, pins, and dead ends

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
  `JacobianOP` operator and GenBaB non-ReLU (sin, cos) branching we depend on are
  NOT on PyPI, so `auto_LiRPA` must be installed from GitHub `main`. Installing the
  PyPI wheel would silently give a verifier that cannot bound the Lie derivative.
  Recorded so nobody "fixes" requirements.txt back to pip.
- GitHub `main` `auto_LiRPA` (installs as version 0.7.2) pins `python ~=3.11.0`,
  but this host is 3.13.9. Installed with `--ignore-requires-python`; the package
  runs correctly on 3.13 (imports, JacobianOP, and CROWN bounds all work). The pin
  is conservative, not a real 3.13 incompatibility.
- Installing `auto_LiRPA` 0.7.2 downgraded torch from 2.13.0 to 2.11.0+cpu (its
  dependency pin). The dynamics port tests were re-run on 2.11.0 and still pass.
- IBM CROWN repo (IBM/CROWN-Robustness-Certification): confirmed dead on a modern
  stack exactly as the spec predicted. It imports
  `from tensorflow.contrib.keras.api.keras.models import Sequential`, and
  `tensorflow.contrib` was removed in TF 2.0, so it needs TF 1.x. It is also wired
  for image classifiers (`setup_mnist`, `setup_cifar`, MNIST/CIFAR data loaders),
  not control systems. Error class: `ModuleNotFoundError` at import against a
  TF1-only `tensorflow.contrib` API. We did not force a TF1 environment, we moved
  to the maintained PyTorch auto_LiRPA as planned.

## ELU is not natively supported, and the fix

- This `auto_LiRPA` build registers `BoundSin` and `BoundCos` (trig works) plus
  `BoundTanh`, `BoundSigmoid`, `BoundGelu`, `BoundSoftplus`, `BoundAtan`,
  `BoundExp`, but there is NO `BoundElu`. Feeding a torch `ELU` gives
  `NotImplementedError: unsupported operation onnx::Elu`. Her Lyapunov net uses
  ELU, so this had to be resolved before any certification.
- Fix, keeping her exact ELU function: express it in natively-boundable ops,
  `ELU(x) = relu(x) - relu(1 - exp(-relu(-x)))`. The exp argument is confined to
  [x_min, 0] so it cannot overflow, and every op (relu, exp, sub) has a bound
  class. Verified: this identity matches `torch.nn.ELU` to 6e-8, and CROWN /
  CROWN-Optimized produce SOUND bounds on it (checked against dense sampling),
  with CROWN-Optimized recovering the exact interval on the test case. sin+cos
  bounding is likewise sound and tight (CROWN-Optimized upper bound 1.647 vs true
  1.644). Capability gate script: scratchpad/cap_test.py, promoted to
  tests/test_verifier_ops.py.
- Consequence for the port: `src/lyapunov.py` uses this ELU identity so the same
  V is both trained in PyTorch and bounded by CROWN without re-expressing anything
  at verification time. tanh is available as a fallback activation if the relu/exp
  ELU ever proves too loose under BaB, noted but not currently needed.

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

## Verification method actually used (and the deviation from the brief)

The brief names JacobianOP + GenBaB + alpha-beta-CROWN. We use auto_LiRPA's CROWN
bounds on a computation graph in which the gradient of the one-hidden-layer ELU
net is written analytically in boundable ops (linear, relu, exp, elementwise mul),
times the sin/cos dynamics, plus an input-space branch-and-bound in
src/verify.py. This is sound (every op has a bound class, and every certificate is
re-checked by an independent multi-start PGD attack in audit_verified; a
certificate is rejected if the attack finds any violation), and for a shallow net
the analytic gradient graph is exact and tighter than JacobianOP graph expansion.
The Jacobian route stays the right tool for deeper nets. This deviation is stated
plainly in the paper's Verification Formulation section, it is not hidden.

## (4a) is genuinely violated on the as-trained net, not just (4b)

E2 finding worth flagging: the as-trained V is not positive definite at the
equilibrium. In a 0.02 ball around x*, 34% of directions have V < V*, and CROWN
finds a genuine (4a) counterexample on the gen-5 annulus (V - V* = -0.008). E0
reported V > V* on 100% of its training samples, so this is the same sampling gap
as (4b), showing up in the condition that was supposed to be trivial. The CEGIS
net certifies (4a) there with margin +0.097.

## Two-bus reduced-coordinate equilibrium trap (dead end, then fix)

First two-bus gate failed (certified rho = 0). Cause: V(x) = net(R x) sees only
the bus differences z = [delta_2 - delta_1, omega_2 - omega_1], so offsetting all
four raw coordinates by [INNER, INNER+rho] sweeps through z = 0 (equal offsets),
which is the equilibrium, where the Lie derivative is structurally 0 and no region
can certify it < 0. The margin loss relu(lie + 0.3) then fought an impossible
target near z = 0 and collapsed V. Fix: pin bus 1 (the reference) at x* and vary
only bus 2, exactly as the gen-5 slice pins every bus but gen 5. After the fix the
bus-2 annulus has dense max Lie -0.41 and certifies.

## Full 20-D (rung 3) failure is genuine, not a verifier timeout

The slice-hardened CEGIS net genuinely violates (4b) in the full 20-D state even
at radius 0.01 (a real counterexample, Lie = +4.52), because CEGIS was scoped to
the gen-5 slice and off-slice behavior is unconstrained, so retraining on the
slice can worsen it elsewhere. Reported as a genuine violation, not incompleteness.
Full-state CEGIS / certified training is the clear next step.

## Verifier cross-check (Task 1): analytic vs JacobianOP, they AGREE

Built a second, independent certification path (src/graphs_jacobian.py) that forms
the Lie derivative through auto_LiRPA's JacobianOP, which expands the gradient nodes
itself and shares nothing with the analytic gradient of src/graphs.py. auto_LiRPA
expands the JacobianOP node through the relu/exp ELU graph fine. Cross-check on seed
0 (results/verifier_crosscheck_seed0.json):
  - Soundness scan, 24 boxes: both paths bound the same F, so a sound lb from one
    must never exceed a sound ub from the other. Contradictions = 0, worst gap 0.0.
    This is the check a PGD audit cannot do, because an unsound verifier certifies
    false properties and PGD failing to find a CE is what that looks like inside.
  - Prop-2 beta: analytic 3.9375, JacobianOP 3.9375 (identical).
  - E5 far region: both return "violation".
  - Certified rho (4b): analytic 2.0 vs JacobianOP 1.25. The analytic path is
    tighter (lb -32.5 vs -125.0 on the fixed box), so it certifies a larger region.
    Looser JacobianOP certifying a smaller region is expected, not a disagreement,
    and the soundness scan rules out the analytic path certifying anything false.
Conclusion: the hand-rolled analytic path is validated as sound AND tighter, so it
stays as the specialized implementation with JacobianOP as the standard cross-check.
JacobianOP eager forward returns zeros (symbolic placeholder), so counterexample
search always uses the analytic function via certify_box(eval_cond=...).

## dReal (Task 2) does not build on Windows

pip has only an sdist for dreal (dreal-4.20.4.1.tar.gz), no Windows wheel, and the
source build needs Bazel + IBEX which do not build on Windows, same class of dead
end as the IBM CROWN repo. The dReal baseline encoding is provided runnable on
Colab/Linux (Colab is Linux, `pip install dreal` works there). Numbers to be
produced on Colab.
