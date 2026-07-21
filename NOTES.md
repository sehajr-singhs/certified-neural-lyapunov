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
Colab/Linux (Colab is Linux, `pip install dreal` works there).

Update (2026-07-15): numbers produced on WSL2 Ubuntu 24.04, not Colab. The pip
`dreal` wheel exists only up to cp311, so the run used a uv-provisioned Python
3.11 venv; the wheel ships only the bindings, so libibex (from ppa:dreal/dreal,
the 24.04 build) and the CoinOR/nlopt shared libs were extracted from their debs
without sudo and put on LD_LIBRARY_PATH. dReal 4.21.06.2, delta=1e-3. Result
(results/dreal_seed0_rung{1,2}.json, dreal_seed0_control.json): rung 1 two-bus
gate delta-unsat in 0.14s, rung 2 gen-5 slice delta-unsat in 0.45s, both agreeing
with CROWN, and a positive control on the as-trained far region returns delta-sat
in 0.37s, agreeing with CROWN's genuine violation there. Both certified regions
are low-dimensional so dReal is sub-second; it does NOT exceed budget, so the
earlier "rung 2 expected to time out" guess was wrong and is not written anywhere.

## Algorithm 2 (Task 3): controller reproduced, fault numbers not, certificate extends

Reproduced her decentralized monotone stacked-ReLU controller and trained it on a
differentiable rollout with her Lyapunov regularizer relu(Lie + beta(V-V*)). Honest
outcome (results/e6_seed0.json):
  - Cost on a fixed eval set: droop 0.695, RNN-no-Lyap 0.715, RNN-Lyap 0.943. Near
    the equilibrium linear droop is close to LQR-optimal, so the learned controller
    matches it (no-Lyap) or trades cost for stability (Lyap). Her 19% reduction NOT
    reproduced. Suspected reason: 19% comes from the large-excursion post-fault fault
    (lines 1-39 and 2-3 tripped at t=6s), and the shipped data is pre-fault Kron only,
    so we cannot re-derive the post-fault reduced network and modeled the disturbance
    as an initial perturbation, where droop is already near-optimal.
  - Transient: both controllers hold at moderate kicks and both lose sync at a large
    kick (2.5), so the selective gen-9 desync is NOT reproduced, same missing-data
    reason. Reported straight.
  - Re-certification (the point): CEGIS V certifies (4b) to rho=2.0 with the learned
    RNN controller in the closed loop, same as droop. The certificate extends to her
    learned controller, not only the droop law used to train V. This is the result
    that answers her question for the learned policy.
Fixed a real bug first: the three controllers were being compared on different
disturbance batches (shared advancing generator); now a fixed eval set.

## Finalization (2026-07-15): certified rho and the bound ladder

Two numbers were inconsistent across documents and are now resolved, each against
the JSON that produces it.

Certified rho for the gen-5 (4b) CEGIS result is 2.0, not 1.5. The 1.5 was a grid
artifact: E3 and E4 scanned rho only up to 1.5, so they reported the grid cap. E5's
finer sweep (results/e5_seed0.json) and the JacobianOP cross-check
(results/verifier_crosscheck_seed0.json) both find the true boundary is 2.0 with a
genuine counterexample at 2.5. Extended E4 and E3 (and E2 for 4a) grids to 2.5 to
match E5. Verified the CEGIS network is byte-identical after the E4 re-run (md5
unchanged), because the retrain frontier is capped at 1.5 independently of the eval
grid, so no downstream re-run was needed. Across five seeds the (4b) rho is
2.0/2.0/1.5/1.5/2.0 = 1.8 +/- 0.24, so 2.0 is the seed-0 value and the modal value,
reported as such, not as uniform. (4a) certifies to 2.5.

Bound ladder: the old site table showed IBP/CROWN/CROWN-Opt = -129.5/-132.2/-54.8,
CROWN looser than IBP, which looks impossible. Investigated: it is NOT a code bug.
On the wide +/-0.5 box centred at the equilibrium with the as-trained network, plain
backward CROWN is genuinely looser than interval propagation because the Lie graph
multiplies grad V by xdot and relaxing a product over a wide range can beat CROWN's
linear relaxation, while CROWN-Optimized recovers. On 3 of 4 boxes tested the ladder
is monotone. We now report it on the certified CEGIS annulus, where it is monotone
and tells a cleaner story: IBP -132.0 (useless), CROWN -32.5, CROWN-Optimized +2.0,
the only bound that clears zero and certifies. e3 now writes bound_ladder_cegis_annulus.

## Certified training direction (2026-07-20): reading Shi et al. and the feasibility probe

New direction from Prof. Cui: replace post-hoc CEGIS with certified training, differentiating
through the CROWN bound so V is verifiable from the start (arXiv:2411.18235, Shi, Li, Hsieh,
Zhang, "Certified Training with Branch-and-Bound for Lyapunov-stable Neural Control", method
name CT-BaB). Read the PDF (extracted to scratchpad), the load-bearing mechanics are:

1. Differentiable certified bound in the loss. They optimize arg min_theta sigma(g_bar(B)),
   where g_bar(B) is the CROWN UPPER bound on a violation model g, computed by auto_LiRPA via
   linear bound propagation, which is differentiable in the weights because every relaxation and
   propagation step is differentiable. g encodes the whole Lyapunov spec as one scalar:
   g(x) = min( rho - V(x), sigma(decrease) + sum_i sigma(box-exit) ), and g <= 0 is the property.
   The min encodes the implication V(x)<rho -> (decrease AND stays in B). This is Eq. (3)-(5).
2. Overall loss Eq. (5): mean over subregions of [ sigma(g_bar + eps) + lambda sigma(g_PGD + eps) ]
   + L_extra. First term is the certified-bound hinge (the point), second is a PGD hinge that
   clears easy counterexamples fast, L_extra controls the region size.
3. Region size in the objective (Eq. 7, L_extra). They FIX rho=1 and grow the fraction of B whose
   samples satisfy V(x)<rho, penalizing V(x)>rho-eps when that fraction is below rho_ratio. This
   is how "size of certified region" enters the objective without differentiating an implicit
   boundary, and they note it is load-bearing: drop it and the spec is met trivially by collapsing
   the ROA to nothing.
4. Training-time BaB (Sec 2.3). They keep a dynamic dataset D of non-overlapping subregions
   covering B, initialized as a grid with max side length l, and each epoch split only the HARD
   subregions (those with g_bar>0) into two along the dimension that minimizes the resulting loss
   (Eq. 6). This is what keeps BaB tractable in the loop, splitting effort concentrates on the hard
   pieces instead of a uniform fine grid.
5. Training-aware verification. Export the final D as "pre-split" regions to warm-start test-time
   alpha,beta-CROWN. Verification (soundness) still comes from the full BaB verifier, not the
   training bound.
Their headline on 2-D quadrotor output feedback: 11x faster verification and 164x larger verified
ROA vs a CEGIS baseline.

What maps and what deviates for our system:
- Their decrease is discrete-time V(x_{t+1})-V(x_t) <= -kappa V(x_t). Ours is continuous-time
  Lie(x) = grad V . f < 0 and Prop-2 Lie <= -beta(V-V*). Same shape, and graphs.py already
  computes F = -(Lie + beta(V-V*)) as its prop2 forward, so the differentiable spec is already
  built, it just needs to be bounded with grad flowing instead of under no_grad.
- They CO-TRAIN u and V. We train V ONLY against the FIXED saturated linear droop law Cui trains
  against, matching the existing CEGIS setup. Logged as a deliberate deviation: our question is
  whether V can be made verifiable-by-construction, not controller co-design. Stated in the report.
- Certified region metric. To compare apples-to-apples with the CEGIS number (rho=2.0 is a BOX
  SIDE of the gen-5 annulus, not a V-sublevel), we measure achieved certified rho with the SAME
  certify_box bisection CEGIS used, and keep the three-verifier cross-check. Certified training
  only changes how V is produced, not how rho is measured or audited.

Feasibility probe (2026-07-20, the one real risk, now cleared). Built the gen-5 slice box
[0.05,0.55]^2 on dims (4,14), ran auto_LiRPA compute_bounds(method='CROWN') on the analytic
LyapunovCondition mode 4b with V params requires_grad and NO no_grad, then backward through
relu(-lb+0.1). Result: CROWN lb = -0.657 (as-trained net fails there, expected), lb.requires_grad
True, and the gradient reaches net.dense1.weight with norm 7.09 and all 500 entries nonzero. So the
CROWN certified margin over a region IS differentiable to V's weights in THIS exact stack (torch
2.11.0+cpu, auto_LiRPA 0.7.2, py 3.13.9) on CPU. The whole certified-training method is executable
here without a GPU. This is the gate for Task 2 and it passed before any training code was written.

## Warm-start guarantee for the controlled comparison (2026-07-21, per Prof. Cui)

Prof. Cui wants the 2-D slice rerun on the SAME network we already have, not a fresh one, so E7
(certified training) vs E4 (CEGIS) is a controlled comparison with the training METHOD as the only
variable. Both start from the identical as-trained lyap_seed{k}.pt weights, same architecture, same
equilibria; E4 repairs that net against finite counterexamples, E7 refines it by differentiating
through the CROWN bound. If E7 quietly initialized a new random net the comparison would be
meaningless, so this is now enforced in code, not left to convention:

- src/certified_train.py grew `_warmstart_V(system, warmstart, require_warmstart)`. With
  require_warmstart=True (the default, and what e7 always passes) a missing or None checkpoint
  RAISES FileNotFoundError instead of silently building a fresh net, which is what the old
  `if warmstart is not None and os.path.exists(...)` guard did. The load is strict=True, so the
  checkpoint must be the same architecture or load_state_dict itself raises.
- It returns a warm-start fingerprint (abspath, warmstart_loaded=True, init_param_l2 = L2 norm of
  all params right after the load) that train() passes through and e7 copies into
  results/e7_seed{k}.json. So each result proves on its face that V was warm-started from the exact
  checkpoint and refined, never reinitialized. require_warmstart=False stays only as an escape hatch
  for future from-scratch scaling experiments and is never used by the 2-D comparison.

Verified on the seed-0 smoke: warmstart_loaded True, the loaded weights are the lyap_seed0.pt net
(not the LyapunovOnState random init), and training descends from that point.

## E7 driver (2026-07-21): closes the SCALING.md forward-reference

experiments/e7_certified_train.py now exists (SCALING.md referenced it before it was written). Per
seed k it: warm-starts training from lyap_seed{k}.pt, saves lyap_certtrain_seed{k}.pt, then measures
the headline certified rho with the IDENTICAL audited certify_box path E4 used (reused verbatim via
`from e4_cegis import max_certifiable_rho, slice_box, INNER, GI`, CROWN, eps=0, min_width=0.03,
time_budget=30, PGD audit reject < -1e-3, rho grid 0.5..2.5), and Prop-2 beta with the same audited
bisection E3 rung2 used (rho=1.0 annulus, min_width=0.04). The headline comes ONLY from certify_box;
the training-time BaB frac/min_lb is written as a diagnostic and labelled as such. It then runs the
JacobianOP cross-check (analytic vs auto_LiRPA gradient path, plus the lb<=ub soundness scan, reused
from verifier_crosscheck) and the dReal (4b) SMT encoding (reused from dreal_baseline, records
dreal_unavailable on this Windows host, runs on Colab/Linux) on the certified-trained V. Writes
results/e7_seed{k}.json with slice_dim as a field. It refuses slice_dim>2, so the 2-D gate is
enforced in code and nothing above 2-D can run from it.

## E7 results (2026-07-21): certified training beats CEGIS on the same network, same verifier

Ran experiments/e7_certified_train.py --all, five seeds, on the anaconda python (3.13.9, torch
2.11.0+cpu, auto_LiRPA 0.7.2), KMP_DUPLICATE_LIB_OK=TRUE. Numbers, all from the identical audited
certify_box E4/E3 used, not the training-time BaB:

  seed  E7 rho  E7 beta   CEGIS rho  CEGIS beta   jac agree  soundness contra  warmstart
  0     2.5     3.9688    2.0        3.9688       True       0                 True (L2 19.09)
  1     2.5     3.9688    2.0        3.9688       True       0                 True (L2 16.84)
  2     2.5     3.9688    1.5        3.9688       True       0                 True (L2 19.46)
  3     2.5     3.9688    1.5        3.9688       True       0                 True (L2 17.31)
  4     2.5     3.9688    2.0        3.9688       True       0                 True (L2 19.39)

- Certified rho: E7 = 2.5 on EVERY seed vs CEGIS 1.8 +/- 0.245 ({2.0,2.0,1.5,1.5,2.0}, the exact
  spread the CEGIS work reported). 2.5 is the grid ceiling (== rho_target, the trained box side), the
  same convention CEGIS measured on, so E7 certifies the whole measured region on all five seeds where
  CEGIS plateaued below it. The true E7 radius may be larger; measuring past 2.5 needs a larger trained
  box, a separate question flagged for the climb, not answered here.
- Prop-2 beta: E7 = 3.9688 on every seed, identical to CEGIS (both saturate the [0,4] bisection near
  the ceiling on the rho=1.0 annulus). beta is verifier-limited, not method-limited, so parity is
  expected and correct.
- JacobianOP cross-check: analytic rho = 2.5 on all five; JacobianOP (looser) rho = 2.5 on seeds 1-4
  and 2.25 on seed 0, a looser verifier certifying a slightly smaller region, which is expected and is
  NOT a contradiction. agreement True on all five, 0 soundness contradictions across 24 boxes x 5 seeds
  (no case of one verifier's lb exceeding the other's ub). This is the soundness cross-check Cui's brief
  names, and it holds for the certified-trained net exactly as it did for the CEGIS net.
- dReal: encoded for the certified region on all five, recorded dreal_unavailable on this host. Note
  the WSL dreal that produced the committed baseline results (dreal 4.21.6.2, WSL2 Ubuntu, Jul 15) is
  currently broken: the wheel dynamically links libibex.so, libClp.so.1, libClpSolver.so.1 and all
  three PPA-installed libs are now missing from the WSL system (ldd on _dreal_py.so shows "not found").
  Not auto-reinstalling system packages under sudo for one cross-check; the encoding is ready and runs
  on Colab or a restored dReal host. The primary independent verifier (JacobianOP) covers soundness.
- Training cost: steps reduced from 1200 to 400 in configs/certified_train.yaml after the first full
  run showed the warm-started net converges by ~step 100 (frac_4b_verified=1.0, loss=0, cells stable
  at 18) and the remaining 1100 steps were identical no-ops. 400 leaves 4x margin. Wall-clock ~6 min
  per seed, most of it the verification and JacobianOP cross-check, not the training.

Bottom line for Prof. Cui: on the identical warm-started network, controller, seeds, and audited
verifier, certified training certifies a LARGER region than CEGIS (2.5 vs 1.8 mean) at the same
exponential rate, verifiable-by-construction rather than repaired, with the soundness cross-check
intact. The 2-D gate holds. The machinery is dimension-agnostic (SliceSpec, one config knob); the
climb is the next session, per SCALING.md.
