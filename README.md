# Certified Neural Lyapunov Functions for Power System Transient Stability

Formal certification of the Lyapunov conditions in Cui and Zhang's
Lyapunov-Regularized RL for frequency control, using bound propagation with
branch-and-bound (auto_LiRPA + alpha-beta-CROWN). This repo is a fork of
[Wenqi-Cui/Lyapunov-Regularized-RL](https://github.com/Wenqi-Cui/Lyapunov-Regularized-RL)
and builds alongside her work, never on top of it.

Sehaj Singh, with Prof. Wenqi Cui (NYU Tandon ECE), 2026. Simulation only.

## The one claim

A neural Lyapunov function trained against randomly sampled violations can't be
certified, because a certificate quantifies over every point in a region while
sampling only ever visits measure zero. The fix is to close the loop, using the
verifier itself as the sampler and feeding its counterexamples back into training,
so that the same architecture that certifies over nothing becomes certifiable over
a real region. Her paper reports the Lie derivative is positive on about 0.1% of
sampled points after training; a certificate has to drive that to zero over an
entire region, not a finite sample.

## Why frequency and not voltage

The lossy Kron-reduced swing dynamics carry a conductance coupling term
`G_ij cos(delta_i - delta_j)` that has no analytic energy function (Chiang 1989),
which is why a valid Lyapunov function for these dynamics has been an open problem
for decades and why the frequency case is the interesting one. The voltage-control
problem is linear, so its limits certify directly without CROWN.

## What is hers, what is ours

Untouched under `upstream/` (MIT, preserved with attribution):
- `Learn_Lyapunov_Trasient_Stability_v2.ipynb` her TensorFlow 2 notebook
- `IEEE_39bus_Kron.mat`, `Sol_lossy_std_0221.mat` the NE39 data and droop gains
- `RNN.png`, `LICENSE`, `README_upstream.md`

Ours:
- `src/` PyTorch port of her dynamics, controllers, and Lyapunov net, plus the
  verification graphs, verifier driver, PGD attack, and CEGIS loop
- `experiments/` E0 reproduce, E1 sampling gap, E2-E3 certification, E4 CEGIS,
  E5 boundary
- `paper/`, `index.html`, `colab/` the whitepaper, site, and Colab notebook

## Data provenance

The `.mat` files come from her repo unchanged. The Kron reduction to 10 generator
buses is from the pg-sync-models MATLAB toolbox, the NE39 data from Chow's power
system toolbox (paper reference [25]), and the linear droop coefficients in
`Sol_lossy_std_0221.mat` were obtained by MATLAB `fmincon`. We do not regenerate
any of these, we only read them.

## The result, in two numbers

Reproducing her Algorithm 1 gives 99.98% of sampled states satisfying the
decrease condition, in line with her ~99.9%. On that same network a verifier
finds genuine counterexamples to both Lyapunov conditions that 200,000 random
samples miss entirely, and counterexample-guided retraining then certifies the
gen-5 slice (her Fig. 2 geometry, a 2-D projection with the other buses fixed at
equilibrium, not the full state) for (4a) out to rho = 2.5, for (4b) out to a
sublevel radius rho = 2.0 (seed 0; 1.8 +/- 0.24 across five seeds, 2.0 on three
and 1.5 on two), and for
Proposition 2 an exponential rate beta up to 3.97, each audited by an independent
attack and cross-checked against a second, JacobianOP-based verifier with zero
soundness contradictions. The certified region collapses as the state dimension
grows, reported not hidden.

## Status

- [x] Repo forked from her repo, structure laid out, her files preserved.
- [x] PyTorch port of the dynamics, controllers, and ELU Lyapunov net, validated
      against an independent numpy transcription to 6.4e-7 and a hand two-bus case
      to 1.6e-8 (`tests/test_dynamics.py`). Paper-vs-code discrepancies in NOTES.md.
- [x] Verifier stack: auto_LiRPA CROWN + input-space branch-and-bound, sin/cos
      native, ELU via a relu/exp identity, every certificate re-checked by an
      independent PGD audit (`tests/test_verifier_ops.py`, `src/verify.py`).
- [x] E0 reproduce (99.98% decrease, 100% V>V*), E1 sampling gap (0 of 200k vs
      6.1%), E2 certify (4a), E3 certify (4b)/Prop-2 staircase (rungs 1-4),
      E4 CEGIS (rho 0 -> 2.0), E5 boundary.
- [x] Verifier cross-check against auto_LiRPA JacobianOP (0 soundness contradictions),
      Algorithm 2 RNN controller reproduced and re-certified, dReal SMT baseline run
      on Linux (WSL): certifies both the 4-D gate (0.14s) and the gen-5 slice (0.45s),
      agreeing with CROWN, and flags the as-trained far-region violation.
- [x] Whitepaper (`paper/main.pdf`), site (`index.html`), Colab (`colab/certify.ipynb`).

Every number in the paper, the site, or this README traces to a per-seed JSON a
script wrote. See NOTES.md.

## Reproduce

```
pip install -r requirements.txt      # plus the verifier from GitHub, see NOTES.md
pip install --ignore-requires-python git+https://github.com/Verified-Intelligence/auto_LiRPA.git
export KMP_DUPLICATE_LIB_OK=TRUE      # Anaconda + torch OpenMP workaround

make quick     # smoke-test every path end to end, minutes on CPU
make full      # the reported numbers: 5 seeds, every rung
make figures   # regenerate every figure from saved JSON
make paper     # compile the whitepaper PDF
```

The Makefile takes `PYTHON=<interpreter>` if `python` is not the one you want.
The dReal SMT baseline runs on Linux or Colab via `colab/certify.ipynb`.
