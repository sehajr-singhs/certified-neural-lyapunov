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

## Status

This is an in-progress research artifact. Current state, honestly:

- [x] Repo forked from her repo, structure laid out, her files preserved.
- [x] PyTorch port of the lossy swing dynamics, saturated droop and stacked-ReLU
      controllers, and the ELU Lyapunov net. Validated against an independent
      numpy transcription of her dynamics to 6.4e-7 and a hand two-bus case to
      1.6e-8 (`tests/test_dynamics.py`). See NOTES.md for the paper-vs-code
      discrepancies this surfaced (the 2*pi angle scaling, the dropped damping
      term, the reduced 18-D coordinates for V).
- [ ] Verifier stack (auto_LiRPA + alpha-beta-CROWN from GitHub main), JacobianOP
      and sin/cos/ELU bounding confirmed.
- [ ] E0 reproduce her Algorithm 1 headline numbers.
- [ ] E1 sampling gap, E2 certify (4a), E3 certify (4b)/Prop-2 staircase,
      E4 CEGIS, E5 boundary.
- [ ] Whitepaper, site, Colab notebook.

Every number that reaches the paper, the site, or this README traces to a JSON a
script wrote. Nothing here is claimed until a run produces it. See NOTES.md.

## Reproduce (foundation, runnable now)

```
pip install -r requirements.txt      # plus the verifier from GitHub, see NOTES.md
export KMP_DUPLICATE_LIB_OK=TRUE      # Anaconda + torch OpenMP workaround
python tests/test_dynamics.py         # port correctness gate
```
