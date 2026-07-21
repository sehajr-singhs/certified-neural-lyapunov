# SCALING.md — how to run the dimensional climb from 2-D to N-D

This session proved certified training on the 2-D gen-5 slice and built the whole
pipeline so the climb to higher dimensions is a config change, not a rewrite. The
2-D anchor is measured, not promised: E7 certifies (4b) to rho = 2.5 (the grid
ceiling) on all five seeds, beating the CEGIS 1.8 +/- 0.24 on the identical
warm-started network and audited verifier, at Prop-2 beta = 3.97, JacobianOP
cross-check clean (see NOTES.md, "E7 results"). This file is the checklist for the
later climb. It is written for future-you, who will run 4-D, 6-D, and toward the
full 20-D in a separate session. Nothing above 2-D was run here.

## The one thing you change

Edit `configs/certified_train.yaml`, the `slice.active_buses` list. It holds
1-indexed generator numbers whose (delta, omega) coordinates vary; every other bus
stays pinned at the equilibrium.

```
active_buses: [5]        # 2-D, this session's result (gen-5 slice, Fig. 2)
active_buses: [5, 9]     # 4-D, gens 5 and 9 vary together
active_buses: [1, 5, 9]  # 6-D
active_buses: [1,2,...,10]  # full 20-D
```

`slice_dim = 2 * len(active_buses)`. That is the whole interface. You do not touch
any `.py` file to raise the dimension.

## Why nothing else needs editing (the machinery that already generalizes)

- `src/slice.py` `SliceSpec` turns `active_buses` into the raw-state dimensions that
  vary (bus b -> delta index b-1 and omega index b-1+n), and builds the box, the
  initial branch-and-bound grid, the sampler, and the embed/extract maps
  programmatically from `slice_dim`. There is no literal `2` anywhere in it. Verified:
  `SliceSpec(s,[5]).active_dims == [4,14]`, `[5,9] -> [4,14,8,18]`, `[1,5,9] ->
  [0,10,4,14,8,18]`.
- `src/certified_train.py` builds the initial dataset with `sp.grid(...)`, bounds all
  subregions in one batched `compute_bounds`, and the training-time BaB `_split_hard`
  iterates over `sp.active_dims`, so it splits along whichever of the N active
  dimensions most raises the child bound. The loss, the PGD hinge, and the logging
  read `sp.slice_dim`, they do not assume 2.
- `src/verify.py` `certify_box` splits on the widest active dimension and leaves the
  zero-width pinned dimensions untouched, so it already certifies an N-D box. It was
  built dimension-agnostic in the CEGIS work and needs no change.
- `experiments/e7_certified_train.py` writes `slice_dim` as a JSON field and measures
  the certified region with the same `certify_box` bisection at every dimension.

## What will actually get hard as the dimension rises (and the knobs for it)

The wall is branch-and-bound cost, which grows roughly exponentially in `slice_dim`
because each added varying coordinate multiplies the number of subregions a box
splits into. This is the whole reason certified training exists, so expect to spend
the climb tuning these, all in the yaml:

- `bab.init_max_side` down and `bab.max_cells` up as `slice_dim` grows, so the
  initial grid resolves the box and the dynamic split has room to work. Watch
  `final_cells` in the JSON; if it pins at `max_cells` with `frac_4b_verified < 1`,
  raise the cap.
- `train.steps` up; higher dimensions need more certified-training steps to drive the
  worst subregion bound non-negative.
- `verify.min_width` and `verify.time_budget_s`: the test-time certifier will need a
  finer `min_width` and a larger budget, and past roughly 4-D you will want the real
  alpha,beta-CROWN with activation splitting rather than the input-space BaB in
  `certify_box`, plus the "pre-split" warm start from the training subregions (Shi et
  al. Sec. 2.3, training-aware verification). The hook for that is exporting the final
  `cells` from `train(...)`; it is returned in the history so it is already available.
- `train.method`: training stays on plain differentiable `CROWN` (linear propagation
  is what carries the gradient). Do NOT switch the training bound to CROWN-Optimized;
  its inner alpha optimization is not what you want to backprop through. CROWN-Optimized
  stays on the verification side only.

## Cross-check and honesty, unchanged at every dimension

Keep the three-verifier cross-check (`experiments/verifier_crosscheck.py`, analytic
CROWN vs JacobianOP; `experiments/dreal_baseline.py` on Linux) running at each new
dimension. A certified region that is not cross-checked is not trusted. Report every
failure as either a genuine violation (a real counterexample) or verifier
incompleteness (a loose bound), never blurred, and never let a slice result read as a
full-state claim. When the certified region shrinks as the dimension rises, which the
CEGIS work already saw, report the shrink, do not hide it.

## Suggested rung order for the climb

1. 4-D, `active_buses: [5, 9]`. Two coupled generators. First real test of the
   training-time BaB doing work along more than two dimensions.
2. 6-D, `[1, 5, 9]`, including the reference bus.
3. Push toward 10-D and eventually the full 20-D. Somewhere in here the input-space
   `certify_box` stops being enough and you move to alpha,beta-CROWN with the pre-split
   warm start. That switch, not the training, is the expected bottleneck.
