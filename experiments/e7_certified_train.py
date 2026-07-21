"""E7 - certified training (CT-BaB) on the gen-5 slice, measured head-to-head
against the CEGIS result on the identical starting network.

This closes the SCALING.md forward-reference. It is the certified-training analogue
of E4 (CEGIS): same gen-5 projected slice, same fixed LinearDroop controller, same
five seeds, same equilibria, and the headline numbers come out of the EXACT same
final verifier. The only variable is the method that produced V:

  E4  : take the as-trained lyap_seed{k}.pt, repair it against the finite
        counterexamples the verifier returns (CEGIS), save lyap_cegis_seed{k}.pt.
  E7  : take the SAME as-trained lyap_seed{k}.pt and REFINE it by differentiating
        through the CROWN bound (certified training), save lyap_certtrain_seed{k}.pt.

Prof. Cui's constraint, enforced here: certified training warm-starts from the exact
lyap_seed{k}.pt weights and never spins up a fresh random net, so E7 vs E4 is a
controlled comparison on one starting network with method as the only difference.
src/certified_train.py::_warmstart_V makes that mandatory (require_warmstart=True,
strict load) and returns a weight fingerprint that we copy into the result JSON.

The headline both methods are judged on - certified rho and Prop-2 beta - is measured
with the IDENTICAL src/verify.py::certify_box branch-and-bound bisection E4 used, at
the same eps, min_width, time budget, and rho grid, gated by the same independent PGD
audit. The training-time BaB is coarser by design and its numbers are logged only as
diagnostics, never as the certificate. Every certified region is then cross-checked
against the JacobianOP verifier (an independent gradient path) and encoded for dReal
(the SMT baseline), exactly as the CEGIS result was.

Holds at the 2-D gate: it refuses any slice above slice_dim=2. Nothing above 2-D runs
here. Writes results/e7_seed{k}.json with slice_dim as a field.

Run one seed:   python experiments/e7_certified_train.py --seed 0
Run all five:   python experiments/e7_certified_train.py --all
"""
import os, sys, json, time, argparse
import numpy as np, torch, yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from data import load_system
from lyapunov import LyapunovOnState
from controller import LinearDroop
from graphs import LyapunovCondition
from graphs_jacobian import LieJacobianCondition
from verify import certify_box, audit_verified
from certified_train import train

# Reuse the CEGIS measurement code verbatim so rho is measured identically.
from e4_cegis import max_certifiable_rho, slice_box, INNER, GI
# Reuse the analytic-vs-JacobianOP cross-check helpers verbatim.
from verifier_crosscheck import soundness_scan, max_rho as xcheck_max_rho, \
    prop2_beta as xcheck_prop2_beta
# Reuse the dReal (4b) encoding verbatim; HAVE_DREAL is False on the Windows host.
from dreal_baseline import build_symbolic, HAVE_DREAL
if HAVE_DREAL:
    from dreal import logical_and, CheckSatisfiability, Config


RHO_GRID = [round(0.5 * k, 2) for k in range(1, 6)]     # 0.5 .. 2.5, exactly E4/E5
CFG_PATH = os.path.join(ROOT, "configs", "certified_train.yaml")


def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def audited_prop2_beta(s, xstar, n, V, seed):
    """Largest beta certifiable for Prop 2 (Lie <= -beta (V - V*)) on the rho=1.0
    gen-5 annulus, measured with the SAME audited certify_box bisection E3 rung2
    used to get the CEGIS beta (up to 3.97). analytic path, the number of record."""
    ctrl = LinearDroop(s)
    beta_lo, beta_hi, beta_max = 0.0, 4.0, 0.0
    xL, xU = slice_box(xstar, n, INNER, INNER + 1.0, INNER, INNER + 1.0)
    for _ in range(7):                       # bisection on beta, matching E3
        beta = 0.5 * (beta_lo + beta_hi)
        cond = LyapunovCondition(s, V, ctrl, mode="prop2", x_star=xstar, beta=beta)
        rr = certify_box(cond, xL, xU, eps=0.0, method="CROWN",
                         min_width=0.04, time_budget=30, seed=seed)
        ok = rr["certified"] and audit_verified(cond, xL, xU, 0.0, seed=seed + 7) >= -1e-3
        if ok:
            beta_max, beta_lo = beta, beta
        else:
            beta_hi = beta
    return round(beta_max, 4)


def dreal_crosscheck(s, xstar, n, V, rho, seed, delta=1e-3, budget=600):
    """Encode the SAME (4b) property over the certified gen-5 region for the
    certified-trained V and hand its negation to dReal. On a host without dReal
    (this Windows box) record that honestly; on Colab/Linux it runs and certifies
    the identical function CROWN did. rho is the achieved certified radius; if that
    is 0 we still encode the rho=1.0 annulus so the artifact is never empty."""
    rho_enc = rho if rho and rho > 0 else 1.0
    out = dict(condition="4b", solver="dReal", delta=delta,
               region=f"gen-5 slice [x*+{INNER}, x*+{INNER}+{rho_enc}]^2 (delta,omega)")
    free = [GI, GI + n]
    xs = xstar.numpy().ravel()
    lo = [xs[i] + INNER for i in free]
    hi = [xs[i] + INNER + rho_enc for i in free]
    if not HAVE_DREAL:
        out["result"] = "dreal_unavailable"
        out["note"] = ("dReal ships no Windows wheel and its source build needs Bazel+IBEX; "
                       "run experiments/e7_certified_train.py on Colab/Linux to execute it. "
                       "The encoding is built from the analytic V/grad in src/graphs.py, so it "
                       "certifies the identical function CROWN does. No number fabricated.")
        return out
    cons, lie = build_symbolic(s, V, xstar, free, lo, hi)
    cfg = Config(); cfg.precision = delta
    formula = logical_and(*cons, lie >= 0.0)          # negation of (4b) over the region
    t0 = time.time()
    res = CheckSatisfiability(formula, delta)
    dt = time.time() - t0
    out["seconds"] = round(dt, 2)
    if res is None:
        out["result"] = "certified"; out["note"] = "delta-unsat: no state with Lie>=0 in region"
    else:
        out["result"] = "counterexample"; out["note"] = "delta-sat"; out["model"] = str(res)
    if dt >= budget:
        out["result"] = "timeout"
    return out


def cegis_reference(seed):
    """Pull the committed CEGIS numbers for a side-by-side, if present."""
    ref = {}
    e4 = os.path.join(ROOT, "results", f"e4_seed{seed}.json")
    e3 = os.path.join(ROOT, "results", f"e3_seed{seed}.json")
    if os.path.exists(e4):
        with open(e4) as f:
            ref["cegis_final_certified_rho"] = json.load(f).get("final_certified_rho")
    if os.path.exists(e3):
        with open(e3) as f:
            ref["cegis_prop2_beta_max"] = json.load(f).get("rung2", {}).get("cegis_prop2_beta_max")
    return ref


def run(seed, cfg, steps_override=None):
    active_buses = list(cfg["slice"]["active_buses"])
    slice_dim = 2 * len(active_buses)
    if slice_dim > 2:
        raise SystemExit(f"E7 holds at the 2-D gate; got active_buses={active_buses} "
                         f"(slice_dim={slice_dim}). Do not run any dimension above 2 here.")

    s = load_system(); n = s["n"]
    xstar_np = np.load(os.path.join(ROOT, "results", f"equilibrium_seed{seed}.npy"))
    xstar = torch.tensor(xstar_np, dtype=torch.float32).reshape(1, -1)
    warm = os.path.join(ROOT, "results", f"lyap_seed{seed}.pt")

    tr = cfg["train"]; bab = cfg["bab"]; sl = cfg["slice"]
    steps = steps_override if steps_override is not None else int(tr["steps"])

    # --- certified training, warm-started from the exact as-trained checkpoint ---
    V, hist = train(
        s, xstar_np, active_buses=active_buses, seed=seed,
        warmstart=warm, require_warmstart=True,        # never a fresh net
        inner=float(sl["inner_offset"]), rho_target=float(sl["rho_target"]),
        init_max_side=float(bab["init_max_side"]), max_cells=int(bab["max_cells"]),
        split_every=int(bab["split_every"]), steps=steps, lr=float(tr["lr"]),
        eps=float(tr["eps"]), margin_4a=float(tr["margin_4a"]),
        w_4a=float(tr["w_4a"]), w_pgd=float(tr["w_pgd"]), method=tr["method"], verbose=True)
    ckpt = os.path.join(ROOT, "results", f"lyap_certtrain_seed{seed}.pt")
    torch.save(V.state_dict(), ckpt)
    V.eval()

    ctrl = LinearDroop(s)
    ana = LyapunovCondition(s, V, ctrl, mode="4b", x_star=xstar)

    # --- HEADLINE: certified rho via the identical audited certify_box E4 used ---
    t0 = time.time()
    rho_max, rho_detail = max_certifiable_rho(ana, xstar, n, RHO_GRID, seed)
    rho_seconds = round(time.time() - t0, 1)

    # --- HEADLINE: Prop-2 beta via the same audited bisection E3 rung2 used ---
    t0 = time.time()
    beta_max = audited_prop2_beta(s, xstar, n, V, seed)
    beta_seconds = round(time.time() - t0, 1)

    # --- cross-check 1: JacobianOP, independent gradient path (soundness) ---
    jac = LieJacobianCondition(s, V, ctrl, mode="4b", x_star=xstar)
    # 0.25 .. 2.5, reaching the same ceiling as the headline rho grid so the
    # JacobianOP path either confirms the full certified region or reveals its own
    # (looser) ceiling, instead of being silently capped below the headline.
    xgrid = [round(0.25 * k, 2) for k in range(1, 11)]
    ra, ta = xcheck_max_rho(ana, jac, xstar, n, xgrid, seed, use_jac=False)
    rj, tj = xcheck_max_rho(ana, jac, xstar, n, xgrid, seed, use_jac=True)
    ac = lambda b: LyapunovCondition(s, V, ctrl, mode="prop2", x_star=xstar, beta=b)
    jc = lambda b: LieJacobianCondition(s, V, ctrl, mode="prop2", x_star=xstar, beta=b)
    beta_ana = xcheck_prop2_beta(ac, jc, xstar, n, seed, False)
    beta_jac = xcheck_prop2_beta(ac, jc, xstar, n, seed, True)
    scan = soundness_scan(ana, jac, xstar, n, seed)
    dis = []
    if scan["contradictions"] > 0:
        dis.append("soundness scan found bound contradictions")
    if rj > ra:
        dis.append("JacobianOP certified a larger rho than the analytic path (looser must be smaller)")
    jac_crosscheck = dict(
        rho_4b=dict(analytic=ra, jacobian=rj, analytic_s=ta, jacobian_s=tj),
        prop2_beta=dict(analytic=beta_ana, jacobian=beta_jac),
        soundness_scan=scan, agreement=(len(dis) == 0), disagreements=dis)

    # --- cross-check 2: dReal SMT encoding on the certified region ---
    dreal = dreal_crosscheck(s, xstar, n, V, rho_max, seed)

    # training-time BaB numbers are DIAGNOSTICS only, never the certificate
    last = hist["history"][-1] if hist["history"] else {}
    train_diag = dict(final_cells=hist["final_cells"],
                      frac_4b_verified_traincells=last.get("frac_4b_verified"),
                      min_lb_4b_traincells=last.get("min_lb_4b"),
                      note="training-time BaB is coarser than the verifier; diagnostic only")

    out = dict(
        experiment="e7_certified_train", seed=seed, method_name="CT-BaB (Shi et al. 2411.18235)",
        slice_dim=slice_dim, active_buses=active_buses, active_dims=hist["active_dims"],
        gen=5, inner_offset=INNER,
        region="gen-5 slice box [x*+INNER, x*+INNER+rho]^2 in (delta,omega)",
        warmstart=hist["warmstart"],                 # proves refine-not-reinit
        certified_rho=rho_max, certified_rho_detail=rho_detail, certified_rho_seconds=rho_seconds,
        prop2_beta_max=beta_max, prop2_beta_seconds=beta_seconds,
        headline_verifier="src/verify.py::certify_box, CROWN, eps=0, min_width=0.03/0.04, "
                          "audited by PGD; identical to E4/E3",
        jacobian_crosscheck=jac_crosscheck,
        dreal_crosscheck=dreal,
        train_seconds=hist["train_seconds"], train_steps=hist["steps"],
        train_diagnostics=train_diag,
        cegis_reference=cegis_reference(seed),
        checkpoint=os.path.relpath(ckpt, ROOT))

    path = os.path.join(ROOT, "results", f"e7_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== E7 seed", seed, "===")
    print(json.dumps({k: out[k] for k in ["certified_rho", "prop2_beta_max", "train_seconds",
          "cegis_reference"]}, indent=2))
    print("jacobian agreement:", jac_crosscheck["agreement"],
          "| soundness contradictions:", scan["contradictions"],
          "| dreal:", dreal["result"])
    print("warmstart:", hist["warmstart"]["warmstart_loaded"],
          "fingerprint L2:", hist["warmstart"]["init_param_l2"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="run seeds from the config's seeds list")
    ap.add_argument("--steps", type=int, default=None, help="override train.steps")
    a = ap.parse_args()
    cfg = load_cfg()
    seeds = cfg.get("seeds", [0, 1, 2, 3, 4]) if a.all else [a.seed]
    for k in seeds:
        run(k, cfg, steps_override=a.steps)
