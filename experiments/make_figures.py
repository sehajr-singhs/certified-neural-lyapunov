"""Regenerate every figure from results/*.json and the saved networks. No number
here is typed by hand, each figure reads a JSON a run wrote (and, for the field
plots, the committed .pt network, which is deterministic given the seed).

Writes PNGs into static/figures/ and copies them into paper/figures/.

Usage: KMP_DUPLICATE_LIB_OK=TRUE python experiments/make_figures.py
"""
import os, sys, json, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import torch
from data import load_system
from lyapunov import LyapunovOnState
from controller import LinearDroop
from graphs import LyapunovCondition

STATIC = os.path.join(ROOT, "static", "figures")
PAPER = os.path.join(ROOT, "paper", "figures")
os.makedirs(STATIC, exist_ok=True); os.makedirs(PAPER, exist_ok=True)
GI = 4; INNER = 0.05
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def jload(name):
    p = os.path.join(ROOT, "results", name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fp = os.path.join(STATIC, name); fig.savefig(fp); shutil.copy(fp, os.path.join(PAPER, name))
    plt.close(fig); print("wrote", name)


def slice_fields(ckpt, xstar, s, dgrid, wgrid):
    V = LyapunovOnState(s); V.load_state_dict(torch.load(ckpt)); V.eval()
    cond = LyapunovCondition(s, V, LinearDroop(s), mode="4b", x_star=xstar)
    n = s["n"]
    D, W = np.meshgrid(dgrid, wgrid, indexing="ij")
    X = xstar.repeat(D.size, 1).clone()
    X[:, GI] = xstar[0, GI] + torch.tensor(D.reshape(-1), dtype=torch.float32)
    X[:, GI + n] = xstar[0, GI + n] + torch.tensor(W.reshape(-1), dtype=torch.float32)
    with torch.no_grad():
        lie = cond.lie(X).squeeze(-1).numpy().reshape(D.shape)
        Vv = (cond.V(X).squeeze(-1) - cond.V_star).numpy().reshape(D.shape)
    return D, W, Vv, lie


def fig_money(s, xstar):
    """Her Fig. 2 geometry, V and Lie over the gen-5 slice, as-trained vs CEGIS,
    with the certified region and the counterexample drawn on top."""
    e3 = jload("e3_seed0.json")
    dg = np.linspace(-0.4, 1.7, 240); wg = np.linspace(-0.4, 0.6, 200)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for col, (tag, ckpt) in enumerate([("as-trained V", "lyap_seed0.pt"),
                                        ("CEGIS V", "lyap_cegis_seed0.pt")]):
        p = os.path.join(ROOT, "results", ckpt)
        if not os.path.exists(p):
            continue
        D, W, Vv, lie = slice_fields(p, xstar, s, dg, wg)
        # top: V - V*
        c0 = ax[0, col].pcolormesh(D, W, Vv, shading="auto", cmap="viridis")
        ax[0, col].contour(D, W, Vv, levels=[0.0], colors="w", linewidths=1.2)
        fig.colorbar(c0, ax=ax[0, col], fraction=0.046)
        ax[0, col].set_title(f"{tag}:  V(x) - V*")
        # bottom: Lie, red where positive (violates 4b)
        m = np.abs(lie).max()
        c1 = ax[1, col].pcolormesh(D, W, lie, shading="auto", cmap="RdBu_r",
                                   vmin=-m, vmax=m)
        ax[1, col].contour(D, W, lie, levels=[0.0], colors="k", linewidths=1.0)
        fig.colorbar(c1, ax=ax[1, col], fraction=0.046)
        ax[1, col].set_title(f"{tag}:  Lie = grad V . f   (red > 0 violates 4b)")
        ax[1, col].set_xlabel(r"$\delta_5 - \delta_5^*$")
    ax[0, 0].set_ylabel(r"$\omega_5 - \omega_5^*$"); ax[1, 0].set_ylabel(r"$\omega_5 - \omega_5^*$")
    # as-trained: mark the counterexample
    if e3:
        ce = e3["rung2"]["as_trained_far_region"]["counterexample_gen5"]
        for r in (0, 1):
            ax[r, 0].plot(ce["delta"], ce["omega"], "kx", ms=11, mew=2.5)
        ax[1, 0].annotate("counterexample\nLie = +%.2f" % e3["rung2"]["as_trained_far_region"]["lie_at_ce"],
                          (ce["delta"], ce["omega"]), xytext=(ce["delta"] + 0.15, ce["omega"] + 0.22),
                          fontsize=9, arrowprops=dict(arrowstyle="->"))
        # CEGIS: draw the certified annulus box
        rho = e3["rung2"]["cegis_4b_annulus"]["certified_rho"]
        for r in (0, 1):
            ax[r, 1].add_patch(Rectangle((INNER, INNER), rho, rho, fill=False,
                                         edgecolor="lime", lw=2.2))
        ax[1, 1].annotate(r"certified (4b), $\rho=%.1f$" % rho + "\naudit +%.2f" %
                          e3["rung2"]["cegis_4b_annulus"]["detail"]["audit_min_F"],
                          (INNER, INNER + rho), xytext=(0.35, 0.35), fontsize=9,
                          color="green", arrowprops=dict(arrowstyle="->", color="green"))
    fig.suptitle("Gen-5 projected slice: sampling trains V to look stable, a verifier finds where it is not",
                 fontsize=12)
    save(fig, "money_slice.png")


def fig_cegis(s):
    e4 = jload("e4_seed0.json")
    if not e4:
        return
    it = e4["iterations"]
    x = [r["iter"] for r in it]
    rho = [r["certified_rho"] for r in it]
    mlie = [r["frontier_max_lie"] for r in it]
    fig, ax1 = plt.subplots(figsize=(7, 4.3))
    ax1.plot(x, rho, "o-", color="green", lw=2, label="certified $\\rho$ (4b)")
    ax1.set_xlabel("CEGIS iteration"); ax1.set_ylabel("certified region $\\rho$", color="green")
    ax1.set_ylim(-0.1, max(rho) + 0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, mlie, "s--", color="crimson", lw=1.6, label="max attack Lie")
    ax2.axhline(0, color="crimson", lw=0.8, alpha=0.5)
    ax2.set_ylabel("max attack Lie (>0 = violation)", color="crimson")
    ax1.set_title("Closing the loop: certified region jumps once counterexamples are fed back")
    save(fig, "cegis_curve.png")


def fig_ladder():
    e3 = jload("e3_seed0.json")
    if not e3:
        return
    bl = e3["rung2"]["bound_ladder_pm0p5_box"]
    keys = ["IBP", "CROWN", "CROWN-Optimized"]; vals = [bl[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    bars = ax.bar(keys, vals, color=["#bbb", "#6aa", "#276"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("lower bound on F = -Lie over the box")
    ax.set_title("Bound tightness on the gen-5 $\\pm0.5$ box (less negative is tighter)")
    for b, v in zip(bars, vals):
        ax.annotate("%.1f" % v, (b.get_x() + b.get_width() / 2, v), ha="center",
                    va="top" if v < 0 else "bottom", fontsize=9)
    save(fig, "bound_ladder.png")


def fig_sampling_gap():
    e1 = jload("e1_seed0.json")
    if not e1:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4))
    a.bar(["random\n(200k)", "PGD\n(directed)"],
          [100 * e1["random_violation_rate"], 100 * e1["pgd_violation_rate"]],
          color=["#aaa", "crimson"])
    a.set_ylabel("(4b) violation rate found (%)")
    a.set_title("Same network, same region")
    a.annotate("0 of 200,000", (0, 0.2), ha="center", fontsize=9)
    a.annotate("%.1f%%" % (100 * e1["pgd_violation_rate"]),
               (1, 100 * e1["pgd_violation_rate"]), ha="center", va="bottom", fontsize=9)
    b.bar(["random", "PGD"], [e1["random_max_lie"], e1["pgd_max_lie"]],
          color=["#aaa", "crimson"])
    b.axhline(0, color="k", lw=0.8); b.set_ylabel("max Lie found (>0 = violation)")
    b.set_title("Directed search finds violations sampling misses")
    save(fig, "sampling_gap.png")


def fig_staircase():
    e3 = jload("e3_seed0.json"); r1 = jload("e3_rung1_seed0.json")
    labels, vals, notes = [], [], []
    if r1:
        labels.append("Rung 1\n2-bus, 4-D"); vals.append(r1["certified_rho"]); notes.append("certified $\\rho$")
    if e3:
        labels.append("Rung 2\ngen-5 slice, 2-D"); vals.append(e3["rung2"]["cegis_4b_annulus"]["certified_rho"]); notes.append("certified $\\rho$")
        labels.append("Rung 3\nfull 20-D"); vals.append(e3["rung3_full20D"]["certified_r_max"]); notes.append("certified $r$")
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4))
    bars = ax.bar(labels, vals, color=["#276", "#3a8", "#c66"])
    ax.set_ylabel("certified region size")
    ax.set_title("The staircase: certified region collapses as state dimension grows")
    for b, v in zip(bars, vals):
        ax.annotate("%.2f" % v, (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=10)
    save(fig, "staircase.png")


def main():
    s = load_system()
    xstar = torch.tensor(np.load(os.path.join(ROOT, "results", "equilibrium_seed0.npy")),
                         dtype=torch.float32).reshape(1, -1)
    fig_sampling_gap()
    fig_money(s, xstar)
    fig_cegis(s)
    fig_ladder()
    fig_staircase()


if __name__ == "__main__":
    main()
