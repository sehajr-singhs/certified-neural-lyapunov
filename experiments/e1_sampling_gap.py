"""E1 - quantify the sampling gap. Random sampling finds violations of (4b) at
roughly her 0.1% rate, while a directed PGD attack finds them far more often,
because violations concentrate near the equilibrium rather than spreading
uniformly. The ratio of attack rate to random rate is the empirical case for
needing a sound verifier rather than more samples.

Writes results/e1_seed{seed}.json.
"""
import os, sys, json, argparse
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from data import load_system
from lyapunov import LyapunovOnState
from controller import LinearDroop
from graphs import LyapunovCondition
from attack import pgd_find_violations, random_violation_rate


def run(seed=0):
    s = load_system()
    xstar = torch.tensor(np.load(os.path.join(ROOT, "results", f"equilibrium_seed{seed}.npy")),
                         dtype=torch.float32).reshape(1, -1)
    V = LyapunovOnState(s)
    V.load_state_dict(torch.load(os.path.join(ROOT, "results", f"lyap_seed{seed}.pt")))
    V.eval()
    cond = LyapunovCondition(s, V, LinearDroop(s), mode="4b", x_star=xstar)

    # her training region: delta std 2, omega std 10 (use as box radius)
    dr, wr = 2.0, 10.0
    rand_rate, rand_max = random_violation_rate(cond, xstar, dr, wr, n=200000, seed=seed)
    pgd_rate, pgd_max, _ = pgd_find_violations(cond, xstar, dr, wr, n_starts=2000, seed=seed)
    ratio = (pgd_rate / rand_rate) if rand_rate > 0 else float("inf")

    out = dict(seed=seed, region={"delta_r": dr, "omega_r": wr},
               random_violation_rate=rand_rate, random_max_lie=rand_max,
               pgd_violation_rate=pgd_rate, pgd_max_lie=pgd_max,
               attack_over_random_ratio=ratio)
    with open(os.path.join(ROOT, "results", f"e1_seed{seed}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args().seed)
