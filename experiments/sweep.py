"""Run the full experiment pipeline for one or more seeds, in dependency order:
E0 (train V) -> E1 (sampling gap) -> E4 (CEGIS, writes the hardened V) ->
E2 (certify 4a, needs the CEGIS V) -> E3 (staircase) -> E3 rung 1 (two-bus).

Used by `make full`. Every step writes its own per-seed JSON to results/, so a
crash mid-sweep leaves every completed step on disk.

Usage: python experiments/sweep.py 0 1 2 3 4
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STEPS = ["e0_reproduce", "e1_sampling_gap", "e4_cegis",
         "e2_certify_4a", "e3_certify_lie", "e3_rung1_twobus"]


def main(seeds):
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    for s in seeds:
        for mod in STEPS:
            print(f"\n=== seed {s} : {mod} ===", flush=True)
            subprocess.run([PY, os.path.join(HERE, f"{mod}.py"), "--seed", str(s)],
                           check=True, env=env)


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 4]
    main(seeds)
