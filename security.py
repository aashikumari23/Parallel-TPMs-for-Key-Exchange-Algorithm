#!/usr/bin/env python3
"""
security.py: attack success rate for one configuration, with a 95 % Wilson
interval (the paper reports point estimates only).

    python security.py --hop-mode paper --L 20 --attack majority --M 50 --trials 100
"""
import argparse
import numpy as np
from sim import simulate
from stats import wilson

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, d in [("K", 3), ("N", 100), ("L", 12), ("B", 10), ("MAX", 200), ("hop", 3),
                 ("M", 50), ("trials", 50), ("seed0", 0), ("P", 1)]:
        ap.add_argument(f"--{k}", type=int, default=d)
    ap.add_argument("--hop-mode", default="paper", choices=["conventional", "paper", "fixed"])
    ap.add_argument("--input-mode", default="fresh", choices=["fresh", "scrn_batch"])
    ap.add_argument("--attack", default="majority", choices=["simple", "geometric", "majority"])
    a = ap.parse_args()
    wins, comm = 0, []
    for t in range(a.trials):
        r = simulate(K=a.K, N=a.N, L=a.L, B=a.B, MAX=a.MAX, hop=a.hop, mode=a.hop_mode,
                     seed=a.seed0 + t, attack=a.attack, M=a.M,
                     input_mode=a.input_mode, P=a.P)
        wins += bool(r["eve_success"])
        comm.append(r["comm_rounds"])
        print(f"trial {t:3d} comm={r['comm_rounds']:6d} best attacker overlap="
              f"{r['eve_best'] * 100:6.2f}% {'ATTACK SUCCESS' if r['eve_success'] else ''}")
    lo, hi = wilson(wins, a.trials)
    print(f"\n{a.attack} attack, M={a.M}: {wins}/{a.trials} = {100 * wins / a.trials:.1f}% "
          f"(95% CI {100 * lo:.1f} to {100 * hi:.1f}%), mean comm rounds {np.mean(comm):.0f}")
