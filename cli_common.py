"""Shared argparse flags for run_alice.py, run_bob.py and demo.py."""
import argparse


def add_protocol_args(ap: argparse.ArgumentParser, seed_required: bool = False):
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--K", type=int, default=3, help="hidden units")
    ap.add_argument("--N", type=int, default=100, help="inputs per hidden unit")
    ap.add_argument("--L", type=int, default=20, help="weight bound")
    ap.add_argument("--B", type=int, default=10, help="batch size")
    ap.add_argument("--MAX", type=int, default=200, help="stopping counter (see --count-mode)")
    ap.add_argument("--hop", type=int, default=3, help="step on odd rounds (paper) or every round (fixed)")
    ap.add_argument("--hop-mode", default="paper", choices=["conventional", "paper", "fixed"])
    ap.add_argument("--input-mode", default="fresh", choices=["fresh", "scrn_batch"],
                    help="fresh random inputs (default), or the paper's SCRN shifted per batch")
    ap.add_argument("--P", type=int, default=1,
                    help="parallel TPM pairs; the key needs all of them (1 = paper protocol)")
    ap.add_argument("--count-mode", default="pseudocode", choices=["pseudocode", "text"])
    ap.add_argument("--ecc-modulus", type=int, default=4,
                    help="residue modulus; corrects diffs below m/2, leaks log2(m) bits/weight")
    ap.add_argument("--seed", type=int, required=seed_required, default=None if seed_required else 42,
                    help="public input seed, identical on both sides")
    return ap


def party_kwargs(args):
    return dict(host=args.host, port=args.port, K=args.K, N=args.N, L=args.L, B=args.B,
                MAX=args.MAX, hop=args.hop, hop_mode=args.hop_mode, input_seed=args.seed,
                ecc_modulus=args.ecc_modulus, count_mode=args.count_mode,
                input_mode=args.input_mode, P=args.P)
