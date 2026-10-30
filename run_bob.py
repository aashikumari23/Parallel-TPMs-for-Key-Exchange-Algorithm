#!/usr/bin/env python3
"""Bob (client side). Example:
    python run_alice.py --seed 2024 --hop-mode paper --L 20
"""
import argparse
from cli_common import add_protocol_args, party_kwargs
from party import run_party

if __name__ == "__main__":
    args = add_protocol_args(argparse.ArgumentParser(), seed_required=True).parse_args()
    run_party("bob", **party_kwargs(args))
