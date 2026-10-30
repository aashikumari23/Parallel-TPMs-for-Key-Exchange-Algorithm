"""
hop_schedule.py
---------------
Weight-step (hop) schedules for the random-walk learning rule.

  FixedHop         conventional random walk (hop = 1 every step) or any
                   constant step.
  AlternatingHop   the paper's 1-h random walk (Section III-A, Algorithm 1):
                   step 1 on even rounds, step `hop` on odd rounds.
"""
from __future__ import annotations


class HopSchedule:
    name = "base"

    def hop_for(self, round_idx: int) -> int:
        raise NotImplementedError

    def observe(self, my_taus, peer_taus) -> None:
        """Feed one exchanged batch of taus (public data). Default: no-op."""

    def describe(self) -> dict:
        return {"mode": self.name}


class FixedHop(HopSchedule):
    name = "fixed"

    def __init__(self, hop: int = 1):
        self.hop = int(hop)

    def hop_for(self, round_idx: int) -> int:
        return self.hop

    def describe(self):
        return {"mode": self.name, "hop": self.hop}


class AlternatingHop(HopSchedule):
    """Paper Algorithm 1: hop=1 when round%2==0, hop=`hop` when round%2==1."""
    name = "paper"

    def __init__(self, hop: int = 3):
        self.hop = int(hop)

    def hop_for(self, round_idx: int) -> int:
        return 1 if round_idx % 2 == 0 else self.hop

    def describe(self):
        return {"mode": self.name, "hop": self.hop}


def make_schedule(mode: str, hop: int = 3) -> HopSchedule:
    mode = mode.lower()
    if mode in ("conventional", "rw"):
        return FixedHop(1)
    if mode == "fixed":
        return FixedHop(hop)
    if mode in ("paper", "1h", "alternating"):
        return AlternatingHop(hop)
    raise ValueError(f"unknown hop mode: {mode}")
