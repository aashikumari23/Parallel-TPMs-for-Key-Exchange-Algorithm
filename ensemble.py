"""
ensemble.py
-----------
Parallel TPM ensemble. NEW.

Instead of one TPM pair with a large weight bound L, Alice and Bob run P
independent TPM pairs side by side. Each pair has its own private weights
and its own public input stream (seed derived from the shared seed), and
follows the paper's protocol unchanged: batch scheme, hop schedule, stop
counter. All P tau batches travel in the SAME message, so the number of
round trips is the slowest pair's, not the sum.

The session key is derived from the weights of ALL P pairs together, so an
eavesdropper learns nothing useful unless she synchronises with every pair.
If her success on each pair is independent with probability p, her chance
of getting the key is about p ** P. Whether that independence holds is an
empirical question; experiments/ensemble_independence.py measures it.

This answers the paper's stated future work (Section VI): a parallel TPM
model is hard because each weight update depends on the previous one. Here
each pair stays strictly sequential; the parallelism is ACROSS pairs.

Stopping: a pair that reaches its stop counter freezes and sends no more
taus. The counter is computed from exchanged taus, so both sides agree on
which pairs are finished without extra messages. The exchange ends when
every pair has finished.
"""
from __future__ import annotations
import numpy as np

from engine import SyncSession
from hop_schedule import make_schedule


def pair_seed(seed: int, i: int) -> int:
    """Public input seed for pair i. Pair 0 uses the shared seed itself."""
    if i == 0:
        return int(seed)
    return int(np.random.SeedSequence([int(seed) % (2 ** 63), i]).generate_state(1)[0])


class EnsembleSession:
    """P SyncSession objects for ONE party, advanced in lock-step."""

    def __init__(self, P: int, K: int, N: int, L: int, B: int, MAX: int, input_seed: int,
                 mode: str = "paper", hop: int = 3, scrn_bits: int = 1,
                 count_mode: str = "pseudocode", weight_rngs=None,
                 input_mode: str = "fresh", role: str = "alice"):
        assert P >= 1
        self.P, self.K, self.N, self.L, self.B, self.MAX = P, K, N, L, B, MAX
        self.mode, self.hop = mode, hop
        rngs = weight_rngs or [None] * P
        self.pairs = [SyncSession(K, N, L, B, MAX, pair_seed(input_seed, i), make_schedule(mode, hop),
                                  scrn_bits, count_mode, rngs[i], input_mode=input_mode, role=role)
                      for i in range(P)]
        self.comm_rounds = 0
        self.finished_at = [None] * P     # communication round at which each pair finished

    # ---------------------------------------------------------------- state
    @property
    def done(self) -> bool:
        return all(p.done for p in self.pairs)

    @property
    def active(self):
        return [i for i, p in enumerate(self.pairs) if not p.done]

    @property
    def weights(self) -> np.ndarray:
        """All weights, shape (P*K, N): the material the key is derived from."""
        return np.concatenate([p.tpm.weights for p in self.pairs], axis=0)

    @property
    def updates(self):
        return sum(p.updates for p in self.pairs)

    @property
    def learning_rounds(self):
        return sum(p.round_idx for p in self.pairs)

    @property
    def bits_sent(self):
        return sum(p.bits_sent for p in self.pairs)

    @property
    def hop_hist(self):
        h = {}
        for p in self.pairs:
            for k, v in p.hop_hist.items():
                h[k] = h.get(k, 0) + v
        return dict(sorted(h.items()))

    def describe(self) -> dict:
        return {"P": self.P, **self.pairs[0].schedule.describe()}

    # ------------------------------------------------------------ protocol
    def make_batches(self):
        """One entry per pair: a list of B taus, or None if that pair is finished."""
        return [p.make_batch() if not p.done else None for p in self.pairs]

    def apply_peer_batches(self, peer):
        """Apply the peer's per-pair taus. Returns hops used per pair (None if finished)."""
        assert len(peer) == self.P
        self.comm_rounds += 1
        hops = []
        for i, (p, t) in enumerate(zip(self.pairs, peer)):
            if p.done:
                if t is not None:
                    raise RuntimeError(f"pair {i}: peer still sending after both sides finished")
                hops.append(None)
                continue
            if t is None:
                raise RuntimeError(f"pair {i}: peer stopped early; the two sides disagree on its state")
            h, _ = p.apply_peer_batch(t)
            hops.append(h)
            if p.done:
                self.finished_at[i] = self.comm_rounds
        return hops
