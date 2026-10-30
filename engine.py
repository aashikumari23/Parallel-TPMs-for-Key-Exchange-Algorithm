"""
engine.py
---------
Transport-independent protocol engine for ONE party (Algorithm 1 of the
paper, minus the socket). The networked runner (party.py) and the
in-process simulator (sim.py) drive this same class.

Per communication round:
    taus = s.make_batch()                      # B outputs from public inputs
    ... exchange taus with the peer ...
    s.apply_peer_batch(peer_taus)              # random-walk updates

Input modes (all generated from the shared public seed)
    "fresh"      : a new random input every output (recommended; default)
    "scrn_batch" : SCRN as Algorithm 1 reads: a set of B inputs drawn once,
                   every row shifted once per batch.
"""
from __future__ import annotations
import numpy as np

from tpm import TreeParityMachine
from hop_schedule import HopSchedule, AlternatingHop

COUNT_MODES = ("pseudocode", "text")
INPUT_MODES = ("fresh", "scrn_batch")


class SyncSession:
    def __init__(self, K: int, N: int, L: int, B: int, MAX: int, input_seed: int,
                 schedule: HopSchedule | None = None, scrn_bits: int = 1,
                 count_mode: str = "pseudocode", weight_rng=None,
                 input_mode: str = "fresh", role: str = "alice"):
        """
        count_mode
          "pseudocode": count += 1 per tau in a fully matched batch, reset on
                        any mismatch (Algorithm 1 lines 29-33).
          "text":       count += 1 per fully matched BATCH (Section III-B text).
        """
        assert count_mode in COUNT_MODES and input_mode in INPUT_MODES and role in ("alice", "bob")
        self.K, self.N, self.L, self.B, self.MAX = K, N, L, B, MAX
        self.tpm = TreeParityMachine(K, N, L, seed=input_seed, rng=weight_rng)
        self.schedule = schedule or AlternatingHop(3)
        self.scrn_bits = scrn_bits
        self.count_mode = count_mode
        self.input_mode = input_mode
        self.role = role
        self.count = 0
        self.round_idx = 0          # learning rounds (one per tau)
        self.comm_rounds = 0        # tau-batch exchanges
        self.updates = 0
        self.hop_hist = {}
        self.bits_sent = 0          # payload bits this party put on the wire
        self._pending = None

    @property
    def done(self) -> bool:
        return self.count >= self.MAX

    def _public_inputs(self, n: int) -> np.ndarray:
        """n public inputs, shape (n, K, N), identical on both sides."""
        if self.input_mode == "fresh":
            bits = self.tpm.input_rng.integers(0, 2, size=(n, self.K, self.N), dtype=np.int8)
            return bits.astype(np.int64) * 2 - 1
        if self.input_mode == "scrn_batch":
            # Algorithm 1 reading: a (B, K, N) input set is drawn once; after each
            # batch EVERY row is shifted by scrn_bits with new bits appended.
            # Inputs within a batch are independent; batch t+1 row i is batch t
            # row i shifted.
            if getattr(self, "_batch_set", None) is None:
                bits = self.tpm.input_rng.integers(0, 2, size=(n, self.K, self.N), dtype=np.int8)
                self._batch_set = bits.astype(np.int64) * 2 - 1
            else:
                new = self.tpm.input_rng.integers(0, 2, size=(n, self.K, self.scrn_bits), dtype=np.int8)
                self._batch_set = np.concatenate(
                    [self._batch_set[:, :, self.scrn_bits:], new.astype(np.int64) * 2 - 1], axis=2)
            return self._batch_set.copy()
        raise ValueError(f"unknown input mode {self.input_mode}")

    def make_batch(self):
        """Compute B outputs from the next B public inputs.

        All B outputs use the weights as they are at the start of the batch
        (that is the batch scheme), so they are computed in one vectorised
        step: h[b, k] = sum_n w[k, n] * x[b, k, n].
        """
        X = self._public_inputs(self.B)
        h = np.einsum("bkn,kn->bk", X, self.tpm.weights)
        sigmas = np.where(h > 0, 1, -1).astype(np.int64)
        taus = [int(t) for t in np.prod(sigmas, axis=1)]
        self.bits_sent += self.B
        self._pending = (taus, sigmas, X)
        return taus

    @property
    def last_batch_inputs(self):
        return self._pending[2] if self._pending else None

    def apply_peer_batch(self, peer_taus):
        taus, sigmas, inputs = self._pending
        self.comm_rounds += 1
        full_match = list(taus) == list(peer_taus)
        hops_used = []
        for i in range(self.B):
            hop = self.schedule.hop_for(self.round_idx)
            hops_used.append(hop)
            if taus[i] == peer_taus[i]:
                mask = (sigmas[i] == taus[i])[:, None]          # units that agree with tau
                w = self.tpm.weights
                w += hop * mask * inputs[i]
                np.clip(w, -self.L, self.L, out=w)
                self.updates += 1
                self.hop_hist[hop] = self.hop_hist.get(hop, 0) + 1
            if self.count_mode == "pseudocode":
                self.count = self.count + 1 if full_match else 0
            self.round_idx += 1
        if self.count_mode == "text":
            self.count = self.count + 1 if full_match else 0
        self.schedule.observe(taus, peer_taus)
        self._pending = None
        return hops_used, full_match
