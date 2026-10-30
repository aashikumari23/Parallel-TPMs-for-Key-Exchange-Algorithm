"""
attacks.py
----------
Passive eavesdropper models for Section V of the paper, vectorised with
numpy so hundreds of trials run in seconds.

Threat model (Kerckhoffs): Eve knows K, N, L, B, MAX, the hop schedule and
the public input seed, and sees every exchanged tau batch. She does NOT see
Alice's or Bob's weights. She uses the same hop schedule as the parties.

  simple     : M independent TPMs, each learns only when its tau matches.
               (This is what the original repo's attack_majority.py did.)
  geometric  : Klimov, Mityagin, Shamir [13]. When Eve's tau disagrees, flip
               the hidden unit with the smallest |local field|, then learn.
  majority   : Shacham et al. [15], the attack the paper uses in Table 2.
               M geometric attackers; the most common corrected internal
               representation across the M is adopted by all of them.

Success criterion (paper, Section V-A): some attacker's weights agree with
Alice's on at least 98 % of positions when Alice and Bob finish.
"""
from __future__ import annotations
import numpy as np


class AttackerEnsemble:
    def __init__(self, kind: str, M: int, K: int, N: int, L: int, rng: np.random.Generator):
        assert kind in ("simple", "geometric", "majority")
        self.kind, self.M, self.K, self.N, self.L = kind, M, K, N, L
        self.w = rng.integers(-L, L + 1, size=(M, K, N)).astype(np.int64)
        self._pending = None
        self._pow2 = 1 << np.arange(K)

    def make_batch(self, inputs):
        """Mirror the parties: compute all B outputs before any update."""
        X = np.asarray(inputs, dtype=np.int64)
        H = np.einsum("mkn,bkn->bmk", self.w, X)          # one call for the whole batch
        S = np.where(H > 0, 1, -1)
        T = np.prod(S, axis=2)
        self._pending = (X, [(H[b], S[b], T[b]) for b in range(X.shape[0])])

    def apply(self, taus_a, taus_b, hops):
        inputs, out = self._pending
        m_idx = np.arange(self.M)
        for x, (h, sigma, tau_e), ta, tb, hop in zip(inputs, out, taus_a, taus_b, hops):
            if ta != tb:
                continue                              # Alice and Bob do not learn
            sigma = sigma.copy()
            if self.kind == "simple":
                learn = tau_e == ta
            else:
                wrong = tau_e != ta
                if np.any(wrong):
                    k_flip = np.argmin(np.abs(h), axis=1)
                    rows = m_idx[wrong]
                    sigma[rows, k_flip[wrong]] *= -1
                learn = np.ones(self.M, dtype=bool)
                if self.kind == "majority":
                    codes = ((sigma > 0) * self._pow2).sum(axis=1)
                    best = int(np.argmax(np.bincount(codes, minlength=1 << self.K)))
                    sigma[:] = np.where((best >> np.arange(self.K)) & 1, 1, -1)
            # random-walk update on units whose sigma equals Alice's tau
            mask = (sigma == ta) & learn[:, None]     # (M, K)
            self.w += hop * mask[:, :, None] * x[None, :, :]
            np.clip(self.w, -self.L, self.L, out=self.w)
        self._pending = None

    def best_overlap(self, w_alice):
        return float(np.max(np.mean(self.w == w_alice[None], axis=(1, 2))))
