"""
ecc.py
------
Error-correcting-code layer used for the "Amplification of Weight
Synchronization Rule" step of the paper (Section IV-B, Figure 8).

The paper uses a BCH(511, 229, t=38) code. To keep this implementation
self-contained and dependency-free (no native/C extensions), we implement a
generic binary Hamming Single-Error-Correcting (SEC) code instead. It plays
exactly the same *protocol* role described in the paper:

  1. Party A encodes its own (already near-synchronized) weight bits and
     sends ONLY the parity bits to Party B -- never the raw weight data.
  2. Party B, whose weight bits should already closely match A's after the
     random-walk phase, combines ITS OWN data bits with A's received parity
     bits into a hybrid codeword and performs standard syndrome decoding.
     Any small residual bit differences (a handful of bit "errors") are
     corrected without A ever having sent its actual weight values.

A (15, 11) Hamming code corrects any single bit error within each 11-bit
data block (4 parity bits per block). This is a smaller correction radius
than the paper's BCH(511,229,t=38), so in this reference implementation we
apply the batch/1-h random-walk phase until synchronization is very close
before invoking ECC (matching the paper's finding that ECC should only be
applied "after the weights are synchronized above a certain rate").
"""

from __future__ import annotations
import numpy as np


class WeightResidueECC:
    """
    A parity-based corrector matched to the ACTUAL noise model of
    near-synchronized TPM weights: after the random-walk phase, weights
    differ (when they differ at all) by a small signed integer amount
    (observed empirically up to roughly +/-5 for L=57), NOT by a bounded
    number of bit-flips. Bit-oriented codes (Hamming/BCH operating on a
    binary serialization) are a poor fit for this: even with Gray coding,
    a true difference of magnitude > 1 can flip multiple bits, which can
    exceed a single-error-correcting code's capacity and cause it to
    "correct" in the wrong direction.

    Instead, this corrector sends only `weight mod modulus` per weight
    (never the weight itself) as parity. The receiver, whose own value for
    that weight should already be close to the true value, reconstructs the
    exact original value via signed modular arithmetic. This exactly and
    deterministically recovers the true weight whenever the true difference
    magnitude is smaller than `modulus // 2` -- independent of how many
    bits that difference happens to touch in binary.
    """

    def __init__(self, modulus: int = 32):
        assert 2 <= modulus <= 256
        self.modulus = modulus

    def encode(self, weights: np.ndarray) -> bytes:
        """Compute and return ONLY the residues (weights mod modulus),
        never the weights themselves."""
        residues = np.mod(weights.astype(np.int64), self.modulus).astype(np.uint8)
        return residues.tobytes()

    def decode(self, local_weights: np.ndarray, parity: bytes):
        """
        Correct `local_weights` (this party's own, near-matching values)
        using the peer's residue parity. Returns (corrected, n_corrected).
        """
        residues = np.frombuffer(parity, dtype=np.uint8)[: len(local_weights)].astype(np.int64)
        local_mod = np.mod(local_weights.astype(np.int64), self.modulus)
        delta = np.mod(residues - local_mod, self.modulus)
        half = self.modulus // 2
        delta = np.where(delta > half, delta - self.modulus, delta)
        corrected = local_weights.astype(np.int64) + delta
        n_corrected = int(np.sum(delta != 0))
        return corrected, n_corrected


class HammingECC:
    def __init__(self, r: int = 4):
        """
        r = number of parity bits per block.
        n = 2^r - 1 total bits per block (codeword length)
        k = n - r data bits per block
        """
        self.r = r
        self.n = (1 << r) - 1
        self.k = self.n - r
        self._parity_positions = [1 << i for i in range(r)]  # 1-indexed positions 1,2,4,8,...
        self._data_positions = [p for p in range(1, self.n + 1) if p not in self._parity_positions]

    # -------------------------------------------------------------- #
    @staticmethod
    def _bytes_to_bits(data: bytes) -> np.ndarray:
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    @staticmethod
    def _bits_to_bytes(bits: np.ndarray) -> bytes:
        pad = (-len(bits)) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        return np.packbits(bits).tobytes()

    def _chunk_bits(self, bits: np.ndarray):
        pad = (-len(bits)) % self.k
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        return bits.reshape(-1, self.k), pad

    # -------------------------------------------------------------- #
    def encode(self, data: bytes) -> bytes:
        """Compute and return ONLY the parity bits for `data` (as bytes)."""
        bits = self._bytes_to_bits(data)
        blocks, _pad = self._chunk_bits(bits)
        all_parity_bits = []
        for block in blocks:
            codeword = np.zeros(self.n + 1, dtype=np.uint8)  # 1-indexed, index 0 unused
            for pos, bit in zip(self._data_positions, block):
                codeword[pos] = bit
            for p in self._parity_positions:
                covered = [pos for pos in range(1, self.n + 1) if (pos & p) and pos != p]
                parity = 0
                for pos in covered:
                    parity ^= int(codeword[pos])
                all_parity_bits.append(parity)
        return self._bits_to_bytes(np.array(all_parity_bits, dtype=np.uint8)), len(blocks)

    def decode(self, local_data: bytes, received_parity: bytes, n_blocks: int):
        """
        Reconstruct a corrected version of `local_data` using the sender's
        parity bits. Returns (corrected_bytes, total_bits_corrected).
        """
        data_bits = self._bytes_to_bits(local_data)
        data_blocks, pad = self._chunk_bits(data_bits)
        parity_bits_flat = self._bytes_to_bits(received_parity)[: n_blocks * self.r]
        parity_blocks = parity_bits_flat.reshape(n_blocks, self.r)

        corrected_blocks = []
        n_corrected = 0
        for block, parity in zip(data_blocks, parity_blocks):
            codeword = np.zeros(self.n + 1, dtype=np.uint8)
            for pos, bit in zip(self._data_positions, block):
                codeword[pos] = bit
            for p, bit in zip(self._parity_positions, parity):
                codeword[p] = bit  # trusted parity from sender

            syndrome = 0
            for i, p in enumerate(self._parity_positions):
                covered = [pos for pos in range(1, self.n + 1) if (pos & p)]
                s = 0
                for pos in covered:
                    s ^= int(codeword[pos])
                syndrome |= (s << i)

            if syndrome != 0 and syndrome <= self.n:
                codeword[syndrome] ^= 1
                n_corrected += 1

            corrected_blocks.append([int(codeword[pos]) for pos in self._data_positions])

        corrected_bits = np.array(corrected_blocks, dtype=np.uint8).flatten()
        if pad:
            corrected_bits = corrected_bits[:-pad]
        return self._bits_to_bytes(corrected_bits), n_corrected
