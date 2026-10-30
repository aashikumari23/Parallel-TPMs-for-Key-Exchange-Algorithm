"""
keyderive.py
------------
Post-synchronisation steps: reconciliation, leakage accounting, key
derivation and key confirmation.

Why this file exists (a security fix to the original repo)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The original code sent `weight mod 64` for every weight as "parity". With
L = 20 a weight takes 41 values, all distinct mod 64, so the "parity" WAS
the weight: a passive listener recovered the whole key from the two parity
messages. Any reconciliation message leaks something; the job is to leak
little, count it, and compress it away:

  1. Reconcile with a SMALL modulus m (default 4). This corrects any weight
     that differs by less than m/2 and leaks H(w mod m) bits per weight.
  2. Account for that leakage explicitly (residual_bits, assuming uniform
     synchronised weights; tau-transcript leakage is NOT counted here, it
     is what the attack simulator measures).
  3. Derive the key with HKDF-SHA256 (privacy amplification): the 256-bit
     key is only safe if residual_bits comfortably exceeds 256.
  4. Key confirmation with HMAC tags, replacing the original "send your raw
     weights to check" debug step, which disclosed the key.
"""
from __future__ import annotations
import hashlib
import hmac
import math
import numpy as np

from ecc import WeightResidueECC


def residue_leak_bits_per_weight(L: int, m: int) -> float:
    """H(w mod m) for w uniform on [-L, L]."""
    vals = np.arange(-L, L + 1)
    _, counts = np.unique(np.mod(vals, m), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def entropy_budget(K: int, N: int, L: int, m: int):
    per_w = math.log2(2 * L + 1)
    leak = residue_leak_bits_per_weight(L, m)
    return K * N * leak, K * N * (per_w - leak)


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm, t, i = b"", b"", 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return okm[:length]


def derive_key(weights: np.ndarray, context: bytes) -> str:
    ikm = weights.astype(np.int64).tobytes()
    salt = hashlib.sha256(b"ntkx-salt|" + context).digest()
    return hkdf_sha256(ikm, salt, b"ntkx session key v1").hex()


def confirm_tag(key_hex: str, role: str) -> str:
    return hmac.new(bytes.fromhex(key_hex), f"confirm|{role}".encode(), hashlib.sha256).hexdigest()


def context_bytes(K, N, L, B, MAX, schedule_desc) -> bytes:
    return f"K={K}|N={N}|L={L}|B={B}|MAX={MAX}|{sorted(schedule_desc.items())}".encode()


def reconcile_ensemble(tpms_a, tpms_b, m: int, context: bytes = b""):
    """
    In-process reconciliation for a P-pair ensemble: the Figure 8 parity
    exchange runs on every pair (all parities travel in one message each
    way), then ONE key is derived from all pairs' weights together.
    """
    ecc = WeightResidueECC(modulus=m)
    corrected = 0
    for ta, tb in zip(tpms_a, tpms_b):
        total = ta.flat_weight_count()
        front, back = slice(0, total // 2), slice(total // 2, total)
        par_a = ecc.encode(ta.get_flat_weights(front))
        par_b = ecc.encode(tb.get_flat_weights(back))
        fixed_b, nb = ecc.decode(tb.get_flat_weights(front), par_a)
        fixed_a, na = ecc.decode(ta.get_flat_weights(back), par_b)
        tb.set_flat_weights(fixed_b, front)
        ta.set_flat_weights(fixed_a, back)
        corrected += na + nb
    wa = np.concatenate([t.weights for t in tpms_a], axis=0)
    wb = np.concatenate([t.weights for t in tpms_b], axis=0)
    leak, residual = entropy_budget(tpms_a[0].K, tpms_a[0].N, tpms_a[0].L, m)
    P = len(tpms_a)
    return {
        "comm_rounds": 1,
        "corrected": corrected,
        "match_post": float(np.mean(wa == wb)),
        "key_a": derive_key(wa, context),
        "key_b": derive_key(wb, context),
        "leak_bits": P * leak, "residual_bits": P * residual,
    }


def reconcile_and_derive(tpm_a, tpm_b, m: int, context: bytes = b""):
    """In-process version of the two-message parity exchange (Figure 8)."""
    ecc = WeightResidueECC(modulus=m)
    total = tpm_a.flat_weight_count()
    front, back = slice(0, total // 2), slice(total // 2, total)
    par_a = ecc.encode(tpm_a.get_flat_weights(front))   # Alice -> Bob
    par_b = ecc.encode(tpm_b.get_flat_weights(back))    # Bob -> Alice
    fixed_b, nb = ecc.decode(tpm_b.get_flat_weights(front), par_a)
    fixed_a, na = ecc.decode(tpm_a.get_flat_weights(back), par_b)
    tpm_b.set_flat_weights(fixed_b, front)
    tpm_a.set_flat_weights(fixed_a, back)
    leak, residual = entropy_budget(tpm_a.K, tpm_a.N, tpm_a.L, m)
    return {
        "comm_rounds": 1,
        "corrected": na + nb,
        "match_post": float(np.mean(tpm_a.weights == tpm_b.weights)),
        "key_a": derive_key(tpm_a.weights, context),
        "key_b": derive_key(tpm_b.weights, context),
        "leak_bits": leak, "residual_bits": residual,
    }
