"""
sim.py
------
In-process simulator: two SyncSession objects (Alice, Bob) plus optional
eavesdroppers, exchanging tau batches through memory instead of TCP. Used
by the benchmark CLI, the security CLI, the tests and the web frontend.

The simulator is an OBSERVER: it can read both weight vectors, so it reports
the true overlap next to the parties' public-transcript estimate. Nothing the
observer sees is fed back into the protocol.
"""
from __future__ import annotations
import time
import numpy as np

from ensemble import EnsembleSession
from hop_schedule import make_schedule
from attacks import AttackerEnsemble
from keyderive import reconcile_ensemble


def true_overlap(wa, wb):
    """Normalised overlap rho = mean over hidden units of wa.wb/|wa||wb|."""
    num = np.sum(wa * wb, axis=1).astype(float)
    den = np.sqrt(np.sum(wa * wa, axis=1) * np.sum(wb * wb, axis=1)).astype(float)
    den[den == 0] = 1.0
    return float(np.mean(num / den))


def _eve_bests(eves, alice):
    return [e.best_overlap(p.tpm.weights) for e, p in zip(eves, alice.pairs)] if eves else None


def _snapshot(alice, bob, eves, hops, with_grid):
    wa, wb = alice.weights, bob.weights
    eq = wa == wb
    per_pair = _eve_bests(eves, alice)
    active_hops = [max(h) for h in hops if h]
    snap = {
        "comm_rounds": alice.comm_rounds,
        "learning_rounds": alice.learning_rounds,
        "match_rate": float(np.mean(eq)),
        "rho_true": float(np.mean([true_overlap(a.tpm.weights, b.tpm.weights)
                                   for a, b in zip(alice.pairs, bob.pairs)])),
        "hop": max(active_hops) if active_hops else 1,
        "pairs_done": sum(p.done for p in alice.pairs),
        # Eve needs EVERY pair, so her progress toward the key is her weakest pair.
        "eve_best": min(per_pair) if per_pair else None,
        "eve_per_pair": per_pair,
    }
    if with_grid:
        snap["agree"] = "".join("1" if v else "0" for v in eq.ravel())
    return snap


def simulate(K=3, N=100, L=20, B=10, MAX=200, hop=3, mode="paper", seed=1,
             count_mode="pseudocode", scrn_bits=1, ecc_modulus=4,
             attack=None, M=20, max_comm_rounds=200_000, snapshot_every=0,
             on_snapshot=None, input_mode="fresh",
             snapshot_min_seconds=0.0, on_progress=None, P=1):
    """
    P                    : number of parallel TPM pairs (ensemble.py); 1 is the
                           paper's single-pair protocol
    snapshot_every       : emit a snapshot every this many communication rounds
    snapshot_min_seconds : the per-weight grid is attached at most this often
                           (keeps long paper-scale runs from flooding the page)
    on_progress          : optional callback(comm_rounds) that may return True
                           to stop early (used by the web server's Stop button)
    """
    ss = np.random.SeedSequence(seed)
    sa, sb, se = ss.spawn(3)
    input_seed = int(ss.generate_state(1)[0])
    rngs = lambda s: [np.random.default_rng(s)] if P == 1 else [np.random.default_rng(c) for c in s.spawn(P)]
    common = dict(mode=mode, hop=hop, scrn_bits=scrn_bits, count_mode=count_mode, input_mode=input_mode)
    alice = EnsembleSession(P, K, N, L, B, MAX, input_seed, weight_rngs=rngs(sa), role="alice", **common)
    bob = EnsembleSession(P, K, N, L, B, MAX, input_seed, weight_rngs=rngs(sb), role="bob", **common)
    eves = [AttackerEnsemble(attack, M, K, N, L, r) for r in rngs(se)] if attack else None

    t0 = time.perf_counter()
    last_snap = -1e9
    hops = [[1]] * P
    while not alice.done and alice.comm_rounds < max_comm_rounds:
        ta = alice.make_batches()
        tb = bob.make_batches()
        if eves is not None:
            for i in alice.active:
                eves[i].make_batch(alice.pairs[i].last_batch_inputs)
        active = alice.active
        hops = alice.apply_peer_batches(tb)
        bob.apply_peer_batches(ta)
        if eves is not None:
            for i in active:
                eves[i].apply(ta[i], tb[i], hops[i])
        if on_progress and alice.comm_rounds % 50 == 0 and on_progress(alice.comm_rounds):
            break
        if on_snapshot and snapshot_every and alice.comm_rounds % snapshot_every == 0:
            # Numbers are cheap and go out every time; the per-weight agreement
            # string (P*K*N characters) goes out at most every snapshot_min_seconds.
            with_grid = time.perf_counter() - last_snap >= snapshot_min_seconds
            if with_grid:
                last_snap = time.perf_counter()
            on_snapshot(_snapshot(alice, bob, eves, hops, with_grid))
    if on_snapshot and snapshot_every and alice.comm_rounds:
        on_snapshot(_snapshot(alice, bob, eves, hops, True))          # final state
    sync_s = time.perf_counter() - t0

    pre_ecc = float(np.mean(alice.weights == bob.weights))
    ctx = f"P={P}|{mode}|{hop}".encode()
    rec = reconcile_ensemble([p.tpm for p in alice.pairs], [p.tpm for p in bob.pairs], ecc_modulus, ctx)
    per_pair = _eve_bests(eves, alice)
    return {
        "mode": mode, "hop": hop, "seed": seed, "P": P,
        "comm_rounds": alice.comm_rounds + rec["comm_rounds"],
        "learning_rounds": alice.learning_rounds,
        "updates": alice.updates,
        "hop_usage": {int(k): int(v) for k, v in alice.hop_hist.items()},
        "timed_out": not alice.done,
        "pair_finish_rounds": alice.finished_at,
        "match_pre_ecc": pre_ecc,
        "match_post_ecc": rec["match_post"],
        "keys_match": rec["key_a"] == rec["key_b"],
        "key_a": rec["key_a"], "key_b": rec["key_b"],
        "leak_bits": rec["leak_bits"], "residual_entropy_bits": rec["residual_bits"],
        "ecc_corrected": rec["corrected"],
        # Eve gets the key only if she broke EVERY pair.
        "eve_per_pair": per_pair,
        "eve_pairs_broken": sum(b >= 0.98 for b in per_pair) if eves else None,
        "eve_best": min(per_pair) if eves else None,
        "eve_success": all(b >= 0.98 for b in per_pair) if eves else None,
        "sync_seconds": sync_s,
        "bits_on_wire": alice.bits_sent + bob.bits_sent,
        "input_mode": input_mode,
    }
