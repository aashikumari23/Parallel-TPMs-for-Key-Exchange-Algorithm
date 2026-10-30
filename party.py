"""
party.py
--------
Algorithm 1 for ONE party over a real TCP socket, driven by engine.SyncSession.

Phases
  1. Handshake: both sides send their parameters and abort on any mismatch
     (the original repo silently desynchronised if flags differed).
  2. Synchronisation: batch scheme + selectable inputs (fresh or SCRN) +
     selectable hop schedule (conventional or paper 1-h).
  3. Reconciliation: residue parity exchange, Alice authoritative for the
     front half, Bob for the back half (Figure 8), with a small modulus.
  4. Key: HKDF-SHA256 over the weights, bound to the session parameters.
  5. Key confirmation: exchange HMAC tags. Replaces the original step that
     sent raw weights over the wire to check the result.
"""
from __future__ import annotations
import time

from ensemble import EnsembleSession
from ecc import WeightResidueECC
from hop_schedule import make_schedule
from keyderive import derive_key, confirm_tag, context_bytes, entropy_budget
from network import Channel, connect_as_server, connect_as_client


class PartyStats:
    def __init__(self):
        self.rounds = 0
        self.communication_rounds = 0
        self.updates = 0
        self.hop_usage = {}
        self.ecc_corrected = 0
        self.leak_bits = 0.0
        self.residual_bits = 0.0
        self.sync_seconds = 0.0
        self.total_seconds = 0.0
        self.confirmed = False
        self.bits_sent = 0


def run_party(role: str, host: str, port: int, K: int = 3, N: int = 100, L: int = 20,
              B: int = 10, MAX: int = 200, hop: int = 3, hop_mode: str = "paper",
              input_seed: int = 42, scrn_bits: int = 1, ecc_modulus: int = 4,
              count_mode: str = "pseudocode", max_comm_rounds: int = 500_000,
              verbose: bool = True, input_mode: str = "fresh", P: int = 1):
    assert role in ("alice", "bob")
    stats = PartyStats()
    t0 = time.perf_counter()
    log = (lambda m: print(f"[{role.upper()}] {m}", flush=True)) if verbose else (lambda m: None)

    log(f"{'Listening on' if role == 'alice' else 'Dialing'} {host}:{port}")
    chan: Channel = connect_as_server(host, port) if role == "alice" else connect_as_client(host, port)

    schedule = make_schedule(hop_mode, hop)
    params = {"K": K, "N": N, "L": L, "B": B, "MAX": MAX, "seed": input_seed,
              "scrn_bits": scrn_bits, "ecc_modulus": ecc_modulus, "count_mode": count_mode,
              "input_mode": input_mode, "P": P, "schedule": schedule.describe()}
    chan.send_msg({"type": "hello", "params": params})
    peer = chan.recv_msg()
    if peer.get("params") != params:
        chan.close()
        raise RuntimeError(f"Parameter mismatch. Mine: {params} Peer: {peer.get('params')}")
    log(f"Handshake ok: {params}")

    s = EnsembleSession(P, K, N, L, B, MAX, input_seed, mode=hop_mode, hop=hop, scrn_bits=scrn_bits,
                        count_mode=count_mode, input_mode=input_mode, role=role)
    while not s.done and s.comm_rounds < max_comm_rounds:
        # One message carries every pair's taus (None for pairs already finished).
        chan.send_msg({"type": "batch_tau", "taus": s.make_batches()})
        s.apply_peer_batches(chan.recv_msg()["taus"])
        if s.comm_rounds % 500 == 0:
            counts = "/".join(str(pp.count) for pp in s.pairs)
            log(f"comm={s.comm_rounds} counts={counts} (MAX {MAX}), pairs done {P - len(s.active)}/{P}")
    if not s.done:
        log("WARNING: stopped at max_comm_rounds before MAX was reached")
    stats.sync_seconds = time.perf_counter() - t0

    # Reconciliation (Figure 8) on every pair; all parities in one message each way.
    ecc = WeightResidueECC(ecc_modulus)
    total = K * N
    mine, theirs = (slice(0, total // 2), slice(total // 2, total))
    if role == "bob":
        mine, theirs = theirs, mine
    chan.send_msg({"type": "ecc_parity",
                   "parity": [ecc.encode(pp.tpm.get_flat_weights(mine)) for pp in s.pairs]})
    peer_par = chan.recv_msg()["parity"]
    n = 0
    for pp, par in zip(s.pairs, peer_par):
        fixed, c = ecc.decode(pp.tpm.get_flat_weights(theirs), par)
        pp.tpm.set_flat_weights(fixed, theirs)
        n += c
    stats.ecc_corrected = n
    leak, residual = entropy_budget(K, N, L, ecc_modulus)
    stats.leak_bits, stats.residual_bits = P * leak, P * residual

    # One key from ALL pairs: Eve needs every pair to learn anything.
    key = derive_key(s.weights, context_bytes(K, N, L, B, MAX, s.describe()))
    other = "bob" if role == "alice" else "alice"
    chan.send_msg({"type": "confirm", "tag": confirm_tag(key, role)})
    stats.confirmed = chan.recv_msg()["tag"] == confirm_tag(key, other)
    chan.close()

    stats.rounds, stats.communication_rounds = s.learning_rounds, s.comm_rounds + 2
    stats.updates, stats.hop_usage = s.updates, s.hop_hist
    stats.pair_finish_rounds = s.finished_at
    stats.bits_sent = s.bits_sent
    stats.total_seconds = time.perf_counter() - t0
    log(f"Sync done: comm_rounds={stats.communication_rounds} updates={stats.updates} hops={stats.hop_usage}")
    log(f"Reconciliation fixed {n} weights; leaked about {stats.leak_bits:.0f} bits, "
        f"residual entropy about {stats.residual_bits:.0f} bits")
    if stats.residual_bits < 384:
        log("WARNING: residual entropy is under 384 bits; raise N or L, or lower ecc_modulus")
    log(f"Key confirmation: {'MATCH' if stats.confirmed else 'FAILED'}")
    log(f"Key: {key}")
    return s, stats, key
