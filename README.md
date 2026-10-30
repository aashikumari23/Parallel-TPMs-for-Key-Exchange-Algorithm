# NT_ensemble

Neural cryptography key exchange with a **parallel ensemble of Tree Parity Machines (TPMs)**.

Instead of one TPM pair with a large weight bound `L`, Alice and Bob run `P` independent TPM pairs side by side and derive **one** session key from the weights of **all** pairs together. An eavesdropper (Eve) learns nothing useful unless she synchronises with every pair. If her success on each pair were independent with probability `p`, her chance of getting the key would be about `p^P`.

Setting `P = 1` reproduces the single-pair protocol of the paper (batch scheme, 1-h random walk, stop counter), so the ensemble is a strict extension.

> Research code. The security claim above is a hypothesis to be measured, not a proven result. See [Limitations](#limitations).

---

## How it works

### Protocol phases (`party.py`)

1. **Handshake.** Both sides send their parameters (including `P`) and abort on any mismatch.
2. **Synchronisation.** Each of the `P` pairs runs the batch scheme with the chosen hop schedule. All `P` tau batches travel in the **same message**, so the number of round trips equals the slowest pair's, not the sum.
3. **Reconciliation.** Residue parity exchange (`weight mod m`, small `m`). Alice is authoritative for the front half of each pair's weights, Bob for the back half. All parities go in one message each way.
4. **Key derivation.** HKDF-SHA256 over the concatenated weights of all pairs, bound to the session parameters.
5. **Key confirmation.** Both sides exchange HMAC tags. Raw weights are never sent.

### Stopping

A pair that reaches its stop counter (`MAX`) freezes and sends no more taus (`None` in the message). The counter is computed from exchanged taus, so both sides agree on which pairs are finished without extra messages. The exchange ends when every pair has finished.

### Why this is not a parallel-weight-update scheme

Within one TPM, each weight update depends on the previous one, so a single pair stays strictly sequential. The parallelism is **across** pairs. This addresses the paper's stated future work (Section VI) without breaking that dependency.

---

## Files

| File | Role |
|---|---|
| `ensemble.py` | `EnsembleSession`: `P` `SyncSession` objects for one party, advanced in lock-step. Derives a per-pair input seed from the shared seed (pair 0 uses the shared seed itself). |
| `engine.py` | `SyncSession`: transport-independent protocol engine for one pair (Algorithm 1 without the socket). |
| `tpm.py` | `TreeParityMachine`: weights, output (tau, sigma), 1-h random-walk update. |
| `hop_schedule.py` | Hop schedules: `conventional`, `fixed`, `paper` (1-h alternating). |
| `party.py` | One party over a real TCP socket, all five phases above. |
| `network.py` | Length-prefixed JSON messaging over TCP. |
| `ecc.py` | `WeightResidueECC`: residue-parity reconciliation. |
| `keyderive.py` | Leakage accounting (`entropy_budget`), HKDF-SHA256, HMAC key confirmation, in-process `reconcile_ensemble`. |
| `attacks.py` | Passive attackers: `simple`, `geometric`, `majority`. |
| `sim.py` | In-process simulator (Alice, Bob, optional Eve) used by the security CLI and the web UI. |
| `security.py` | CLI: attack success rate with a 95% Wilson interval. |
| `stats.py` | Wilson interval helper. |
| `server.py` + `frontend/` | Flask backend and browser UI streaming a live exchange over Server-Sent Events. |
| `run_alice.py`, `run_bob.py` | Networked entry points. |
| `cli_common.py` | Shared argparse flags. |

---

## Setup

Python 3.10 or newer (the code uses `X | None` type syntax).

```bash
pip install -r requirements.txt     # numpy>=1.24, flask>=2.3
```

---

## Usage

### 1. Networked key exchange (two terminals)

```bash
# Terminal 1
python run_alice.py --seed 2024 --P 4 --hop-mode paper --L 20

# Terminal 2 (same parameters, same seed)
python run_bob.py   --seed 2024 --P 4 --hop-mode paper --L 20
```

Alice listens, Bob dials. Both print `Key confirmation: MATCH` when the derived keys agree. `--seed` is required and must be identical on both sides; any parameter mismatch aborts at the handshake.

### 2. Attack success rate

```bash
python security.py --hop-mode paper --L 20 --attack majority --M 50 --trials 100 --P 4
```

Reports, per trial, the communication rounds and the attacker's best overlap, then the success rate with a 95% Wilson interval.

### 3. Web UI

```bash
python server.py            # then open http://127.0.0.1:5000
```

Live weight-agreement grid, overlap and hop charts, attacker progress (weakest pair), and a final result panel. Choose `P` from 1 to 8 in the form.

---

## Parameters

| Flag | Default | Meaning |
|---|---|---|
| `--P` | 1 | Parallel TPM pairs. The key needs all of them. `1` is the paper protocol. |
| `--K` | 3 | Hidden units per TPM |
| `--N` | 100 | Inputs per hidden unit |
| `--L` | 20 | Weight bound (weights in `[-L, L]`) |
| `--B` | 10 | Batch size |
| `--MAX` | 200 | Stop counter |
| `--hop` | 3 | Step on odd rounds (`paper`) or every round (`fixed`) |
| `--hop-mode` | `paper` | `conventional`, `paper`, or `fixed` |
| `--input-mode` | `fresh` | `fresh` random inputs, or `scrn_batch` (paper's shift-and-change per batch) |
| `--count-mode` | `pseudocode` | `pseudocode`: count per tau in a fully matched batch. `text`: count per fully matched batch. |
| `--ecc-modulus` | 4 | Residue modulus. Corrects differences below `m/2`, leaks about `log2(m)` bits per weight. |
| `--seed` | 42 (required for `run_alice`/`run_bob`) | Public input seed, identical on both sides |
| `--host`, `--port` | `127.0.0.1`, 9999 | Socket endpoint |

`security.py` additionally takes `--attack {simple,geometric,majority}`, `--M` (attacker networks, default 50), `--trials` (default 50), `--seed0`.

---

## Security model

- **Threat model (Kerckhoffs).** Eve knows `K, N, L, B, MAX`, the hop schedule and the public input seed, and sees every tau batch. She does not see either party's weights.
- **Attacks simulated.** `simple` (independent TPMs), `geometric` (Klimov, Mityagin, Shamir), `majority` (Shacham et al., the attack used in the paper's Table 2).
- **Success criterion.** For one pair, some attacker's weights agree with Alice's on at least 98% of positions. For the ensemble, Eve succeeds only if she does so on **every** pair; the reported `eve_best` is her weakest pair.
- **Reconciliation leakage** is counted explicitly (`leak_bits`, `residual_entropy_bits`) and scales with `P`. `party.py` warns if residual entropy falls under 384 bits. Leakage through the tau transcript is not counted there; the attack simulator measures it instead.
- **Fixes over the original repo.** The original sent `weight mod 64` as "parity", which for `L = 20` equalled the weight and exposed the key to a passive listener. It also sent raw weights for confirmation. Both are replaced by small-modulus reconciliation with leakage accounting, HKDF privacy amplification, and HMAC key confirmation.

---

## Limitations

- **Independence is assumed, not shown.** The `p^P` estimate only holds if Eve's success on different pairs is independent. `ensemble.py` refers to `experiments/ensemble_independence.py` to measure this, but that script is **not included** in this archive.
- Reconciliation corrects only differences smaller than `m/2` per weight; larger residual mismatches break key agreement.
- Simulation results use the observer's view of both weight vectors and are not a proof of security.
- The networked parties print the derived key to the log (`Key: ...`). Remove that line outside demos.
- No automated tests are included.

---

## Known inconsistencies in the source

- `cli_common.py` says it is shared with `demo.py`, but `NT_ensemble` has no `demo.py` (the sibling `NT_adaptive_batch` does).
- The docstring in `run_bob.py` shows a `run_alice.py` example command.
- `security.py` defaults to `L = 12`, while `run_alice.py`/`run_bob.py` default to `L = 20` and the web UI to `L = 12`.
