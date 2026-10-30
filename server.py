"""
server.py
---------
Flask backend for the web frontend (frontend/).

Start the server and open:
http://127.0.0.1:5000

API:
  GET /api/run
      Runs a single TPM key-exchange simulation and streams
      intermediate results using Server-Sent Events (SSE).
"""
from __future__ import annotations

import json
import threading
import time

from flask import Flask, Response, request, send_from_directory

from sim import simulate

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path=""
)

LIMITS = {
    "K": (1, 7),
    "N": (4, 1000),
    "L": (2, 80),
    "B": (1, 50),
    "MAX": (1, 5000),
    "hop": (1, 8),
    "M": (1, 200),
    "ecc_modulus": (2, 64),
    "limit": (100, 2_000_000),
    "P": (1, 8),
}

def _int(args, name, default):
    """Read an integer parameter and clamp it to its allowed range."""
    value = int(args.get(name, default))
    low, high = LIMITS.get(name, (-(10 ** 9), 10 ** 9))
    return max(low, min(high, value))

def _common(args):
    """Build common simulation parameters from request arguments."""
    return {
        "K": _int(args, "K", 3),
        "N": _int(args, "N", 100),
        "L": _int(args, "L", 12),
        "B": _int(args, "B", 10),
        "MAX": _int(args, "MAX", 200),
        "hop": _int(args, "hop", 3),
        "ecc_modulus": _int(args, "ecc_modulus", 4),
        "count_mode": args.get(
            "count_mode",
            "pseudocode"
        ),
        "P": _int(args, "P", 1),
        "input_mode": args.get("input_mode", "fresh") if args.get("input_mode") in ("fresh", "scrn_batch") else "fresh",
    }

def _sse(obj, event="message"):
    """Format an object as a Server-Sent Event."""
    return (
        f"event: {event}\n"
        f"data: {json.dumps(obj)}\n\n"
    )

@app.get("/")
def index():
    """Serve the frontend index.html."""
    return send_from_directory(
        app.static_folder,
        "index.html"
    )

@app.get("/api/run")
def api_run():
    """
    Run one live TPM exchange.

    Results are streamed to the frontend as SSE events:
      - snap: intermediate simulation snapshot
      - done: final simulation result
    """
    args = request.args
    params = _common(args)
    mode = args.get("mode", "paper")
    if mode not in ("conventional", "paper"):
        mode = "paper"
    attack = args.get("attack") or None
    seed = int(args.get("seed", 7))
    M = _int(args,"M",50)
    limit = _int(args, "limit", 200_000)

    def generate():
        queue = []
        finished = threading.Event()
        stop = threading.Event()          # set when the browser disconnects or presses Stop
        result = {}
        def run_simulation():
            try:
                result["data"] = simulate(
                    **params,
                    mode=mode,
                    seed=seed,
                    attack=attack,
                    M=M,
                    snapshot_every=int(
                        args.get("every", 2)
                    ),
                    on_snapshot=queue.append,
                    snapshot_min_seconds=0.2,
                    on_progress=lambda _r: stop.is_set(),
                    max_comm_rounds=limit,
                )
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                finished.set()
        threading.Thread(
            target=run_simulation,
            daemon=True
        ).start()
        try:
            while not finished.is_set() or queue:
                while queue:
                    snapshot = queue.pop(0)
                    yield _sse(
                        snapshot,
                        "snap"
                    )
                time.sleep(0.03)
            yield _sse(
                result.get("data")
                or {
                    "error": result.get(
                        "error",
                        "Unknown error"
                    )
                },
                "done"
            )
        finally:
            stop.set()          # frees the CPU if the page is closed or stopped

    return Response(generate(), mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)