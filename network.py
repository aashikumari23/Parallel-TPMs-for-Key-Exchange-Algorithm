"""
network.py
----------
Minimal length-prefixed JSON messaging over a TCP socket, used as the
communication channel between the two parties (Alice / Bob) of the key
exchange. Also wraps numpy int arrays <-> plain lists and bytes <-> base64
for JSON transport.
"""

from __future__ import annotations
import socket
import struct
import json
import base64
import numpy as np


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _json_object_hook(d):
    if "__bytes__" in d:
        return base64.b64decode(d["__bytes__"])
    return d


class Channel:
    """Wraps a connected TCP socket with send_msg / recv_msg helpers."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send_msg(self, obj):
        payload = json.dumps(obj, default=_json_default).encode("utf-8")
        header = struct.pack(">I", len(payload))
        self.sock.sendall(header + payload)

    def recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Peer closed connection")
            buf += chunk
        return buf

    def recv_msg(self):
        header = self.recv_exact(4)
        (length,) = struct.unpack(">I", header)
        payload = self.recv_exact(length)
        return json.loads(payload.decode("utf-8"), object_hook=_json_object_hook)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def connect_as_server(host: str, port: int) -> Channel:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    conn, _addr = srv.accept()
    srv.close()
    return Channel(conn)


def connect_as_client(host: str, port: int, retries: int = 50, delay: float = 0.2) -> Channel:
    import time
    last_exc = None
    for _ in range(retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            return Channel(sock)
        except (ConnectionRefusedError, OSError) as e:
            last_exc = e
            time.sleep(delay)
    raise last_exc
