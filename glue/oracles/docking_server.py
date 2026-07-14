"""Persistent docking server — construct the docking oracle ONCE, serve many batches.

Why
---
At publication scale (5,000 training steps; campaign Logs/030) the cross-env baselines
(FragGFN / RxnFlow / SCENT) can no longer afford the old per-step pattern of spawning
``conda run -n rgfn scripts/score_batch.py`` every step: constructing the docking oracle
(loading the receptors + gnina + QuickVina2-GPU context) costs ~35-44 s, which over 5,000
steps is ~2 days of pure startup per run. This server pays that construction cost **once**,
then answers docking requests over a Unix-domain socket for the life of the SLURM job.

Two payoffs:
  1. **Speed** — a request is just SMILES in, scores out; no re-construction.
  2. **Utilization** — the server measures how much of its uptime it actually spends docking
     (busy) vs. waiting for requests (idle). A low busy fraction means the docker has spare
     capacity, so additional random seeds can share one server (interleaving their docking)
     for error bars at little extra wall-clock — only more GPUs (the campaign's seed decision).

Design
------
* Transport: newline-delimited JSON over ``AF_UNIX`` (localhost only; no ports, no auth
  needed — the socket lives in the job's run dir on ``$SCRATCH``).
* Concurrency: connections are handled on threads, but the actual docking is serialized by a
  lock — one GPU docks one batch at a time regardless. That serialization is exactly what lets
  N seeds share the server: their training runs concurrently, their docking queues here.
* The **client half** of this module (``DockingServerClient`` + protocol constants) imports
  nothing from ``glue``/``rgfn`` (glue imports are lazy, inside the server functions), so it is
  safe to import from any conda env. The baselines carry a copy of the same tiny client.

Protocol (one JSON object per line, both directions):
    -> {"cmd": "dock", "smiles": ["CCO", ...]}
    <- {"labels": [<float|null>, ...], "details": [<dict|null>, ...] | null}
    -> {"cmd": "ping"}                 <- {"ok": true, "oracle": "docking_6td3_gpu"}
    -> {"cmd": "stats"}                <- {"uptime_s":.., "busy_s":.., "idle_s":.., "utilization":.., "n_requests":.., "n_mols":..}
    -> {"cmd": "shutdown"}             <- {"ok": true}   (server writes final stats, then exits)

CLI (started by the submit script before the training client):
    python -m glue.oracles.docking_server --oracle docking_6td3_gpu \
        --socket $RUN/dock.sock --stats $RUN/dock_server_stats.json \
        --oracle-arg num_modes=9
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- protocol
# Kept import-light (stdlib only above) so the client half is importable from any env.
_RECV_BUF = 1 << 20  # 1 MiB readline cap per message; a 200-SMILES batch is a few KB.


class DockingServerClient:
    """Minimal stdlib client for a :class:`DockingServer` (safe to import from any env).

    A docking reward generator holds one of these for the whole run and calls :meth:`dock`
    every training step, reusing a single persistent connection instead of spawning a
    subprocess. ``dock`` returns raw oracle scores (lower-is-better dvina/Vina; ``None`` on a
    per-molecule oracle failure), mirroring ``scripts/score_batch.py`` semantics.
    """

    def __init__(self, socket_path: str, timeout: float = 3600.0):
        self.socket_path = str(socket_path)
        self.timeout = timeout

    def _roundtrip(self, msg: Dict) -> Dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(self.timeout)
            s.connect(self.socket_path)
            f = s.makefile("rwb", buffering=0)
            f.write((json.dumps(msg) + "\n").encode())
            line = f.readline(_RECV_BUF)
            if not line:
                raise ConnectionError("docking server closed the connection with no reply")
            return json.loads(line.decode())

    def dock(self, smiles: List[str]):
        """Return ``(labels, details)`` for a batch; ``details`` is None or per-mol dicts."""
        resp = self._roundtrip({"cmd": "dock", "smiles": list(smiles)})
        return resp.get("labels", []), resp.get("details")

    def ping(self) -> Dict:
        return self._roundtrip({"cmd": "ping"})

    def stats(self) -> Dict:
        return self._roundtrip({"cmd": "stats"})

    def shutdown(self) -> Dict:
        return self._roundtrip({"cmd": "shutdown"})

    def wait_until_ready(self, timeout: float = 600.0, poll: float = 1.0) -> bool:
        """Block until the socket exists and answers a ping (server finished constructing the
        oracle), or ``timeout`` elapses. Returns True on success."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if Path(self.socket_path).exists() and self.ping().get("ok"):
                    return True
            except (OSError, ConnectionError, ValueError):
                pass
            time.sleep(poll)
        return False


def client_from_env(env_var: str = "RGFN_DOCK_SOCKET", timeout: float = 3600.0):
    """Return a :class:`DockingServerClient` if ``env_var`` names a socket, else ``None``.

    A docking reward bridge calls this at construction: if the submit script launched a
    persistent server and exported ``RGFN_DOCK_SOCKET``, the bridge talks to it (fast, no
    per-step subprocess); otherwise the bridge falls back to its per-step ``score_batch.py``
    spawn (backward-compatible). Import-safe from any conda env (stdlib only)."""
    path = os.environ.get(env_var)
    if not path:
        return None
    return DockingServerClient(path, timeout=timeout)


# ----------------------------------------------------------------------------- server
class _ServerState:
    """Shared docking state: the oracle, a GPU-serialization lock, and busy/idle accounting."""

    def __init__(self, oracle, oracle_name: str, stats_path: Optional[Path]):
        self.oracle = oracle
        self.oracle_name = oracle_name
        self.stats_path = stats_path
        self.lock = threading.Lock()  # serializes docking (one GPU batch at a time)
        self.start = time.monotonic()
        self.busy_s = 0.0
        self.n_requests = 0
        self.n_mols = 0
        # Detailed oracles (the 6TD3 differential) expose score_detailed; mirror score_batch.
        sd = getattr(oracle, "score_detailed", None)
        self._score_detailed = sd if callable(sd) else None

    def dock(self, smiles: List[str]):
        # Only the docking call is under the lock + counted as busy; JSON I/O stays parallel.
        with self.lock:
            t0 = time.monotonic()
            if self._score_detailed is not None:
                details = self._score_detailed(smiles)
                labels = [d.get("dvina", None) for d in details]
            else:
                labels = list(self.oracle.score(smiles))
                details = None
            self.busy_s += time.monotonic() - t0
            self.n_requests += 1
            self.n_mols += len(smiles)
        # NaN -> null so the JSON is valid and the client sees a clean failure sentinel.
        labels = [None if (v is None or v != v) else float(v) for v in labels]
        self._flush_stats()
        return labels, details

    def stats(self) -> Dict:
        uptime = time.monotonic() - self.start
        return {
            "oracle": self.oracle_name,
            "uptime_s": round(uptime, 3),
            "busy_s": round(self.busy_s, 3),
            "idle_s": round(max(uptime - self.busy_s, 0.0), 3),
            "utilization": round(self.busy_s / uptime, 4) if uptime > 0 else 0.0,
            "n_requests": self.n_requests,
            "n_mols": self.n_mols,
            "s_per_mol": round(self.busy_s / self.n_mols, 4) if self.n_mols else 0.0,
        }

    def _flush_stats(self) -> None:
        if self.stats_path is not None:
            try:
                self.stats_path.write_text(json.dumps(self.stats(), indent=2))
            except OSError:
                pass


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        state: _ServerState = self.server.state  # type: ignore[attr-defined]
        for raw in self.rfile:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode())
            except (ValueError, UnicodeDecodeError):
                self._send({"error": "bad json"})
                continue
            cmd = msg.get("cmd")
            if cmd == "dock":
                labels, details = state.dock(list(msg.get("smiles", [])))
                self._send({"labels": labels, "details": details})
            elif cmd == "ping":
                self._send({"ok": True, "oracle": state.oracle_name})
            elif cmd == "stats":
                self._send(state.stats())
            elif cmd == "shutdown":
                self._send({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            else:
                self._send({"error": f"unknown cmd {cmd!r}"})

    def _send(self, obj: Dict) -> None:
        self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.flush()


class _ThreadingUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def _load_oracle(name: str, oracle_kwargs: Dict):
    """Registry mirroring scripts/score_batch.py (lazy glue import — server side only)."""
    from glue.oracles import (
        Docking6TD3GpuOracle,
        Docking6TD3Oracle,
        DockingClpPOracle,
        DockingSEHOracle,
        MockGlueOracle,
    )

    registry = {
        "docking_6td3_gpu": Docking6TD3GpuOracle,
        "docking_6td3": Docking6TD3Oracle,
        "docking_seh": DockingSEHOracle,
        "docking_clpp": DockingClpPOracle,
        "mock": MockGlueOracle,
    }
    if name not in registry:
        raise SystemExit(f"unknown oracle {name!r}; choices: {sorted(registry)}")
    return registry[name](**oracle_kwargs)


def serve(socket_path: str, oracle, oracle_name: str, stats_path: Optional[str] = None) -> None:
    """Bind ``socket_path`` and serve docking requests until a shutdown command arrives."""
    sp = Path(socket_path)
    # AF_UNIX paths are capped at ~108 bytes; a deep $SCRATCH run dir blows past it. Put the
    # socket in a short, node-local dir (e.g. /tmp/rgfn_dock_$SLURM_JOB_ID.sock — shared across
    # all processes of the job on that node) and keep only the stats file in the run dir.
    if len(str(sp)) > 100:
        raise SystemExit(
            f"socket path too long ({len(str(sp))} > 100 bytes for AF_UNIX): {sp}\n"
            "Pass a short --socket like /tmp/rgfn_dock_$SLURM_JOB_ID.sock (node-local)."
        )
    if sp.exists():
        sp.unlink()  # stale socket from a previous (crashed) run
    sp.parent.mkdir(parents=True, exist_ok=True)
    state = _ServerState(oracle, oracle_name, Path(stats_path) if stats_path else None)
    server = _ThreadingUnixServer(str(sp), _Handler)
    server.state = state  # type: ignore[attr-defined]
    print(f"[dock-server] oracle={oracle_name} listening on {sp}", flush=True)
    state._flush_stats()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        state._flush_stats()
        try:
            sp.unlink()
        except OSError:
            pass
        s = state.stats()
        print(
            f"[dock-server] shutdown: {s['n_requests']} requests / {s['n_mols']} mols, "
            f"busy {s['busy_s']:.0f}s of {s['uptime_s']:.0f}s uptime "
            f"(utilization {s['utilization'] * 100:.1f}%)",
            flush=True,
        )


def _coerce(val: str):
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    return val


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oracle", required=True, help="oracle name (e.g. docking_6td3_gpu)")
    ap.add_argument("--socket", required=True, help="AF_UNIX socket path (in the run dir)")
    ap.add_argument("--stats", default=None, help="JSON path to flush utilization stats to")
    ap.add_argument(
        "--oracle-arg", action="append", default=[], help="oracle kwarg KEY=VAL (repeatable)"
    )
    args = ap.parse_args()
    kwargs: Dict = {}
    for item in args.oracle_arg:
        if "=" not in item:
            raise SystemExit(f"--oracle-arg must be KEY=VAL, got {item!r}")
        k, v = item.split("=", 1)
        kwargs[k.strip()] = _coerce(v.strip())
    oracle = _load_oracle(args.oracle, kwargs)
    serve(args.socket, oracle, args.oracle, args.stats)


if __name__ == "__main__":
    main()
