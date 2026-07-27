"""Local standard-library studio server: drives the real engine from a browser.

No third-party web framework and no new dependencies — just ``http.server`` from
the standard library serving a single-page UI and a small JSON API backed by the
actual :class:`~usct.engine.pipeline.PipelineRunner`. Everything runs on
``127.0.0.1``; nothing leaves the machine. Because one runner instance is kept
for the process, its dependency-aware cache makes slider drags cheap: only the
stages a change actually touches are re-solved.

Endpoints
---------
``GET  /``         the studio page (``studio.html`` next to this module)
``GET  /schema``   the component registry as JSON (drives every control)
``POST /run``      body ``{stages, view, pattern}`` -> solved artifacts as JSON
"""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from usct.engine import PipelineRunner, PipelineSpec, StageChoice, registry
from usct.engine.contracts import STAGES
from usct.engine.sampling import boundary_iq, domain_radius, field_values, rasterize

_STUDIO_HTML = Path(__file__).with_name("studio.html")

# One runner per process so the incremental cache survives across requests.
_runner = PipelineRunner()
_lock = threading.Lock()


def _spec_from(stages: dict[str, Any]) -> PipelineSpec:
    choices = {}
    for stage in STAGES:
        entry = stages[stage]
        choices[stage] = StageChoice(
            component=str(entry["component"]),
            params=dict(entry.get("params", {})),
        )
    return PipelineSpec(**choices)


def _grid_to_json(grid: np.ndarray) -> list[list]:
    """Convert an ``n x n`` float grid to nested lists, ``None`` for ``NaN``."""

    return [[None if not np.isfinite(v) else round(float(v), 5) for v in row] for row in grid]


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    spec = _spec_from(payload["stages"])
    view = payload.get("view", "specimen")
    pattern = int(payload.get("pattern", 0))
    grid_n = int(payload.get("grid", 150))

    with _lock:
        result = _runner.run(spec)

    coords = result.domain.coordinates
    radius = domain_radius(coords)
    speed = result.medium.sound_speed
    values, pattern, _ = field_values(result, view, pattern)
    grid = _grid_to_json(rasterize(coords, values, radius, grid_n))
    theta, channel_i, channel_q = boundary_iq(result, pattern)

    freq = float(spec.forward.params.get("frequency", 0.0))
    c_min = float(np.min(speed))
    kr = 2.0 * np.pi * freq * radius / max(c_min, 1e-12)

    return {
        "view": view,
        "grid": grid,
        "grid_range": [float(np.min(values)), float(np.max(values))],
        "speed_range": [float(np.min(speed)), float(np.max(speed))],
        "pattern": pattern,
        "pattern_names": list(result.forcing.names),
        "response": {
            "theta": [round(float(t), 5) for t in theta],
            "i": [round(float(v), 6) for v in channel_i],
            "q": [round(float(v), 6) for v in channel_q],
        },
        "meta": {
            "dof": int(result.domain.basis.N),
            "triangles": int(result.domain.triangles.shape[0]),
            "h_max": float(result.domain.h_max),
            "radius": radius,
            "frequency": freq,
            "kR": kr,
            "c_min": c_min,
            "c_max": float(np.max(speed)),
            "medium": spec.medium.component,
        },
        "reports": [
            {"stage": r.stage, "component": r.component,
             "ms": round(r.seconds * 1000.0, 2), "recomputed": r.recomputed}
            for r in result.reports
        ],
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, _STUDIO_HTML.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/schema":
            body = json.dumps(registry.schema()).encode("utf-8")
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/run":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            body = json.dumps(_run(payload)).encode("utf-8")
        except Exception as error:  # surface the message to the UI, do not crash
            body = json.dumps({"error": str(error)}).encode("utf-8")
            self._send(400, body, "application/json")
            return
        self._send(200, body, "application/json")


def serve(host: str = "127.0.0.1", port: int = 8730, open_browser: bool = True) -> None:
    """Start the local studio server and (optionally) open it in a browser."""

    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print(f"USCT studio running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping studio")
    finally:
        httpd.server_close()
