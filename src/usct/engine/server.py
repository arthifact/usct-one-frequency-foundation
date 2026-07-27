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
from scipy.interpolate import griddata

from usct.engine import PipelineRunner, PipelineSpec, StageChoice, registry
from usct.engine.contracts import STAGES

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


def _rasterize(coords: np.ndarray, values: np.ndarray, radius: float, n: int) -> list[list]:
    """Sample nodal FEM values onto an ``n x n`` grid, ``None`` outside the disk."""

    axis = np.linspace(-radius, radius, n)
    gx, gy = np.meshgrid(axis, axis)
    grid = griddata(coords, values, (gx, gy), method="linear")
    outside = (gx**2 + gy**2) > radius**2
    out: list[list] = []
    for row_v, row_out in zip(grid, outside, strict=True):
        out.append([None if (o or not np.isfinite(v)) else round(float(v), 5)
                    for v, o in zip(row_v, row_out, strict=True)])
    return out


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    spec = _spec_from(payload["stages"])
    view = payload.get("view", "specimen")
    pattern = int(payload.get("pattern", 0))
    grid_n = int(payload.get("grid", 150))

    with _lock:
        result = _runner.run(spec)

    domain = result.domain
    coords = domain.coordinates
    radius = float(np.max(np.hypot(coords[:, 0], coords[:, 1])))
    speed = result.medium.sound_speed
    field = result.forward.solution.field
    n_patterns = field.shape[1]
    pattern = max(0, min(pattern, n_patterns - 1))

    values = field[:, pattern].real if view == "wavefield" else speed
    grid = _rasterize(coords, values, radius, grid_n)

    response = result.response
    order = np.argsort(response.theta)
    theta = response.theta[order]
    channel = response.complex_response[order, pattern]

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
            "i": [round(float(v), 6) for v in channel.real],
            "q": [round(float(v), 6) for v in channel.imag],
        },
        "meta": {
            "dof": int(domain.basis.N),
            "triangles": int(domain.triangles.shape[0]),
            "h_max": float(domain.h_max),
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
