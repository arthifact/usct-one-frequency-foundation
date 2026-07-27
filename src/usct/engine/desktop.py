"""Native Qt studio: a desktop control surface that drives the real engine.

``python -m usct studio`` opens this window (falls back with a clear message if
PySide6 is not installed; ``pip install -e '.[gui]'``). It embeds Matplotlib for
the field and boundary-response plots, keeps one in-process
:class:`~usct.engine.pipeline.PipelineRunner` (so its incremental cache makes
slider drags cheap), and can open/save configs and export PNG/NPZ directly to
disk — everything in the same Python process as your project.

Aesthetic follows the parchment / forest-green / amber "Forest Carbon"
reference: a serif wordmark, green pill selectors, monospace labels, an amber
sound-speed field, and a green boundary-response waveform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from usct.config import load_config  # noqa: E402
from usct.engine import PipelineRunner, PipelineSpec, StageChoice, registry  # noqa: E402
from usct.engine.contracts import STAGES  # noqa: E402
from usct.engine.sampling import boundary_iq, domain_radius, field_values  # noqa: E402

# ---- palette (Forest Carbon) ----
PAPER = "#f4ecd6"
PAPER2 = "#efe6cc"
INK = "#22301d"
GREEN = "#1c4a34"
ORANGE = "#ee7a18"
FILE = "#d9481f"

_CMAP_STOPS: dict[str, list[list[int]]] = {
    "amber": [[247, 240, 220], [244, 196, 90], [240, 150, 40], [196, 86, 26], [120, 42, 16]],
    "inferno": [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]],
    "viridis": [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
    "ice": [[8, 17, 34], [28, 72, 120], [64, 145, 190], [150, 205, 220], [240, 249, 252]],
    "moss": [[24, 38, 26], [36, 74, 52], [74, 120, 72], [150, 180, 110], [233, 231, 200]],
    "sand": [[38, 30, 18], [110, 84, 44], [176, 140, 86], [220, 196, 150], [245, 236, 214]],
}


def _make_cmap(name: str, stops: list[list[int]]) -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(name, [[c / 255 for c in s] for s in stops])
    cmap.set_bad(PAPER)  # outside the disk blends into the parchment
    return cmap


CMAPS = {name: _make_cmap(name, stops) for name, stops in _CMAP_STOPS.items()}

_QSS = f"""
QWidget {{ background: {PAPER}; color: {INK};
  font-family: "SF Mono", "Menlo", "Consolas", monospace; font-size: 12px; }}
QLabel#logo {{ font-family: "New York", "Georgia", serif; font-weight: 700; font-size: 26px; }}
QLabel#section {{ color: #655c3c; font-size: 11px; }}
QLabel#metricval {{ font-size: 16px; }}
QFrame#panel {{ border-right: 1px solid #d9cca6; }}
QFrame#hline {{ background: #d9cca6; max-height: 1px; min-height: 1px; }}
QPushButton[pill="true"] {{ border: 1px solid {GREEN}; color: {GREEN}; background: transparent;
  padding: 4px 12px; border-radius: 6px; }}
QPushButton[pill="true"]:checked {{ background: {GREEN}; color: {PAPER}; }}
QPushButton#run {{ border: 1px solid {INK}; border-radius: 15px;
  min-width: 30px; min-height: 30px; }}
QPushButton#swatch {{ border: 2px solid transparent; border-radius: 11px; min-width: 20px;
  max-width: 20px; min-height: 20px; max-height: 20px; }}
QPushButton#swatch:checked {{ border: 2px solid {INK}; }}
QComboBox {{ background: {PAPER2}; border: 1px solid #d9cca6;
  border-radius: 6px; padding: 5px 8px; }}
QSlider::groove:horizontal {{ height: 2px; background: #d9cca6; }}
QSlider::handle:horizontal {{ background: {GREEN}; width: 12px; height: 12px; margin: -6px 0;
  border-radius: 6px; border: 2px solid {PAPER}; }}
QMenuBar, QMenu {{ background: {PAPER2}; }}
QMenu::item:selected {{ background: {GREEN}; color: {PAPER}; }}
"""


class FloatSlider(QtWidgets.QWidget):
    """A short labelled float slider: ``key  ====o====  value``."""

    changed = QtCore.Signal(float)

    def __init__(self, key: str, lo: float, hi: float, value: float, digits: int) -> None:
        super().__init__()
        self._lo, self._hi, self._digits = lo, hi, digits
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        klab = QtWidgets.QLabel(key)
        klab.setFixedWidth(26)
        klab.setStyleSheet("font-style: italic; color: #7c7350;")
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.setValue(self._to_slider(value))
        self._vlab = QtWidgets.QLabel(f"{value:.{digits}f}")
        self._vlab.setFixedWidth(56)
        self._vlab.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._vlab.setStyleSheet(f"color: {GREEN};")
        row.addWidget(klab)
        row.addWidget(self._slider, 1)
        row.addWidget(self._vlab)
        self._slider.valueChanged.connect(self._on)

    def _to_slider(self, v: float) -> int:
        return int(round((v - self._lo) / (self._hi - self._lo) * 1000))

    def _from_slider(self, s: int) -> float:
        return self._lo + s / 1000 * (self._hi - self._lo)

    def _on(self, s: int) -> None:
        v = self._from_slider(s)
        self._vlab.setText(f"{v:.{self._digits}f}")
        self.changed.emit(v)


class Studio(QtWidgets.QMainWindow):
    """The native engine studio window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("USCT Engine — Studio")
        self.resize(1180, 760)
        self.setStyleSheet(_QSS)

        self._runner = PipelineRunner()
        self._state: dict[str, dict[str, Any]] = {}
        for stage in STAGES:
            comp = self._default_component(stage)
            self._state[stage] = {"component": comp["name"], "params": _defaults(comp)}
        self._view = "specimen"
        self._colormap = "amber"
        self._pattern = 0
        self._last: Any = None

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(130)
        self._debounce.timeout.connect(self._solve)

        self._build_ui()
        self._build_menu()
        self._rebuild_params()
        self._solve()

    # ---- registry helpers ----
    def _schema(self, stage: str) -> list[dict]:
        return registry.schema()[stage]

    def _component(self, stage: str, name: str) -> dict:
        return next(c for c in self._schema(stage) if c["name"] == name)

    def _default_component(self, stage: str) -> dict:
        if stage == "medium":
            return self._component("medium", "feature_phantom")
        return self._schema(stage)[0]

    # ---- UI construction ----
    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QHBoxLayout(root)
        outer.setContentsMargins(20, 18, 20, 18)
        outer.setSpacing(22)

        # ---------- left panel ----------
        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(300)
        col = QtWidgets.QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 20, 0)
        col.setSpacing(6)

        logo = QtWidgets.QLabel("USCT\nENGINE")
        logo.setObjectName("logo")
        col.addWidget(logo)
        col.addSpacing(8)

        col.addWidget(_section("Engine Settings"))
        self._mesh_group = self._pill_row(col, "Mesh", [3, 4, 5, 6],
                                          self._state["mesh"]["params"]["refinements"],
                                          self._on_mesh, str)
        medium_names = [c["name"] for c in self._schema("medium")]
        labels = {"constant": "const", "circular_inclusion": "disk", "feature_phantom": "phantom"}
        self._medium_group = self._pill_row(col, "Medium", medium_names,
                                            self._state["medium"]["component"],
                                            self._on_medium, lambda n: labels.get(n, n))

        col.addWidget(_hline())
        col.addWidget(_section("Phantom"))
        self._preset = QtWidgets.QComboBox()
        self._preset.currentTextChanged.connect(self._on_preset)
        col.addWidget(self._preset)

        col.addWidget(_hline())
        col.addWidget(_section("Colour Scheme"))
        sw_row = QtWidgets.QHBoxLayout()
        sw_row.setSpacing(8)
        self._swatch_group = QtWidgets.QButtonGroup(self)
        self._swatch_group.setExclusive(True)
        for name in CMAPS:
            b = QtWidgets.QPushButton()
            b.setObjectName("swatch")
            b.setCheckable(True)
            b.setChecked(name == self._colormap)
            stops = _CMAP_STOPS[name]
            mid, hi = stops[len(stops) // 2], stops[-1]
            b.setStyleSheet(
                f"#swatch{{background: qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                f"stop:0 rgb({','.join(map(str, stops[1]))}),"
                f"stop:0.5 rgb({','.join(map(str, mid))}),"
                f"stop:1 rgb({','.join(map(str, hi))}));}}"
            )
            b.clicked.connect(lambda _c=False, n=name: self._on_colormap(n))
            self._swatch_group.addButton(b)
            sw_row.addWidget(b)
        sw_row.addStretch(1)
        col.addLayout(sw_row)

        col.addWidget(_hline())
        col.addWidget(_section("Physics Parameters"))
        self._param_box = QtWidgets.QVBoxLayout()
        self._param_box.setSpacing(2)
        col.addLayout(self._param_box)

        col.addWidget(_hline())
        col.addWidget(_section("Forcing"))
        self._pattern_combo = QtWidgets.QComboBox()
        self._pattern_combo.currentIndexChanged.connect(self._on_pattern)
        col.addWidget(self._pattern_combo)

        col.addStretch(1)
        krow = QtWidgets.QHBoxLayout()
        kl = QtWidgets.QLabel("kR (unit disk)")
        kl.setObjectName("section")
        self._metric = QtWidgets.QLabel("—")
        self._metric.setObjectName("metricval")
        krow.addWidget(kl)
        krow.addStretch(1)
        krow.addWidget(self._metric)
        col.addLayout(krow)
        rrow = QtWidgets.QHBoxLayout()
        rl = QtWidgets.QLabel("resolution")
        rl.setObjectName("section")
        self._resolution = QtWidgets.QLabel("—")
        self._resolution.setObjectName("section")
        rrow.addWidget(rl)
        rrow.addStretch(1)
        rrow.addWidget(self._resolution)
        col.addLayout(rrow)
        outer.addWidget(panel)

        # ---------- right column ----------
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)
        self._fig = Figure(figsize=(6, 3.6), facecolor=PAPER)
        self._ax = self._fig.add_axes((0, 0, 1, 1))
        self._ax.set_facecolor(PAPER)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumHeight(360)
        right.addWidget(self._canvas, 1)

        transport = QtWidgets.QHBoxLayout()
        run = QtWidgets.QPushButton("▶")
        run.setObjectName("run")
        run.clicked.connect(self._solve)
        transport.addWidget(run)
        self._view_group = QtWidgets.QButtonGroup(self)
        self._view_group.setExclusive(True)
        for label, mode in (
            ("Specimen c(x)", "specimen"),
            ("Wavefield", "wavefield"),
            ("Mesh", "mesh"),
        ):
            b = QtWidgets.QPushButton(label)
            b.setProperty("pill", True)
            b.setCheckable(True)
            b.setChecked(mode == self._view)
            b.clicked.connect(lambda _c=False, m=mode: self._on_view(m))
            self._view_group.addButton(b)
            transport.addWidget(b)
        transport.addStretch(1)
        self._clock = QtWidgets.QLabel("— ms")
        self._clock.setObjectName("section")
        transport.addWidget(self._clock)
        right.addLayout(transport)

        self._resp_label = QtWidgets.QLabel("Boundary response   D = I + iQ")
        self._resp_label.setObjectName("section")
        right.addWidget(self._resp_label)
        self._wave_fig = Figure(figsize=(6, 1.2), facecolor=PAPER)
        self._wave_ax = self._wave_fig.add_axes((0.01, 0.18, 0.98, 0.8))
        self._wave_canvas = FigureCanvas(self._wave_fig)
        self._wave_canvas.setFixedHeight(130)
        right.addWidget(self._wave_canvas)
        outer.addLayout(right, 1)

    def _pill_row(self, col, label, values, current, handler, textfn):
        row = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel(label)
        lab.setFixedWidth(64)
        lab.setStyleSheet("color: #7c7350;")
        row.addWidget(lab)
        row.addStretch(1)
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for v in values:
            b = QtWidgets.QPushButton(textfn(v))
            b.setProperty("pill", True)
            b.setCheckable(True)
            b.setChecked(str(v) == str(current))
            b.clicked.connect(lambda _c=False, val=v: handler(val))
            group.addButton(b)
            row.addWidget(b)
        col.addLayout(row)
        return group

    def _build_menu(self) -> None:
        bar = self.menuBar()
        filemenu = bar.addMenu("File")
        for label, slot, keys in (
            ("Open config (.toml)…", self._open_config, "Ctrl+O"),
            ("Save spec (.json)…", self._save_spec, "Ctrl+S"),
            ("Export field (.png)…", self._export_png, "Ctrl+E"),
            ("Export result (.npz)…", self._export_npz, "Ctrl+Shift+E"),
        ):
            act = QtGui.QAction(label, self)
            act.setShortcut(keys)
            act.triggered.connect(slot)
            filemenu.addAction(act)

    # ---- parameter panel (depends on selected medium) ----
    def _rebuild_params(self) -> None:
        while self._param_box.count():
            item = self._param_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        fwd = self._state["forward"]["params"]
        self._add_slider("f", 0.05, 4.0, fwd["frequency"], 3,
                         lambda v: self._set("forward", "frequency", v))
        self._add_slider("eta", 0.0, 0.05, fwd["loss"], 3,
                         lambda v: self._set("forward", "loss", v))
        med = self._state["medium"]
        p = med["params"]
        if med["component"] == "feature_phantom" and p.get("preset") == "custom":
            self._add_slider("s", 1.0, 200.0, p["sharpness"], 0,
                             lambda v: self._set("medium", "sharpness", v))
            self._add_slider("c", 0.5, 2.0, p["background_speed"], 2,
                             lambda v: self._set("medium", "background_speed", v))
        elif med["component"] == "circular_inclusion":
            self._add_slider("r", 0.02, 0.9, p["radius"], 3,
                             lambda v: self._set("medium", "radius", v))
            self._add_slider("c", 0.5, 2.5, p["inclusion_speed"], 2,
                             lambda v: self._set("medium", "inclusion_speed", v))
        elif med["component"] == "constant":
            self._add_slider("c", 0.5, 2.5, p["background_speed"], 2,
                             lambda v: self._set("medium", "background_speed", v))

        # phantom preset combo enabled only for feature_phantom
        self._preset.blockSignals(True)
        self._preset.clear()
        preset_param = next((q for q in self._component("medium", med["component"])["params"]
                             if q["name"] == "preset"), None)
        if preset_param:
            self._preset.addItems(preset_param["choices"])
            self._preset.setCurrentText(p.get("preset", preset_param["choices"][0]))
            self._preset.setEnabled(True)
        else:
            self._preset.addItem("— n/a for " + med["component"])
            self._preset.setEnabled(False)
        self._preset.blockSignals(False)

        # forcing patterns
        self._pattern_combo.blockSignals(True)
        self._pattern_combo.clear()
        self._pattern_combo.addItems(self._state["forcing"]["params"]["patterns"])
        self._pattern = min(self._pattern, self._pattern_combo.count() - 1)
        self._pattern_combo.setCurrentIndex(self._pattern)
        self._pattern_combo.blockSignals(False)
        self._refresh_kr()

    def _add_slider(self, key, lo, hi, value, digits, setter) -> None:
        s = FloatSlider(key, lo, hi, value, digits)
        s.changed.connect(setter)
        self._param_box.addWidget(s)

    # ---- state mutation ----
    def _set(self, stage: str, key: str, value: float) -> None:
        self._state[stage]["params"][key] = value
        if key == "frequency":
            self._refresh_kr()
        self._debounce.start()

    def _on_mesh(self, value: int) -> None:
        self._state["mesh"]["params"]["refinements"] = value
        self._debounce.start()

    def _on_medium(self, name: str) -> None:
        comp = self._component("medium", name)
        self._state["medium"] = {"component": name, "params": _defaults(comp)}
        self._rebuild_params()
        self._debounce.start()

    def _on_preset(self, name: str) -> None:
        if name.startswith("—"):
            return
        self._state["medium"]["params"]["preset"] = name
        self._rebuild_params()
        self._debounce.start()

    def _on_colormap(self, name: str) -> None:
        self._colormap = name
        if self._last is not None:
            self._paint(self._last)

    def _on_pattern(self, index: int) -> None:
        self._pattern = max(0, index)
        self._debounce.start()

    def _on_view(self, mode: str) -> None:
        self._view = mode
        self._debounce.start()

    def _refresh_kr(self) -> None:
        f = self._state["forward"]["params"]["frequency"]
        r = self._state["mesh"]["params"].get("radius", 1.0)
        c = self._bg_speed()
        self._metric.setText(f"{2 * np.pi * f * r / max(c, 1e-9):.2f}")

    def _bg_speed(self) -> float:
        med = self._state["medium"]
        p = med["params"]
        if med["component"] == "feature_phantom":
            return p["background_speed"] if p.get("preset") == "custom" else 1.0
        if med["component"] == "circular_inclusion":
            return min(p["background_speed"], p["inclusion_speed"])
        return p["background_speed"]

    # ---- run the real engine ----
    def _spec(self) -> PipelineSpec:
        return PipelineSpec(**{
            stage: StageChoice(self._state[stage]["component"], dict(self._state[stage]["params"]))
            for stage in STAGES
        })

    def _solve(self) -> None:
        self.setCursor(QtCore.Qt.CursorShape.BusyCursor)
        timer = QtCore.QElapsedTimer()
        timer.start()
        try:
            result = self._runner.run(self._spec())
        except Exception as error:  # keep the window alive, show the message
            self._clock.setText(f"engine error: {error}")
            self.unsetCursor()
            return
        coords = result.domain.coordinates
        radius = domain_radius(coords)
        values, self._pattern, _ = field_values(result, self._view, self._pattern)
        theta, ii, qq = boundary_iq(result, self._pattern)
        c_min = float(np.min(result.medium.sound_speed))
        freq = self._state["forward"]["params"]["frequency"]
        wavelength = c_min / max(freq, 1e-9)
        self._last = {
            "x": coords[:, 0], "y": coords[:, 1], "triangles": result.domain.triangles,
            "values": values, "radius": radius,
            "range": (float(np.nanmin(values)), float(np.nanmax(values))),
            "theta": theta, "i": ii, "q": qq,
            "names": list(result.forcing.names), "pattern": self._pattern,
            "dof": int(result.domain.basis.N),
            "kR": 2 * np.pi * freq * radius / max(c_min, 1e-9),
            "points_per_wavelength": wavelength / result.domain.h_max,
        }
        self._paint(self._last)
        self._paint_wave(self._last)
        self._metric.setText(f"{self._last['kR']:.2f}")
        self._set_resolution(self._last["points_per_wavelength"])
        self._clock.setText(f"{self._last['dof']:,} dof · {timer.elapsed()} ms")
        self.unsetCursor()

    def _set_resolution(self, ppw: float) -> None:
        # points per shortest wavelength; below ~8 the forward solve is suspect.
        colour = INK if ppw >= 8.0 else FILE
        self._resolution.setText(f"{ppw:.1f} pts/λ")
        self._resolution.setStyleSheet(f"color: {colour};")

    # ---- rendering ----
    def _paint(self, data: dict) -> None:
        ax, R = self._ax, data["radius"]
        ax.clear()
        ax.set_facecolor(PAPER)
        x, y, tri = data["x"], data["y"], data["triangles"]
        if self._view == "mesh":
            ax.triplot(x, y, tri, color=GREEN, linewidth=0.35, alpha=0.75)
            tag = f"mesh · {data['dof']:,} nodes · {len(tri):,} triangles"
        else:
            lo, hi = data["range"]
            if self._view == "wavefield":
                m = max(abs(lo), abs(hi)) or 1.0
                lo, hi = -m, m
            # true P1 field on the actual triangulation — no rasterization
            ax.tripcolor(x, y, tri, data["values"], shading="gouraud",
                         cmap=CMAPS[self._colormap], vmin=lo, vmax=hi)
            tag = (f"Re u(x) · {data['names'][data['pattern']]}" if self._view == "wavefield"
                   else f"specimen c(x) · {self._colormap}")
        ax.add_patch(Circle((0, 0), R, fill=False, edgecolor=INK, linewidth=1.4, alpha=0.6))
        ax.set_xlim(-R * 1.02, R * 1.02)
        ax.set_ylim(-R * 1.02, R * 1.02)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.02, 0.97, tag, transform=ax.transAxes, va="top", ha="left",
                fontsize=9, color=INK, family="monospace")
        self._canvas.draw_idle()

    def _paint_wave(self, data: dict) -> None:
        ax = self._wave_ax
        ax.clear()
        ax.set_facecolor(PAPER)
        theta, ii, qq = data["theta"], data["i"], data["q"]
        deg = np.degrees(theta)
        maxa = max(1e-9, float(np.max(np.abs(ii))), float(np.max(np.abs(qq))))
        ax.bar(deg, ii / maxa, width=360 / len(deg) * 0.7, color="#1d5c42", align="center")
        ax.plot(deg, qq / maxa, color=ORANGE, linewidth=1.4, alpha=0.85)
        ax.axhline(0, color="#d9cca6", linewidth=1)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([])
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.tick_params(colors="#a89a6e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        name = data["names"][data["pattern"]]
        self._resp_label.setText(f"Boundary response   D = I + iQ · {name}")
        self._wave_canvas.draw_idle()

    # ---- file operations ----
    def _open_config(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open config", "", "TOML (*.toml)")
        if not path:
            return
        try:
            config = load_config(Path(path))
        except Exception as error:
            QtWidgets.QMessageBox.warning(self, "Cannot load config", str(error))
            return
        self._state["mesh"]["params"].update(
            radius=config.domain.radius, refinements=config.domain.refinements)
        self._state["forward"]["params"].update(
            frequency=config.wave.frequency, loss=config.wave.loss)
        self._state["forcing"]["params"]["patterns"] = list(config.forcing.patterns)
        med = config.medium
        if med.kind == "constant":
            self._state["medium"] = {"component": "constant",
                                     "params": {"background_speed": med.background_speed}}
        elif med.kind == "circular_inclusion":
            self._state["medium"] = {"component": "circular_inclusion", "params": {
                "background_speed": med.background_speed,
                "center": list(med.inclusion_center),
                "radius": med.inclusion_radius, "inclusion_speed": med.inclusion_speed}}
        self._sync_pills()
        self._rebuild_params()
        self._solve()

    def _save_spec(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save spec", "spec.json", "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        self.statusBar().showMessage(f"wrote {path}", 4000)

    def _export_png(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export field", "field.png", "PNG (*.png)")
        if not path:
            return
        self._fig.savefig(path, dpi=200, facecolor=PAPER)
        self.statusBar().showMessage(f"wrote {path}", 4000)

    def _export_npz(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export result", "result.npz", "NPZ (*.npz)")
        if not path:
            return
        result = self._runner.run(self._spec())
        theta, ii, qq = boundary_iq(result, self._pattern)
        np.savez_compressed(
            path,
            coordinates=result.domain.coordinates,
            triangles=result.domain.triangles,
            sound_speed=result.medium.sound_speed,
            fields=result.forward.solution.field,
            boundary_theta=theta,
            boundary_response=result.response.complex_response,
        )
        self.statusBar().showMessage(f"wrote {path}", 4000)

    def _sync_pills(self) -> None:
        for b in self._mesh_group.buttons():
            b.setChecked(b.text() == str(self._state["mesh"]["params"]["refinements"]))
        labels = {"constant": "const", "circular_inclusion": "disk", "feature_phantom": "phantom"}
        want = labels.get(self._state["medium"]["component"], self._state["medium"]["component"])
        for b in self._medium_group.buttons():
            b.setChecked(b.text() == want)


def _section(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("section")
    return lab


def _hline() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setObjectName("hline")
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    return line


def _defaults(component: dict) -> dict[str, Any]:
    return {p["name"]: json.loads(json.dumps(p["default"])) for p in component["params"]}


def launch() -> int:
    """Create the Qt application and show the studio window."""

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = Studio()
    window.show()
    return app.exec()
