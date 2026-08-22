"""
Battery Cycling Voltage Plotter
================================

A drag-and-drop GUI for plotting exported voltage-vs-time (.txt) data files,
modeled after CellGraphingNotebook.ipynb (the voltage-plotting portion).

Features
--------
* Drag & drop (or browse for) exported .txt data files.
* Plot voltage vs time, or voltage vs capacity with a selectable cycle range.
* Toggle each file's visibility, rename its legend label, recolor it, remove it.
* Customize the axes: title, axis labels, x/y limits, line width, legend,
  figure size, and plot style.
* Add points of interest (e.g. critical current density) with an arrow-annotated
  label.  Tick "Calculate CCD" and the label auto-computes the critical current
  density from the point's time via the current schedule and electrode area.
* Input parameters like the lithium electrode diameter are fully customizable and
  feed the CCD calculation live.
* Save the figure to PNG/PDF/SVG.

Data formats (one header row, comma-separated), auto-detected by header name:
  * ``TestTime/h,Voltage/V`` — voltage vs time only, and
  * ``TestTime/h,Current/mA,Capacity/mAh,Voltage/V,CycleNo`` — also enables the
    voltage-vs-capacity plot and cycle-range selection.

Run with the interpreter that has matplotlib / numpy / tkinterdnd2 installed, e.g.::

    ~/PyCharmMiscProject/.venv/bin/python3 voltage_plotter.py
"""

import os
import sys
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.style          # required for matplotlib.style.context(...)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.axes_grid1 import Divider, Size

# Optional OS-level drag-and-drop.  Falls back gracefully to the Browse button.
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False


# Default line color cycle (same spirit as the notebook's palette).
DEFAULT_COLORS = ["red", "orange", "yellow", "green", "blue", "purple",
                  "brown", "magenta", "teal", "black"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
class CellFile:
    """One loaded .txt data file.

    Two export formats are supported, distinguished by their header:
      * ``TestTime/h,Voltage/V`` (voltage vs time only), and
      * ``TestTime/h,Current/mA,Capacity/mAh,Voltage/V,CycleNo`` which also
        carries current, capacity, and cycle number.
    Missing columns are stored as ``None``."""

    def __init__(self, path, label, color):
        self.path = path
        self.label = label
        self.color = color
        self.visible = True
        self.linewidth = 0.5
        # time, volt, current, capacity, cycle (any of the last three may be None)
        self.time = self.volt = self.current = self.capacity = self.cycle = None
        self._load(path)

    def _load(self, path):
        # Read the header so columns can be located by name (order-independent).
        with open(path) as fh:
            header = fh.readline().strip()
        cols = [c.strip() for c in header.split(",")]
        idx = {name: i for i, name in enumerate(cols)}

        data = np.loadtxt(path, skiprows=1, delimiter=",")
        data = np.atleast_2d(data)

        def col(name, positional=None):
            if name in idx:
                return data[:, idx[name]]
            if positional is not None and data.shape[1] > positional:
                return data[:, positional]
            return None

        # Fall back to positional [time, voltage] for headerless/renamed 2-col files.
        self.time = col("TestTime/h", positional=0)
        self.volt = col("Voltage/V", positional=1)
        self.current = col("Current/mA")
        self.capacity = col("Capacity/mAh")
        self.cycle = col("CycleNo")

        # Omitted resting time can make the first point start at, e.g., t=20.
        # Shift so every trace begins at t=0 by subtracting its first timestamp.
        if self.time is not None and self.time.size:
            self.time = self.time - self.time[0]

    @property
    def has_capacity(self):
        return self.capacity is not None


class POI:
    """A point of interest annotated on the plot (e.g. a critical current point)."""

    _counter = 0

    def __init__(self):
        POI._counter += 1
        self.name = f"#{POI._counter}"
        self.x = 0.0
        self.y = 0.0
        self.calc_ccd = False      # when True, CCD is derived from x via the
                                   # current schedule (custom label is ignored)
        self.custom_label = ""     # shown when CCD is not being calculated
        self.x_offset = 5.0        # label text offset from the marker (x)
        self.y_offset = 0.05       # label text offset from the marker (y)
        self.color = "black"
        self.visible = True

    def label_text(self, li_area, current_ma=None):
        # When "Calculate CCD" is on, the app passes the current (mA) it derived
        # from this point's time, and the label reports the current density.
        if self.calc_ccd and current_ma is not None and li_area:
            ccd = round(current_ma / li_area, 2)
            return f"{self.name} {ccd} mA/cm^2 CCD"
        if self.custom_label.strip():
            return self.custom_label
        return self.name


# ---------------------------------------------------------------------------
# Scrollable block + a single, global mouse-wheel dispatcher
# ---------------------------------------------------------------------------
class ScrollBlock(ttk.Frame):
    """A self-contained vertically scrollable block.  Put content in ``self.inner``.

    Each block owns a canvas + scrollbar and manages only its own vertical
    scroll.  Wheel scrolling is handled by one global dispatcher
    (:func:`install_wheel_dispatch`) that routes each wheel event to whichever
    block the pointer is physically over — so there is no per-widget binding to
    stack, and no Enter/Leave logic to misfire when the pointer sits on a child.
    """

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>",
                        lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfigure(self._win, width=e.width))

    def contains(self, widget):
        """True if ``widget`` lives anywhere inside this block."""
        while widget is not None:
            if widget is self:
                return True
            widget = getattr(widget, "master", None)
        return False

    def scroll(self, steps):
        # Don't bother (or fight the other block) if there's nothing to scroll.
        first, last = self.canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return
        self.canvas.yview_scroll(steps, "units")


def wheel_steps(event):
    """Scroll amount (in 'units') for a wheel event, cross-platform.

    Linux sends Button-4/5; Windows uses ±120 deltas; macOS uses small integer
    deltas (often ±1) that a naive `delta/120` would truncate to 0."""
    if getattr(event, "num", None) == 4:
        return -3
    if getattr(event, "num", None) == 5:
        return 3
    if abs(event.delta) >= 120:
        return int(-event.delta / 120) * 3
    return -event.delta


def install_wheel_dispatch(root, blocks):
    """Route every wheel event to whichever ScrollBlock the pointer is over."""

    def on_wheel(event):
        target = root.winfo_containing(event.x_root, event.y_root)
        if target is None:
            return
        for block in blocks:
            if block.winfo_exists() and block.contains(target):
                s = wheel_steps(event)
                if s:
                    block.scroll(s)
                return "break"
        return None

    root.bind_all("<MouseWheel>", on_wheel, add="+")   # Windows / macOS
    root.bind_all("<Button-4>", on_wheel, add="+")     # Linux up
    root.bind_all("<Button-5>", on_wheel, add="+")     # Linux down


# ---------------------------------------------------------------------------
# Plot navigation toolbar without the built-in "Save" button
# ---------------------------------------------------------------------------
class _PlotToolbar(NavigationToolbar2Tk):
    """Matplotlib nav toolbar with its Save (and Subplots) buttons removed.

    Saving is done exclusively through the app's "Save figure…" button, which
    honours the Export graph size fields; the toolbar's own save would export
    the raw on-screen figure and ignore those settings, which is confusing."""

    toolitems = [t for t in NavigationToolbar2Tk.toolitems
                 if t[0] not in ("Save", "Subplots")]


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class VoltagePlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Battery Cycling Voltage Plotter")
        self._fit_to_screen()

        self.files = []   # list[CellFile]
        self.pois = []    # list[POI]

        # ---- top-level layout: controls (left) | plot (right) ----
        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, width=460)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        self._build_controls(left)
        self._build_plot(right)

        # Pressing Enter in any text box should commit AND leave the box.  Each
        # field already commits on its own <Return> (widget-tier binding); this
        # class-tier binding runs afterwards and drops keyboard focus.
        for cls in ("TEntry", "Entry"):
            self.root.bind_class(cls, "<Return>", self._defocus_entry, add="+")

        self.redraw()

    def _defocus_entry(self, _event=None):
        # Move focus off the entry (to the plot) so the caret leaves the box.
        self.canvas.get_tk_widget().focus_set()

    def _combobox_wheel_passthrough(self, event):
        # Prevent the combobox's own value-changing wheel behavior, but keep the
        # panel scrolling by forwarding the wheel to the bottom block.
        self._bottom_block.scroll(wheel_steps(event))
        return "break"

    def _on_mode_change(self):
        """React to switching between Voltage-vs-Time and Voltage-vs-Capacity."""
        capacity = self.plot_mode.get() == "capacity"
        # Cycle-range fields only apply to the capacity plot.
        state = "normal" if capacity else "disabled"
        self.cyc_from_entry.configure(state=state)
        self.cyc_to_entry.configure(state=state)
        # Update the x-axis label to match the mode.
        if hasattr(self, "xlabel_var"):
            self.xlabel_var.set("Capacity / mAh" if capacity else "Time / hour")
        if hasattr(self, "canvas"):
            self.redraw()

    def _fit_to_screen(self):
        """Open expanded to fill the usable screen (below the menu bar), not
        fullscreen and not overlapping the macOS menu bar."""
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        if sys.platform == "darwin":
            top = 28          # leave room for the macOS menu bar
            bottom = 8        # small breathing room above the Dock
            self.root.geometry(f"{sw}x{sh - top - bottom}+0+{top}")
        else:
            try:
                self.root.state("zoomed")        # Windows / most Linux WMs
            except Exception:
                self.root.geometry(f"{sw}x{sh}+0+0")

    # ------------------------------------------------------------------
    # Control-panel construction
    # ------------------------------------------------------------------
    def _build_controls(self, parent):
        # The side panel is split into two independently scrolling blocks:
        #   * top    -> drag/drop + file list
        #   * bottom -> everything else (parameters, axes, points of interest)
        # A draggable sash lets the user rebalance the two.
        split = ttk.PanedWindow(parent, orient="vertical")
        split.pack(fill="both", expand=True)

        top_block = ScrollBlock(split)
        bottom_block = ScrollBlock(split)
        split.add(top_block, weight=1)
        split.add(bottom_block, weight=3)
        self._split = split
        self._top_block = top_block
        self._bottom_block = bottom_block

        # Don't let the sash drop below the top block's content: clamp during and
        # after a drag so no empty gap can open above the bottom block.
        split.bind("<B1-Motion>", lambda _e: self._clamp_sash(), add="+")
        split.bind("<ButtonRelease-1>", lambda _e: self._clamp_sash(), add="+")

        # One global wheel dispatcher scrolls whichever block the pointer is over.
        install_wheel_dispatch(self.root, [top_block, bottom_block])

        top = top_block.inner       # parent for the files block
        c = bottom_block.inner      # parent for everything else

        # ---- Drop zone / Browse (top block) ----
        drop = tk.Label(
            top,
            text=("⬇  Drag & drop .txt files here" if DND_AVAILABLE
                  else "Drag & drop unavailable — use Browse"),
            relief="ridge", borderwidth=2, height=3,
            bg="#eef3fb", fg="#20406a",
        )
        drop.pack(fill="x", padx=8, pady=(8, 4))
        if DND_AVAILABLE:
            drop.drop_target_register(DND_FILES)
            drop.dnd_bind("<<Drop>>", self._on_drop)

        ttk.Button(top, text="Browse for files…", command=self._browse).pack(
            fill="x", padx=8, pady=(0, 8))

        # ---- Files list (top block) ----
        ttk.Label(top, text="Data files", font=("", 11, "bold")).pack(anchor="w", padx=8)
        self.files_frame = ttk.Frame(top)
        self.files_frame.pack(fill="x", padx=8, pady=(2, 8))

        # ---- Input parameters (bottom block) ----
        params = ttk.LabelFrame(c, text="Input parameters")
        params.pack(fill="x", padx=8, pady=4)
        ttk.Label(params, text="Li electrode diameter (mm):").grid(
            row=0, column=0, sticky="w", padx=4, pady=3)
        self.li_diam = tk.StringVar(value="8")
        e = ttk.Entry(params, textvariable=self.li_diam, width=8)
        e.grid(row=0, column=1, sticky="w", padx=4)
        e.bind("<FocusOut>", lambda _e: self.redraw())
        e.bind("<Return>", lambda _e: self.redraw())
        self.li_area_lbl = ttk.Label(params, text="")
        self.li_area_lbl.grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 3))

        # Current schedule used by "Calculate CCD": an alternating-sign step that
        # steps up in magnitude every two steps (discharge then charge at each
        # level).  Defaults: 0.05 mA per level, 1 hour per step, discharge first.
        ttk.Label(params, text="Current step (mA):").grid(
            row=2, column=0, sticky="w", padx=4, pady=3)
        self.curr_step = tk.StringVar(value="0.05")
        cse = ttk.Entry(params, textvariable=self.curr_step, width=8)
        cse.grid(row=2, column=1, sticky="w", padx=4)
        cse.bind("<FocusOut>", lambda _e: self.redraw())
        cse.bind("<Return>", lambda _e: self.redraw())

        ttk.Label(params, text="Step duration (h):").grid(
            row=3, column=0, sticky="w", padx=4, pady=3)
        self.step_dur = tk.StringVar(value="1")
        sde = ttk.Entry(params, textvariable=self.step_dur, width=8)
        sde.grid(row=3, column=1, sticky="w", padx=4)
        sde.bind("<FocusOut>", lambda _e: self.redraw())
        sde.bind("<Return>", lambda _e: self.redraw())

        self.discharge_first = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="Discharge first (negative)",
                        variable=self.discharge_first,
                        command=self.redraw).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 3))

        # ---- Plot type (bottom block) ----
        ptype = ttk.LabelFrame(c, text="Plot type")
        ptype.pack(fill="x", padx=8, pady=4)
        self.plot_mode = tk.StringVar(value="time")
        ttk.Radiobutton(ptype, text="Voltage vs Time", variable=self.plot_mode,
                        value="time", command=self._on_mode_change).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=4, pady=1)
        ttk.Radiobutton(ptype, text="Voltage vs Capacity", variable=self.plot_mode,
                        value="capacity", command=self._on_mode_change).grid(
            row=1, column=0, columnspan=3, sticky="w", padx=4, pady=1)

        ttk.Label(ptype, text="Cycles (from, to):").grid(
            row=2, column=0, sticky="w", padx=4, pady=2)
        self.cyc_from = tk.StringVar(value="")
        self.cyc_to = tk.StringVar(value="")
        self.cyc_from_entry = ttk.Entry(ptype, textvariable=self.cyc_from, width=6)
        self.cyc_to_entry = ttk.Entry(ptype, textvariable=self.cyc_to, width=6)
        self.cyc_from_entry.grid(row=2, column=1, sticky="w", padx=2)
        self.cyc_to_entry.grid(row=2, column=2, sticky="w", padx=2)
        for e in (self.cyc_from_entry, self.cyc_to_entry):
            e.bind("<FocusOut>", lambda _e: self.redraw())
            e.bind("<Return>", lambda _e: self.redraw())
        ttk.Label(ptype, text="(blank = all cycles; capacity mode only)",
                  foreground="#888").grid(row=3, column=0, columnspan=3,
                                          sticky="w", padx=4)
        # Start in time mode: cycle range not applicable, so disable it.
        self.cyc_from_entry.configure(state="disabled")
        self.cyc_to_entry.configure(state="disabled")

        # ---- Axes / appearance ----
        axes = ttk.LabelFrame(c, text="Axes & appearance")
        axes.pack(fill="x", padx=8, pady=4)

        self.title_var = self._labeled_entry(axes, 0, "Title:", "")
        self.xlabel_var = self._labeled_entry(axes, 1, "X label:", "Time / hour")
        self.ylabel_var = self._labeled_entry(axes, 2, "Y label:", "Voltage / V")

        ttk.Label(axes, text="X limits (min, max):").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        xlim = ttk.Frame(axes); xlim.grid(row=3, column=1, sticky="w")
        self.xmin = self._small_entry(xlim); self.xmax = self._small_entry(xlim)

        ttk.Label(axes, text="Y limits (min, max):").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ylim = ttk.Frame(axes); ylim.grid(row=4, column=1, sticky="w")
        self.ymin = self._small_entry(ylim); self.ymax = self._small_entry(ylim)
        ttk.Label(axes, text="(blank = auto)", foreground="#888").grid(
            row=5, column=0, columnspan=2, sticky="w", padx=4)

        ttk.Label(axes, text="Default line width:").grid(row=6, column=0, sticky="w", padx=4, pady=2)
        self.lw_var = tk.StringVar(value="0.5")
        lwe = ttk.Entry(axes, textvariable=self.lw_var, width=8)
        lwe.grid(row=6, column=1, sticky="w", padx=4)
        lwe.bind("<FocusOut>", lambda _e: self.redraw()); lwe.bind("<Return>", lambda _e: self.redraw())

        ttk.Label(axes, text="Export graph size (w, h in):").grid(row=7, column=0, sticky="w", padx=4, pady=2)
        figf = ttk.Frame(axes); figf.grid(row=7, column=1, sticky="w")
        self.fig_w = self._small_entry(figf, "8"); self.fig_h = self._small_entry(figf, "5")

        ttk.Label(axes, text="Plot style:").grid(row=8, column=0, sticky="w", padx=4, pady=2)
        self.style_var = tk.StringVar(value="default")
        styles = ["default", "dark_background"]
        sc = ttk.Combobox(axes, textvariable=self.style_var, values=styles,
                          width=18, state="readonly")
        sc.grid(row=8, column=1, sticky="w", padx=4)
        sc.bind("<<ComboboxSelected>>", lambda _e: self.redraw())
        # A read-only combobox changes its value when scrolled over — annoying
        # while scrolling the panel.  Swallow the wheel here (so the selection
        # never changes) but still scroll the block so the panel keeps moving.
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            sc.bind(seq, self._combobox_wheel_passthrough, add="+")

        self.show_legend = tk.BooleanVar(value=True)
        ttk.Checkbutton(axes, text="Show legend", variable=self.show_legend,
                        command=self.redraw).grid(row=9, column=0, sticky="w", padx=4, pady=2)
        self.show_grid = tk.BooleanVar(value=False)
        ttk.Checkbutton(axes, text="Grid", variable=self.show_grid,
                        command=self.redraw).grid(row=9, column=1, sticky="w", padx=4, pady=2)

        for var in (self.title_var, self.xlabel_var, self.ylabel_var):
            pass  # already bound in _labeled_entry

        # ---- Points of interest ----
        poi_head = ttk.Frame(c); poi_head.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(poi_head, text="Points of interest", font=("", 11, "bold")).pack(side="left")
        ttk.Button(poi_head, text="+ Add", width=8, command=self._add_poi).pack(side="right")
        ttk.Label(c, text="Tick “Calculate CCD” to label a point with the current "
                           "density derived from its time; otherwise set a custom label.",
                  foreground="#888", wraplength=430).pack(anchor="w", padx=8)
        self.poi_frame = ttk.Frame(c)   # plain frame; scrolls with the outer panel
        self.poi_frame.pack(fill="x", padx=8, pady=(2, 8))

        # ---- Actions ----
        actions = ttk.Frame(c); actions.pack(fill="x", padx=8, pady=8)
        ttk.Button(actions, text="Redraw", command=self.redraw).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(actions, text="Save figure…", command=self._save_fig).pack(side="left", expand=True, fill="x", padx=2)

        self._refresh_files()
        self._refresh_pois()
        # Give the top (files) block a sensible starting height (but never taller
        # than its content); the user can drag the sash to rebalance afterwards.
        self.root.after(0, self._init_sash)

    # ------------------------------------------------------------------
    # Sash management (top block never taller than its content)
    # ------------------------------------------------------------------
    def _max_top(self):
        """Tallest the top (files) block may be before dead space appears."""
        return max(1, self._top_block.inner.winfo_reqheight() + 6)

    def _clamp_sash(self):
        try:
            mx = self._max_top()
            if self._split.sashpos(0) > mx:
                self._split.sashpos(0, mx)
        except Exception:
            pass

    def _init_sash(self):
        try:
            self._split.sashpos(0, min(220, self._max_top()))
        except Exception:
            pass

    def _labeled_entry(self, parent, row, label, default):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        var = tk.StringVar(value=default)
        e = ttk.Entry(parent, textvariable=var, width=26)
        e.grid(row=row, column=1, sticky="w", padx=4)
        e.bind("<FocusOut>", lambda _e: self.redraw())
        e.bind("<Return>", lambda _e: self.redraw())
        return var

    def _small_entry(self, parent, default=""):
        var = tk.StringVar(value=default)
        e = ttk.Entry(parent, textvariable=var, width=7)
        e.pack(side="left", padx=2)
        e.bind("<FocusOut>", lambda _e: self.redraw())
        e.bind("<Return>", lambda _e: self.redraw())
        return var

    # ------------------------------------------------------------------
    # Plot area
    # ------------------------------------------------------------------
    def _build_plot(self, parent):
        # Constrained layout re-fits the margins on every draw — including the
        # redraws FigureCanvasTkAgg fires on window resize — so the axis labels
        # stay inside the canvas and the plot fills the available space.
        self.fig = Figure(figsize=(8, 5), layout="constrained")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = _PlotToolbar(self.canvas, parent)
        toolbar.update()

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    def _browse(self):
        paths = filedialog.askopenfilenames(
            title="Select exported .txt data files",
            filetypes=[("Text data", "*.txt"), ("All files", "*.*")])
        self._add_paths(paths)

    def _on_drop(self, event):
        # tkinterdnd2 gives a brace-wrapped, space-separated list of paths.
        paths = self.root.tk.splitlist(event.data)
        self._add_paths(paths)

    def _add_paths(self, paths):
        added = 0
        for p in paths:
            p = p.strip()
            if not p or not os.path.isfile(p):
                continue
            if any(f.path == p for f in self.files):
                continue
            color = DEFAULT_COLORS[len(self.files) % len(DEFAULT_COLORS)]
            label = os.path.splitext(os.path.basename(p))[0]
            try:
                self.files.append(CellFile(p, label, color))
                added += 1
            except Exception as exc:
                messagebox.showerror("Could not load file",
                                     f"{os.path.basename(p)}:\n{exc}")
        if added:
            self._refresh_files()
            self.redraw()

    def _refresh_files(self):
        frame = self.files_frame
        for w in frame.winfo_children():
            w.destroy()

        if self.files:
            for i, f in enumerate(self.files):
                row = ttk.Frame(frame)
                row.pack(fill="x", pady=1)

                vis = tk.BooleanVar(value=f.visible)
                ttk.Checkbutton(row, variable=vis,
                                command=lambda ff=f, v=vis: self._set_visible(ff, v)).pack(side="left")

                swatch = tk.Button(row, width=2, bg=f.color, relief="groove",
                                   command=lambda ff=f: self._pick_color(ff))
                swatch.pack(side="left", padx=2)

                name = tk.StringVar(value=f.label)
                ent = ttk.Entry(row, textvariable=name, width=26)
                ent.pack(side="left", padx=2, fill="x", expand=True)
                ent.bind("<FocusOut>", lambda _e, ff=f, n=name: self._set_label(ff, n))
                ent.bind("<Return>", lambda _e, ff=f, n=name: self._set_label(ff, n))

                ttk.Button(row, text="✕", width=3,
                           command=lambda ff=f: self._remove_file(ff)).pack(side="left", padx=2)
        else:
            ttk.Label(frame, text="No files loaded.", foreground="#888").pack(
                anchor="w", padx=4, pady=6)

        # The list height just changed; drop the sash if it now sits in dead space.
        if hasattr(self, "_split"):
            self.root.after_idle(self._clamp_sash)

    def _set_visible(self, f, var):
        f.visible = var.get()
        self.redraw()

    def _set_label(self, f, var):
        f.label = var.get()
        self.redraw()

    def _pick_color(self, f):
        rgb, hx = colorchooser.askcolor(color=f.color, title="Line color")
        if hx:
            f.color = hx
            self._refresh_files()
            self.redraw()

    def _remove_file(self, f):
        self.files.remove(f)
        self._refresh_files()
        self.redraw()

    # ------------------------------------------------------------------
    # POI handling
    # ------------------------------------------------------------------
    def _add_poi(self):
        self.pois.append(POI())
        self._refresh_pois()
        self.redraw()

    def _remove_poi(self, p):
        self.pois.remove(p)
        self._refresh_pois()
        self.redraw()

    def _refresh_pois(self):
        frame = self.poi_frame
        for w in frame.winfo_children():
            w.destroy()

        if not self.pois:
            ttk.Label(frame, text="No points of interest.", foreground="#888").pack(
                anchor="w", padx=4, pady=6)
            return

        for p in self.pois:
            box = ttk.LabelFrame(frame, text="")
            box.pack(fill="x", pady=3, padx=1)

            # Header row: visibility, name, color, remove.
            hdr = ttk.Frame(box); hdr.pack(fill="x")
            vis = tk.BooleanVar(value=p.visible)
            ttk.Checkbutton(hdr, variable=vis,
                            command=lambda pp=p, v=vis: self._poi_set(pp, "visible", v.get())).pack(side="left")
            nm = tk.StringVar(value=p.name)
            ne = ttk.Entry(hdr, textvariable=nm, width=6)
            ne.pack(side="left", padx=2)
            self._bind_str(ne, nm, p, "name")
            tk.Button(hdr, width=2, bg=p.color, relief="groove",
                      command=lambda pp=p: self._poi_color(pp)).pack(side="left", padx=2)
            ttk.Button(hdr, text="✕", width=3,
                       command=lambda pp=p: self._remove_poi(pp)).pack(side="right", padx=2)

            # Grid of numeric / text fields.
            g = ttk.Frame(box); g.pack(fill="x", padx=2, pady=2)

            self._poi_num(g, 0, 0, "x:", p, "x")
            self._poi_num(g, 0, 2, "y:", p, "y")
            self._poi_num(g, 1, 0, "label dx:", p, "x_offset")
            self._poi_num(g, 1, 2, "label dy:", p, "y_offset")

            # "Calculate CCD": derive the current density from x (time) via the
            # current schedule.  While on, the custom label is grayed out.
            ccd = tk.BooleanVar(value=p.calc_ccd)
            ccd_chk = ttk.Checkbutton(g, text="Calculate CCD", variable=ccd)
            ccd_chk.grid(row=2, column=0, columnspan=2, sticky="w", padx=2, pady=1)

            ttk.Label(g, text="Custom label:").grid(row=3, column=0, sticky="w", padx=2, pady=1)
            cl = tk.StringVar(value=p.custom_label)
            cle = ttk.Entry(g, textvariable=cl, width=24)
            cle.grid(row=3, column=1, columnspan=3, sticky="we", padx=2)
            self._bind_str(cle, cl, p, "custom_label")

            def toggle_ccd(pp=p, v=ccd, entry=cle):
                pp.calc_ccd = v.get()
                entry.configure(state=("disabled" if pp.calc_ccd else "normal"))
                self.redraw()
            ccd_chk.configure(command=toggle_ccd)
            if p.calc_ccd:                       # reflect initial state
                cle.configure(state="disabled")

    def _poi_num(self, parent, r, col, label, p, attr, allow_blank=False):
        ttk.Label(parent, text=label).grid(row=r, column=col, sticky="w", padx=2, pady=1)
        cur = getattr(p, attr)
        var = tk.StringVar(value="" if cur is None else str(cur))
        e = ttk.Entry(parent, textvariable=var, width=8)
        e.grid(row=r, column=col + 1, sticky="w", padx=2)

        def commit(_e=None):
            txt = var.get().strip()
            if txt == "" and allow_blank:
                setattr(p, attr, None)
            else:
                try:
                    setattr(p, attr, float(txt))
                except ValueError:
                    pass
            self.redraw()

        e.bind("<FocusOut>", commit)
        e.bind("<Return>", commit)

    def _bind_str(self, entry, var, obj, attr):
        def commit(_e=None):
            setattr(obj, attr, var.get())
            self.redraw()
        entry.bind("<FocusOut>", commit)
        entry.bind("<Return>", commit)

    def _poi_set(self, p, attr, value):
        setattr(p, attr, value)
        self.redraw()

    def _poi_color(self, p):
        rgb, hx = colorchooser.askcolor(color=p.color, title="Point color")
        if hx:
            p.color = hx
            self._refresh_pois()
            self.redraw()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _li_area(self):
        """Lithium electrode area in cm^2 from the diameter in mm."""
        try:
            d = float(self.li_diam.get())
        except ValueError:
            d = 8.0
        # diameter mm -> radius cm is (d/10)/2 = d/20; area = pi r^2.
        return np.pi * (d / 20.0) ** 2

    def _current_at_time(self, t):
        """Cell current (mA) at time ``t`` (hours) for the cycling schedule.

        The protocol is an alternating-sign step whose magnitude increases every
        two steps: at level k (k = 0, 1, 2, ...) the magnitude is
        ``curr_step * (k + 1)``; within a level the first step is discharge and
        the second is charge.  Discharge is negative by default."""
        step = self._parse(self.curr_step)
        dur = self._parse(self.step_dur)
        if not step or not dur or dur <= 0 or t is None or t < 0:
            return None
        s = int(t // dur)                    # step index: 0,1,2,...
        level = s // 2                       # magnitude level: 0,0,1,1,2,2,...
        magnitude = step * (level + 1)
        discharge = (s % 2 == 0)             # first step of each level is discharge
        if not self.discharge_first.get():
            discharge = not discharge
        sign = -1.0 if discharge else 1.0
        return sign * magnitude

    @staticmethod
    def _parse(var):
        try:
            return float(var.get())
        except (ValueError, AttributeError):
            return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def redraw(self):
        li_area = self._li_area()
        self.li_area_lbl.config(text=f"→ Li area = {li_area:.4f} cm²")

        # Apply the chosen style, then rebuild the figure/axes.
        style = self.style_var.get()
        try:
            with matplotlib.style.context(style):
                self._render(li_area)
        except Exception:
            self._render(li_area)

    def _render(self, li_area):
        self.fig.clear()
        # The on-screen figure was created before any style context, so its
        # background doesn't follow the style on its own — sync it to the active
        # style (whatever rcParams currently say) so dark styles apply fully.
        self.fig.set_facecolor(matplotlib.rcParams["figure.facecolor"])
        self.ax = self.fig.add_subplot(111)
        self._draw_into(self.ax, li_area)

        # NOTE: we deliberately do NOT call set_size_inches on the on-screen
        # figure.  Doing so pins it to a fixed pixel size and defeats
        # FigureCanvasTkAgg's resize-to-fit.  The on-screen figure always fills
        # the canvas widget; the "Saved fig size" fields apply only on export.
        # Margins are handled by the figure's constrained layout (see
        # _build_plot), which also re-fits on resize — no tight_layout here.
        self.canvas.draw_idle()

    def _draw_into(self, ax, li_area):
        """Draw the full plot (traces, POIs, labels, limits) into ``ax``.

        Shared by the on-screen figure and the fixed-size export figure so both
        look identical."""
        try:
            default_lw = float(self.lw_var.get())
        except ValueError:
            default_lw = 0.5

        capacity_mode = self.plot_mode.get() == "capacity"
        if capacity_mode:
            self._draw_capacity(ax, default_lw)
        else:
            for f in self.files:
                if not f.visible:
                    continue
                ax.plot(f.time, f.volt, lw=default_lw, c=f.color, label=f.label)

        # Points of interest.
        for p in self.pois:
            if not p.visible:
                continue
            ax.plot(p.x, p.y, marker="o", color=p.color, markersize=4, linestyle="none")
            current = self._current_at_time(p.x) if p.calc_ccd else None
            text = p.label_text(li_area, current)
            if text:
                ax.annotate(
                    text,
                    xy=(p.x, p.y),
                    xytext=(p.x + p.x_offset, p.y + p.y_offset),
                    color=p.color,
                    arrowprops=dict(arrowstyle="->", color=p.color),
                    ha="center",
                )

        # Labels / title.
        ax.set_xlabel(self.xlabel_var.get())
        ax.set_ylabel(self.ylabel_var.get())
        if self.title_var.get().strip():
            ax.set_title(self.title_var.get())

        # Limits.  A user-set bound still gets the usual 5% breathing room so the
        # trace never runs flush against the frame; an unset bound falls back to
        # the raw data extent (then also gets the margin).
        MARGIN = 0.05
        xmin, xmax = self._parse(self.xmin), self._parse(self.xmax)
        if xmin is not None or xmax is not None:
            dlo, dhi = ax.dataLim.intervalx
            lo = xmin if xmin is not None else dlo
            hi = xmax if xmax is not None else dhi
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = MARGIN * (hi - lo)
                ax.set_xlim(lo - pad, hi + pad)
        ymin, ymax = self._parse(self.ymin), self._parse(self.ymax)
        if ymin is not None or ymax is not None:
            dlo, dhi = ax.dataLim.intervaly
            lo = ymin if ymin is not None else dlo
            hi = ymax if ymax is not None else dhi
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                pad = MARGIN * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)

        if self.show_grid.get():
            ax.grid(True, alpha=0.3)
        if self.show_legend.get() and any(f.visible for f in self.files):
            ax.legend()

    def _draw_capacity(self, ax, default_lw):
        """Plot voltage vs capacity, one segment per cycle within the selected
        cycle range (blank = all).  Each cycle is drawn separately so the trace
        lifts between cycles instead of jumping back when capacity resets."""
        lo = self._parse(self.cyc_from)
        hi = self._parse(self.cyc_to)
        for f in self.files:
            if not f.visible or not f.has_capacity:
                continue
            cap, volt, cyc = f.capacity, f.volt, f.cycle
            if cyc is None:
                ax.plot(cap, volt, lw=default_lw, c=f.color, label=f.label)
                continue
            cycles = [c for c in np.unique(cyc)
                      if (lo is None or c >= lo) and (hi is None or c <= hi)]
            first = True
            for cn in cycles:
                m = cyc == cn
                ax.plot(cap[m], volt[m], lw=default_lw, c=f.color,
                        label=(f.label if first else None))
                first = False

    def _build_export_figure(self, li_area, w, h):
        """Figure whose *plot area* (axes) is exactly ``w`` x ``h`` inches.

        The axes is pinned to a fixed physical size, so two exports with the
        same limits produce graphs of identical dimensions.  Titles/labels sit
        in the surrounding margin, so only the overall image size varies."""
        fig = Figure()
        if w and h and w > 0 and h > 0:
            # Generous fixed margins (inches) hold the tick labels, axis titles,
            # and plot title; savefig(bbox_inches="tight") trims the leftover
            # whitespace afterwards without changing the w x h axes rectangle.
            left, right, bottom, top = 1.1, 0.5, 0.8, 0.6
            fig.set_size_inches(left + w + right, bottom + h + top)
            horiz = [Size.Fixed(left), Size.Fixed(w), Size.Fixed(right)]
            vert = [Size.Fixed(bottom), Size.Fixed(h), Size.Fixed(top)]
            div = Divider(fig, (0.0, 0.0, 1.0, 1.0), horiz, vert, aspect=False)
            ax = fig.add_axes(div.get_position(),
                              axes_locator=div.new_locator(nx=1, ny=1))
            self._draw_into(ax, li_area)
        else:
            # Blank size fields: fall back to a normally-laid-out figure.
            ax = fig.add_subplot(111)
            self._draw_into(ax, li_area)
            try:
                fig.tight_layout()
            except Exception:
                pass
        return fig

    def _save_fig(self):
        path = filedialog.asksaveasfilename(
            title="Save figure",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            # Export from a separate figure whose plot area is exactly w x h
            # inches; the on-screen figure is untouched.
            li_area = self._li_area()
            w, h = self._parse(self.fig_w), self._parse(self.fig_h)
            style = self.style_var.get()
            try:
                try:
                    with matplotlib.style.context(style):
                        fig = self._build_export_figure(li_area, w, h)
                        fig.savefig(path, dpi=300, bbox_inches="tight")
                except Exception:
                    fig = self._build_export_figure(li_area, w, h)
                    fig.savefig(path, dpi=300, bbox_inches="tight")
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))


def main():
    root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
    VoltagePlotterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
