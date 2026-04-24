#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pycontrol.py - modern, Matlab-ish Control Systems GUI (ASCII-safe)

Run without hardware (three UI profiles):
    # Classic ttk
    PYCSB_DEMO=1 python pycontrol.py

    # Bootstrap (requires ttkbootstrap installed)
    PYCONTROL_TTKB=1 PYCSB_DEMO=1 python pycontrol.py

    # Mac-style (light, unified surfaces, SF/Helvetica, accent blue)
    PYCONTROL_MACUI=1 PYCSB_DEMO=1 python pycontrol.py

    # alternatively:
    PYCONTROL_UI=mac PYCSB_DEMO=1 python pycontrol.py
"""

from __future__ import annotations

import os
import json
import math
import time
from pathlib import Path
from typing import List, Tuple, Optional

# tkinter / ttk
import tkinter as tk
from tkinter import messagebox, filedialog
import tkinter.ttk as ttk
from tkinter import font as tkfont

# ----- UI profile switches ----------------------------------------------------
ENV_UI = os.getenv("PYCONTROL_UI", "").lower().strip()
USE_MACUI = os.getenv("PYCONTROL_MACUI", "0") == "1" or ENV_UI in ("mac", "macos")

# Optional ttkbootstrap only when explicitly enabled (ignored if Mac UI selected)
USE_TTKB = (os.getenv("PYCONTROL_TTKB", "0") == "1") and not USE_MACUI
if USE_TTKB:
    try:
        from ttkbootstrap import Style as TBStyle  # type: ignore
        HAVE_TTKBOOTSTRAP = True
    except Exception:
        TBStyle = None
        HAVE_TTKBOOTSTRAP = False
else:
    TBStyle = None
    HAVE_TTKBOOTSTRAP = False

# Plotting
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation
from matplotlib.backend_bases import NavigationToolbar2

# Control + data
import control as co
from scipy.optimize import curve_fit
import pandas as pd

# Serial
import serial  # type: ignore
import serial.tools.list_ports  # type: ignore

# Images
from PIL import Image, ImageDraw, ImageTk


# ------------------------------
# Preferences
# ------------------------------
PREFS_PATH = Path.home() / ".pycontrol_prefs.json"
DEFAULT_PREFS = {
    "theme": "clam",     # classic ttk look (ignored for ttkbootstrap/mac)
    "baud": 9600,
    "palette": "Matlab",
    "ui_profile": "mac" if USE_MACUI else ("bootstrap" if USE_TTKB else "classic"),
}


def load_prefs() -> dict:
    try:
        if PREFS_PATH.exists():
            with open(PREFS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {**DEFAULT_PREFS, **data}
    except Exception:
        pass
    return DEFAULT_PREFS.copy()


def save_prefs(prefs: dict) -> None:
    try:
        with open(PREFS_PATH, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
    except Exception:
        pass


# ------------------------------
# Serial helpers
# ------------------------------
def _is_bluetooth_port(dev: str, desc: str) -> bool:
    dev = (dev or "").lower()
    desc = (desc or "").lower()
    return (
        "bluetooth" in dev
        or "bluetooth" in desc
        or dev.endswith(".blth")
        or "bluetooth-incoming-port" in dev
    )


def list_port_items() -> List[Tuple[str, str]]:
    """Return [(device, label), ...] with 'device - description'.
    Filters out macOS Bluetooth pseudo-ports.
    """
    items: List[Tuple[str, str]] = []
    for p in serial.tools.list_ports.comports():
        dev = p.device or ""
        desc = p.description or ""
        if _is_bluetooth_port(dev, desc):
            continue
        label = f"{dev} - {desc}".strip()
        items.append((dev, label))
    return items


def parse_device_from_label(label: str) -> str:
    if not label:
        return ""
    parts = [s.strip() for s in label.split("-", 1)]
    if parts and parts[0].startswith("/dev/"):
        return parts[0]
    return label.strip()


class DemoSerial:
    """Stub for serial.Serial when running without hardware.
    Generates a smooth step + ringing as ASCII lines via .readline().
    """
    def __init__(self):
        self.is_open = True
        self.t0 = time.time()

    def close(self):
        self.is_open = False

    def readline(self) -> bytes:
        t = max(0.0, time.time() - self.t0)
        y = 1.0 - math.exp(-t) + 0.12 * math.sin(2 * math.pi * 0.7 * t)
        y += np.random.normal(0, 0.003)
        return f"{y:.6f}\n".encode("ascii")


# ------------------------------
# Palettes (axes colors etc.)
# ------------------------------
# palette: (axes_face, grid, line, text, spine)
PALETTES = {
    "Matlab":    ("#ffffff", "#cccccc", "#1f77b4", "#111111", "#999999"),
    "Light":     ("#fbfbfb", "#d0d0d0", "#2d6cdf", "#111111", "#b0b0b0"),
    "Grayscale": ("#ffffff", "#c9c9c9", "#4d4d4d", "#111111", "#9a9a9a"),
    "Midnight":  ("#101316", "#2b3a44", "#c9d6df", "#e6e6e6", "#5f7483"),
    "Monokai":   ("#272822", "#49483e", "#a6e22e", "#f8f8f2", "#75715e"),
    # Subtle Mac-inspired: light canvas, soft grid, accent-blue line, graphite text
    "MacOS":     ("#ffffff", "#d9d9de", "#007aff", "#1c1c1e", "#b4b4b9"),
}


# ------------------------------
# Compact Toolbar
# ------------------------------
class CompactToolbar(ttk.Frame):
    """Tiny wrapper that uses NavigationToolbar2 methods."""
    def __init__(self, parent, canvas, style_prefix: str = ""):
        super().__init__(parent, style=f"{style_prefix}Toolbar.TFrame" if style_prefix else "TFrame")
        self.canvas = canvas

        class _Hidden(NavigationToolbar2):
            def __init__(self, canvas):
                super().__init__(canvas)
            def set_message(self, s):
                pass
        self._tb = _Hidden(canvas)

        btnstyle = f"{style_prefix}Tool.TButton" if style_prefix else "TButton"
        buttons = [
            ("Home", self._tb.home),
            ("Back", self._tb.back),
            ("Fwd",  self._tb.forward),
            ("Pan",  self._tb.pan),
            ("Zoom", self._tb.zoom),
            ("Save", self._tb.save_figure),
        ]
        for i, (txt, cmd) in enumerate(buttons):
            ttk.Button(self, text=txt, width=5, command=cmd, style=btnstyle).grid(row=0, column=i, padx=2)


# ------------------------------
# Main App
# ------------------------------
class PyControlApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        # prefs first (lets us pick default palette per profile)
        self.prefs = load_prefs()

        # window basics
        self.title("PyControl - Control Systems Lab")
        self.geometry("1200x760+80+40")
        self.resizable(True, True)

        # style / theme
        self.style, self.ui_profile = self._init_style(self.prefs.get("theme", "clam"))

        # vars
        self.demo_mode_var = tk.BooleanVar(value=(os.getenv("PYCSB_DEMO", "0") == "1"))
        self.ser = None  # type: ignore
        self.ani_obj = None
        self.com_port_label = ""
        self.com_xdata = np.array([], dtype=float)
        self.com_ydata = np.array([], dtype=float)

        # palette
        default_palette = "MacOS" if self.ui_profile == "mac" else self.prefs.get("palette", "Matlab")
        self.palette_name = default_palette
        (self.axes_face,
         self.grid_color,
         self.line_color,
         self.text_color,
         self.spine_color) = PALETTES.get(self.palette_name, PALETTES["Matlab"])

        # UI
        self._build_menu()
        self._build_layout()
        self._apply_palette(self.palette_name)
        self._status("Ready.")

        # populate
        self._refresh_ports()
        self.baud_combo.set(str(self.prefs.get("baud", 9600)))

        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    # --- style
    def _init_style(self, theme_name: str) -> Tuple[ttk.Style, str]:
        """
        Returns (style, ui_profile) where ui_profile is one of: 'classic', 'bootstrap', 'mac'
        """
        # --- Mac-style profile (pure ttk, custom styling)
        if USE_MACUI:
            ui_profile = "mac"
            style = ttk.Style()
            try:
                style.theme_use("clam")
            except Exception:
                pass

            # Window & base colors (Big Sur-like neutrals)
            window_bg = "#f5f5f7"
            control_bg = "#ffffff"
            subtle_border = "#d6d6d6"
            graphite = "#1c1c1e"
            accent_blue = "#007aff"

            # set Tk default fonts similar to SF Pro Text / Helvetica
            def _tweak_fonts():
                try:
                    base = tkfont.nametofont("TkDefaultFont")
                    base.configure(family="SF Pro Text", size=12)
                except Exception:
                    base = tkfont.nametofont("TkDefaultFont")
                    base.configure(family="Helvetica", size=12)

                tkfont.nametofont("TkTextFont").configure(size=12)
                tkfont.nametofont("TkHeadingFont").configure(size=13, weight="bold")

            _tweak_fonts()
            try:
                # Mildly increase scaling on mac for clarity
                if str(self.call("tk", "windowingsystem")).lower() == "aqua":
                    self.call("tk", "scaling", 1.2)
            except Exception:
                pass

            # Base widget surfaces
            self.configure(bg=window_bg)
            style.configure(".", background=window_bg)
            style.configure("TFrame", background=window_bg)
            style.configure("TLabel", background=window_bg, foreground=graphite)
            style.configure("Card.TLabelframe",
                            background=window_bg, relief="flat", borderwidth=0, padding=8)
            style.configure("Card.TLabelframe.Label",
                            background=window_bg, foreground=graphite, font=("Helvetica", 13, "bold"))

            # Entries
            style.configure("TEntry", fieldbackground=control_bg, background=control_bg,
                            foreground=graphite, bordercolor=subtle_border, lightcolor=subtle_border,
                            darkcolor=subtle_border, borderwidth=1, padding=4)

            # Buttons (flat, rounded-ish via padding; ttk has no real radius)
            style.configure("TButton", background=control_bg, foreground=graphite,
                            relief="flat", borderwidth=1, padding=(10, 6))
            style.map("TButton",
                      background=[("active", "#ececec"), ("pressed", "#e5e5e5")])
            style.configure("Accent.TButton", background=accent_blue, foreground="#ffffff",
                            padding=(10, 6))
            style.map("Accent.TButton",
                      background=[("active", "#1676ff"), ("pressed", "#0a61df")])

            # Toolbar
            style.configure("MacToolbar.TFrame", background=window_bg)
            style.configure("MacTool.TButton", background=control_bg, padding=(8, 4))
            style.map("MacTool.TButton",
                      background=[("active", "#ececec"), ("pressed", "#e5e5e5")])

            # Combobox
            style.configure("TCombobox", fieldbackground=control_bg, background=control_bg,
                            bordercolor=subtle_border, lightcolor=subtle_border,
                            darkcolor=subtle_border, arrowcolor=graphite)

            # Checkbutton / Radiobutton
            style.configure("TRadiobutton", background=window_bg, foreground=graphite)
            style.configure("TCheckbutton", background=window_bg, foreground=graphite)

            # Separator line subtle
            style.configure("TSeparator", background=subtle_border)

            # record UI profile
            return style, ui_profile

        # --- ttkbootstrap profile
        if HAVE_TTKBOOTSTRAP:
            ui_profile = "bootstrap"
            chosen = "flatly"
            try:
                style = TBStyle(chosen)  # type: ignore
            except Exception:
                style = ttk.Style()
                style.theme_use("clam")
            # some common aliases used in the rest of the code
            style.configure("Card.TLabelframe", padding=8)
            style.configure("Card.TLabelframe.Label", font=("Helvetica", 12, "bold"))
            return style, ui_profile

        # --- plain ttk classic
        ui_profile = "classic"
        style = ttk.Style()
        try:
            style.theme_use(theme_name)
        except Exception:
            style.theme_use("clam")
        style.configure("TLabel", font=("Helvetica", 11))
        style.configure("Header.TLabel", font=("Helvetica", 13, "bold"))
        style.configure("TButton", font=("Helvetica", 11))
        style.configure("Accent.TButton", font=("Helvetica", 11, "bold"))
        style.configure("Card.TLabelframe", padding=8)
        style.configure("Card.TLabelframe.Label", font=("Helvetica", 12, "bold"))
        return style, ui_profile

    # --- menu
    def _build_menu(self) -> None:
        m = tk.Menu(self)

        fm = tk.Menu(m, tearoff=0)
        fm.add_command(label="Load CSV...", command=self.load_file)
        fm.add_command(label="Save Live Trace", command=self.save_live_trace)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.on_exit)
        m.add_cascade(label="File", menu=fm)

        vm = tk.Menu(m, tearoff=0)
        for name in sorted(PALETTES.keys()):
            vm.add_command(label=f"Palette: {name}",
                           command=lambda n=name: self._apply_palette(n))
        m.add_cascade(label="View", menu=vm)

        uim = tk.Menu(m, tearoff=0)
        uim.add_command(label="Classic (restart; default ttk)",
                        command=lambda: self._explain_switch("classic"))
        uim.add_command(label="Bootstrap (restart; requires ttkbootstrap)",
                        command=lambda: self._explain_switch("bootstrap"))
        uim.add_command(label="Mac-style (restart; PYCONTROL_MACUI=1)",
                        command=lambda: self._explain_switch("mac"))
        m.add_cascade(label="UI Profile", menu=uim)

        hm = tk.Menu(m, tearoff=0)
        hm.add_command(label="Ports Help (macOS)", command=self._ports_help)
        hm.add_command(label="About", command=self._about)
        m.add_cascade(label="Help", menu=hm)

        self.config(menu=m)

    def _explain_switch(self, target: str):
        msg = (
            "Switching UI profile requires restart.\n\n"
            "How to launch:\n"
            "  • Classic: python pycontrol.py\n"
            "  • Bootstrap: PYCONTROL_TTKB=1 python pycontrol.py\n"
            "  • Mac-style: PYCONTROL_MACUI=1 python pycontrol.py\n"
        )
        self.prefs["ui_profile"] = target
        save_prefs(self.prefs)
        messagebox.showinfo("UI Profile", msg)

    # --- layout
    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1, uniform="col")
        self.columnconfigure(1, weight=2, uniform="col")
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # left
        left = ttk.Frame(self, style="TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left.columnconfigure(0, weight=1)

        # logo
        lf = ttk.Frame(left, style="TFrame")
        lf.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._logo_img = self._make_logo_img()
        ttk.Label(lf, image=self._logo_img).pack(side=tk.LEFT)
        ttk.Label(lf, text="PyControl",
                  style="Header.TLabel" if self.ui_profile != "mac" else "TLabel").pack(side=tk.LEFT, padx=8)

        self._build_model_card(left).grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self._build_pid_card(left).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self._build_sweep_card(left).grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self._build_serial_sysid_card(left).grid(row=4, column=0, sticky="ew", pady=(0, 10))

        # right
        right = ttk.Frame(self, style="TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)
        right.rowconfigure(2, weight=0)

        self.fig = Figure(figsize=(7.6, 4.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        tb_prefix = "Mac" if self.ui_profile == "mac" else ""
        self.ctb = CompactToolbar(right, self.canvas, style_prefix=tb_prefix)
        self.ctb.grid(row=1, column=0, sticky="w", pady=(6, 0))

        quick = ttk.Frame(right, style="TFrame")
        quick.grid(row=2, column=0, sticky="ew")
        ttk.Label(quick, text="Palette:").pack(side=tk.LEFT)
        self.palette_combo = ttk.Combobox(
            quick, state="readonly", values=sorted(PALETTES.keys()), width=12
        )
        self.palette_combo.set(self.palette_name)
        self.palette_combo.pack(side=tk.LEFT, padx=6)
        self.palette_combo.bind("<<ComboboxSelected>>",
                                lambda e: self._apply_palette(self.palette_combo.get()))

        self.status = ttk.Label(self, text="", anchor="w", style="TLabel")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))

    # --- logo
    def _make_logo_img(self):
        W, H = 140, 34
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        # subtle rounded rectangle frame (Mac light)
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=6, outline=(200, 200, 205, 255))
        x_prev, y_prev = 6, H // 2
        for x in range(6, W - 6):
            t = (x - 6) / float(W - 12)
            y = 0.65 * H - 0.22 * H * math.exp(-5 * t) * math.cos(8 * t)
            d.line([x_prev, y_prev, x, y], fill=(0, 122, 255, 255), width=2)
            x_prev, y_prev = x, y
        return ImageTk.PhotoImage(img)

    # --- cards
    def _build_model_card(self, parent: ttk.Frame) -> ttk.Labelframe:
        card = ttk.Labelframe(parent, text="System Modeling", style="Card.TLabelframe")
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)
        card.columnconfigure(2, weight=1)

        self.model_mode = tk.IntVar(value=1)
        ttk.Radiobutton(card, text="Frequency-domain Transfer Function",
                        variable=self.model_mode, value=1,
                        command=self._on_tf_mode).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(card, text="Time-domain State-Space",
                        variable=self.model_mode, value=2,
                        command=self._on_ss_mode).grid(row=1, column=0, columnspan=3, sticky="w")

        ttk.Label(card, text="G(s)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.tf_num = ttk.Entry(card, width=18); self.tf_den = ttk.Entry(card, width=18)
        self.tf_num.insert(0, "3,1"); self.tf_den.insert(0, "1,10,20")
        self.tf_num.grid(row=2, column=1, sticky="w"); self.tf_den.grid(row=2, column=2, sticky="w")

        ttk.Label(card, text="A [n x n]").grid(row=3, column=0, sticky="e", pady=(6, 0))
        self.ss_A = ttk.Entry(card, width=22); self.ss_A.grid(row=3, column=1, columnspan=2, sticky="w")
        ttk.Label(card, text="B [n x 1]").grid(row=4, column=0, sticky="e")
        self.ss_B = ttk.Entry(card, width=22); self.ss_B.grid(row=4, column=1, columnspan=2, sticky="w")
        ttk.Label(card, text="C [1 x n]").grid(row=5, column=0, sticky="e")
        self.ss_C = ttk.Entry(card, width=22); self.ss_C.grid(row=5, column=1, columnspan=2, sticky="w")
        ttk.Label(card, text="D [1 x 1]").grid(row=6, column=0, sticky="e")
        self.ss_D = ttk.Entry(card, width=12); self.ss_D.grid(row=6, column=1, sticky="w")

        btns = ttk.Frame(card); btns.grid(row=0, column=3, rowspan=7, padx=(10, 0), sticky="ns")
        for r in range(7): btns.rowconfigure(r, weight=1)
        ttk.Button(btns, text="Pole-Zero", style="Accent.TButton", command=self.pzplot).grid(row=0, column=0, sticky="ew", pady=2)
        ttk.Button(btns, text="Root Locus", command=self.rootlocus_plot).grid(row=1, column=0, sticky="ew", pady=2)
        ttk.Button(btns, text="Step", command=self.stepplot).grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Button(btns, text="Impulse", command=self.impulseplot).grid(row=3, column=0, sticky="ew", pady=2)
        ttk.Button(btns, text="Transient", command=self.transient).grid(row=4, column=0, sticky="ew", pady=2)

        ttk.Label(card, text="Time [s]").grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.time_entry = ttk.Entry(card, width=8); self.time_entry.insert(0, "3"); self.time_entry.grid(row=7, column=1, sticky="w")

        return card

    def _build_pid_card(self, parent: ttk.Frame) -> ttk.Labelframe:
        card = ttk.Labelframe(parent, text="PID Controller", style="Card.TLabelframe")
        for i in range(3): card.rowconfigure(i, weight=1)
        card.columnconfigure(1, weight=1)

        def add_row(row, label, scale, entry, val):
            ttk.Label(card, text=label).grid(row=row, column=0, sticky="w")
            scale.grid(row=row, column=1, sticky="ew")
            ttk.Label(card, text="x").grid(row=row, column=2)
            entry.grid(row=row, column=3, sticky="w")
            ttk.Label(card, text="=").grid(row=row, column=4)
            val.grid(row=row, column=5, sticky="w")

        self.kp_scale = ttk.Scale(card, from_=0, to=10, orient="horizontal", command=lambda e: self._update_pid_labels())
        self.ki_scale = ttk.Scale(card, from_=0, to=10, orient="horizontal", command=lambda e: self._update_pid_labels())
        self.kd_scale = ttk.Scale(card, from_=0, to=10, orient="horizontal", command=lambda e: self._update_pid_labels())
        self.kp_res = ttk.Entry(card, width=6); self.kp_res.insert(0, "1")
        self.ki_res = ttk.Entry(card, width=6); self.ki_res.insert(0, "1")
        self.kd_res = ttk.Entry(card, width=6); self.kd_res.insert(0, "1")
        self.kp_val = ttk.Label(card, text="0.00"); self.ki_val = ttk.Label(card, text="0.00"); self.kd_val = ttk.Label(card, text="0.00")

        add_row(0, "Kp", self.kp_scale, self.kp_res, self.kp_val)
        add_row(1, "Ki", self.ki_scale, self.ki_res, self.ki_val)
        add_row(2, "Kd", self.kd_scale, self.kd_res, self.kd_val)
        self._update_pid_labels()
        return card

    def _build_sweep_card(self, parent: ttk.Frame) -> ttk.Labelframe:
        card = ttk.Labelframe(parent, text="Parameter Sweep", style="Card.TLabelframe")
        for r, name in enumerate(["Kp", "Ki", "Kd"]):
            ttk.Label(card, text=name).grid(row=r, column=0, sticky="w")
            e = ttk.Entry(card, width=20); e.grid(row=r, column=1, sticky="w")
            if r == 0: e.insert(0, "1,10,100"); self.kp_sw = e
            elif r == 1: e.insert(0, "0"); self.ki_sw = e
            else: e.insert(0, "0"); self.kd_sw = e
        ttk.Button(card, text="Sweep", style="Accent.TButton", command=self.sweep).grid(row=0, column=2, rowspan=3, sticky="ns", padx=8)
        return card

    def _build_serial_sysid_card(self, parent: ttk.Frame) -> ttk.Labelframe:
        card = ttk.Labelframe(parent, text="I/O & Identification", style="Card.TLabelframe")

        serial_frame = ttk.Frame(card); serial_frame.grid(row=0, column=0, sticky="ew")
        serial_frame.columnconfigure(1, weight=1)
        ttk.Label(serial_frame, text="ComPort").grid(row=0, column=0, sticky="w")
        self.port_combo = ttk.Combobox(serial_frame, state="readonly", width=36)
        self.port_combo.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        ttk.Label(serial_frame, text="Baud").grid(row=0, column=2, sticky="e")
        self.baud_combo = ttk.Combobox(serial_frame, state="readonly", width=8,
                                       values=[9600, 19200, 38400, 57600, 115200])
        self.baud_combo.grid(row=0, column=3, sticky="w")
        self.baud_combo.bind("<<ComboboxSelected>>", lambda e: self._on_baud_change())

        self.demo_chk = ttk.Checkbutton(serial_frame, text="DEMO mode",
                                        variable=self.demo_mode_var,
                                        command=lambda: self._status(f"DEMO={self.demo_mode_var.get()}"))
        self.demo_chk.grid(row=0, column=4, padx=(8, 0))

        btns = ttk.Frame(card); btns.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(btns, text="Refresh", command=self._refresh_ports).grid(row=0, column=0, padx=2)
        self.read_btn = ttk.Button(btns, text="Start", command=self.start_stream); self.read_btn.grid(row=0, column=1, padx=2)
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop_stream, state="disabled"); self.stop_btn.grid(row=0, column=2, padx=2)
        ttk.Button(btns, text="Save", command=self.save_live_trace).grid(row=0, column=3, padx=2)
        ttk.Button(btns, text="Load CSV...", command=self.load_file).grid(row=0, column=4, padx=2)
        ttk.Button(btns, text="SysID", command=self.sys_id).grid(row=0, column=5, padx=2)

        sysid = ttk.Frame(card); sysid.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(sysid, text="Controller H(s):").grid(row=0, column=0, sticky="w")
        self.H_kp = ttk.Entry(sysid, width=6); self.H_kp.insert(0, "0")
        self.H_ki = ttk.Entry(sysid, width=6); self.H_ki.insert(0, "0")
        self.H_kd = ttk.Entry(sysid, width=6); self.H_kd.insert(0, "0")
        ttk.Label(sysid, text="kp").grid(row=0, column=1); self.H_kp.grid(row=0, column=2)
        ttk.Label(sysid, text="ki").grid(row=0, column=3); self.H_ki.grid(row=0, column=4)
        ttk.Label(sysid, text="kd").grid(row=0, column=5); self.H_kd.grid(row=0, column=6)
        return card

    # --- status
    def _status(self, msg: str) -> None:
        self.status.config(text=msg)
        self.status.update_idletasks()

    # --- palette
    def _apply_palette(self, name: str):
        self.palette_name = name
        self.prefs["palette"] = name
        save_prefs(self.prefs)
        (self.axes_face,
         self.grid_color,
         self.line_color,
         self.text_color,
         self.spine_color) = PALETTES.get(name, PALETTES["Matlab"])
        self.ax.set_facecolor(self.axes_face)
        self.ax.tick_params(colors=self.text_color)
        self.ax.xaxis.label.set_color(self.text_color)
        self.ax.yaxis.label.set_color(self.text_color)
        self.ax.title.set_color(self.text_color)
        for spine in self.ax.spines.values():
            spine.set_color(self.spine_color)
        self.ax.grid(True, color=self.grid_color, alpha=0.6)
        self.canvas.draw_idle()
        self._status(f"Palette -> {name}")

    # --- modeling
    def _get_tf(self):
        n = [float(s) for s in self.tf_num.get().split(",") if s.strip() != ""]
        d = [float(s) for s in self.tf_den.get().split(",") if s.strip() != ""]
        return np.array(n, dtype=float), np.array(d, dtype=float)

    def _get_ss(self):
        aa = np.array(np.asmatrix(self.ss_A.get()))
        bb = np.array(np.asmatrix(self.ss_B.get()))
        cc = np.array(np.asmatrix(self.ss_C.get()))
        dd = np.array(np.asmatrix(self.ss_D.get()))
        TF_G = co.ss2tf(aa, bb, cc, dd)
        n, d = co.tfdata(TF_G)
        return np.array(n[0][0], dtype=float), np.array(d[0][0], dtype=float)

    def _tf_from_mode(self):
        return self._get_tf() if self.model_mode.get() == 1 else self._get_ss()

    # --- PID
    def _pid_values(self):
        def val(scale: ttk.Scale, entry: ttk.Entry):
            x = float(scale.get()) * float(entry.get())
            return float(f"{x:.2f}")
        return val(self.kp_scale, self.kp_res), val(self.ki_scale, self.ki_res), val(self.kd_scale, self.kd_res)

    def _update_pid_labels(self):
        kp, ki, kd = self._pid_values()
        self.kp_val.config(text=f"{kp:.2f}")
        self.ki_val.config(text=f"{ki:.2f}")
        self.kd_val.config(text=f"{kd:.2f}")

    def _apply_pid(self, num, den):
        kp, ki, kd = self._pid_values()
        G = co.tf(num, den)
        if kp > 0 or ki > 0 or kd > 0:
            s = co.tf('s')
            C = kp + ki/s + kd*s
            negfeed = co.feedback(C*G, 1, sign=-1)
            n, d = co.tfdata(negfeed)
            return n[0][0], d[0][0]
        return num, den

    # --- actions
    def pzplot(self):
        o_n, o_d = self._tf_from_mode()
        num, den = self._apply_pid(o_n, o_d)
        z = np.roots(num); p = np.roots(den)
        self.ax.cla(); self._apply_palette(self.palette_name)
        self.ax.scatter(np.real(p), np.imag(p), marker='x', s=80, color=self.line_color)
        self.ax.scatter(np.real(z), np.imag(z), marker='o', s=80, facecolors='none', edgecolors=self.line_color)
        self.ax.axhline(0, color=self.grid_color, lw=0.7); self.ax.axvline(0, color=self.grid_color, lw=0.7)
        self.ax.set_xlabel('Real (s^-1)'); self.ax.set_ylabel('Imag (s^-1)'); self.ax.set_title('Pole-Zero Map')
        vals_re = np.concatenate([np.real(p), np.real(z)]) if p.size + z.size else np.array([0])
        vals_im = np.concatenate([np.imag(p), np.imag(z)]) if p.size + z.size else np.array([0])
        self.ax.set_xlim(vals_re.min() - 1, vals_re.max() + 1); self.ax.set_ylim(vals_im.min() - 1, vals_im.max() + 1)
        self.canvas.draw_idle(); self._status("Pole-Zero plotted.")

    def stepplot(self):
        o_n, o_d = self._tf_from_mode()
        num, den = self._apply_pid(o_n, o_d)
        sys = co.tf(num, den)
        t_end = float(self.time_entry.get())
        t = np.linspace(0, t_end, max(50, int(t_end/0.01)+1))
        t, y = co.step_response(sys, t)
        self.ax.cla(); self._apply_palette(self.palette_name)
        self.ax.plot(t, y, lw=1.4, color=self.line_color)
        self.ax.set_xlabel('time (s)'); self.ax.set_ylabel('Amplitude'); self.ax.set_title('Step Response')
        self.canvas.draw_idle()
        np.savetxt('step_data.csv', np.column_stack([t, y]), delimiter=',')
        self._status("Step computed -> step_data.csv saved.")

    def impulseplot(self):
        o_n, o_d = self._tf_from_mode()
        num, den = self._apply_pid(o_n, o_d)
        sys = co.tf(num, den)
        t_end = float(self.time_entry.get())
        t = np.linspace(0, t_end, max(50, int(t_end/0.01)+1))
        t, y = co.impulse_response(sys, t)
        self.ax.cla(); self._apply_palette(self.palette_name)
        self.ax.plot(t, y, lw=1.2, color=self.line_color)
        self.ax.set_xlabel('time (s)'); self.ax.set_ylabel('Amplitude'); self.ax.set_title('Impulse Response')
        self.canvas.draw_idle()
        self._status("Impulse plotted.")

    def rootlocus_plot(self):
        o_n, o_d = self._tf_from_mode()
        num, den = self._apply_pid(o_n, o_d)
        sys = co.tf(num, den)
        self.ax.cla(); self._apply_palette(self.palette_name)
        co.root_locus(sys, ax=self.ax)
        self.ax.grid(True)
        self.ax.set_title('Root Locus')
        self.canvas.draw_idle()
        self._status("Root Locus plotted.")

    def transient(self):
        path = Path('step_data.csv')
        if not path.exists():
            messagebox.showwarning("No step data", "Run a Step plot first to generate step_data.csv")
            return
        data = pd.read_csv(path, header=None, names=["Time", "Response"]).apply(pd.to_numeric, errors='coerce')
        ss = data["Response"].iloc[-1]
        ten, ninety = 0.1 * ss, 0.9 * ss
        idx = data[(data["Response"] >= ten) & (data["Response"] <= ninety)].index
        if idx.empty:
            rise_time = float('nan'); t10 = t90 = data["Time"].iloc[-1]
        else:
            rise_time = data["Time"].iloc[idx[-1]] - data["Time"].iloc[idx[0]]
            t10 = data["Time"].iloc[idx[0]]; t90 = data["Time"].iloc[idx[-1]]
        peak_val = data["Response"].max(); peak_time = data["Time"].iloc[data["Response"].idxmax()]
        os_ = ((peak_val - ss) / (ss if ss != 0 else 1.0)) * 100.0
        band = 0.02 * abs(ss)
        within = data[(data["Response"] >= ss - band) & (data["Response"] <= ss + band)]
        settling = within["Time"].iloc[0] if not within.empty else float('nan')

        self.ax.cla(); self._apply_palette(self.palette_name)
        self.ax.plot(data["Time"], data["Response"], lw=1.2, color=self.line_color)
        self.ax.axhline(ss, color=self.grid_color, ls='--', lw=0.9)
        self.ax.vlines([t10, t90], ymin=min(data["Response"]), ymax=[ten, ninety],
                       colors=self.grid_color, linestyles='--', lw=0.9)
        self.ax.set_xlabel('time (s)'); self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Transient & Steady-State Metrics')
        self.canvas.draw_idle()
        self._status(f"Rise={rise_time:.3f}s, PeakT={peak_time:.3f}s, OS={os_:.2f}%, Settle~{settling:.3f}s")

    # --- sweep
    def sweep(self):
        def csv_to_array(s: str) -> np.ndarray:
            vals = [float(x) for x in s.split(',') if x.strip() != '']
            return np.array(vals, dtype=float) if vals else np.array([0.0])

        kps = csv_to_array(self.kp_sw.get())
        kis = csv_to_array(self.ki_sw.get())
        kds = csv_to_array(self.kd_sw.get())

        o_n, o_d = self._tf_from_mode()
        G = co.tf(o_n, o_d)
        s_ = co.tf('s')
        t_end = float(self.time_entry.get())
        t = np.linspace(0, t_end, max(50, int(t_end/0.01)+1))

        self.ax.cla(); self._apply_palette(self.palette_name)
        lens = [len(kps), len(kis), len(kds)]
        argmax = int(np.argmax(lens))

        def one_curve(_kp, _ki, _kd):
            C = _kp + _ki/s_ + _kd*s_
            neg = co.feedback(C*G, 1, sign=-1)
            tt, yy = co.step_response(neg, t)
            self.ax.plot(tt, yy, lw=1.0,
                         label=f"kp={_kp:.2f}, ki={_ki:.2f}, kd={_kd:.2f}")

        if argmax == 0:
            ki0, kd0 = float(kis[0]), float(kds[0])
            for kp0 in kps: one_curve(float(kp0), ki0, kd0)
        elif argmax == 1:
            kp0, kd0 = float(kps[0]), float(kds[0])
            for ki0 in kis: one_curve(kp0, float(ki0), kd0)
        else:
            kp0, ki0 = float(kps[0]), float(kis[0])
            for kd0 in kds: one_curve(kp0, ki0, float(kd0))

        self.ax.set_xlabel('time (s)'); self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Step Response - Parameter Sweep')
        self.ax.legend(fontsize=8, loc='best')
        self.canvas.draw_idle()
        self._status("Sweep done.")

    # --- serial / streaming
    def _on_baud_change(self):
        try:
            self.prefs["baud"] = int(self.baud_combo.get())
            save_prefs(self.prefs)
        except Exception:
            pass

    def _refresh_ports(self):
        pairs = list_port_items()
        labels = [lbl for _, lbl in pairs] or ["None"]
        self.port_combo["values"] = labels
        self.port_combo.set(labels[0])
        self._status("Ports: " + ", ".join(labels))

    def _open_serial(self, label: str, baud: int):
        if self.demo_mode_var.get() or os.getenv("PYCSB_DEMO", "0") == "1":
            self._status("DEMO mode active.")
            return DemoSerial()
        dev = parse_device_from_label(label)
        try:
            self._status(f"Opening serial: {dev} @ {baud}")
            return serial.Serial(dev, baud, timeout=0.1)
        except Exception as e:
            messagebox.showwarning("Serial",
                                   f"Could not open {dev}: {e}\nSwitching to DEMO mode.")
            self.demo_mode_var.set(True)
            return DemoSerial()

    def start_stream(self):
        if self.ani_obj:
            try: self.ani_obj.event_source.stop()
            except Exception: pass
            self.ani_obj = None

        label = self.port_combo.get().strip()
        if (not label or label == "None") and not self.demo_mode_var.get():
            if not messagebox.askyesno("No device", "No serial device selected. Start in DEMO mode?"):
                return
            self.demo_mode_var.set(True)

        baud = int(self.baud_combo.get())
        self.ser = self._open_serial(label, baud)
        self.com_port_label = parse_device_from_label(label) if not self.demo_mode_var.get() else "DEMO"
        self.com_xdata = np.array([], dtype=float)
        self.com_ydata = np.array([], dtype=float)

        self.ax.cla(); self._apply_palette(self.palette_name)
        self.ax.set_title(f"{self.com_port_label} - Live Stream")
        self.ax.set_xlabel('sample'); self.ax.set_ylabel('Amplitude')
        (self.live_line,) = self.ax.plot([], [], lw=1.0, color=self.line_color)

        self.ani_obj = animation.FuncAnimation(self.fig, self._animate, interval=60, blit=False)
        self.read_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._status("Streaming...")

    def stop_stream(self):
        if self.ani_obj:
            try: self.ani_obj.event_source.stop()
            except Exception: pass
            self.ani_obj = None
        if self.ser:
            try: self.ser.close()
            except Exception: pass
        self.read_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._status("Stopped.")

    def _read_one_float_line(self) -> Optional[float]:
        try:
            raw = self.ser.readline()  # type: ignore
            if not raw: return None
            s = raw.decode("ascii", errors="ignore").strip()
            if not s: return None
            return float(s)
        except Exception:
            return None

    def _animate(self, _i):
        val = self._read_one_float_line()
        if val is None:
            return
        idx = len(self.com_xdata)
        self.com_xdata = np.append(self.com_xdata, float(idx))
        self.com_ydata = np.append(self.com_ydata, val)
        self.live_line.set_data(self.com_xdata, self.com_ydata)
        if self.com_xdata.size >= 2:
            xmin, xmax = float(self.com_xdata.min()), float(self.com_xdata.max() + 1)
        else:
            xmin, xmax = 0.0, 10.0
        ylo = float(np.min(self.com_ydata)) if self.com_ydata.size else -1.0
        yhi = float(np.max(self.com_ydata)) if self.com_ydata.size else 1.0
        pad = max(1e-3, 0.05 * max(1.0, abs(yhi - ylo)))
        self.ax.set_xlim(xmin, xmax); self.ax.set_ylim(ylo - pad, yhi + pad)
        self.canvas.draw_idle()

    def save_live_trace(self):
        if self.com_xdata.size == 0:
            messagebox.showwarning("No data", "No live data to save yet.")
            return
        data = np.column_stack([self.com_xdata, self.com_ydata])
        np.savetxt('data.csv', data, delimiter=',')
        self._status("Saved data.csv")

    # --- file / sys-id
    def load_file(self):
        f = filedialog.askopenfilename(title="Select CSV",
                                       filetypes=(("CSV", "*.csv"), ("Text", "*.txt")))
        if not f: return
        data = np.loadtxt(f, delimiter=",")
        t_data = data[:, 0]; y_data = data[:, 1]
        self.ax.cla(); self._apply_palette(self.palette_name)
        self.ax.plot(t_data, y_data, lw=1.0, color=self.line_color)
        self.ax.set_xlabel('time (s)'); self.ax.set_ylabel('Amplitude')
        self.ax.set_title('System Step Response (Loaded)')
        self.canvas.draw_idle()
        self.time_data = t_data; self.response_data = y_data
        self._status(f"Loaded {Path(f).name}")

    def second_order_model(self, t, K, wn, zeta):
        tf_ = co.tf([K * wn**2], [1, 2 * zeta * wn, wn**2])
        tout, yout = co.step_response(tf_, t)
        return np.interp(t, tout, yout)

    def sys_id(self):
        if not hasattr(self, 'time_data'):
            path = Path('step_data.csv')
            if not path.exists():
                messagebox.showwarning("No data", "Load a CSV or run a Step first.")
                return
            data = np.loadtxt(path, delimiter=",")
            self.time_data = data[:, 0]; self.response_data = data[:, 1]

        initial = [1.0, 1.0, 0.5]
        try:
            params, _ = curve_fit(self.second_order_model, self.time_data, self.response_data, p0=initial)
            Kf, wnf, zf = params
            fit = self.second_order_model(self.time_data, Kf, wnf, zf)

            self.ax.cla(); self._apply_palette(self.palette_name)
            self.ax.plot(self.time_data, self.response_data, label="Original")
            self.ax.plot(self.time_data, fit, '--', label=f"Fit: K={Kf:.2f}, wn={wnf:.2f}, zeta={zf:.2f}")
            self.ax.set_xlabel('time (s)'); self.ax.set_ylabel('Amplitude')
            self.ax.set_title('Step Response - 2nd Order Fit')
            self.ax.legend(); self.canvas.draw_idle()

            hkp = float(self.H_kp.get()); hki = float(self.H_ki.get()); hkd = float(self.H_kd.get())
            s = co.tf('s'); H = hkp + hki/s + hkd*s
            TT = co.tf([Kf * wnf**2], [1, 2 * zf * wnf, wnf**2])
            GG = co.minreal((TT / (1 - TT)) / H)
            print("SysID TF:", TT)
            print("Controller H(s):", H)
            print("Estimated Plant:", GG)
            self._status("SysID complete (console shows plant TF).")
        except Exception as e:
            messagebox.showerror("SysID", f"Error fitting model: {e}")

    # --- misc
    def _on_tf_mode(self):
        self._status("TF mode.")

    def _on_ss_mode(self):
        self.ss_A.delete(0, 'end'); self.ss_A.insert(0, "-10,-20;1,0")
        self.ss_B.delete(0, 'end'); self.ss_B.insert(0, "1;0")
        self.ss_C.delete(0, 'end'); self.ss_C.insert(0, "3,1")
        self.ss_D.delete(0, 'end'); self.ss_D.insert(0, "0")
        self._status("SS mode (prefilled example).")

    def _about(self):
        messagebox.showinfo(
            "About PyControl",
            "PyControl - compact, Matlab-ish control toolbox in Python.\n"
            "- DEMO-safe serial streaming\n"
            "- Pole-Zero, Root Locus, Step/Impulse, Transient, Sweep\n"
            "- Quick SysID\n\n"
            "UI profiles: classic • bootstrap • mac\n"
            "Built with tkinter + matplotlib + python-control."
        )

    def _ports_help(self):
        messagebox.showinfo(
            "Ports Help (macOS)",
            "On macOS, USB-serial devices appear as /dev/cu.*\n\n"
            "Common names:\n"
            "  - /dev/cu.usbserial-XXXX (FTDI)\n"
            "  - /dev/cu.usbmodemXXXX (CDC/Arduino)\n"
            "  - /dev/cu.SLAB_USBtoUART (CP210x)\n"
            "  - /dev/cu.wchusbserial* (CH340/CH341)\n\n"
            "Use:  python -m serial.tools.list_ports -v\n"
            "USB-C works with any USB-C serial adapter or a hub."
        )

    def on_exit(self):
        try:
            if self.ani_obj:
                self.ani_obj.event_source.stop()
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = PyControlApp()
    app.mainloop()
