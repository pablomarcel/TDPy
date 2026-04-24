#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
gui_core_dpg

Dear PyGui frontend for the TDpy package — Polished Layout Edition.

New focus:
- Modern, welcoming, student-friendly look
- Light theme by default
- Left-to-right layout: Inputs (left) / Outputs (right)
- Outputs organized with tabs: Results / Parsed JSON / Logs
- Bigger controls + better spacing + a bit of cuteness ✨

Run:
    runroot python -m gui_core_dpg
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Any, Dict, Mapping, Optional, Callable, List

# ---------- Import shim so `python gui_core_dpg.py` works ----------
if __package__ in (None, ""):
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

try:
    import dearpygui.dearpygui as dpg
except Exception as e:  # pragma: no cover
    raise SystemExit("Dear PyGui is required. Install with: pip install dearpygui") from e

from gui_log_dpg import LogPanel
from gui_utils_dpg import (
    CmdResult,
    ensure_dir,
    extract_dpg_file_dialog_path,
    find_repo_root,
    in_dir,
    last_nonempty_line,
    load_json,
    load_text,
    open_path,
    out_dir,
    preview_cmd,
    rel_to_in,
    resolve_input_pattern,
    run_cmd_async,
    save_json,
    save_text,
    unique_path,
)

# ------------------------------ state ------------------------------

@dataclass
class AppState:
    repo_root: Path
    in_root: Path
    out_root: Path

    # Fonts (optional). If not found/loaded, these remain None.
    # - ui_font_default is used for Light/Dark and as fallback reset.
    # - ui_font_macos is used for the macOS palette.
    # - ui_font_labview is used for the LabVIEW palette.
    # - mono_font is used for editors/outputs/logs (bound per-item).
    ui_font_default: int | None = None
    ui_font_macos: int | None = None
    ui_font_labview: int | None = None
    mono_font: int | None = None

    eqn_loaded_problem: Dict[str, Any] | None = None
    eqn_guess_widgets: Dict[str, Dict[str, str]] | None = None
    eqn_last_out_file: str = ""

    props_last_out_file: str = ""


def t(prefix: str, name: str) -> str:
    return f"##{prefix}_{name}"


LEFT_PANEL_WIDTH = 650
LOG_PANEL_HEIGHT = 420
TABBAR_TAG = "##tabbar"
TAB_HOME = "##tab_home"
TAB_PROPS = "##tab_props"
TAB_EQN = "##tab_eqn"

THEME_TOGGLE_TAG = "##app_theme_mode"
SCALE_SLIDER_TAG = "##app_ui_scale"

DEFAULT_UI_SCALE = 1.12

UiTask = Callable[[], None]


def enqueue_task(q: SimpleQueue[UiTask], fn: UiTask) -> None:
    q.put(fn)


def drain_tasks(q: SimpleQueue[UiTask], *, max_tasks: int = 50) -> None:
    for _ in range(max_tasks):
        try:
            fn = q.get_nowait()
        except Exception:
            return
        try:
            fn()
        except Exception:
            return


# ------------------------------ themes ------------------------------

def _make_theme_light() -> int:
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            # Light theme goal:
            # - "Real estate" (windows/child areas) = light gray
            # - Input/output frames (text areas, inputs) = white
            # This matches the PySide-style look you referenced.
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (240, 242, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (236, 240, 245, 255))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (255, 255, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (17, 24, 39, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (107, 114, 128, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 255, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (249, 250, 251, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (243, 244, 246, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (203, 213, 225, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Separator, (226, 232, 240, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (59, 130, 246, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (37, 99, 235, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (29, 78, 216, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (229, 231, 235, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (191, 219, 254, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (59, 130, 246, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (219, 234, 254, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (191, 219, 254, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (147, 197, 253, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (241, 245, 249, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (148, 163, 184, 255))

            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 10)

            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 14)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 7)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 6)

        # Button label contrast for the blue buttons in Light theme.
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))
    return theme


def _make_theme_dark_soft() -> int:
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (22, 24, 28, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (28, 30, 35, 255))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (28, 30, 35, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (233, 236, 239, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (156, 163, 175, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (41, 45, 52, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (55, 60, 69, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (70, 76, 87, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border, (55, 60, 69, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (99, 102, 241, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (79, 70, 229, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (67, 56, 202, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (41, 45, 52, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (70, 76, 87, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (99, 102, 241, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (55, 60, 69, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (70, 76, 87, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (99, 102, 241, 255))

            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 14)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 7)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 6)
    return theme


def _make_theme_macos() -> int:
    """macOS-like light theme: unified surfaces + accent blue.

    Inspired by `pycontrol_macos.py` (Tk style):
      window_bg      = #f5f5f7
      control_bg     = #ffffff
      subtle_border  = #d6d6d6
      graphite       = #1c1c1e
      accent_blue    = #007aff
    """

    window_bg = (245, 245, 247, 255)      # #f5f5f7
    control_bg = (255, 255, 255, 255)     # #ffffff
    subtle_border = (214, 214, 214, 255)  # #d6d6d6
    graphite = (28, 28, 30, 255)          # #1c1c1e
    accent = (0, 122, 255, 255)           # #007aff

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, window_bg)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, window_bg)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, control_bg)

            dpg.add_theme_color(dpg.mvThemeCol_Text, graphite)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (120, 120, 125, 255))

            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, control_bg)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (248, 248, 250, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (240, 240, 243, 255))

            dpg.add_theme_color(dpg.mvThemeCol_Border, subtle_border)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, subtle_border)

            dpg.add_theme_color(dpg.mvThemeCol_Button, accent)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (22, 118, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (10, 97, 223, 255))

            dpg.add_theme_color(dpg.mvThemeCol_Tab, (230, 230, 233, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (210, 225, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, accent)

            dpg.add_theme_color(dpg.mvThemeCol_Header, (219, 234, 254, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (191, 219, 254, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (147, 197, 253, 255))

            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (241, 245, 249, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (148, 163, 184, 255))

            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 10)

            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 14)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 7)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 6)

        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255, 255))

    return theme


def _make_theme_labview() -> int:
    """LabVIEW-like front panel theme: silver + graphite + amber accents.

    Inspired by `pycontrol_labview.py` (Tk style):
      window_bg   = #dfe1e5
      panel_bg    = #d0d2d6
      control_bg  = #f2f3f5
      graphite    = #1e1f22
      border      = #8b8f94
      accent      = #f2c200
    """

    window_bg = (223, 225, 229, 255)   # #dfe1e5
    panel_bg = (208, 210, 214, 255)    # #d0d2d6
    control_bg = (242, 243, 245, 255)  # #f2f3f5
    graphite = (30, 31, 34, 255)       # #1e1f22
    border = (139, 143, 148, 255)      # #8b8f94
    accent = (242, 194, 0, 255)        # #f2c200

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, window_bg)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, panel_bg)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, control_bg)

            dpg.add_theme_color(dpg.mvThemeCol_Text, graphite)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (110, 115, 120, 255))

            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, control_bg)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (233, 235, 238, 255))  # #e9ebee-ish
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (224, 226, 230, 255))

            dpg.add_theme_color(dpg.mvThemeCol_Border, border)
            dpg.add_theme_color(dpg.mvThemeCol_Separator, border)

            # Default buttons are gray; hover/active swings toward amber (LabVIEW-ish).
            dpg.add_theme_color(dpg.mvThemeCol_Button, (230, 232, 235, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 213, 77, 255))  # #ffd54d
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, accent)

            dpg.add_theme_color(dpg.mvThemeCol_Tab, (215, 217, 221, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (245, 224, 140, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, accent)

            dpg.add_theme_color(dpg.mvThemeCol_Header, (220, 222, 226, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (245, 224, 140, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (255, 213, 77, 255))

            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, panel_bg)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (160, 165, 170, 255))

            # LabVIEW feels more rectangular/tight than macOS.
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 14, 12)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 7)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 5)

    return theme


@dataclass(frozen=True)
class ThemeSpec:
    label: str
    key: str
    theme_id: int
    font_id: int | None


def _norm_theme_key(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "")
    if s in ("lab", "labview", "lv", "ni", "lab-view"):
        return "labview"
    if s in ("mac", "macos", "osx", "macosx"):
        return "macos"
    if s in ("dark", "night"):
        return "dark"
    if s in ("light", "day"):
        return "light"
    return s


def _file_dialog_ext_tags(prefix: str) -> Dict[str, str]:
    return {
        "all": t(prefix, "fdext_all"),
        "json": t(prefix, "fdext_json"),
        "txt": t(prefix, "fdext_txt"),
        "csv": t(prefix, "fdext_csv"),
        "yaml": t(prefix, "fdext_yaml"),
        "yml": t(prefix, "fdext_yml"),
    }


def _apply_file_dialog_extension_colors(mode: str) -> None:
    """
    Dear PyGui file dialogs color registered file extensions separately from the
    regular theme text. That is why .txt/.json rows were white in light mode,
    while unregistered extensions like .csv/.yaml looked normal.

    We explicitly recolor the registered extension filters/items whenever the
    app theme changes.
    """
    key = _norm_theme_key(mode)
    if key == "dark":
        ext_color = (233, 236, 239, 255)
    else:
        ext_color = (17, 24, 39, 255)

    for prefix in ("eqn", "props"):
        tags = list(_file_dialog_ext_tags(prefix).values()) + [
            t(prefix, "fdext_interp_json"),
            t(prefix, "fdext_solve_all"),
            t(prefix, "fdext_solve_json"),
            t(prefix, "fdext_solve_out"),
        ]
        for tag in tags:
            if dpg.does_item_exist(tag):
                try:
                    dpg.configure_item(tag, color=ext_color)
                except Exception:
                    pass


def _apply_theme(mode: str, themes: Mapping[str, ThemeSpec]) -> None:
    key = _norm_theme_key(mode)
    spec = themes.get(key) or themes.get("light")
    if spec is None:
        return

    try:
        dpg.bind_theme(spec.theme_id)
    except Exception:
        pass

    # Bind UI font (theme-specific) if available.
    # If font_id is None, we intentionally keep the current font.
    if spec.font_id is not None:
        try:
            dpg.bind_font(spec.font_id)
        except Exception:
            pass

    _apply_file_dialog_extension_colors(mode)


def _set_ui_scale(scale: float) -> None:
    try:
        dpg.set_global_font_scale(float(scale))
    except Exception:
        pass



def _load_fonts(state: AppState, *, ui_point_size: int = 14, mono_point_size: int = 13) -> None:
    """Load UI + monospace fonts (best-effort) and store their IDs in `state`.

    Dear PyGui fonts require explicit font files. We try common system locations
    on macOS / Windows / Linux. If a font can't be found/loaded, that entry
    remains None — themes still work (colors/styles), you just won't get the
    intended typeface.
    """

    def _try_add_font(path: str, size: int) -> int | None:
        fp = Path(path)
        if not fp.exists():
            return None
        try:
            return dpg.add_font(str(fp), size)
        except Exception:
            return None

    # Candidate lists: keep them conservative and OS-friendly.
    ui_default_candidates: list[str] = []
    ui_macos_candidates: list[str] = []
    ui_labview_candidates: list[str] = []
    mono_candidates: list[str] = []

    # --- macOS ---
    if sys.platform.startswith("darwin"):
        ui_default_candidates += [
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/SFNSText.ttf",
            "/System/Library/Fonts/SFNSDisplay.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
        ]
        ui_macos_candidates += [
            "/System/Library/Fonts/SFNSText.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        ui_labview_candidates += [
            "/System/Library/Fonts/Supplemental/Verdana.ttf",
            "/System/Library/Fonts/Supplemental/Tahoma.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        mono_candidates += [
            "/System/Library/Fonts/Monaco.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/Library/Fonts/Menlo.ttc",
        ]

    # --- Windows ---
    if os.name == "nt":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        ui_default_candidates += [
            str(Path(windir) / "Fonts" / "segoeui.ttf"),
            str(Path(windir) / "Fonts" / "arial.ttf"),
        ]
        ui_macos_candidates += [
            str(Path(windir) / "Fonts" / "segoeui.ttf"),
            str(Path(windir) / "Fonts" / "arial.ttf"),
        ]
        ui_labview_candidates += [
            str(Path(windir) / "Fonts" / "verdana.ttf"),
            str(Path(windir) / "Fonts" / "tahoma.ttf"),
            str(Path(windir) / "Fonts" / "arial.ttf"),
        ]
        mono_candidates += [
            str(Path(windir) / "Fonts" / "consola.ttf"),
            str(Path(windir) / "Fonts" / "lucon.ttf"),
            str(Path(windir) / "Fonts" / "cour.ttf"),
        ]

    # --- Linux (and others) ---
    ui_default_candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    ui_macos_candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    ui_labview_candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    mono_candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    ]

    # Create ONE font registry and add fonts into it.
    try:
        with dpg.font_registry():
            state.ui_font_default = next((fid for fid in (_try_add_font(p, ui_point_size) for p in ui_default_candidates) if fid), None)
            state.ui_font_macos = next((fid for fid in (_try_add_font(p, ui_point_size) for p in ui_macos_candidates) if fid), None)
            state.ui_font_labview = next((fid for fid in (_try_add_font(p, ui_point_size) for p in ui_labview_candidates) if fid), None)
            state.mono_font = next((fid for fid in (_try_add_font(p, mono_point_size) for p in mono_candidates) if fid), None)
    except Exception:
        # If font registry creation fails for any reason, keep fonts as None.
        return

    # Ensure we have a "reset" UI font if any UI font loaded.
    if state.ui_font_default is None:
        state.ui_font_default = state.ui_font_macos or state.ui_font_labview

    # Fall back theme-specific fonts to default when missing.
    if state.ui_font_macos is None:
        state.ui_font_macos = state.ui_font_default
    if state.ui_font_labview is None:
        state.ui_font_labview = state.ui_font_default


def _load_mono_font(*, point_size: int = 13) -> int | None:
    """Load a monospace font for editors/outputs only.

    Dear PyGui requires a font file path. We try common system locations on
    macOS/Windows/Linux. If none are found or loading fails, return None.
    """

    candidates: list[str] = []

    # macOS
    if sys.platform.startswith("darwin"):
        candidates += [
            "/System/Library/Fonts/Monaco.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/Library/Fonts/Menlo.ttc",
        ]

    # Windows
    if os.name == "nt":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        candidates += [
            str(Path(windir) / "Fonts" / "consola.ttf"),
            str(Path(windir) / "Fonts" / "lucon.ttf"),
            str(Path(windir) / "Fonts" / "cour.ttf"),
        ]

    # Linux
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    ]

    for p in candidates:
        fp = Path(p)
        if not fp.exists():
            continue
        try:
            # Must be created after dpg.create_context().
            with dpg.font_registry():
                return dpg.add_font(str(fp), point_size)
        except Exception:
            continue

    return None


# ------------------------------ guess editor ------------------------------

def _extract_variables_mapping(payload: Mapping[str, Any]) -> Dict[str, Any]:
    v = payload.get("variables")
    if isinstance(v, Mapping):
        return dict(v)
    v = payload.get("vars")
    if isinstance(v, Mapping):
        return dict(v)
    return {}


def _rebuild_guess_table(prefix: str, state: AppState, log: LogPanel) -> None:
    group_tag = t(prefix, "guess_group")

    if dpg.does_item_exist(group_tag):
        kids = dpg.get_item_children(group_tag, 1) or []
        for c in kids:
            try:
                dpg.delete_item(c)
            except Exception:
                pass

    state.eqn_guess_widgets = {}

    payload = state.eqn_loaded_problem
    if not isinstance(payload, Mapping):
        with dpg.group(parent=group_tag):
            dpg.add_text("Load/interpret an equations JSON to edit guesses.")
        return

    if str(payload.get("problem_type", "")).strip() != "equations":
        with dpg.group(parent=group_tag):
            dpg.add_text("Guess editor is available for problem_type='equations' only.")
        return

    variables = _extract_variables_mapping(payload)
    if not variables:
        with dpg.group(parent=group_tag):
            dpg.add_text("No variables found in JSON.")
        return

    with dpg.group(parent=group_tag):
        dpg.add_text("Initial guesses ✍️ (edit → Apply to write *_guess.json):")
        with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(label="Variable", width_fixed=True)
            dpg.add_table_column(label="Fixed")
            dpg.add_table_column(label="Value")
            dpg.add_table_column(label="Guess")

            for name, spec in variables.items():
                nm = str(name)
                if not isinstance(spec, Mapping):
                    fixed = False
                    value = ""
                    guess = str(spec)
                else:
                    fixed = bool(spec.get("fixed", False)) or str(spec.get("kind", "")).lower() == "fixed"
                    value = "" if spec.get("value", None) is None else str(spec.get("value"))
                    guess = "" if spec.get("guess", None) is None else str(spec.get("guess"))
                    if not guess and (spec.get("value", None) is not None) and (not fixed):
                        guess = str(spec.get("value"))

                tag_fixed = t(prefix, f"fixed_{nm}")
                tag_val = t(prefix, f"val_{nm}")
                tag_guess = t(prefix, f"guess_{nm}")
                state.eqn_guess_widgets[nm] = {"fixed": tag_fixed, "value": tag_val, "guess": tag_guess}

                with dpg.table_row():
                    dpg.add_text(nm)
                    dpg.add_checkbox(tag=tag_fixed, default_value=bool(fixed))
                    dpg.add_input_text(tag=tag_val, default_value=value, width=-1)
                    dpg.add_input_text(tag=tag_guess, default_value=guess, width=-1)

        dpg.add_spacer(height=6)
        with dpg.group(horizontal=True):
            dpg.add_button(label="✅ Apply guesses", callback=lambda: _apply_guesses(prefix, state, log))
            dpg.add_text("→ writes *_guess.json and selects it for solve")


def _apply_guesses(prefix: str, state: AppState, log: LogPanel) -> None:
    solve_in_tag = t(prefix, "solve_in")
    infile = (dpg.get_value(solve_in_tag) or "").strip()
    if not infile:
        log.warn("No solve input JSON selected.")
        return

    src0 = Path(infile).expanduser()
    src = resolve_input_pattern(src0, prefer_exts=(".json", ".txt"))
    if src is None or not src.exists():
        log.warn(f"Solve input not found: {src0}")
        return

    try:
        payload = load_json(src)
    except Exception as e:
        log.error(f"Failed to load JSON: {src} ({e})")
        return

    variables = _extract_variables_mapping(payload)
    if not variables:
        log.warn("No variables mapping found in JSON.")
        return

    for nm, tags in (state.eqn_guess_widgets or {}).items():
        fixed = bool(dpg.get_value(tags["fixed"]))
        val_s = str(dpg.get_value(tags["value"]) or "").strip()
        guess_s = str(dpg.get_value(tags["guess"]) or "").strip()

        spec: Dict[str, Any] = {}
        if fixed:
            spec["fixed"] = True
            if val_s:
                try:
                    spec["value"] = float(val_s)
                except Exception:
                    spec["value"] = val_s
        else:
            spec["fixed"] = False
            if guess_s:
                try:
                    spec["guess"] = float(guess_s)
                except Exception:
                    spec["guess"] = guess_s
            if val_s and "guess" not in spec:
                try:
                    spec["guess"] = float(val_s)
                except Exception:
                    spec["guess"] = val_s

        variables[nm] = spec

    payload["variables"] = variables
    dst = unique_path(src.with_name(src.stem + "_guess.json"))
    save_json(dst, payload, indent=2)

    dpg.set_value(solve_in_tag, str(dst))
    log.info(f"Wrote guess-updated JSON: {dst}")

    state.eqn_loaded_problem = payload
    _rebuild_guess_table(prefix, state, log)
    _refresh_previews(prefix, state)


# ------------------------------ command builders ------------------------------

def _build_solve_cmd(prefix: str, state: AppState) -> List[str]:
    kind = str(dpg.get_value(t(prefix, "solve_kind")) or "run").strip()
    infile = (dpg.get_value(t(prefix, "solve_in")) or "").strip()
    if not infile:
        return []

    in_path0 = Path(infile).expanduser()
    in_path = resolve_input_pattern(in_path0, prefer_exts=(".json", ".txt")) or in_path0

    cmd = [sys.executable, "-m", "cli", kind]

    in_arg = rel_to_in(in_path, state.repo_root) if in_path.exists() else str(in_path)
    cmd += ["--in", in_arg]

    out_s = (dpg.get_value(t(prefix, "solve_out")) or "").strip()
    if out_s:
        cmd += ["--out", out_s]

    backend = str(dpg.get_value(t(prefix, "ov_backend")) or "(spec)")
    if backend != "(spec)":
        cmd += ["--backend", backend]

    method = str(dpg.get_value(t(prefix, "ov_method")) or "(spec)")
    if method != "(spec)":
        cmd += ["--method", method]

    tol_s = (dpg.get_value(t(prefix, "ov_tol")) or "").strip()
    if tol_s:
        cmd += ["--tol", tol_s]

    mi_s = (dpg.get_value(t(prefix, "ov_max_iter")) or "").strip()
    if mi_s:
        cmd += ["--max-iter", mi_s]

    mr_s = (dpg.get_value(t(prefix, "ov_max_restarts")) or "").strip()
    if mr_s:
        cmd += ["--max-restarts", mr_s]

    units = str(dpg.get_value(t(prefix, "ov_units")) or "(spec)")
    if units == "use-units":
        cmd += ["--use-units"]
    elif units == "no-units":
        cmd += ["--no-units"]

    if bool(dpg.get_value(t(prefix, "ov_dry_run"))):
        cmd += ["--dry-run"]

    return cmd


def _refresh_previews(prefix: str, state: AppState) -> None:
    mode = str(dpg.get_value(t(prefix, "input_mode")) or "Editor").strip().lower()
    if mode.startswith("editor"):
        prev = "runroot python -m interpreter.cli --in <editor_text> --out <auto>"
    else:
        in_s = (dpg.get_value(t(prefix, "input_path")) or "").strip()
        out_s = (dpg.get_value(t(prefix, "interp_out")) or "").strip()
        prev = f"runroot python -m interpreter.cli --in {in_s or '<...>'} --out {out_s or '<auto>'}"
    if dpg.does_item_exist(t(prefix, "interp_preview")):
        dpg.set_value(t(prefix, "interp_preview"), prev)

    scmd = _build_solve_cmd(prefix, state)
    if dpg.does_item_exist(t(prefix, "solve_preview")):
        dpg.set_value(t(prefix, "solve_preview"), preview_cmd(scmd, prefix="runroot"))


def _set_parsed_view(prefix: str, path: Optional[Path], log: LogPanel) -> None:
    tag = t(prefix, "parsed_json")
    if not dpg.does_item_exist(tag):
        return
    if path is None or not path.exists():
        dpg.set_value(tag, "")
        return
    try:
        txt = load_text(path)
        if len(txt) > 300_000:
            txt = txt[:300_000] + "\n... (truncated) ..."
        dpg.set_value(tag, txt)
    except Exception as e:
        log.warn(f"Could not read parsed JSON: {path} ({e})")
        dpg.set_value(tag, "")


# ------------------------------ callbacks ------------------------------

def _pick_input_cb(sender, app_data, user_data) -> None:
    prefix, state, log = user_data["prefix"], user_data["state"], user_data["log"]
    raw = extract_dpg_file_dialog_path(app_data)
    if not raw:
        return
    chosen0 = Path(raw).expanduser()
    chosen = resolve_input_pattern(chosen0, prefer_exts=(".txt", ".json")) or chosen0

    dpg.set_value(t(prefix, "input_path"), str(chosen))
    dpg.set_value(t(prefix, "solve_in"), str(chosen))
    log.info(f"Selected input: {chosen}")
    _refresh_previews(prefix, state)

    if prefix == "eqn" and str(chosen).lower().endswith(".json"):
        try:
            state.eqn_loaded_problem = load_json(Path(chosen))
        except Exception:
            state.eqn_loaded_problem = None
        _rebuild_guess_table(prefix, state, log)


def _pick_interp_out_cb(sender, app_data, user_data) -> None:
    prefix, state = user_data["prefix"], user_data["state"]
    raw = extract_dpg_file_dialog_path(app_data)
    if raw:
        dpg.set_value(t(prefix, "interp_out"), raw)
        _refresh_previews(prefix, state)


def _pick_solve_out_cb(sender, app_data, user_data) -> None:
    prefix, state = user_data["prefix"], user_data["state"]
    raw = extract_dpg_file_dialog_path(app_data)
    if raw:
        dpg.set_value(t(prefix, "solve_out"), raw)
        _refresh_previews(prefix, state)


def _open_input_folder(prefix: str, state: AppState) -> None:
    s = (dpg.get_value(t(prefix, "input_path")) or "").strip()
    if not s:
        open_path(state.in_root)
        return
    p = Path(s).expanduser()
    open_path(p.parent if p.parent.exists() else state.in_root)


def _open_output_folder(state: AppState) -> None:
    open_path(state.out_root)


def _interpret(prefix: str, state: AppState, log: LogPanel, uiq: SimpleQueue[UiTask], *, after_ok: Optional[Callable[[], None]] = None) -> None:
    mode = str(dpg.get_value(t(prefix, "input_mode")) or "Editor").strip().lower()

    if mode.startswith("file"):
        in_s = (dpg.get_value(t(prefix, "input_path")) or "").strip()
        if not in_s:
            log.warn("Choose an input TXT file (or switch to Editor).")
            return
        in0 = Path(in_s).expanduser()
        in_path = resolve_input_pattern(in0, prefer_exts=(".txt", ".json")) or in0
        if not in_path.exists():
            log.warn(f"Input file not found: {in_path}")
            return
    else:
        text = str(dpg.get_value(t(prefix, "editor")) or "")
        if not text.strip():
            log.warn("Editor is empty.")
            return
        scratch_dir = ensure_dir(state.in_root / "gui_scratch")
        in_path = unique_path(scratch_dir / f"{prefix}_gui_equations.txt")
        save_text(in_path, text)
        dpg.set_value(t(prefix, "input_path"), str(in_path))
        log.info(f"Saved editor text to: {in_path}")

    out_s = (dpg.get_value(t(prefix, "interp_out")) or "").strip()
    if out_s:
        out_path = Path(out_s).expanduser()
    else:
        cand = in_path.with_suffix(".json") if in_path.suffix.lower() == ".txt" else in_path.with_name(in_path.stem + "_interpreted.json")
        out_path = unique_path(cand)
        dpg.set_value(t(prefix, "interp_out"), str(out_path))

    cmd = [sys.executable, "-m", "interpreter.cli", "--in", str(in_path), "--out", str(out_path)]
    dpg.set_value(t(prefix, "interp_preview"), preview_cmd(cmd, prefix="runroot"))

    log.set_status("Interpreting…")
    log.info("Interpreting…")

    def _on_line(line: str) -> None:
        # Interpreter uses stderr for warnings; don't scare the user with "ERR:".
        if line.startswith("STDERR:"):
            log.warn(line.replace("STDERR:", "", 1).strip())
        elif line.startswith("ERR:"):
            log.warn(line.replace("ERR:", "", 1).strip())
        else:
            log.info(line)

    def _on_done(res: CmdResult) -> None:
        def _apply_done() -> None:
            if res.returncode != 0:
                log.error(f"Interpret failed (rc={res.returncode}).")
                if res.stderr.strip():
                    log.error(res.stderr.strip())
                log.set_status("Interpret failed")
                return

            log.info("Interpret complete ✅")
            log.set_status("Interpret complete")
            dpg.set_value(t(prefix, "solve_in"), str(out_path))
            _set_parsed_view(prefix, out_path, log)
            _refresh_previews(prefix, state)

            if prefix == "eqn":
                try:
                    state.eqn_loaded_problem = load_json(out_path)
                except Exception as e:
                    state.eqn_loaded_problem = None
                    log.warn(f"Could not load interpreted JSON for guess editor: {e}")
                _rebuild_guess_table(prefix, state, log)

            if after_ok is not None:
                after_ok()

        enqueue_task(uiq, _apply_done)

    run_cmd_async(cmd, cwd=state.repo_root, on_line=_on_line, on_done=_on_done)


def _solve(prefix: str, state: AppState, log: LogPanel, uiq: SimpleQueue[UiTask]) -> None:
    cmd = _build_solve_cmd(prefix, state)
    if not cmd:
        log.warn("Choose a solve input file first.")
        return

    dpg.set_value(t(prefix, "solve_preview"), preview_cmd(cmd, prefix="runroot"))
    log.set_status("Solving…")
    log.info("Solving…")

    def _on_line(line: str) -> None:
        if line.startswith("STDERR:"):
            log.warn(line.replace("STDERR:", "", 1).strip())
        elif line.startswith("ERR:"):
            log.warn(line.replace("ERR:", "", 1).strip())
        else:
            log.info(line)

    def _on_done(res: CmdResult) -> None:
        def _apply_done() -> None:
            if res.returncode != 0:
                log.error(f"Solve failed (rc={res.returncode}).")
                if res.stderr.strip():
                    log.error(res.stderr.strip())
                log.set_status("Solve failed")
                return

            outp = last_nonempty_line(res.stdout)
            out_path: Optional[Path] = None
            if outp:
                p = Path(outp).expanduser()
                out_path = p if p.is_absolute() else (state.repo_root / p).resolve()

            if out_path is None:
                asked = (dpg.get_value(t(prefix, "solve_out")) or "").strip()
                out_path = Path(asked).expanduser() if asked else None

            if out_path and out_path.exists():
                try:
                    txt = load_text(out_path)
                except Exception as e:
                    log.warn(f"Could not read output file: {out_path} ({e})")
                    txt = ""
                if len(txt) > 300_000:
                    txt = txt[:300_000] + "\n... (truncated) ..."
                dpg.set_value(t(prefix, "results"), txt)

                if prefix == "eqn":
                    state.eqn_last_out_file = str(out_path)
                else:
                    state.props_last_out_file = str(out_path)

                log.info(f"Output: {out_path}")
                log.set_status("Solved ✅")
            else:
                log.warn("Solve completed, but output file path could not be determined.")
                log.set_status("Solved (output unknown)")

        enqueue_task(uiq, _apply_done)

    run_cmd_async(cmd, cwd=state.repo_root, on_line=_on_line, on_done=_on_done)


def _open_last_output(prefix: str, state: AppState) -> None:
    p = state.eqn_last_out_file if prefix == "eqn" else state.props_last_out_file
    if p and Path(p).exists():
        open_path(p)


# ------------------------------ UI builders ------------------------------

def _build_file_dialogs(prefix: str, state: AppState, log: LogPanel) -> None:
    extt = _file_dialog_ext_tags(prefix)

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_pick_input_cb,
        tag=t(prefix, "fd_input"),
        user_data={"prefix": prefix, "state": state, "log": log},
        width=760,
        height=460,
    ):
        dpg.add_file_extension(".*", tag=extt["all"])
        dpg.add_file_extension(".json", tag=extt["json"])
        dpg.add_file_extension(".txt", tag=extt["txt"])
        dpg.add_file_extension(".csv", tag=extt["csv"])
        dpg.add_file_extension(".yaml", tag=extt["yaml"])
        dpg.add_file_extension(".yml", tag=extt["yml"])

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_pick_interp_out_cb,
        tag=t(prefix, "fd_interp_out"),
        user_data={"prefix": prefix, "state": state, "log": log},
        width=760,
        height=460,
    ):
        dpg.add_file_extension(".json", tag=t(prefix, "fdext_interp_json"))

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=_pick_solve_out_cb,
        tag=t(prefix, "fd_solve_out"),
        user_data={"prefix": prefix, "state": state, "log": log},
        width=760,
        height=460,
    ):
        dpg.add_file_extension(".*", tag=t(prefix, "fdext_solve_all"))
        dpg.add_file_extension(".json", tag=t(prefix, "fdext_solve_json"))
        dpg.add_file_extension(".out", tag=t(prefix, "fdext_solve_out"))


def _inputs_block(prefix: str, state: AppState, log: LogPanel, uiq: SimpleQueue[UiTask], *, default_mode: str) -> None:
    dpg.add_text("Equation source")
    with dpg.group(horizontal=True):
        dpg.add_radio_button(items=["Editor", "File"], default_value=("Editor" if default_mode == "editor" else "File"), tag=t(prefix, "input_mode"))
        dpg.add_spacer(width=10)
        dpg.add_button(label="📂 Browse", callback=lambda: dpg.show_item(t(prefix, "fd_input")))
        dpg.add_button(label="🗂 Open folder", callback=lambda: _open_input_folder(prefix, state))

    with dpg.group(horizontal=True):
        dpg.add_text("Input path:")
        dpg.add_input_text(tag=t(prefix, "input_path"), width=-1)

    dpg.add_spacer(height=6)
    dpg.add_text("Equation editor ✨")
    dpg.add_input_text(
        tag=t(prefix, "editor"),
        multiline=True,
        height=260,
        width=-1,
        show=(default_mode == "editor"),
        hint="Type one equation per line.\nExample:\n  sin(x) + y = 1\n  cos(x) - y = 0\n",
    )
    # Monospace only for editor (per UX request).
    if state.mono_font is not None:
        try:
            dpg.bind_item_font(t(prefix, "editor"), state.mono_font)
        except Exception:
            pass

    with dpg.collapsing_header(label="Parse / Interpret (TXT → JSON)", default_open=True):
        with dpg.group(horizontal=True):
            dpg.add_text("Output JSON:")
            dpg.add_input_text(tag=t(prefix, "interp_out"), width=-1, hint="(auto)")
            dpg.add_button(label="Pick…", callback=lambda: dpg.show_item(t(prefix, "fd_interp_out")))
        with dpg.group(horizontal=True):
            dpg.add_button(label="🧠 Interpret", callback=lambda: _interpret(prefix, state, log, uiq))
            if prefix == "eqn":
                dpg.add_button(
                    label="🚀 Interpret + Solve",
                    callback=lambda: _interpret(prefix, state, log, uiq, after_ok=lambda: _solve(prefix, state, log, uiq)),
                )
            dpg.add_spacer(width=10)
            dpg.add_text("Preview:")
            dpg.add_input_text(tag=t(prefix, "interp_preview"), readonly=True, width=-1)

    with dpg.collapsing_header(label="Solve options", default_open=True):
        with dpg.group(horizontal=True):
            dpg.add_text("Mode:")
            dpg.add_combo(items=["run", "props", "eqn"], default_value=("run" if prefix == "eqn" else "props"), width=100, tag=t(prefix, "solve_kind"))
            dpg.add_spacer(width=10)
            dpg.add_text("Input:")
            dpg.add_input_text(tag=t(prefix, "solve_in"), width=-1)

        with dpg.group(horizontal=True):
            dpg.add_text("Output:")
            dpg.add_input_text(tag=t(prefix, "solve_out"), width=-1, hint="(optional)")
            dpg.add_button(label="Pick…", callback=lambda: dpg.show_item(t(prefix, "fd_solve_out")))
            dpg.add_button(label="🗂 Output folder", callback=lambda: _open_output_folder(state))

        with dpg.group(horizontal=True):
            dpg.add_button(label="▶ Solve", callback=lambda: _solve(prefix, state, log, uiq))
            dpg.add_spacer(width=10)
            dpg.add_text("Preview:")
            dpg.add_input_text(tag=t(prefix, "solve_preview"), readonly=True, width=-1)

    with dpg.collapsing_header(label="Advanced solver overrides", default_open=False):
        dpg.add_text("Only applied when set (leave '(spec)' to use problem defaults).")

        with dpg.group(horizontal=True):
            dpg.add_text("Backend:")
            dpg.add_combo(items=["(spec)", "auto", "scipy", "gekko"], default_value="(spec)", width=120, tag=t(prefix, "ov_backend"))
            dpg.add_spacer(width=10)
            dpg.add_text("Method:")
            dpg.add_combo(items=["(spec)", "hybr", "lm", "broyden1", "krylov"], default_value="(spec)", width=120, tag=t(prefix, "ov_method"))
            dpg.add_spacer(width=10)
            dpg.add_text("Units:")
            dpg.add_combo(items=["(spec)", "use-units", "no-units"], default_value="(spec)", width=130, tag=t(prefix, "ov_units"))
            dpg.add_spacer(width=10)
            dpg.add_checkbox(label="dry-run", tag=t(prefix, "ov_dry_run"), default_value=False)

        with dpg.group(horizontal=True):
            dpg.add_text("tol:")
            dpg.add_input_text(tag=t(prefix, "ov_tol"), width=140, hint="(blank)")
            dpg.add_spacer(width=10)
            dpg.add_text("max_iter:")
            dpg.add_input_text(tag=t(prefix, "ov_max_iter"), width=140, hint="(blank)")
            dpg.add_spacer(width=10)
            dpg.add_text("max_restarts:")
            dpg.add_input_text(tag=t(prefix, "ov_max_restarts"), width=140, hint="(blank)")

    def _refresh_all() -> None:
        mode = str(dpg.get_value(t(prefix, "input_mode")) or "Editor").strip().lower()
        dpg.configure_item(t(prefix, "editor"), show=mode.startswith("editor"))
        _refresh_previews(prefix, state)

    def _refresh_cb(*_args) -> None:
        _refresh_all()

    for tag in (
        t(prefix, "input_mode"),
        t(prefix, "input_path"),
        t(prefix, "interp_out"),
        t(prefix, "solve_kind"),
        t(prefix, "solve_in"),
        t(prefix, "solve_out"),
        t(prefix, "ov_backend"),
        t(prefix, "ov_method"),
        t(prefix, "ov_units"),
        t(prefix, "ov_dry_run"),
        t(prefix, "ov_tol"),
        t(prefix, "ov_max_iter"),
        t(prefix, "ov_max_restarts"),
    ):
        if dpg.does_item_exist(tag):
            try:
                dpg.set_item_callback(tag, _refresh_cb)
            except Exception:
                pass

    _refresh_all()


def _outputs_block(prefix: str, state: AppState, log: LogPanel) -> None:
    with dpg.group(horizontal=True):
        dpg.add_text("Output")
        dpg.add_spacer(width=10)
        dpg.add_button(label="📄 Open last output", callback=lambda: _open_last_output(prefix, state))

    with dpg.tab_bar():
        with dpg.tab(label="Results"):
            dpg.add_input_text(tag=t(prefix, "results"), multiline=True, readonly=True, height=-1, width=-1)
            if state.mono_font is not None:
                try:
                    dpg.bind_item_font(t(prefix, "results"), state.mono_font)
                except Exception:
                    pass
        with dpg.tab(label="Parsed JSON"):
            dpg.add_input_text(tag=t(prefix, "parsed_json"), multiline=True, readonly=True, height=-1, width=-1)
            if state.mono_font is not None:
                try:
                    dpg.bind_item_font(t(prefix, "parsed_json"), state.mono_font)
                except Exception:
                    pass
        with dpg.tab(label="Logs"):
            with dpg.child_window(height=-1, width=-1, border=False):
                log.build(height=-1)
            dpg.add_text("Tip: set log level to DEBUG when troubleshooting.")

def _build_equations_tab(state: AppState, log: LogPanel, uiq: SimpleQueue[UiTask]) -> None:
    prefix = "eqn"
    with dpg.group(horizontal=True):
        with dpg.child_window(width=LEFT_PANEL_WIDTH, height=-1, border=False):
            _inputs_block(prefix, state, log, uiq, default_mode="editor")
            with dpg.collapsing_header(label="Guess editor (equations only)", default_open=False):
                with dpg.group(tag=t(prefix, "guess_group")):
                    dpg.add_text("Load/interpret an equations JSON to edit guesses.")
        with dpg.child_window(width=-1, height=-1, border=False):
            _outputs_block(prefix, state, log)


def _build_props_tab(state: AppState, log: LogPanel, uiq: SimpleQueue[UiTask]) -> None:
    prefix = "props"
    with dpg.group(horizontal=True):
        with dpg.child_window(width=LEFT_PANEL_WIDTH, height=-1, border=False):
            dpg.add_text("Thermo Props 🧊🔥")
            dpg.add_text("Pick a thermo_props JSON/TXT input, then Solve.")
            dpg.add_spacer(height=6)
            _inputs_block(prefix, state, log, uiq, default_mode="file")
        with dpg.child_window(width=-1, height=-1, border=False):
            _outputs_block(prefix, state, log)


def _build_home_tab() -> None:
    dpg.add_text("Welcome to TDpy ✨")
    dpg.add_text("Thermodynamics is fun. Numerical thermodynamics is even funnier 😄")
    dpg.add_spacer(height=10)

    with dpg.group(horizontal=True):
        dpg.add_button(label="🧪 Thermo Props", width=260, height=90, callback=lambda: dpg.set_value(TABBAR_TAG, TAB_PROPS))
        dpg.add_spacer(width=14)
        dpg.add_button(label="🧮 Equations", width=260, height=90, callback=lambda: dpg.set_value(TABBAR_TAG, TAB_EQN))

    dpg.add_spacer(height=10)
    dpg.add_text("Tip: Start with Equations → type a system → Interpret + Solve 🚀")


# ------------------------------ main ------------------------------

def main() -> int:
    repo = find_repo_root()
    state = AppState(repo_root=repo, in_root=in_dir(repo), out_root=out_dir(repo))
    ensure_dir(state.in_root)
    ensure_dir(state.out_root)

    log_eqn = LogPanel(tag_level="##eqn_log_level", tag_box="##eqn_log_box", tag_status="##eqn_log_status", tag_clear_btn="##eqn_log_clear")
    log_props = LogPanel(tag_level="##props_log_level", tag_box="##props_log_box", tag_status="##props_log_status", tag_clear_btn="##props_log_clear")

    uiq: SimpleQueue[UiTask] = SimpleQueue()

    dpg.create_context()
    # Fonts: per-theme UI fonts + mono for editors/outputs (best-effort).
    _load_fonts(state, ui_point_size=14, mono_point_size=13)

    dpg.create_viewport(title="TDpy GUI (Dear PyGui)", width=1320, height=860)

    theme_light = _make_theme_light()
    theme_dark = _make_theme_dark_soft()
    theme_labview = _make_theme_labview()
    theme_macos = _make_theme_macos()

    themes: dict[str, ThemeSpec] = {
        "light": ThemeSpec(label="Light", key="light", theme_id=theme_light, font_id=state.ui_font_default),
        "dark": ThemeSpec(label="Dark", key="dark", theme_id=theme_dark, font_id=state.ui_font_default),
        "labview": ThemeSpec(label="LabVIEW", key="labview", theme_id=theme_labview, font_id=state.ui_font_labview),
        "macos": ThemeSpec(label="MacOS", key="macos", theme_id=theme_macos, font_id=state.ui_font_macos),
    }

    _apply_theme("Light", themes)
    _set_ui_scale(DEFAULT_UI_SCALE)

    _build_file_dialogs("eqn", state, log_eqn)
    _build_file_dialogs("props", state, log_props)

    with dpg.window(label="TDpy GUI", width=-1, height=-1):
        with dpg.group(horizontal=True):
            dpg.add_text("TDpy • Thermodynamics Playground ✨")
            dpg.add_spacer(width=18)
            dpg.add_text("Theme:")
            dpg.add_combo(items=["Light", "Dark", "LabVIEW", "MacOS"], default_value="Light", width=110, tag=THEME_TOGGLE_TAG,
                          callback=lambda s, a, u: _apply_theme(a, themes))
            dpg.add_spacer(width=14)
            dpg.add_text("UI scale:")
            dpg.add_slider_float(tag=SCALE_SLIDER_TAG, default_value=DEFAULT_UI_SCALE, min_value=0.85, max_value=1.60, width=200,
                                 callback=lambda s, a, u: _set_ui_scale(a))

        with dpg.collapsing_header(label="Paths (click to expand)", default_open=False):
            dpg.add_text(f"root: {state.repo_root}")
            dpg.add_text(f"in:   {state.in_root}")
            dpg.add_text(f"out:  {state.out_root}")

        dpg.add_separator()

        with dpg.tab_bar(tag=TABBAR_TAG):
            with dpg.tab(label="Home", tag=TAB_HOME):
                _build_home_tab()
            with dpg.tab(label="Thermo Props", tag=TAB_PROPS):
                _build_props_tab(state, log_props, uiq)
            with dpg.tab(label="Equations", tag=TAB_EQN):
                _build_equations_tab(state, log_eqn, uiq)

    def _on_frame() -> None:
        log_eqn.drain(max_lines=200)
        log_props.drain(max_lines=200)
        drain_tasks(uiq, max_tasks=50)

    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        _on_frame()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
