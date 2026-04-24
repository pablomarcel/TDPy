#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
gui_log_dpg

A friendly logging panel for Dear PyGui GUIs.

Design goals:
- standalone helper module
- no relative-import dependence
- queue-based ingestion from worker threads
- GUI-thread draining
- level filtering
- clear button + status line
- optional monospace binding for the log box
"""

from dataclasses import dataclass, field
from datetime import datetime
from queue import SimpleQueue
from typing import Optional

import re

try:
    import dearpygui.dearpygui as dpg
except Exception:  # pragma: no cover
    dpg = None  # type: ignore[assignment]


_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


@dataclass
class LogPanel:
    tag_level: str = "##log_level"
    tag_box: str = "##log_box"
    tag_status: str = "##log_status"
    tag_clear_btn: str = "##log_clear_btn"
    default_level: str = "INFO"
    max_chars: int = 250_000

    _queue: SimpleQueue[str] = field(default_factory=SimpleQueue, init=False)
    _level_value: str = field(default="INFO", init=False)
    _text: str = field(default="", init=False)

    def build(
        self,
        parent: object | None = None,
        *,
        height: int = 240,
        mono_font: int | None = None,
    ) -> None:
        if dpg is None:  # pragma: no cover
            raise RuntimeError("Dear PyGui is not installed.")

        group_kwargs = {"parent": parent} if parent is not None else {}

        with dpg.group(**group_kwargs):
            with dpg.group(horizontal=True):
                dpg.add_text("Log:")
                dpg.add_combo(
                    items=_LEVELS,
                    default_value=self.default_level,
                    width=110,
                    tag=self.tag_level,
                    callback=self._combo_callback,
                )
                dpg.add_spacer(width=8)
                dpg.add_button(
                    label="🧹 Clear",
                    tag=self.tag_clear_btn,
                    callback=self._clear_callback,
                )
                dpg.add_spacer(width=12)
                dpg.add_text("", tag=self.tag_status)

            dpg.add_input_text(
                tag=self.tag_box,
                multiline=True,
                readonly=True,
                width=-1,
                height=height,
            )

        self._level_value = str(self.default_level).upper().strip()
        if self._level_value not in _LEVELS:
            self._level_value = "INFO"

        dpg.set_value(self.tag_box, "")

        if mono_font is not None and dpg.does_item_exist(self.tag_box):
            try:
                dpg.bind_item_font(self.tag_box, mono_font)
            except Exception:
                pass

    # ------------------------------ public actions ------------------------------

    def clear(self) -> None:
        self._text = ""
        if dpg is not None and dpg.does_item_exist(self.tag_box):
            dpg.set_value(self.tag_box, "")

    def debug(self, msg: str) -> None:
        self._enqueue("DEBUG", msg)

    def info(self, msg: str) -> None:
        self._enqueue("INFO", msg)

    def warn(self, msg: str) -> None:
        self._enqueue("WARN", msg)

    def error(self, msg: str) -> None:
        self._enqueue("ERROR", msg)

    def set_status(self, msg: str) -> None:
        if dpg is None:  # pragma: no cover
            return
        if dpg.does_item_exist(self.tag_status):
            dpg.set_value(self.tag_status, str(msg))

    # ------------------------------ callbacks ------------------------------

    def _combo_callback(self, sender, app_data, user_data=None) -> None:
        self._on_level_changed(app_data)

    def _clear_callback(self, sender=None, app_data=None, user_data=None) -> None:
        self.clear()

    def _on_level_changed(self, value: str) -> None:
        level = str(value).upper().strip()
        self._level_value = level if level in _LEVELS else "INFO"

    # ------------------------------ internals ------------------------------

    def _enqueue(self, level: str, msg: str) -> None:
        level = str(level).upper().strip()
        if level not in _LEVELS:
            level = "INFO"
        self._queue.put(f"[{_ts()}] {level}: {msg}")

    def _level_allows(self, level: str) -> bool:
        try:
            idx = _LEVELS.index(level)
            cur = _LEVELS.index(self._level_value)
            return idx >= cur
        except Exception:
            return True

    def drain(self, *, max_lines: int = 200) -> None:
        if dpg is None:  # pragma: no cover
            return

        added = 0
        while added < max_lines:
            try:
                line = self._queue.get_nowait()
            except Exception:
                break

            m = re.match(r"^\[\d\d:\d\d:\d\d\]\s+([A-Z]+):", line)
            lvl = m.group(1) if m else "INFO"
            if self._level_allows(lvl):
                self._text += line + "\n"
            added += 1

        if added == 0:
            return

        if len(self._text) > self.max_chars:
            self._text = self._text[-self.max_chars:]

        if dpg.does_item_exist(self.tag_box):
            dpg.set_value(self.tag_box, self._text)


__all__ = ["LogPanel"]
