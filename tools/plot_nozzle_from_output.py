from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover
    go = None


def _load(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Support both wrapped and bare outputs
    if "result" in data and isinstance(data["result"], dict):
        return data["result"]
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.plot_nozzle_from_output",
        description="Plot nozzle output JSON produced by tdpy.",
    )

    # Back-compat flags:
    p.add_argument("--in", dest="in_path", default=None, help="Input JSON path")
    p.add_argument("--out", dest="out_html", default=None, help="Output HTML path")

    # Nicer UX:
    p.add_argument("input", nargs="?", help="Input JSON (positional alternative)")
    p.add_argument("--out_html", dest="out_html2", default=None, help="Output HTML (preferred flag)")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if go is None:
        raise SystemExit("plotly is not installed; cannot generate HTML plots.")

    in_arg = args.in_path or args.input
    out_arg = args.out_html2 or args.out_html

    if not in_arg or not out_arg:
        raise SystemExit("Provide input and output: either --in/--out or <input> --out_html <file>")

    in_path = Path(in_arg).expanduser().resolve()
    out_html = Path(out_arg).expanduser().resolve()

    payload = _load(in_path)
    x = payload["x_mm"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=payload["M"], mode="lines+markers", name="Mach"))
    fig.add_trace(go.Scatter(x=x, y=payload["P_Pa"], mode="lines+markers", name="Pressure (Pa)", yaxis="y2"))
    fig.add_trace(go.Scatter(x=x, y=payload["T_K"], mode="lines+markers", name="Temperature (K)", yaxis="y3"))

    fig.update_layout(
        title="Nozzle — Mach / Pressure / Temperature vs x",
        xaxis=dict(title="x (mm)"),
        yaxis=dict(title="Mach"),
        yaxis2=dict(title="P (Pa)", overlaying="y", side="right"),
        yaxis3=dict(
            title="T (K)",
            anchor="free",
            overlaying="y",
            side="right",
            position=0.98,
        ),
        legend=dict(orientation="h"),
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(str(out_html))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
