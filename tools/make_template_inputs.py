from __future__ import annotations

import argparse
from pathlib import Path


NOZZLE_CASE_YAML = """\
# Demo nozzle case (ideal-gas, quasi-1D, isentropic)
problem_type: nozzle_ideal
T0_K: 298.15
P0_Pa: 500000
k: 1.4
R: 287.058
geometry_csv: nozzle_geometry.csv
branch_after_throat: sup
"""

NOZZLE_CASE_TXT = """\
! EES-ish key=value input (alternative format)
problem_type = nozzle_ideal
T0_K = 298.15
P0_Pa = 500000
k = 1.4
R = 287.058
geometry_csv = nozzle_geometry.csv
branch_after_throat = sup
"""

NOZZLE_GEOM_CSV = """\
x_mm,D_mm
0,8.0
5,7.2
10,6.8
15,6.4
20,6.132
25,6.4
30,6.9
35,7.2
40,7.5
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.make_template_inputs",
        description="Write bundled demo inputs (YAML/TXT/CSV) into a directory.",
    )
    p.add_argument("--dir", dest="dir", default=None, help="Directory to write templates into")
    # Back-compat alias that people naturally try:
    p.add_argument("--outdir", dest="outdir", default=None, help="Alias for --dir")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = args.dir or args.outdir
    if out_dir is None:
        # default: in/demo relative to current working directory
        out = Path("in/demo")
    else:
        out = Path(out_dir)

    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    (out / "nozzle_case.yaml").write_text(NOZZLE_CASE_YAML, encoding="utf-8")
    (out / "nozzle_case.txt").write_text(NOZZLE_CASE_TXT, encoding="utf-8")
    (out / "nozzle_geometry.csv").write_text(NOZZLE_GEOM_CSV, encoding="utf-8")
    (out / "README.txt").write_text("Demo inputs for tdpy.\n", encoding="utf-8")

    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
