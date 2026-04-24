from __future__ import annotations

from pathlib import Path

from app import TdpyApp
from apis import RunRequest


def test_nozzle_demo_runs(tmp_path: Path) -> None:
    pkg_demo = Path(__file__).resolve().parents[1] / "in" / "demo"
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    for f in pkg_demo.iterdir():
        (in_dir / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")

    app = TdpyApp(in_dir=in_dir, out_dir=out_dir)
    res = app.run(RunRequest(in_path=Path("nozzle_case.yaml"), make_plots=False))

    assert res.ok is True
    assert res.out_path is not None and res.out_path.exists()
    payload = res.payload
    assert payload["mdot_kgps"] > 0
    assert len(payload["M"]) == len(payload["x_mm"])
    i_throat = payload["meta"]["i_throat"]
    assert abs(payload["M"][i_throat] - 1.0) < 1e-12
