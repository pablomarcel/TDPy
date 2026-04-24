from __future__ import annotations

from pathlib import Path

from in_out import load_text_kv


def test_load_text_kv_basic(tmp_path: Path) -> None:
    p = tmp_path / "case.txt"
    p.write_text(
        """# comment
problem_type = nozzle_ideal
T0_K = 298.15
P0_Pa = 500000
k = 1.4
flag = true
name: Air
""",
        encoding="utf-8",
    )
    d = load_text_kv(p)
    assert d["problem_type"] == "nozzle_ideal"
    assert abs(d["T0_K"] - 298.15) < 1e-9
    assert d["P0_Pa"] == 500000
    assert abs(d["k"] - 1.4) < 1e-12
    assert d["flag"] is True
    assert d["name"] == "Air"
