from __future__ import annotations

import subprocess
import sys


def test_cli_list_inputs_runs() -> None:
    cmd = [sys.executable, "-m", "cli", "list-inputs"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0
    assert "tdpy" in p.stdout.lower()
