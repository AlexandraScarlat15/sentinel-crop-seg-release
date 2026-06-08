from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PUBLIC_CLI_SCRIPTS = [
    "scripts/generate_dataset.py",
    "scripts/convert_pastis_dataset.py",
    "scripts/convert_dacia5_dataset.py",
    "scripts/train_unet.py",
    "scripts/train_random_forest.py",
    "scripts/evaluate.py",
    "scripts/predict.py",
]


def test_public_cli_scripts_show_help():
    for script in PUBLIC_CLI_SCRIPTS:
        script_path = PROJECT_ROOT / script

        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"{script} --help failed\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        output = result.stdout.lower()
        assert "usage:" in output