from __future__ import annotations

import subprocess


INSTALLED_CLI_COMMANDS = [
    "sentinel-crop-generate",
    "sentinel-crop-convert-pastis",
    "sentinel-crop-convert-dacia5",
    "sentinel-crop-train-unet",
    "sentinel-crop-train-rf",
    "sentinel-crop-evaluate",
    "sentinel-crop-predict",
]


def test_installed_cli_commands_show_help():
    for command in INSTALLED_CLI_COMMANDS:
        result = subprocess.run(
            [command, "--help"],
            text=True,
            capture_output=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"{command} --help failed\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        assert "usage:" in result.stdout.lower()