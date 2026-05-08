from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TYPING_DIR = Path(__file__).resolve().parent / "typing"


def _run_mypy(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mypy", *map(str, paths)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "pipeline_builder_linear_ok.py",
        "pipeline_builder_split_ok.py",
        "pipeline_builder_broadcast_ok.py",
        "pipeline_builder_apply_ok.py",
    ],
)
def test_typing_fixtures_pass_mypy(fixture_name: str) -> None:
    pytest.importorskip("mypy")

    fixture = TYPING_DIR / fixture_name
    result = _run_mypy(fixture)

    assert result.returncode == 0, result.stdout + result.stderr


def test_invalid_typing_fixture_is_checked_by_mypy() -> None:
    pytest.importorskip("mypy")

    fixture = TYPING_DIR / "pipeline_builder_invalid_usage.py"
    result = _run_mypy(fixture)

    assert fixture.exists()
    assert result.returncode != 0
    assert 'Argument 2 to "then"' in result.stdout
    assert 'Argument 2 to "sink"' in result.stdout
