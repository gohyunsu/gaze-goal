from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from goalexit.utils import git_commit


def test_git_commit_returns_frozen_sha_only_for_clean_tree() -> None:
    clean = [
        subprocess.CompletedProcess([], 0, "abc123\n", ""),
        subprocess.CompletedProcess([], 0, "", ""),
    ]
    with patch("goalexit.utils.subprocess.run", side_effect=clean):
        assert git_commit(Path(".")) == "abc123"
    dirty = [
        subprocess.CompletedProcess([], 0, "abc123\n", ""),
        subprocess.CompletedProcess([], 0, " M source.py\n", ""),
    ]
    with patch("goalexit.utils.subprocess.run", side_effect=dirty):
        assert git_commit(Path(".")) == "UNCOMMITTED"
