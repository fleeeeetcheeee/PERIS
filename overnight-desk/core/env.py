"""Load overnight-desk/.env into os.environ (dependency-free).

launchd jobs don't inherit shell-profile exports, so API keys live in a .env file
next to pyproject.toml. Real environment variables always win — .env only fills
in unset ones. Called once on `import core`.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path | None = None) -> None:
    f = path or ENV_FILE
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split("#", 1)[0].strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
