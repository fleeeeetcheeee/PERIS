"""Native macOS notifications via osascript. Best-effort: a notification that
can't be delivered logs a warning and never disturbs the worker."""

from __future__ import annotations

import json
import logging
import subprocess

logger = logging.getLogger(__name__)


def mac_notify(title: str, subtitle: str, message: str) -> bool:
    # ensure_ascii=False: AppleScript string literals have no \uXXXX escapes,
    # so non-ASCII must be passed through raw (json quoting still escapes ").
    def q(s: str) -> str:
        return json.dumps(s, ensure_ascii=False)

    script = (
        f"display notification {q(message)} "
        f"with title {q(title)} subtitle {q(subtitle)} "
        f'sound name "Glass"'
    )
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5, check=True)
        return True
    except Exception as exc:
        logger.warning("notification failed: %s", exc)
        return False
