"""core.env: .env parsing, precedence (real env wins), comment handling."""

from __future__ import annotations

import os

from core.env import load_env


def test_load_env_parses_and_respects_existing(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text(
        "# comment line\n"
        "TIINGO_TEST_KEY=abc123\n"
        "QUOTED_TEST_KEY='qval'\n"
        "INLINE_TEST_KEY=xyz  # trailing comment\n"
        "EMPTY_TEST_KEY=\n"
        "ALREADY_SET_KEY=from_file\n"
    )
    monkeypatch.delenv("TIINGO_TEST_KEY", raising=False)
    monkeypatch.setenv("ALREADY_SET_KEY", "from_env")

    load_env(f)
    try:
        assert os.environ["TIINGO_TEST_KEY"] == "abc123"
        assert os.environ["QUOTED_TEST_KEY"] == "qval"
        assert os.environ["INLINE_TEST_KEY"] == "xyz"
        assert "EMPTY_TEST_KEY" not in os.environ  # blank values are not set
        assert os.environ["ALREADY_SET_KEY"] == "from_env"  # real env wins
    finally:
        for k in ("TIINGO_TEST_KEY", "QUOTED_TEST_KEY", "INLINE_TEST_KEY"):
            os.environ.pop(k, None)


def test_load_env_missing_file_is_noop(tmp_path):
    load_env(tmp_path / "nonexistent.env")  # must not raise
