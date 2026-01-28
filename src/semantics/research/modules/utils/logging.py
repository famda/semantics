"""Logging utilities for research modules.

Provides consistent debug output and gray-styled terminal output matching
the patterns used in audio and video CLIs.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Generator

from colorama import Fore, Style, init as colorama_init

colorama_init()

GRAY = Fore.LIGHTBLACK_EX
RESET = Style.RESET_ALL


def debug_print(message: str, *, debug: bool) -> None:
    """Print a message only when debug mode is enabled.

    Args:
        message: The message to print.
        debug: If True, print the message; otherwise do nothing.
    """
    if debug:
        print(f"{GRAY}DEBUG: {message}{RESET}")


def info_print(message: str) -> None:
    """Print an info-level message."""
    print(f"INFO: {message}")


@contextmanager
def gray_debug_output(debug: bool) -> Generator[None, None, None]:
    """Context manager that grays out stdout/stderr when debug is enabled.

    Useful for wrapping library calls that produce verbose output.

    Args:
        debug: If True, colorize output in gray; otherwise pass through.
    """
    if not debug:
        yield
        return

    class _GrayWriter:
        def __init__(self, stream):
            self._stream = stream
            self._in_gray = False

        def write(self, text: str) -> int:
            if text:
                if not self._in_gray:
                    self._stream.write(GRAY)
                    self._in_gray = True
                self._stream.write(text)
                if text.endswith("\n"):
                    self._stream.write(RESET)
                    self._in_gray = False
            return len(text)

        def flush(self) -> None:
            if self._in_gray:
                self._stream.write(RESET)
                self._in_gray = False
            self._stream.flush()

        def __getattr__(self, name):
            return getattr(self._stream, name)

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = _GrayWriter(old_stdout)
        sys.stderr = _GrayWriter(old_stderr)
        yield
    finally:
        if hasattr(sys.stdout, "flush"):
            sys.stdout.flush()
        if hasattr(sys.stderr, "flush"):
            sys.stderr.flush()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
