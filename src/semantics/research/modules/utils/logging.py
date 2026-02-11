"""Logging utilities for research modules.

Re-exports shared UI functions from the platform-level ``ui`` module and
provides research-specific helpers (gray debug output, external-logging config).
"""

from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import contextmanager
from typing import Generator

from colorama import Fore, Style, init as colorama_init

colorama_init()

GRAY = Fore.LIGHTBLACK_EX
RESET = Style.RESET_ALL


# ---------------------------------------------------------------------------
# Re-export shared UI — every consumer can import from here as before
# ---------------------------------------------------------------------------
from ui import (  # noqa: F401 — re-exported
    debug_print,
    format_duration,
    get_all_timings,
    get_resource_status,
    info_print,
    install_abort_handler,
    is_debug,
    is_plain,
    is_spinner_active,
    print_header,
    print_summary_table,
    register_planned_modules,
    reset_timings,
    restore_abort_handler,
    run_module,
    set_debug,
    set_input_subtitle,
    set_plain,
    skip_module,
    start_pipeline,
    start_resource_monitor,
    stop_pipeline,
    stop_resource_monitor,
    update_sub_progress,
)


# ---------------------------------------------------------------------------
# Research-specific: gray debug output (simple version)
# ---------------------------------------------------------------------------

@contextmanager
def gray_debug_output(debug: bool) -> Generator[None, None, None]:
    """Context manager that grays out stdout/stderr when debug is enabled.

    When debug is False, fully suppresses output (including fd-level)
    to keep the CLI UI clean from third-party library noise.
    """
    original_stdout, original_stderr = sys.stdout, sys.stderr

    if debug:
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

        try:
            sys.stdout = _GrayWriter(original_stdout)
            sys.stderr = _GrayWriter(original_stderr)
            yield
        finally:
            if hasattr(sys.stdout, "flush"):
                sys.stdout.flush()
            if hasattr(sys.stderr, "flush"):
                sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        return

    # Non-debug: fully suppress all output including fd-level
    buffer_out = io.StringIO()
    buffer_err = io.StringIO()
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)

    devnull_fd = saved_stdout_fd = saved_stderr_fd = None
    if os.name == "posix":
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            saved_stdout_fd = os.dup(1)
            saved_stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
        except Exception:
            if devnull_fd is not None:
                os.close(devnull_fd)
            devnull_fd = saved_stdout_fd = saved_stderr_fd = None

    sys.stdout = buffer_out
    sys.stderr = buffer_err

    try:
        yield
    except Exception:
        out = buffer_out.getvalue()
        err = buffer_err.getvalue()
        if out:
            original_stdout.write(out)
        if err:
            original_stderr.write(err)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logging.disable(previous_disable_level)
        if os.name == "posix":
            try:
                if saved_stdout_fd is not None:
                    os.dup2(saved_stdout_fd, 1)
                    os.close(saved_stdout_fd)
                if saved_stderr_fd is not None:
                    os.dup2(saved_stderr_fd, 2)
                    os.close(saved_stderr_fd)
                if devnull_fd is not None:
                    os.close(devnull_fd)
            except Exception:
                pass


def configure_external_logging(debug: bool) -> None:
    """Tune third-party logging verbosity for the research CLI."""
    target_level = logging.DEBUG if debug else logging.ERROR
    for name in ("sentence_transformers", "transformers", "transformers_modules",
                 "sentence_transformers.SentenceTransformer", "huggingface_hub",
                 "nltk"):
        lg = logging.getLogger(name)
        if lg:
            lg.setLevel(target_level)
