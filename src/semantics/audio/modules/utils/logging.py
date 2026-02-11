"""Logging utilities for audio modules.

Re-exports shared UI functions from the platform-level ``ui`` module and
provides audio-specific helpers (gray debug output, external-logging config).
"""

from __future__ import annotations

import io
import logging
import os
import sys
import warnings
from contextlib import contextmanager

from colorama import Fore, Style, init

init()

GRAY = Fore.LIGHTBLACK_EX

# ---------------------------------------------------------------------------
# Re-export shared UI — every consumer can import from here as before
# ---------------------------------------------------------------------------
# NOTE: ui.py lives at /platform/ui.py which is already on sys.path.
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
# Audio-specific: gray debug output
# ---------------------------------------------------------------------------

class _GrayStream:
    def __init__(self, original):
        self._original = original

    def write(self, data):
        if not data:
            return 0
        self._original.write(f"{GRAY}{data}{Style.RESET_ALL}")
        return len(data)

    def flush(self):
        self._original.flush()

    def isatty(self):
        return self._original.isatty()

    def fileno(self):
        return self._original.fileno()


@contextmanager
def gray_debug_output(enabled: bool):
    """Interpose stdout/stderr so dependency chatter becomes debug-only."""
    original_stdout, original_stderr = sys.stdout, sys.stderr
    original_showwarning = warnings.showwarning

    if enabled:
        gray_stdout = _GrayStream(original_stdout)
        gray_stderr = _GrayStream(original_stderr)
        sys.stdout, sys.stderr = gray_stdout, gray_stderr

        handler_streams = []

        def _wrap_logger_handlers(logger):
            if not isinstance(logger, logging.Logger):
                return
            for handler in logger.handlers:
                if hasattr(handler, "setStream") and handler.stream is not None:
                    original_stream = handler.stream
                    if isinstance(original_stream, _GrayStream):
                        continue
                    handler.setStream(_GrayStream(original_stream))
                    handler_streams.append((handler, original_stream))

        _wrap_logger_handlers(logging.root)
        for logger in logging.Logger.manager.loggerDict.values():
            try:
                _wrap_logger_handlers(logger)
            except Exception:
                continue

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            target = file if file is not None else sys.stderr
            if not isinstance(target, _GrayStream):
                target = _GrayStream(target)
            original_showwarning(message, category, filename, lineno, file=target, line=line)

        warnings.showwarning = _showwarning
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr
            for handler, stream in handler_streams:
                try:
                    handler.setStream(stream)
                except Exception:
                    pass
            warnings.showwarning = original_showwarning
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

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        target = file if file is not None else buffer_err
        original_showwarning(message, category, filename, lineno, file=target, line=line)

    warnings.showwarning = _showwarning

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
        warnings.showwarning = original_showwarning
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


# ---------------------------------------------------------------------------
# Audio-specific: external logging configuration
# ---------------------------------------------------------------------------

def configure_external_logging(debug: bool) -> None:
    """Tune third-party logging verbosity to align with the debug flag."""

    os.environ["ORT_LOGGING_LEVEL"] = "INFO" if debug else "FATAL"
    os.environ["NEMO_LOG_LEVEL"] = "INFO" if debug else "ERROR"

    severity = 1 if debug else 4
    with gray_debug_output(debug):
        try:
            import onnxruntime as ort
            if hasattr(ort, "set_default_logger_severity"):
                ort.set_default_logger_severity(severity)
        except Exception:
            pass

        try:
            from transformers import logging as hf_logging
            if debug:
                hf_logging.set_verbosity_info()
            else:
                hf_logging.set_verbosity_error()
        except Exception:
            pass

    target_level = logging.INFO if debug else logging.ERROR
    for name in ("nemo", "nemo.collections", "nemo.utils", "transformers"):
        logger = logging.getLogger(name)
        if logger:
            logger.setLevel(target_level)
