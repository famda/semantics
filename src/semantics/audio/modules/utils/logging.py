import io
import logging
import os
import sys
import warnings
from contextlib import contextmanager

from colorama import Fore, Style, init

init()

GRAY = Fore.LIGHTBLACK_EX


def debug_print(message: str, *, debug: bool) -> None:
    """Emit a debug message in gray when debugging is enabled."""
    if debug:
        print(f"{GRAY}{message}{Style.RESET_ALL}")


class _GrayStream:
    def __init__(self, original):
        self._original = original

    def write(self, data):  # pragma: no cover - passthrough wrapper
        if not data:
            return 0
        self._original.write(f"{GRAY}{data}{Style.RESET_ALL}")
        return len(data)

    def flush(self):  # pragma: no cover - passthrough wrapper
        self._original.flush()

    def isatty(self):  # pragma: no cover - passthrough wrapper
        return self._original.isatty()

    def fileno(self):  # pragma: no cover - passthrough wrapper
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
        for logger in logging.Logger.manager.loggerDict.values():  # pragma: no cover - defensive iteration
            try:
                _wrap_logger_handlers(logger)
            except Exception:
                continue

        def _showwarning(message, category, filename, lineno, file=None, line=None):  # pragma: no cover - wrapper
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

    buffer_out = io.StringIO()
    buffer_err = io.StringIO()
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)

    devnull_fd = saved_stdout_fd = saved_stderr_fd = None
    if os.name == "posix":  # pragma: no cover - fd redirection only on POSIX
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

    def _showwarning(message, category, filename, lineno, file=None, line=None):  # pragma: no cover - wrapper
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
        if os.name == "posix":  # pragma: no cover - fd restoration only on POSIX
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
    """Tune third-party logging verbosity to align with the debug flag."""

    os.environ["ORT_LOGGING_LEVEL"] = "INFO" if debug else "FATAL"
    os.environ["NEMO_LOG_LEVEL"] = "INFO" if debug else "ERROR"

    severity = 1 if debug else 4  # onnxruntime: 0=verbose, 1=info, 2=warning, 3=error, 4=fatal
    with gray_debug_output(debug):
        try:  # pragma: no cover - optional dependency
            import onnxruntime as ort

            if hasattr(ort, "set_default_logger_severity"):
                ort.set_default_logger_severity(severity)
        except Exception:
            pass

        try:  # pragma: no cover - optional dependency
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
