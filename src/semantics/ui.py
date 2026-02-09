"""Shared CLI UI utilities for all Semantics workers.

Provides progress bars, spinners, summary tables, resource monitoring,
abort/failure handling, and consistent formatting across audio, video,
and research CLIs.
"""

from __future__ import annotations

import io
import os
import signal
import sys
import threading
import time
from typing import Any, Callable, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_plain_mode: bool = False
_debug_mode: bool = False
_spinner_active: bool = False
_rich_console = None

# Timing / status tracking
_all_timings: list[tuple[str, float, str]] = []
_planned_modules: list[str] = []
_aborted: bool = False

# Pipeline live display
_pipeline_live: Optional["PipelineLive"] = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def set_plain(flag: bool) -> None:
    """Enable plain text output (no rich formatting)."""
    global _plain_mode
    _plain_mode = flag or os.environ.get("NO_COLOR") is not None


def set_debug(flag: bool) -> None:
    """Track whether debug mode is active (disables spinners)."""
    global _debug_mode
    _debug_mode = flag


def is_plain() -> bool:
    return _plain_mode


def is_debug() -> bool:
    return _debug_mode


def is_spinner_active() -> bool:
    return _spinner_active


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------

def _get_console():
    """Get or create a cached Rich Console pinned to the real terminal."""
    global _rich_console
    if _rich_console is None:
        from rich.console import Console
        _rich_console = Console(file=sys.__stdout__)
    return _rich_console


# ---------------------------------------------------------------------------
# Timing & planning
# ---------------------------------------------------------------------------

def reset_timings() -> None:
    """Clear the accumulated module timing records and planned modules."""
    _all_timings.clear()
    _planned_modules.clear()
    global _aborted
    _aborted = False


def register_planned_modules(modules: list[str]) -> None:
    """Register the full list of module labels that will be executed."""
    _planned_modules.clear()
    _planned_modules.extend(modules)


def skip_module(label: str, reason: str = "") -> None:
    """Record a module as skipped with an optional reason."""
    entry = (label, 0.0, "skip")
    _all_timings.append(entry)
    if _pipeline_live and _pipeline_live._live:
        _pipeline_live.add_result(label, 0.0, "skip", reason=reason)
    elif not _spinner_active:
        msg = f"Skipped: {label}"
        if reason:
            msg += f" ({reason})"
        if _plain_mode:
            print(f"  ⊘ {msg}")
        else:
            _get_console().print(f"  [yellow]⊘[/yellow] [dim]{msg}[/dim]")


def get_all_timings() -> list[tuple[str, float, str]]:
    return list(_all_timings)


# ---------------------------------------------------------------------------
# Abort handling
# ---------------------------------------------------------------------------

_original_sigint = None


def _abort_handler(signum, frame):
    """Handle Ctrl+C gracefully — mark as aborted and re-raise."""
    global _aborted
    _aborted = True
    raise KeyboardInterrupt


def install_abort_handler() -> None:
    """Install a SIGINT handler that sets the aborted flag."""
    global _original_sigint
    _original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _abort_handler)


def restore_abort_handler() -> None:
    """Restore the original SIGINT handler."""
    if _original_sigint is not None:
        signal.signal(signal.SIGINT, _original_sigint)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def info_print(message: str) -> None:
    """Print an info-level message.

    In rich mode the ``INFO:`` prefix is omitted for a cleaner look.
    Messages are suppressed entirely while a spinner is active.
    """
    if _spinner_active:
        return
    if _plain_mode:
        print(f"INFO: {message}")
    else:
        _get_console().print(f"  {message}")


def debug_print(message: str, *, debug: bool) -> None:
    """Emit a debug message in gray when debugging is enabled."""
    if not debug:
        return
    if _plain_mode:
        print(f"DEBUG: {message}")
    else:
        _get_console().print(f"  [dim]{message}[/dim]")


def print_header(cli_name: str, input_path: str = "") -> None:
    """Print a styled CLI header banner.

    In rich mode the input file is shown by PipelineLive, so only the
    rule is printed here.  In plain/debug mode the input is included.
    """
    if _plain_mode:
        print()
        print("=" * 55)
        print(f"  Semantics [{cli_name}]")
        if input_path:
            print(f"  Input: {os.path.basename(input_path)}")
        print("=" * 55)
        print()
    else:
        console = _get_console()
        console.print()
        console.rule(
            f"[bold]Semantics[/bold] [dim]\\[{cli_name}][/dim]",
            style="cyan",
        )
        if _debug_mode and input_path:
            console.print(f"  [dim]Input:[/dim] {os.path.basename(input_path)}")
        console.print()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to human-readable string."""
    secs = int(seconds)
    if secs < 60:
        return f"{seconds:.1f}s"
    if secs < 3600:
        m, s = divmod(secs, 60)
        return f"{m}m {s}s"
    h, remainder = divmod(secs, 3600)
    m, s = divmod(remainder, 60)
    return f"{h}h {m}m {s}s"


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Background thread that periodically samples CPU, RAM, and GPU usage."""

    def __init__(self, interval: float = 1.0):
        self._interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cpu: float = 0.0
        self._ram: float = 0.0
        self._ram_used_gb: float = 0.0
        self._gpu_util: float = 0.0
        self._gpu_mem: float = 0.0
        self._gpu_available: bool = False

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        try:
            import psutil
        except ImportError:
            return

        # Try to initialise GPU monitoring
        nvml_ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_available = True
            nvml_ok = True
        except Exception:
            self._gpu_available = False

        while not self._stop_event.is_set():
            try:
                self._cpu = psutil.cpu_percent(interval=None)
                mem = psutil.virtual_memory()
                self._ram = mem.percent
                self._ram_used_gb = mem.used / (1024 ** 3)
            except Exception:
                pass

            if nvml_ok:
                try:
                    import pynvml
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self._gpu_util = util.gpu
                    self._gpu_mem = (mem_info.used / mem_info.total) * 100 if mem_info.total else 0
                except Exception:
                    pass

            self._stop_event.wait(self._interval)

        if nvml_ok:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def format_status(self) -> str:
        """Return a single-line plain text status."""
        parts = [f"CPU {self._cpu:.0f}%", f"RAM {self._ram_used_gb:.1f}GB ({self._ram:.0f}%)"]
        if self._gpu_available:
            parts.append(f"GPU {self._gpu_util:.0f}%")
            parts.append(f"VRAM {self._gpu_mem:.0f}%")
        return " | ".join(parts)

    def format_rich(self) -> str:
        """Return a Rich-markup status string."""
        cpu_color = "green" if self._cpu < 60 else ("yellow" if self._cpu < 85 else "red")
        ram_color = "green" if self._ram < 60 else ("yellow" if self._ram < 85 else "red")
        parts = [
            f"[{cpu_color}]CPU {self._cpu:.0f}%[/{cpu_color}]",
            f"[{ram_color}]RAM {self._ram_used_gb:.1f}GB ({self._ram:.0f}%)[/{ram_color}]",
        ]
        if self._gpu_available:
            gpu_color = "green" if self._gpu_util < 60 else ("yellow" if self._gpu_util < 85 else "red")
            vram_color = "green" if self._gpu_mem < 60 else ("yellow" if self._gpu_mem < 85 else "red")
            parts.append(f"[{gpu_color}]GPU {self._gpu_util:.0f}%[/{gpu_color}]")
            parts.append(f"[{vram_color}]VRAM {self._gpu_mem:.0f}%[/{vram_color}]")
        return " [dim]|[/dim] ".join(parts)


# Backward-compatible no-ops (resource monitor is now embedded in PipelineLive)

def start_resource_monitor(interval: float = 2.0) -> None:
    """No-op — resource monitoring is managed by PipelineLive."""
    pass


def stop_resource_monitor() -> None:
    """No-op — resource monitoring is managed by PipelineLive."""
    pass


def get_resource_status() -> str:
    """Get the current resource status string (rich or plain)."""
    if _pipeline_live is not None and _pipeline_live._monitor is not None:
        if _plain_mode:
            return _pipeline_live._monitor.format_status()
        return _pipeline_live._monitor.format_rich()
    return ""


# ---------------------------------------------------------------------------
# Pipeline live display
# ---------------------------------------------------------------------------

class PipelineLive:
    """Unified live display: header row, progress bar, module results, active spinner.

    In rich mode a single ``Rich.Live`` renders all components.  The class
    implements ``__rich_console__`` so that ``Rich.Live`` calls
    ``_build_display()`` on every refresh cycle — this keeps the resource
    monitor stats updating fluidly even while a module is executing.

    The console is backed by a duplicated file descriptor so that fd-level
    redirects in ``_run_module_rich`` do not blank the display.

    In plain / debug mode the ``Live`` is not created; the resource monitor
    still runs so ``get_resource_status()`` works.
    """

    def __init__(self, total_modules: int, cli_name: str = "", input_path: str = ""):
        self._total = total_modules
        self._completed = 0
        self._cli_name = cli_name
        self._input_path = input_path
        self._live = None
        self._progress = None
        self._task_id = None
        self._results: list = []  # pre-rendered Text objects
        self._errors: list[tuple[str, float, str]] = []  # (label, elapsed, error_msg)
        self._active_label: str = ""
        self._active_config: dict[str, Any] = {}
        self._active_spinner = None  # reused Spinner instance
        self._sub_current: int = 0
        self._sub_total: int = 0
        self._sub_unit: str = ""
        self._monitor = ResourceMonitor(interval=1.0)
        self._tty_fd: Optional[int] = None
        self._tty_file = None
        self._console = None
        self._stopping: bool = False  # set True during final render

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        self._monitor.start()
        if _plain_mode or _debug_mode or self._total == 0:
            return
        # Dup the real terminal fd so our console survives fd-level redirects
        try:
            self._tty_fd = os.dup(1)
            self._tty_file = os.fdopen(self._tty_fd, "w")
            from rich.console import Console
            self._console = Console(file=self._tty_file)
        except OSError:
            self._console = _get_console()

        from rich.progress import (
            Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn,
        )
        from rich.live import Live

        self._progress = Progress(
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        desc = f"Pipeline [{self._cli_name}]" if self._cli_name else "Pipeline"
        self._task_id = self._progress.add_task(desc, total=self._total)

        # Pass *self* as the renderable so __rich_console__ is called on
        # every refresh cycle, keeping resource stats fluid.
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            try:
                self._active_label = ""
                self._active_config = {}
                self._active_spinner = None
                self._sub_current = self._sub_total = 0
                self._stopping = True
                if self._progress is not None and self._task_id is not None:
                    self._progress.update(self._task_id, completed=self._total)
                self._live.stop()
            except Exception:
                pass
            self._live = None
            self._stopping = False
        self._monitor.stop()
        # Errors are already persisted in the Live display (transient=False)
        # via _results, so no need to re-print them here.
        if self._tty_file is not None:
            try:
                self._tty_file.close()
            except Exception:
                pass

    # -- Rich protocol -------------------------------------------------------

    def __rich_console__(self, console, options):
        """Called by Rich.Live on every refresh to get the current renderable."""
        yield self._build_display()

    # -- display building ----------------------------------------------------

    def _build_display(self):
        from rich.table import Table
        from rich.text import Text
        from rich.console import Group
        from rich.progress import Progress, BarColumn, TextColumn
        from rich.progress_bar import ProgressBar

        parts: list = []

        # Row 1: input file (left) + live resource stats (right)
        if self._input_path:
            header = Table.grid(expand=True)
            header.add_column(ratio=1)
            header.add_column(justify="right")
            left = f"  [dim]Input:[/dim] {os.path.basename(self._input_path)}"
            # Hide resource stats from the final persisted view
            right = "" if self._stopping else self._monitor.format_rich()
            header.add_row(left, right)
            parts.append(header)

        # Blank line between header and progress bar
        if self._input_path and self._progress is not None:
            parts.append(Text(""))

        # Row 2: overall progress bar (full width)
        if self._progress is not None:
            parts.append(self._progress)

        # Blank line after progress bar before module results
        if self._progress is not None and (self._results or self._active_label):
            parts.append(Text(""))

        # Completed / skipped module result lines (pre-rendered Text objects)
        for text_obj in self._results:
            parts.append(text_obj)

        # Active module: spinner + config sub-messages + sub-progress bar
        if self._active_label and not self._stopping:
            if self._active_spinner is not None:
                parts.append(self._active_spinner)
            for k, v in self._active_config.items():
                parts.append(Text.from_markup(f"    [dim]{k}:[/dim] {v}"))
            # Compact sub-progress bar
            if self._sub_total > 0:
                unit_str = f" {self._sub_unit}" if self._sub_unit else ""
                pct = (self._sub_current / self._sub_total * 100) if self._sub_total else 0
                bar = ProgressBar(total=self._sub_total, completed=self._sub_current, width=20)
                sub_row = Table.grid(padding=(0, 1))
                sub_row.add_column()  # indent
                sub_row.add_column()  # bar
                sub_row.add_column()  # label
                sub_row.add_row(
                    Text("   "),
                    bar,
                    Text.from_markup(
                        f"[dim]{self._sub_current}/{self._sub_total}{unit_str} ({pct:.0f}%)[/dim]"
                    ),
                )
                parts.append(sub_row)

        return Group(*parts) if parts else Text("")

    # -- module lifecycle ----------------------------------------------------

    def set_active_module(
        self, label: str, config_values: dict[str, Any] | None = None,
    ) -> None:
        """Mark *label* as the currently running module with optional config."""
        from rich.spinner import Spinner
        self._active_label = label
        self._active_config = config_values or {}
        self._active_spinner = Spinner("dots", text=f"  [bold]{label}[/bold]")
        self._sub_current = self._sub_total = 0
        self._sub_unit = ""

    def clear_active_module(self) -> None:
        self._active_label = ""
        self._active_config = {}
        self._active_spinner = None
        self._sub_current = self._sub_total = 0
        self._sub_unit = ""

    def update_sub_progress(self, current: int, total: int, unit: str = "") -> None:
        """Update the sub-progress counter for the active module."""
        self._sub_current = current
        self._sub_total = total
        if unit:
            self._sub_unit = unit

    def add_result(
        self,
        label: str,
        elapsed: float,
        status: str,
        *,
        reason: str = "",
        error: Any = None,
    ) -> None:
        """Record a finished / skipped module and advance the progress bar."""
        from rich.text import Text

        icon = _status_icon_rich(status)
        if elapsed > 0:
            self._results.append(
                Text.from_markup(f"  {icon} {label} [dim]{format_duration(elapsed)}[/dim]")
            )
        else:
            msg = label
            if reason:
                msg += f" [dim]({reason})[/dim]"
            self._results.append(Text.from_markup(f"  {icon} [dim]{msg}[/dim]"))
        if error:
            self._results.append(Text.from_markup(f"    [red dim]{error}[/red dim]"))
            self._errors.append((label, elapsed, str(error)))
        self._active_label = ""
        self._active_config = {}
        self._active_spinner = None
        self._sub_current = self._sub_total = 0
        self._sub_unit = ""
        self._completed += 1
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=self._completed)

    def advance_only(self) -> None:
        """Advance the counter without adding a result line (debug / plain)."""
        self._completed += 1
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=self._completed)
        elif _plain_mode:
            print(f"  [{self._completed}/{self._total}]")

    # -- internals -----------------------------------------------------------

    def _refresh(self) -> None:
        """No-op — Live auto-refreshes via __rich_console__."""
        pass


def start_pipeline(total_modules: int, cli_name: str = "", input_path: str = "") -> None:
    """Create and start the unified pipeline live display."""
    global _pipeline_live
    _pipeline_live = PipelineLive(total_modules, cli_name, input_path)
    _pipeline_live.start()


def stop_pipeline() -> None:
    """Stop and clean up the pipeline live display."""
    global _pipeline_live
    if _pipeline_live is not None:
        _pipeline_live.stop()
        _pipeline_live = None


def update_sub_progress(current: int, total: int, unit: str = "") -> None:
    """Update a sub-progress counter on the active module.

    In rich mode the progress is rendered inside `PipelineLive` beneath
    the config sub-messages.  In plain / debug mode a periodic line is
    printed (every 10 % or every final item).
    """
    if _pipeline_live is not None:
        _pipeline_live.update_sub_progress(current, total, unit)
    elif _plain_mode and total > 0:
        # Print a progress dot every ~10 % to keep the user informed
        step = max(1, total // 10)
        if current % step == 0 or current == total:
            print(f"    [{current}/{total}]", flush=True)


# ---------------------------------------------------------------------------
# Module execution with progress indication
# ---------------------------------------------------------------------------

def run_module(
    label: str,
    fn: Callable,
    *args: Any,
    config_display: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Tuple[Any, Tuple[str, float, str]]:
    """Execute a module function with progress indication.

    Rich mode shows an animated spinner inside the unified ``PipelineLive``
    and suppresses module stdout/stderr (including fd-level output from
    third-party libraries).  Debug mode leaves all output visible.
    Plain mode prints a simple text indicator.

    The *config_display* dict, when present, is rendered as transient
    sub-messages under the active module spinner.  If omitted it is
    auto-extracted from a ``config`` keyword argument that exposes
    ``model_dump()`` (Pydantic models).
    """
    global _spinner_active

    # Auto-extract config display from the Pydantic config kwarg
    if config_display is None and "config" in kwargs and kwargs["config"] is not None:
        try:
            cfg = kwargs["config"]
            if hasattr(cfg, "model_dump"):
                config_display = cfg.model_dump(exclude_none=True) or None
        except Exception:
            config_display = None

    if _debug_mode:
        return _run_module_debug(label, fn, *args, config_display=config_display, **kwargs)

    if _plain_mode:
        return _run_module_plain(label, fn, *args, config_display=config_display, **kwargs)

    return _run_module_rich(label, fn, *args, config_display=config_display, **kwargs)


def _run_module_debug(
    label: str, fn: Callable, *args: Any,
    config_display: dict[str, Any] | None = None, **kwargs: Any,
) -> Tuple[Any, Tuple[str, float, str]]:
    """Debug mode — all output visible, no spinner."""
    console = _get_console()
    if _plain_mode:
        print(f"\n  > {label}")
    else:
        res = get_resource_status()
        if res:
            console.print(f"  [dim]{res}[/dim]")
        console.print(f"\n  [dim]▸[/dim] [bold]{label}[/bold]")

    # Show config sub-messages
    if config_display:
        for k, v in config_display.items():
            if _plain_mode:
                print(f"    {k}: {v}")
            else:
                console.print(f"    [dim]{k}:[/dim] {v}")

    t0 = time.perf_counter()
    result = None
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        entry = (label, elapsed, "done")
        _all_timings.append(entry)
        if _plain_mode:
            print(f"  done ({format_duration(elapsed)})")
        else:
            console.print(
                f"  [green]✓[/green] {label} [dim]{format_duration(elapsed)}[/dim]"
            )
        if _pipeline_live:
            _pipeline_live.advance_only()
        return result, entry
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t0
        entry = (label, elapsed, "abort")
        _all_timings.append(entry)
        if _pipeline_live:
            _pipeline_live.advance_only()
        raise
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        entry = (label, elapsed, "fail")
        _all_timings.append(entry)
        if _plain_mode:
            print(f"  FAILED ({format_duration(elapsed)})")
            print(f"    Error: {exc}")
        else:
            console.print(
                f"  [red]✗[/red] {label} [dim]{format_duration(elapsed)}[/dim]"
            )
            console.print(f"    [red dim]{exc}[/red dim]")
        if _pipeline_live:
            _pipeline_live.advance_only()
        return None, entry


def _run_module_plain(
    label: str, fn: Callable, *args: Any,
    config_display: dict[str, Any] | None = None, **kwargs: Any,
) -> Tuple[Any, Tuple[str, float, str]]:
    """Plain mode — simple text progress."""
    global _spinner_active
    res = get_resource_status()
    if res:
        print(f"  [{res}]")
    print(f"  > {label}...", end="" if not config_display else "\n", flush=True)
    if config_display:
        for k, v in config_display.items():
            print(f"    {k}: {v}")
    _spinner_active = True
    t0 = time.perf_counter()
    result = None
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _spinner_active = False
        entry = (label, elapsed, "done")
        _all_timings.append(entry)
        if config_display:
            print(f"  done ({format_duration(elapsed)})")
        else:
            print(f" done ({format_duration(elapsed)})")
        if _pipeline_live:
            _pipeline_live.advance_only()
        return result, entry
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t0
        _spinner_active = False
        entry = (label, elapsed, "abort")
        _all_timings.append(entry)
        if _pipeline_live:
            _pipeline_live.advance_only()
        raise
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _spinner_active = False
        entry = (label, elapsed, "fail")
        _all_timings.append(entry)
        if config_display:
            print(f"  FAILED ({format_duration(elapsed)})")
        else:
            print(f" FAILED ({format_duration(elapsed)})")
        print(f"    Error: {exc}")
        if _pipeline_live:
            _pipeline_live.advance_only()
        return None, entry


def _run_module_rich(
    label: str, fn: Callable, *args: Any,
    config_display: dict[str, Any] | None = None, **kwargs: Any,
) -> Tuple[Any, Tuple[str, float, str]]:
    """Rich mode — PipelineLive spinner + fd-level noise suppression."""
    global _spinner_active
    console = _get_console()
    _spinner_active = True

    # Show active module in the unified live display
    if _pipeline_live:
        _pipeline_live.set_active_module(label, config_display)

    # Save Python-level streams
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()

    # Attempt fd-level redirect to suppress third-party noise (tqdm, yt-dlp)
    fd_redirected = False
    saved_fd1 = saved_fd2 = None
    try:
        saved_fd1 = os.dup(1)
        saved_fd2 = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        fd_redirected = True
    except OSError:
        # Clean up any fds that were successfully duped before the failure
        if saved_fd1 is not None:
            os.close(saved_fd1)
            saved_fd1 = None
        if saved_fd2 is not None:
            os.close(saved_fd2)
            saved_fd2 = None

    t0 = time.perf_counter()
    result = None
    exc_caught = None
    try:
        result = fn(*args, **kwargs)
    except KeyboardInterrupt:
        exc_caught = "keyboard"
    except Exception as exc:
        exc_caught = exc
    finally:
        elapsed = time.perf_counter() - t0
        _spinner_active = False

        # Restore fd-level streams
        if fd_redirected:
            os.dup2(saved_fd1, 1)
            os.dup2(saved_fd2, 2)
            os.close(saved_fd1)
            os.close(saved_fd2)

        # Restore Python-level streams
        sys.stdout, sys.stderr = saved_out, saved_err

    if exc_caught == "keyboard":
        entry = (label, elapsed, "abort")
        _all_timings.append(entry)
        if _pipeline_live and _pipeline_live._live:
            _pipeline_live.add_result(label, elapsed, "abort")
        else:
            console.print(
                f"  [yellow]⚠[/yellow] {label}"
                f" [dim]{format_duration(elapsed)}[/dim]"
            )
        raise KeyboardInterrupt
    elif exc_caught is not None:
        entry = (label, elapsed, "fail")
        _all_timings.append(entry)
        if _pipeline_live and _pipeline_live._live:
            _pipeline_live.add_result(label, elapsed, "fail", error=str(exc_caught))
        else:
            console.print(
                f"  [red]✗[/red] {label}"
                f" [dim]{format_duration(elapsed)}[/dim]"
            )
            console.print(f"    [red dim]{exc_caught}[/red dim]")
        return None, entry
    else:
        entry = (label, elapsed, "done")
        _all_timings.append(entry)
        if _pipeline_live and _pipeline_live._live:
            _pipeline_live.add_result(label, elapsed, "done")
        else:
            console.print(
                f"  [green]✓[/green] {label}"
                f" [dim]{format_duration(elapsed)}[/dim]"
            )
        return result, entry


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    timings: "list[tuple[str, float, str]] | None" = None,
    total_elapsed: float = 0,
) -> None:
    """Print module execution summary as a styled table.

    If *timings* is ``None`` the internally accumulated ``_all_timings``
    list is used, which includes entries for failed and skipped modules.
    Planned modules that were never executed are shown as "skip" when the
    pipeline was aborted, otherwise "not_run".
    """
    if timings is None:
        timings = list(_all_timings)

    # Add any planned modules that were not executed (due to abort/failure)
    executed_labels = {t[0] for t in timings}
    for mod in _planned_modules:
        if mod not in executed_labels:
            status = "skip" if _aborted else "not_run"
            timings.append((mod, 0.0, status))

    if not timings:
        return

    if _plain_mode:
        print()
        print("=" * 62)
        print(f"  {'Module':<30} {'Duration':>10} {'Status':>12}")
        print("-" * 62)
        for name, elapsed, status in timings:
            icon = _status_icon_plain(status)
            dur = format_duration(elapsed) if elapsed > 0 else "-"
            print(f"  {name:<30} {dur:>10} {icon:>12}")
        print("-" * 62)
        print(f"  {'Total':<30} {format_duration(total_elapsed):>10}")
        print("=" * 62)
        if _aborted:
            print("  ⚠ Aborted by user")
    else:
        from rich.panel import Panel
        from rich.table import Table

        console = _get_console()
        table = Table(
            show_header=True,
            header_style="bold magenta",
            padding=(0, 1),
            expand=True,
        )
        table.add_column("Module", style="cyan", no_wrap=True)
        table.add_column("Duration", style="green", justify="right")
        table.add_column("Status", justify="center")
        for name, elapsed, status in timings:
            icon = _status_icon_rich(status)
            dur = format_duration(elapsed) if elapsed > 0 else "[dim]-[/dim]"
            table.add_row(name, dur, icon)
        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold]{format_duration(total_elapsed)}[/bold]",
            "",
        )
        console.print()

        subtitle = "[yellow]⚠ Aborted by user[/yellow]" if _aborted else None

        console.print(
            Panel(
                table,
                title="[bold]Execution Summary[/bold]",
                subtitle=subtitle,
                border_style="dim",
                padding=(0, 1),
            )
        )


def _status_icon_rich(status: str) -> str:
    if status == "done":
        return "[green]✓[/green]"
    if status == "fail":
        return "[red]✗[/red]"
    if status == "abort":
        return "[yellow]⚠[/yellow]"
    if status == "skip":
        return "[yellow]⊘[/yellow]"
    if status == "not_run":
        return "[dim]—[/dim]"
    return status


def _status_icon_plain(status: str) -> str:
    if status == "done":
        return "ok"
    if status == "fail":
        return "FAIL"
    if status == "abort":
        return "ABORTED"
    if status == "skip":
        return "SKIPPED"
    if status == "not_run":
        return "NOT RUN"
    return status



