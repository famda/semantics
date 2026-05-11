"""Document Processing CLI Tool.

A document processing pipeline for structured extraction and analysis
of PDF, DOCX, PPTX, and other document formats.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import rich_click as click

# Setup path for imports
try:
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    platform_root = os.path.dirname(script_dir)

    if platform_root not in sys.path:
        sys.path.insert(0, platform_root)

    from global_helpers import DOCUMENT_FILE_TYPES

except ImportError as e:
    print("\n****** ERROR: Failed to import required modules ******", file=sys.stderr)
    print(f"Reason: {e}", file=sys.stderr)
    sys.exit(1)

except Exception as e:
    print(f"An unexpected error occurred during initial setup: {e}", file=sys.stderr)
    sys.exit(1)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-i", "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input document file (PDF, DOCX, PPTX, TXT, MD)",
)
@click.option(
    "-o", "--output",
    "output_folder",
    required=True,
    type=click.Path(),
    help="Output folder path",
)
@click.option(
    "-s", "--structured",
    is_flag=True,
    default=False,
    help="Extract structured content from the document",
)
@click.option(
    "-im", "--images",
    is_flag=True,
    default=False,
    help="Extract images from the document (PDF only)",
)
@click.option(
    "-t", "--tables",
    is_flag=True,
    default=False,
    help="Extract tables to CSV/HTML files",
)
@click.option(
    "-m", "--markdown",
    is_flag=True,
    default=False,
    help="Convert document to Markdown format",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable verbose debug logging",
)
@click.option(
    "--plain",
    is_flag=True,
    default=False,
    help="Disable rich formatting, use plain text output",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML config file",
)
def main(
    input_file: str,
    output_folder: str,
    structured: bool,
    images: bool,
    tables: bool,
    markdown: bool,
    debug: bool,
    plain: bool,
    config: Optional[str],
) -> None:
    """
    \b
    Semantics CLI [docs] - Unified interface for media intelligence
    -------------------------------------------
    Extract meaning, not just metadata. Composable AI operations designed for developers.
    \b
    Examples:
        semantics-docs -i document.pdf -o ./output -s
        semantics-docs -i document.pdf -o ./output -im -t -m
        semantics-docs -i report.docx -o ./output -s -t -m --debug
        semantics-docs -i paper.pdf -o ./output -s -im -t -m --config config.yml
    """
    from docs.modules.utils.logging import (
        configure_external_logging,
        debug_print,
        info_print,
        print_header,
        print_summary_table,
        reset_timings,
        run_module,
        set_debug,
        set_plain,
        skip_module,
        install_abort_handler,
        restore_abort_handler,
        register_planned_modules,
        start_pipeline,
        stop_pipeline,
    )

    set_plain(plain)
    set_debug(debug)
    configure_external_logging(debug)
    reset_timings()
    _start_time = time.perf_counter()

    # Validate input file type
    ext = os.path.splitext(input_file)[1].lstrip(".").lower()
    if ext not in DOCUMENT_FILE_TYPES:
        raise click.UsageError(
            f"Unsupported file type '.{ext}'. Supported: {', '.join(DOCUMENT_FILE_TYPES)}"
        )

    # Validate that at least one processing flag is set
    if not any([structured, images, tables, markdown]):
        raise click.UsageError(
            "At least one processing flag must be set (e.g., -s, -im, -t, -m)."
        )

    # Load config if provided
    docs_config = None
    if config:
        from config import load_docs_config
        docs_config = load_docs_config(config)
    if docs_config is None:
        from config import DocsConfig
        docs_config = DocsConfig()

    # Setup directories
    output_dir = os.path.abspath(output_folder)
    os.makedirs(output_dir, exist_ok=True)
    print_header("docs", input_file)

    # Build planned module list
    planned: list[str] = []
    if structured:
        planned.append("Structured Extraction")
    if images:
        planned.append("Image Extraction")
    if tables:
        planned.append("Table Extraction")
    if markdown:
        planned.append("Markdown Export")

    register_planned_modules(planned)
    start_pipeline(len(planned), "docs", input_file)
    install_abort_handler()

    try:
        if structured:
            from docs.modules import structured as structured_module

            structured_cfg = docs_config.structured if docs_config else None
            _, _ = run_module(
                "Structured Extraction", structured_module.handle,
                input_file,
                output_folder=output_dir,
                config=structured_cfg,
                debug=debug,
            )

        if images:
            from docs.modules import images as images_module

            images_cfg = docs_config.images if docs_config else None
            _, _ = run_module(
                "Image Extraction", images_module.handle,
                input_file,
                output_folder=output_dir,
                config=images_cfg,
                debug=debug,
            )

        if tables:
            from docs.modules import tables as tables_module

            tables_cfg = docs_config.tables if docs_config else None
            _, _ = run_module(
                "Table Extraction", tables_module.handle,
                input_file,
                output_folder=output_dir,
                config=tables_cfg,
                debug=debug,
            )

        if markdown:
            from docs.modules import markdown as markdown_module

            markdown_cfg = docs_config.markdown if docs_config else None
            _, _ = run_module(
                "Markdown Export", markdown_module.handle,
                input_file,
                output_folder=output_dir,
                config=markdown_cfg,
                debug=debug,
            )

    except KeyboardInterrupt:
        pass  # abort — summary table will show remaining as "not run"
    finally:
        restore_abort_handler()
        stop_pipeline()
        total_elapsed = time.perf_counter() - _start_time
        print_summary_table(total_elapsed=total_elapsed)


if __name__ == "__main__":
    main()
