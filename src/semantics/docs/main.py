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
    "-ner", "--named-entities",
    is_flag=True,
    default=False,
    help="Extract named entities (requires -s)",
)
@click.option(
    "-cl", "--classify",
    is_flag=True,
    default=False,
    help="Classify the document type (requires -s)",
)
@click.option(
    "-ov", "--overview",
    is_flag=True,
    default=False,
    help="Generate document overview for RAG/semantic search (requires -s)",
)
@click.option(
    "-f", "--forms",
    is_flag=True,
    default=False,
    help="Extract form fields and key-value pairs",
)
@click.option(
    "-cap", "--captions",
    is_flag=True,
    default=False,
    help="Caption extracted images using BLIP + Qwen3-VL with NER (requires -im)",
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
    named_entities: bool,
    classify: bool,
    overview: bool,
    forms: bool,
    captions: bool,
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
        semantics-docs -i paper.pdf -o ./output -s -ner -cl -ov -f
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
    if not any([structured, images, tables, markdown, named_entities, classify, overview, forms, captions]):
        raise click.UsageError(
            "At least one processing flag must be set (e.g., -s, -im, -t, -m, -ner, -cl, -ov, -f, -cap)."
        )

    # NER and classify require structured extraction
    if named_entities and not structured:
        raise click.UsageError(
            "Named entity recognition (-ner) requires structured extraction (-s)."
        )
    if classify and not structured:
        raise click.UsageError(
            "Classification (-cl) requires structured extraction (-s)."
        )
    if overview and not structured:
        raise click.UsageError(
            "Overview (-ov) requires structured extraction (-s)."
        )
    if captions and not images:
        raise click.UsageError(
            "Image captioning (-cap) requires image extraction (-im)."
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
    if captions:
        planned.append("Image Captioning")
    if tables:
        planned.append("Table Extraction")
    if markdown:
        planned.append("Markdown Export")
    if named_entities:
        planned.append("Named Entity Recognition")
    if overview:
        planned.append("Document Overview")
    if classify:
        planned.append("Document Classification")
    if forms:
        planned.append("Form Extraction")

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

        if captions:
            from docs.modules import captions as captions_module

            captions_cfg = docs_config.captions if docs_config else None
            _, _ = run_module(
                "Image Captioning", captions_module.handle,
                input_file,
                output_folder=output_dir,
                config=captions_cfg,
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

        if named_entities:
            from docs.modules import entities as entities_module

            ner_cfg = docs_config.ner if docs_config else None
            _, _ = run_module(
                "Named Entity Recognition", entities_module.handle,
                input_file,
                output_folder=output_dir,
                config=ner_cfg,
                debug=debug,
            )

        if overview:
            from docs.modules import overview as overview_module

            overview_cfg = docs_config.overview if docs_config else None
            _, _ = run_module(
                "Document Overview", overview_module.handle,
                input_file,
                output_folder=output_dir,
                config=overview_cfg,
                debug=debug,
            )

        if classify:
            from docs.modules import classify as classify_module

            classify_cfg = docs_config.classify if docs_config else None
            _, _ = run_module(
                "Document Classification", classify_module.handle,
                input_file,
                output_folder=output_dir,
                config=classify_cfg,
                debug=debug,
            )

        if forms:
            from docs.modules import forms as forms_module

            forms_cfg = docs_config.forms if docs_config else None
            _, _ = run_module(
                "Form Extraction", forms_module.handle,
                input_file,
                output_folder=output_dir,
                config=forms_cfg,
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

