"""Console/file reporting shared by the pipeline drivers.

run_pipeline.py, run_comparison.py and run_user_s1_s2.py each carried their own
copy of logging setup and their own hand-rolled "print the schema" loop, all
three slightly different. This is that code, once.

Logging setup deliberately configures a FILE handler before returning, not as
an afterthought: CLAUDE.md's Live LLM Run Discipline requires a live run to be
logged from the first API call, because a run that produced no on-disk log
cannot be diagnosed once the process ends.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_BANNER_WIDTH = 62


def configure_logging(
    log_dir: Path,
    *,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    run_name: str = "run",
) -> Path:
    """Attach a DEBUG file handler plus a console handler to the root logger
    and return the log file's path.

    Called BEFORE the first provider call, never after. The file always gets
    DEBUG regardless of what the console shows -- the point is that the full
    trace survives on disk even when the terminal is quiet.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name}_{stamp}.log"

    root = logging.getLogger()
    root.setLevel(min(console_level, file_level))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    )
    root.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    root.addHandler(console)

    logger.info(
        "Logging to %s (file=DEBUG, console=%s)",
        log_path,
        logging.getLevelName(console_level),
    )
    return log_path


def banner(title: str) -> None:
    logger.info("=" * _BANNER_WIDTH)
    logger.info("  %s", title)
    logger.info("=" * _BANNER_WIDTH)


def report_facts(facts: Iterable[Any], *, show_segments: bool = True) -> None:
    """One line per extracted fact: id, tags, text, and its source span."""
    facts = list(facts)
    logger.info("Facts: %d", len(facts))
    for f in facts:
        tags = ", ".join(t.value for t in getattr(f, "tags", []) or [])
        segment = ""
        if show_segments and getattr(f, "segment_text", None):
            segment = f'  <- "{f.segment_text[:50]}"'
        logger.info("  [%s] (%s) %s%s", f.id, tags, f.fact, segment)


def report_chunks(plan: Any) -> None:
    chunks = getattr(plan, "chunks", None) or []
    logger.info("Chunks: %d", len(chunks))
    for i, chunk in enumerate(chunks, 1):
        ids = sorted(cf.id for cf in chunk)
        logger.info("  chunk %d (%d facts): %s", i, len(chunk), ids)


def report_schema(schema: Any, *, label: str = "Schema") -> None:
    """Tables with PK and typed columns, then foreign keys."""
    if schema is None:
        logger.warning("%s: none produced", label)
        return
    logger.info("%s: %d tables", label, len(schema.tables))
    for table in schema.tables:
        cols = ", ".join(f"{c.name}:{_type_name(c.data_type)}" for c in table.columns)
        logger.info("  %s  PK=%s  [%s]", table.name, table.primary_key, cols)
    for fk in schema.relationships or []:
        logger.info(
            "  FK: %s.%s -> %s",
            fk.referencing_table,
            fk.referencing_column,
            fk.referred_table,
        )


def report_constraints(stage3_output: Any) -> None:
    """Stage 3's per-category counts plus the analysis report's headline
    numbers -- the probes are Stage 4's actual input, so they matter more than
    the raw constraint count."""
    if stage3_output is None:
        logger.warning("Stage 3: no output")
        return
    logger.info("Constraints: %d total", stage3_output.total_constraints)
    for label, items in (
        ("distributions", stage3_output.distributions),
        ("moment targets", stage3_output.moment_targets),
        ("correlations", stage3_output.correlations),
        ("structural", stage3_output.structural_constraints),
        ("logic", stage3_output.logic_constraints),
        ("derived columns", stage3_output.derived_columns),
        ("state sequences", stage3_output.state_sequences),
    ):
        if items:
            logger.info("  %-16s %d", label, len(items))

    report = stage3_output.analysis_report
    logger.info(
        "DOF: %d square, %d loose probes | conflicts: %d unresolved, "
        "%d dismissed, %d cycles",
        len(report.square_variables),
        len(report.loose_variable_probes),
        len(report.conflicts),
        len(report.dismissed_conflicts),
        len(report.derived_cycle_conflicts),
    )
    for note in report.unsupported:
        logger.warning("  unsupported: %s", note)


def _type_name(data_type: Any) -> str:
    return getattr(data_type, "value", str(data_type))


def write_json(path: Path, model: Any, *, label: Optional[str] = None) -> None:
    """Dump a Pydantic model to disk, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        model.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
    )
    logger.info("Wrote %s%s", path, f" ({label})" if label else "")
