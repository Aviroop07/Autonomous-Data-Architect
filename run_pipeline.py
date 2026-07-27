"""
ScribbleDB -- pipeline driver

Runs Stages 1-3 on one natural-language spec and reports what each produced.

Replaces three near-identical ad-hoc drivers (run_pipeline.py,
run_comparison.py, run_user_s1_s2.py) that each hardcoded their own NL blob,
their own logging setup, and their own schema-printing loop. The specs they
carried now live in dataset/handcrafted/specs/ and are selectable by name.

Usage
-----
  # Named spec from dataset/handcrafted/specs/
  python run_pipeline.py --input hospital

  # Any file
  python run_pipeline.py --input path/to/spec.txt

  # Stages 1 and 2 only
  python run_pipeline.py --input hospital --stages 1,2

  # Pick a provider/model explicitly
  python run_pipeline.py --input hospital --provider deepseek --model deepseek-v4-flash

  # List the built-in specs
  python run_pipeline.py --list-specs

THIS MAKES REAL, BILLABLE LLM CALLS. Every run writes a DEBUG log to
artifacts/logs/ before the first call, per CLAUDE.md's Live LLM Run Discipline.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.util.core.providers import PROVIDERS  # noqa: E402  (needs sys.path above)
from src.util.observability.report import (  # noqa: E402
    banner,
    configure_logging,
    report_chunks,
    report_constraints,
    report_facts,
    report_schema,
    write_json,
)

SPEC_DIR = PROJECT_ROOT / "dataset" / "handcrafted" / "specs"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"

logger = logging.getLogger("run_pipeline")


def available_specs() -> List[str]:
    if not SPEC_DIR.is_dir():
        return []
    return sorted(p.stem for p in SPEC_DIR.glob("*.txt"))


def resolve_spec(value: str) -> str:
    """`--input` is either a named spec in dataset/handcrafted/specs/ or a path."""
    named = SPEC_DIR / f"{value}.txt"
    if named.is_file():
        return named.read_text(encoding="utf-8").strip()
    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    raise SystemExit(
        f"No spec named {value!r} in {SPEC_DIR} and no file at that path.\n"
        f"Available: {', '.join(available_specs()) or '(none)'}"
    )


async def run(args: argparse.Namespace) -> int:
    from src.orchestration.stage1.entry import orchestrate as stage1
    from src.orchestration.stage2.entry import orchestrate as stage2
    from src.orchestration.stage3.entry import orchestrate as stage3

    nl = resolve_spec(args.input)
    stages = {int(s) for s in args.stages.split(",") if s.strip()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out) if args.out else ARTIFACT_ROOT / "runs" / f"pipeline_{stamp}"
    )

    logger.info("Spec: %s (%d chars)", args.input, len(nl))
    logger.info("Stages: %s | model=%s | out=%s", sorted(stages), args.model, out_dir)

    total_tokens = 0
    t_start = time.time()

    # ---- Stage 1 ----------------------------------------------------------
    banner("STAGE 1 -- Fact Extraction")
    t0 = time.time()
    s1_out, s1_tokens = await stage1(nl_description=nl, model=args.model)
    total_tokens += s1_tokens
    logger.info("Stage 1 done in %.1fs | tokens=%d", time.time() - t0, s1_tokens)
    logger.info("Domain: %s", s1_out.domain)
    logger.info("Analytical goal: %s", s1_out.analytical_goal)
    report_facts(s1_out.final_facts)
    report_chunks(s1_out.plan)
    write_json(out_dir / "stage1.json", s1_out, label="Stage 1")

    if 2 not in stages:
        logger.info("Stopping after Stage 1 (--stages %s)", args.stages)
        return total_tokens

    # ---- Stage 2 ----------------------------------------------------------
    banner("STAGE 2 -- Schema Generation")
    t0 = time.time()
    s2_out, s2_tokens, registry = await stage2(
        plan=s1_out.plan,
        facts=s1_out.final_facts,
        domain=s1_out.domain,
        analytical_goal=s1_out.analytical_goal,
        nl_query=nl,
        model=args.model,
        artifact_dir=out_dir if args.dump_artifacts else None,
    )
    total_tokens += s2_tokens
    logger.info("Stage 2 done in %.1fs | tokens=%d", time.time() - t0, s2_tokens)
    logger.info("ER shards: %d", len(s2_out.segments))
    schema = s2_out.final_global_schema
    report_schema(schema, label="Final schema")
    logger.info("Uncovered facts: %s", s2_out.uncovered_fact_ids)
    for table_name, fact_ids in registry.table_to_facts.items():
        logger.info("  provenance %s <- %s", table_name, fact_ids)
    write_json(out_dir / "stage2.json", s2_out, label="Stage 2")

    if 3 not in stages:
        logger.info("Stopping after Stage 2 (--stages %s)", args.stages)
        return total_tokens
    if schema is None:
        logger.error("Stage 2 produced no schema -- cannot run Stage 3.")
        return total_tokens

    # ---- Stage 3 ----------------------------------------------------------
    # No previous driver ran this stage at all.
    banner("STAGE 3 -- Constraint Modelling")
    t0 = time.time()
    s3_out, s3_tokens = await stage3(
        schema=schema,
        facts=s1_out.final_facts,
        model=args.model,
        artifact_dir=out_dir if args.dump_artifacts else None,
    )
    total_tokens += s3_tokens
    logger.info("Stage 3 done in %.1fs | tokens=%d", time.time() - t0, s3_tokens)
    report_constraints(s3_out)
    write_json(out_dir / "stage3.json", s3_out, label="Stage 3")

    logger.info("")
    logger.info("Total: %.1fs, %d tokens", time.time() - t_start, total_tokens)
    return total_tokens


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ScribbleDB -- run Stages 1-3 on one NL spec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        help="Named spec from dataset/handcrafted/specs/, or a path to a .txt file",
    )
    p.add_argument(
        "--stages",
        default="1,2,3",
        help="Comma-separated stages to run, e.g. '1,2' (default: 1,2,3)",
    )
    p.add_argument(
        "--out", default=None, help="Output dir (default: artifacts/runs/pipeline_<ts>)"
    )
    p.add_argument(
        "--model", default=None, help="Model name (default: provider default)"
    )
    p.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default=None,
        help="Provider (sets the PROVIDER env var for this run)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="CONSOLE level. The on-disk log is always DEBUG.",
    )
    p.add_argument(
        "--dump-artifacts",
        action="store_true",
        help="Dump per-phase intermediate state, so a mid-run crash stays inspectable",
    )
    p.add_argument(
        "--list-specs", action="store_true", help="List built-in specs and exit"
    )
    return p


def main() -> None:
    if sys.platform == "win32":
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")

    args = _build_parser().parse_args()

    if args.list_specs:
        specs = available_specs()
        print(f"Specs in {SPEC_DIR}:")
        for name in specs:
            print(f"  {name}")
        if not specs:
            print("  (none)")
        return

    if not args.input:
        raise SystemExit("--input is required (or use --list-specs)")

    if args.provider:
        os.environ["PROVIDER"] = args.provider

    # Before the first provider call, never after -- see report.configure_logging.
    configure_logging(
        ARTIFACT_ROOT / "logs",
        console_level=getattr(logging, args.log_level),
        run_name=f"pipeline_{args.input}",
    )

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
