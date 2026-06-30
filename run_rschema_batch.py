"""
Batch runner: ScribbleDB stages 1+2 on RSchema annotation.jsonl.

Produces a JSONL file readable by evaluate_rschema.py --scribble.
Each output line: {"id": "<sample_id>", "schema": <Schema.model_dump()>}

With --trace, also writes per-sample debug dirs under output/rschema_traces/<id>/:
  nl.txt                  -- input NL description
  golden.txt              -- golden schema (tables, PKs, FKs)
  stage1_facts.tsv        -- all facts (id, source, tags, text)
  stage1_enrichment.txt   -- enrichment accept/reject summary
  stage2_shards.txt       -- initial per-shard schemas
  stage2_merged.txt       -- post-initial-merge schema
  stage2_final.txt        -- final schema (human-readable)
  stage2_final.json       -- final schema (model_dump, re-loadable)
  metrics.json            -- per-sample metric scores vs golden
  pipeline.log            -- captured stdout from both stages

Usage:
    python run_rschema_batch.py --start_pos 0 --end_pos 10
    python run_rschema_batch.py --start_pos 0 --end_pos 10 --trace
    python run_rschema_batch.py --start_pos 0 --end_pos 10 --output output/run2.jsonl --trace
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "RSchema" / "annotation.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "output" / "scribble_rschema_annotation.jsonl"
DEFAULT_TRACE_DIR = PROJECT_ROOT / "output" / "rschema_traces"


def _import(mod_path: str) -> Callable[..., Any]:
    mod = importlib.import_module(mod_path)
    return mod.orchestrate  # type: ignore[return-value]


def _load_cases(start: int, end: int, dataset_path: Path) -> List[dict]:
    cases = []
    with dataset_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if end != -1 and i >= end:
                break
            row = json.loads(line)
            cases.append(
                {"idx": i, "id": str(row.get("id", "")), "nl": row.get("question", "")}
            )
    return cases


def _load_golden_schemas(n_samples: int, dataset_path: Path) -> Dict[str, Any]:
    """Returns {id: raw_answer_dict} for the first n_samples from the dataset."""
    golden = {}
    with dataset_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            row = json.loads(line)
            sid = str(row.get("id", ""))
            golden[sid] = row.get("answer", {})
    return golden


def _format_golden(answer: dict) -> str:
    lines = []
    for table_name, info in answer.items():
        pk_list = info.get("Primary key", [])
        pk_str = ", ".join(pk_list)
        attrs = info.get("Attributes", [])
        fk_dict = info.get("Foreign key", {})
        lines.append(f"  {table_name}  pk=[{pk_str}]")
        lines.append(f"    cols: {attrs}")
        if fk_dict:
            for col, ref in fk_dict.items():
                for ref_table, ref_col in ref.items():
                    lines.append(f"    FK: {col} -> {ref_table}.{ref_col}")
    return "\n".join(lines)


def _facts_tsv(facts: List[Any]) -> str:
    rows = ["id\tsource\ttags\tfact"]
    for f in facts:
        source = "external" if getattr(f, "is_external", False) else "nl"
        tags = ", ".join(t.value for t in getattr(f, "tags", []))
        text = getattr(f, "fact", str(f))
        rows.append(f"{f.id}\t{source}\t{tags}\t{text}")
    return "\n".join(rows)


def _schema_summary(schema: Any) -> str:
    if schema is None:
        return "(none)"
    lines = []
    for t in schema.tables:
        fks = [r for r in (schema.relationships or []) if r.referencing_table == t.name]
        fk_str = "  ".join(f"{r.referencing_column}->{r.referred_table}" for r in fks)
        cols = ", ".join(c.name for c in t.columns)
        lines.append(f"  {t.name}  pk={t.pk}  cols=[{cols}]")
        if fk_str:
            lines.append(f"    FKs: {fk_str}")
    return "\n".join(lines) if lines else "(empty)"


def _shards_summary(segments: List[Any]) -> str:
    parts = []
    for i, shard in enumerate(segments):
        parts.append(f"--- Shard {i + 1} ---")
        parts.append(_schema_summary(shard))
    return "\n".join(parts)


def _compute_metrics(pred_schema: Any, golden_answer: dict) -> dict:
    """Compute schema metrics between a predicted Schema and a raw golden answer dict."""
    try:
        from src.evaluation.schema_level.RSchemaAPI import map_rschema_to_pydantic
        from src.evaluation.schema_level.schema_eval import SchemaEvaluator

        gt = map_rschema_to_pydantic(golden_answer)
        evaluator = SchemaEvaluator()
        scores = evaluator.evaluate_schema(pred_schema, gt)
        # Round to 3dp, drop dt_acc (always None here)
        return {
            k: round(v, 3) for k, v in scores.items() if k != "dt_acc" and v is not None
        }
    except Exception as exc:
        return {"error": str(exc)}


async def run_case(
    case: dict,
    model: Optional[str],
    trace_dir: Optional[Path],
    golden_answer: Optional[dict],
) -> Optional[dict]:
    idx, sid, nl = case["idx"], case["id"], case["nl"]
    print(f"\n[{idx}] {sid} -- {nl[:60]!r}")

    log_buf = io.StringIO()

    # Stage 1
    t0 = time.time()
    try:
        s1_fn = _import("src.orchestration.stage1.entry")
        with redirect_stdout(log_buf):
            s1_out, s1_tok = await s1_fn(nl_description=nl, model=model)
        print(
            f"  Stage 1 done in {time.time() - t0:.1f}s ({s1_tok} tok, {len(s1_out.final_facts)} facts)"
        )
    except Exception as e:
        print(f"  Stage 1 FAILED: {e}")
        if trace_dir:
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / "pipeline.log").write_text(
                log_buf.getvalue() + f"\nStage 1 FAILED: {e}\n", encoding="utf-8"
            )
        return None

    # Stage 2
    t1 = time.time()
    try:
        s2_fn = _import("src.orchestration.stage2.entry")
        with redirect_stdout(log_buf):
            result = await s2_fn(
                facts=s1_out.final_facts,
                domain=s1_out.domain,
                analytical_goal=s1_out.analytical_goal,
                model=model,
                nl_query=nl,
            )
        if isinstance(result, tuple):
            s2_out = result[0]
        else:
            s2_out = result
        print(f"  Stage 2 done in {time.time() - t1:.1f}s")
    except Exception as e:
        print(f"  Stage 2 FAILED: {e}")
        if trace_dir:
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / "pipeline.log").write_text(
                log_buf.getvalue() + f"\nStage 2 FAILED: {e}\n", encoding="utf-8"
            )
        return None

    schema = getattr(s2_out, "final_global_schema", None) or getattr(
        s2_out, "merged_schema", None
    )
    if schema is None:
        print("  No schema in Stage 2 output -- skipping")
        return None

    tables = [t.name for t in schema.tables]
    print(f"  Schema: {tables}")

    # Write trace files
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)

        (trace_dir / "pipeline.log").write_text(log_buf.getvalue(), encoding="utf-8")
        (trace_dir / "nl.txt").write_text(nl, encoding="utf-8")

        if golden_answer is not None:
            (trace_dir / "golden.txt").write_text(
                _format_golden(golden_answer), encoding="utf-8"
            )

        # Stage 1 facts
        nl_facts = [
            f for f in s1_out.final_facts if not getattr(f, "is_external", False)
        ]
        ext_facts = [f for f in s1_out.final_facts if getattr(f, "is_external", False)]
        (trace_dir / "stage1_facts.tsv").write_text(
            _facts_tsv(s1_out.final_facts), encoding="utf-8"
        )
        enrich_report = s1_out.enrichment_filter_report
        accepted = getattr(enrich_report, "accepted_facts", [])
        rejected = getattr(enrich_report, "rejected_facts", [])
        enrich_lines = [
            f"NL-extracted facts: {len(nl_facts)}",
            f"External facts accepted: {len(ext_facts)} (of which {len(accepted)} via filter)",
            f"External facts rejected by filter: {len(rejected)}",
            "",
            "--- Accepted external facts ---",
        ]
        for f in ext_facts:
            kind = getattr(f, "external_kind", None)
            reason = getattr(f, "novelty_reason", "")
            enrich_lines.append(f"  [{kind}] {f.fact}")
            if reason:
                enrich_lines.append(f"    reason: {reason}")
        if rejected:
            enrich_lines.append("")
            enrich_lines.append("--- Rejected external facts ---")
            for f in rejected:
                enrich_lines.append(f"  {f.fact}")
        (trace_dir / "stage1_enrichment.txt").write_text(
            "\n".join(enrich_lines), encoding="utf-8"
        )

        # Stage 2 intermediates
        segments = getattr(s2_out, "segments", []) or []
        merged = getattr(s2_out, "merged_schema", None)
        final = schema

        (trace_dir / "stage2_shards.txt").write_text(
            _shards_summary(segments), encoding="utf-8"
        )
        (trace_dir / "stage2_merged.txt").write_text(
            _schema_summary(merged), encoding="utf-8"
        )
        (trace_dir / "stage2_final.txt").write_text(
            _schema_summary(final), encoding="utf-8"
        )
        (trace_dir / "stage2_final.json").write_text(
            json.dumps(final.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Per-sample metrics
        if golden_answer is not None:
            metrics = _compute_metrics(schema, golden_answer)
            (trace_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            # Print key metrics inline
            key = [
                "table_f1",
                "attr_f1",
                "attr_coverage_f1",
                "pk_acc",
                "fk_acc",
                "fk_fd_coverage",
            ]
            score_str = "  ".join(f"{m}={metrics.get(m, 0) * 100:.0f}" for m in key)
            print(f"  Metrics: {score_str}")

        # Diff summary: extra tables vs missing tables
        if golden_answer is not None:
            golden_table_names = {
                k.replace(" ", "_").upper() for k in golden_answer.keys()
            }
            pred_table_names = {t.name for t in schema.tables}
            extra = pred_table_names - golden_table_names
            missing = golden_table_names - pred_table_names
            if extra:
                print(f"  Extra tables (not in golden): {sorted(extra)}")
            if missing:
                print(f"  Missing tables (in golden):   {sorted(missing)}")

    return {"id": sid, "schema": schema.model_dump()}


async def main_async(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trace_root = Path(args.trace_dir) if args.trace else None

    # Load already-processed IDs to support resume
    done_ids: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resuming -- {len(done_ids)} already done")

    dataset_path = Path(args.dataset)
    cases = _load_cases(args.start_pos, args.end_pos, dataset_path)
    end_actual = args.end_pos if args.end_pos != -1 else args.start_pos + len(cases)
    golden_map = _load_golden_schemas(end_actual, dataset_path)
    print(f"Running {len(cases)} cases (positions {args.start_pos}--{args.end_pos})")
    if trace_root:
        print(f"Tracing enabled -> {trace_root}")

    with out_path.open("a", encoding="utf-8") as out_f:
        for case in cases:
            if case["id"] in done_ids:
                print(f"[{case['idx']}] {case['id']} -- already done, skipping")
                continue
            trace_dir = (trace_root / case["id"]) if trace_root else None
            result = await run_case(
                case, args.model, trace_dir, golden_map.get(case["id"])
            )
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"\nDone. Output: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="ScribbleDB batch runner for RSchema")
    p.add_argument("--start_pos", type=int, default=0)
    p.add_argument("--end_pos", type=int, default=6)
    p.add_argument("--model", type=str, default=None)
    p.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "gemini", "openrouter"],
        help="LLM provider override (sets PROVIDER env var)",
    )
    p.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    p.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Path to annotation.jsonl dataset",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Save per-sample trace files (Stage 1 facts, Stage 2 intermediates, metrics)",
    )
    p.add_argument(
        "--trace_dir",
        type=str,
        default=str(DEFAULT_TRACE_DIR),
        help="Root directory for trace output (default: output/rschema_traces/)",
    )
    args = p.parse_args()

    if args.provider:
        os.environ["PROVIDER"] = args.provider

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
