"""
Batch runner: LLM4DBdesign stages on RSchema annotation.jsonl.

Mirrors run_rschema_batch.py's interface exactly. Produces a JSONL file
readable by evaluate_rschema.py --llm4db.
Each output line: {"id": "<sample_id>", "schema": <Schema.model_dump()>}

Usage:
    python run_llm4db_batch.py --start_pos 0 --end_pos 10
    python run_llm4db_batch.py --start_pos 0 --end_pos 10 --model gpt-4o --provider openai
    python run_llm4db_batch.py --start_pos 0 --end_pos 10 --model gemini-2.0-flash --trace
    python run_llm4db_batch.py --start_pos 0 --end_pos 10 --model openai/gpt-4o-mini --provider openrouter

--model accepts:
  - LLM4DBdesign aliases:  gpt4, chatgpt, gemini, gemini-flash, gemini-pro, deepseek, glm4
  - Raw model IDs:         gpt-4o, gpt-4o-mini, gemini-2.0-flash, openai/gpt-4o, etc.
  - Default (None):        gpt4 (gpt-4o-2024-08-06 via OpenAI)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, List, Optional

PROJECT_ROOT = Path(__file__).parent
# LLM4DBdesign modules use bare imports (no package prefix) -- add its dir to path first.
sys.path.insert(0, str(PROJECT_ROOT / "LLM4DBdesign"))
sys.path.insert(1, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "RSchema" / "annotation.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "output" / "llm4db_rschema.jsonl"
DEFAULT_TRACE_DIR = PROJECT_ROOT / "output" / "llm4db_traces"

# Per-provider default models (used when --model is not supplied)
_PROVIDER_DEFAULT_MODEL = {
    "gemini": "gemini",  # alias -> reads GEMINI_BASE_MODEL env, falls back to gemini-3.1-flash-lite
    "openai": "gpt4",  # alias -> gpt-4o-2024-08-06
    "openrouter": os.getenv("OPENROUTER_BASE_MODEL", "openai/gpt-4o"),
}


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
                {
                    "idx": i,
                    "id": str(row.get("id", "")),
                    "nl": row.get("question", ""),
                    "answer": row.get("answer", {}),
                }
            )
    return cases


def _format_golden(answer: dict) -> str:
    lines = []
    for table_name, info in answer.items():
        pk_str = ", ".join(info.get("Primary key", []))
        lines.append(f"  {table_name}  pk=[{pk_str}]")
        lines.append(f"    cols: {info.get('Attributes', [])}")
        for col, ref in info.get("Foreign key", {}).items():
            for ref_table, ref_col in ref.items():
                lines.append(f"    FK: {col} -> {ref_table}.{ref_col}")
    return "\n".join(lines)


def _compute_metrics(pred_schema: Any, golden_answer: dict) -> dict:
    try:
        from src.evaluation.schema_level.RSchemaAPI import map_rschema_to_pydantic
        from src.evaluation.schema_level.schema_eval import SchemaEvaluator

        gt = map_rschema_to_pydantic(golden_answer)
        evaluator = SchemaEvaluator()
        scores = evaluator.evaluate_schema(pred_schema, gt)
        return {
            k: round(v, 3) for k, v in scores.items() if k != "dt_acc" and v is not None
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_case(
    case: dict,
    handler: Any,
    method_args: Any,
    trace_dir: Optional[Path],
) -> Optional[dict]:
    idx, sid, nl = case["idx"], case["id"], case["nl"]
    golden_answer = case["answer"]
    print(f"\n[{idx}] {sid} -- {nl[:60]!r}")

    t0 = time.time()
    try:
        from agent_format import fully_decode  # type: ignore[import]

        data_info = fully_decode(nl, handler, method_args)
    except Exception as e:
        print(f"  FAILED: {e}")
        if trace_dir:
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / "pipeline.log").write_text(f"FAILED: {e}\n", encoding="utf-8")
        return None

    elapsed = time.time() - t0
    answer = data_info.get("answer", {})
    schema_dict = answer.get("schema", answer) if isinstance(answer, dict) else {}

    try:
        from src.evaluation.schema_level.RSchemaAPI import map_rschema_to_pydantic

        pred_schema = map_rschema_to_pydantic(schema_dict)
    except Exception as e:
        print(f"  Schema parse failed: {e}")
        pred_schema = None

    tables = [t.name for t in pred_schema.tables] if pred_schema else []
    print(f"  Done in {elapsed:.1f}s  Schema: {tables}")

    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "nl.txt").write_text(nl, encoding="utf-8")
        (trace_dir / "golden.txt").write_text(
            _format_golden(golden_answer), encoding="utf-8"
        )
        (trace_dir / "raw_output.json").write_text(
            json.dumps(data_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if pred_schema is not None:
            metrics = _compute_metrics(pred_schema, golden_answer)
            (trace_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
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
            golden_table_names = {
                k.replace(" ", "_").upper() for k in golden_answer.keys()
            }
            pred_table_names = {t.name for t in pred_schema.tables}
            extra = pred_table_names - golden_table_names
            missing = golden_table_names - pred_table_names
            if extra:
                print(f"  Extra tables (not in golden): {sorted(extra)}")
            if missing:
                print(f"  Missing tables (in golden):   {sorted(missing)}")

    if pred_schema is None:
        return None
    return {"id": sid, "schema": pred_schema.model_dump()}


def main() -> None:
    p = argparse.ArgumentParser(description="LLM4DBdesign batch runner for RSchema")
    p.add_argument("--start_pos", type=int, default=0)
    p.add_argument("--end_pos", type=int, default=10)
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID or alias (e.g. gpt-4o, gemini-2.0-flash, gpt4). Default: gpt4.",
    )
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
        "--method",
        type=str,
        default="base_direct",
        choices=[
            "expert_analyse",
            "domain_analyse",
            "pseudo_code_analyse",
            "base_direct",
            "base_cot",
        ],
    )
    p.add_argument(
        "--trace",
        action="store_true",
        default=False,
        help="Save per-sample trace files (raw output, metrics)",
    )
    p.add_argument(
        "--trace_dir",
        type=str,
        default=str(DEFAULT_TRACE_DIR),
        help="Root directory for trace output (default: output/llm4db_traces/)",
    )
    args = p.parse_args()

    if args.provider:
        os.environ["PROVIDER"] = args.provider

    from api_utils import api_handler  # type: ignore[import]

    effective_provider = args.provider or os.getenv("PROVIDER", "gemini")
    model_arg = args.model or _PROVIDER_DEFAULT_MODEL.get(effective_provider, "gpt4")
    handler = api_handler(model_arg)

    method_args = types.SimpleNamespace(
        method=args.method,
        few_shot=False,
        log_history=True,
        max_attempt_vote=1,
        verification=(
            "entity_verification, entity_denpendency_verification, "
            "relation_denpendency_verification"
        ),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_root = Path(args.trace_dir) if args.trace else None

    done_ids: set = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resuming -- {len(done_ids)} already done")

    cases = _load_cases(args.start_pos, args.end_pos, Path(args.dataset))
    print(f"Running {len(cases)} cases (positions {args.start_pos}--{args.end_pos})")
    if trace_root:
        print(f"Tracing enabled -> {trace_root}")

    with out_path.open("a", encoding="utf-8") as out_f:
        for case in cases:
            if case["id"] in done_ids:
                print(f"[{case['idx']}] {case['id']} -- already done, skipping")
                continue
            trace_dir = (trace_root / case["id"]) if trace_root else None
            result = run_case(case, handler, method_args, trace_dir)
            if result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"\nDone. Output: {out_path}")


if __name__ == "__main__":
    main()
