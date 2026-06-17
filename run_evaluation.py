"""
ScribbleDB -- Evaluation Harness

Runs the full pipeline on every case in a dataset and computes published metrics:
  Schema-level : Table F1/Acc, Attr F1/Acc, PK Acc, FK Acc, DT Acc
  Data-level   : MRE, NLL, KS, FA
  Smoke test   : pass rate

Usage examples
--------------
  # Evaluate on the 20 handcrafted cases
  python run_evaluation.py --dataset handcrafted

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

  # Evaluate on RSchema (first 50 cases)
  python run_evaluation.py --dataset rschema --limit 50

  # Ablation: no-sharding
  python run_evaluation.py --dataset handcrafted --no-sharding

  # Save results to a specific directory
  python run_evaluation.py --dataset handcrafted --output-dir eval_results/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_ROOT = PROJECT_ROOT / "dataset"

DATASET_PATHS: Dict[str, Path] = {
    "rschema":           DATASET_ROOT / "RSchema" / "annotation.jsonl",
    "handcrafted":       DATASET_ROOT / "handcrafted" / "cases.jsonl",
    "benchmark_imdb":    DATASET_ROOT / "benchmark" / "imdb" / "ground_truth.jsonl",
    "benchmark_tpch":    DATASET_ROOT / "benchmark" / "tpch" / "ground_truth.jsonl",
    "benchmark_tpcds":   DATASET_ROOT / "benchmark" / "tpcds" / "ground_truth.jsonl",
    "benchmark_mimiciv": DATASET_ROOT / "benchmark" / "mimiciv" / "ground_truth.jsonl",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_cases(dataset: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load all cases from a dataset JSONL file."""
    path = DATASET_PATHS.get(dataset)
    if path is None:
        raise ValueError(f"Unknown dataset: {dataset!r}. "
                         f"Valid: {list(DATASET_PATHS)}")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    cases = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def extract_nl(case: Dict[str, Any]) -> str:
    return str(case.get("nl_description") or case.get("question") or "")


# ---------------------------------------------------------------------------
# Single-case pipeline runner
# ---------------------------------------------------------------------------

async def run_case(
    case: Dict[str, Any],
    model: str,
    ablation_config: Any,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], List[str]]:
    """
    Run the full 4-stage pipeline on a single case.

    Returns (s1_output, s2_output, s3_output, s4_result, logs).
    Any stage that fails returns None for that output and all downstream outputs.
    """
    from src.orchestration.stage1.entry import orchestrate as stage1
    from src.orchestration.stage2.entry import orchestrate as stage2
    from src.orchestration.stage3.entry import orchestrate as stage3
    from src.orchestration.stage4.entry import orchestrate as stage4

    nl = extract_nl(case)
    logs: List[str] = []

    # Stage 1
    try:
        s1_out, _ = await stage1(
            nl_description=nl,
            model=model,
            ablation_config=ablation_config,
        )
        logs.append("[Stage 1] OK")
    except Exception as e:
        logs.append(f"[Stage 1] FAILED: {e}")
        return None, None, None, None, logs

    # Stage 2
    try:
        result2 = await stage2(
            facts=s1_out.final_facts,
            domain=s1_out.domain,
            analytical_goal=s1_out.analytical_goal,
            model=model,
            ablation_config=ablation_config,
        )
        # stage2 always returns (Output, int, TableFactRegistry)
        s2_out, _, s2_registry = result2  # type: ignore[misc]
        logs.append("[Stage 2] OK")
    except Exception as e:
        logs.append(f"[Stage 2] FAILED: {e}")
        return s1_out, None, None, None, logs

    global_schema = (
        getattr(s2_out, "final_global_schema", None)
        or getattr(s2_out, "merged_schema", None)
    )
    if global_schema is None:
        logs.append("[Stage 2] ERROR: no usable schema in output")
        return s1_out, s2_out, None, None, logs

    # Stage 3
    try:
        kw: Dict[str, Any] = dict(
            global_schema=global_schema,
            all_facts=s1_out.final_facts,
            model=model,
            ablation_config=ablation_config,
        )
        if s2_registry is not None:
            kw["registry"] = s2_registry
        result3 = await stage3(**kw)
        s3_out = result3[0] if isinstance(result3, tuple) else result3
        logs.append("[Stage 3] OK")
    except Exception as e:
        logs.append(f"[Stage 3] FAILED: {e}")
        return s1_out, s2_out, None, None, logs

    manifest = getattr(s3_out, "global_manifest", None)
    if manifest is None:
        logs.append("[Stage 3] ERROR: no manifest in output")
        return s1_out, s2_out, s3_out, None, logs

    # Stage 4
    try:
        s4_result, _ = await stage4(
            global_schema=global_schema,
            manifest=manifest,
            business_facts=s1_out.final_facts,
            model=model,
            ablation_config=ablation_config,
        )
        logs.append(f"[Stage 4] OK (smoke={'PASSED' if s4_result.success else 'FAILED'})")
    except Exception as e:
        logs.append(f"[Stage 4] FAILED: {e}")
        return s1_out, s2_out, s3_out, None, logs

    return s1_out, s2_out, s3_out, s4_result, logs


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _schema_metrics(pred_schema: Any, gt_case: Dict[str, Any]) -> Dict[str, Any]:
    """Compute schema-level metrics for one case."""
    from src.evaluation.schema_level.schema_eval import SchemaEvaluator
    from src.pipeline.stage2.models.schema import Schema, Table, Column, ForeignKey

    try:
        gt_raw = gt_case.get("ground_truth_schema", {})

        # Build GT Schema object + type maps for DT Acc
        gt_tables = []
        gt_col_types: Dict[str, str] = {}
        for t in gt_raw.get("tables", []):
            t_name: str = t["name"]
            default_pk = t_name.lower() + "_id"
            cols = []
            for c in t.get("columns", []):
                dt = c.get("data_type") or "VARCHAR"
                cols.append(Column(name=c["name"], data_type=dt))
                gt_col_types[f"{t_name}.{c['name']}"] = dt
            gt_tables.append(Table(
                name=t_name,
                columns=cols,
                pk=t.get("pk") or default_pk,
            ))
        gt_rels = [
            ForeignKey(
                referencing_table=r["referencing_table"],
                referencing_column=r["referencing_column"],
                referred_table=r["referred_table"],
            )
            for r in gt_raw.get("relationships", [])
        ]
        gt_schema = Schema(tables=gt_tables, relationships=gt_rels)

        # Extract type map from predicted schema
        pred_col_types: Dict[str, str] = {
            f"{t.name}.{c.name}": (c.data_type or "VARCHAR")
            for t in pred_schema.tables
            for c in t.columns
        }

        evaluator = SchemaEvaluator()
        return evaluator.evaluate_schema(
            pred_schema, gt_schema,
            gt_col_types=gt_col_types,
            pred_col_types=pred_col_types,
        )
    except Exception as e:
        return {"error": str(e)}


def _data_metrics(
    smoke_dfs: Dict[str, Any],
    gt_case: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute data-level metrics for one case."""
    from src.evaluation.data_level.data_eval import evaluate_data

    gt_dists = gt_case.get("ground_truth_distributions", {})
    if not gt_dists or not smoke_dfs:
        return {"mre": 1.0, "nll": 0.0, "ks": 1.0, "fa": 0.0,
                "n_evaluated": 0, "n_missing": 0}
    try:
        return evaluate_data(smoke_dfs, gt_dists)
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# Aggregate metric computation
# ---------------------------------------------------------------------------

def _aggregate(scores: List[Dict[str, Any]], key: str) -> float:
    vals = [s[key] for s in scores if key in s and s[key] is not None]
    return float(np.mean(vals)) if vals else float("nan")


def compute_aggregate_metrics(case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema_scores = [r["schema_metrics"] for r in case_results if "schema_metrics" in r]
    data_scores = [r["data_metrics"] for r in case_results if "data_metrics" in r]
    smoke_results = [r.get("smoke_passed") for r in case_results]

    agg: Dict[str, Any] = {
        "n_cases": len(case_results),
        "schema": {
            "table_f1":  _aggregate(schema_scores, "table_f1"),
            "table_acc": _aggregate(schema_scores, "table_acc"),
            "attr_f1":   _aggregate(schema_scores, "attr_f1"),
            "attr_acc":  _aggregate(schema_scores, "attr_acc"),
            "pk_acc":    _aggregate(schema_scores, "pk_acc"),
            "fk_acc":    _aggregate(schema_scores, "fk_acc"),
            "dt_acc":    _aggregate(schema_scores, "dt_acc"),
        },
        "data": {
            "mre": _aggregate(data_scores, "mre"),
            "nll": _aggregate(data_scores, "nll"),
            "ks":  _aggregate(data_scores, "ks"),
            "fa":  _aggregate(data_scores, "fa"),
        },
        "smoke_pass_rate": (
            sum(1 for r in smoke_results if r is True) / len(smoke_results)
            if smoke_results else float("nan")
        ),
    }
    return agg


def _print_aggregate(agg: Dict[str, Any], label: str = "ScribbleDB") -> None:
    print(f"\n{'=' * 62}")
    print(f"  {label} -- Aggregate Metrics  (n={agg['n_cases']})")
    print(f"{'=' * 62}")
    s = agg["schema"]
    d = agg["data"]
    print(f"  Schema")
    print(f"    Table F1 / Acc  : {s['table_f1']:.3f} / {s['table_acc']:.3f}")
    print(f"    Attr  F1 / Acc  : {s['attr_f1']:.3f} / {s['attr_acc']:.3f}")
    print(f"    PK Acc          : {s['pk_acc']:.3f}")
    print(f"    FK Acc          : {s['fk_acc']:.3f}")
    print(f"    DT Acc          : {s['dt_acc'] if s['dt_acc'] == s['dt_acc'] else 'N/A'}")
    print(f"  Data")
    print(f"    MRE             : {d['mre']:.3f}")
    print(f"    NLL             : {d['nll']:.3f}")
    print(f"    KS              : {d['ks']:.3f}")
    print(f"    FA              : {d['fa']:.3f}")
    print(f"  Smoke pass rate   : {agg['smoke_pass_rate']:.2%}")
    print(f"{'=' * 62}\n")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def evaluate(args: argparse.Namespace) -> None:
    from src.util.config.ablation import AblationConfig
    from src.pipeline.stage4.smoke_test import run_smoke_test

    ablation = AblationConfig(
        enable_enrichment=not args.no_enrichment,
        enable_sharding=not args.no_sharding,
        enable_logical_constraints=not args.no_logical_constraints,
    )

    model = args.model or "gpt-4o"
    cases = load_cases(args.dataset, limit=args.limit)
    print(f"\n[Eval] Dataset: {args.dataset}  ({len(cases)} cases)")
    print(f"[Eval] Model  : {model}")
    print(f"[Eval] Ablation: enrichment={ablation.enable_enrichment}, "
          f"sharding={ablation.enable_sharding}, "
          f"logical_constraints={ablation.enable_logical_constraints}\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_tag = (
        "no_enrichment" if not ablation.enable_enrichment else
        "no_sharding" if not ablation.enable_sharding else
        "no_logical" if not ablation.enable_logical_constraints else
        "full"
    )
    out_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "output" / "eval" / f"{ts}_{args.dataset}_{ablation_tag}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    case_results: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        case_id = case.get("id", i)
        nl = extract_nl(case)
        print(f"[Case {i+1}/{len(cases)}] id={case_id}")

        t0 = time.time()
        result: Dict[str, Any] = {"id": case_id, "nl": nl[:120]}

        try:
            s1_out, s2_out, s3_out, s4_result, logs = await run_case(
                case, model, ablation
            )
            result["pipeline_logs"] = logs
            elapsed = time.time() - t0
            result["elapsed"] = elapsed

            # Schema metrics
            pred_schema = None
            if s2_out is not None:
                pred_schema = (
                    getattr(s2_out, "final_global_schema", None)
                    or getattr(s2_out, "merged_schema", None)
                )
            if pred_schema is not None and case.get("ground_truth_schema"):
                result["schema_metrics"] = _schema_metrics(pred_schema, case)
            else:
                result["schema_metrics"] = {
                    "table_f1": 0.0, "table_acc": 0.0,
                    "attr_f1": 0.0, "attr_acc": 0.0,
                    "pk_acc": 0.0, "fk_acc": 0.0, "dt_acc": None,
                }

            # Smoke test + data metrics
            smoke_passed = False
            smoke_dfs: Dict[str, Any] = {}
            if s4_result is not None:
                smoke_passed = bool(s4_result.success)
                # Re-run smoke test at full scale to collect DataFrames for metrics
                if smoke_passed and case.get("ground_truth_distributions"):
                    try:
                        _, smoke_dfs, _ = run_smoke_test(
                            s4_result.generated_code, scale_factor=1.0
                        )
                    except Exception:
                        smoke_dfs = {}

            result["smoke_passed"] = smoke_passed
            if case.get("ground_truth_distributions"):
                result["data_metrics"] = _data_metrics(smoke_dfs, case)
            else:
                result["data_metrics"] = {"mre": 1.0, "nll": 0.0, "ks": 1.0, "fa": 0.0,
                                          "n_evaluated": 0, "n_missing": 0}

            sm = result["schema_metrics"]
            dm = result["data_metrics"]
            print(f"  table_f1={sm.get('table_f1', 0):.2f}  "
                  f"attr_f1={sm.get('attr_f1', 0):.2f}  "
                  f"mre={dm.get('mre', 1):.2f}  "
                  f"ks={dm.get('ks', 1):.2f}  "
                  f"smoke={'P' if smoke_passed else 'F'}  "
                  f"({elapsed:.1f}s)")

        except Exception as e:
            result["error"] = traceback.format_exc()
            result["elapsed"] = time.time() - t0
            print(f"  ERROR: {e}")

        case_results.append(result)

    # Aggregate and save
    agg = compute_aggregate_metrics(case_results)
    _print_aggregate(agg)

    (out_dir / "case_results.json").write_text(
        json.dumps(case_results, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "aggregate_metrics.json").write_text(
        json.dumps(agg, indent=2, default=str), encoding="utf-8"
    )

    print(f"[Eval] Results saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ScribbleDB -- evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset to evaluate: handcrafted | rschema | "
                        "benchmark_tpch | benchmark_imdb | benchmark_tpcds | benchmark_mimiciv")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of cases to evaluate (default: all)")
    p.add_argument("--model", type=str, default="gpt-4o",
                   help="LLM model (default: gpt-4o)")
    p.add_argument("--output-dir", type=str, default=None, dest="output_dir",
                   help="Output directory (default: output/eval/{timestamp}_{dataset})")
    p.add_argument("--no-enrichment", action="store_true", dest="no_enrichment")
    p.add_argument("--no-sharding", action="store_true", dest="no_sharding")
    p.add_argument("--no-logical-constraints", action="store_true",
                   dest="no_logical_constraints")
    return p


def main() -> None:
    if sys.platform == "win32":
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")

    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
