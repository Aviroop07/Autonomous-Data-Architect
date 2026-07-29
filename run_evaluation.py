"""
ScribbleDB -- Evaluation Harness

Runs the full pipeline on every case in a dataset and computes published metrics:
  Schema-level : IC F1 (recall/precision), Structural score (FK topology,
                 table recall, column types), KDC (normalisation) --
                 all name-blind.
                 See docs/design/EVALUATION_METRICS.md
  Data-level   : MRE, NLL, KS, FA
  Smoke test   : pass rate

Usage examples
--------------
  # Evaluate on the 20 handcrafted cases
  python run_evaluation.py --dataset handcrafted

  # Evaluate on RSchema (first 50 cases)
  python run_evaluation.py --dataset rschema --limit 50

  # Ablation: no-sharding
  python run_evaluation.py --dataset handcrafted --no-sharding

  # Save results to a specific directory
  python run_evaluation.py --dataset handcrafted --output-dir eval_results/

Stage 4 does not exist yet. Its metrics (smoke pass rate, and the data-level
MRE/NLL/KS/FA that need generated data) are reported as their empty defaults
until it lands; Stages 1-3 are scored normally.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore[import]

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.util.core.providers import PROVIDERS  # noqa: E402  (needs sys.path above)

DATASET_ROOT = PROJECT_ROOT / "dataset"

DATASET_PATHS: Dict[str, Path] = {
    "rschema": DATASET_ROOT / "RSchema" / "annotation.jsonl",
    "handcrafted": DATASET_ROOT / "handcrafted" / "cases.jsonl",
    "benchmark_imdb": DATASET_ROOT / "benchmark" / "imdb" / "ground_truth.jsonl",
    "benchmark_tpch": DATASET_ROOT / "benchmark" / "tpch" / "ground_truth.jsonl",
    "benchmark_tpcds": DATASET_ROOT / "benchmark" / "tpcds" / "ground_truth.jsonl",
    "benchmark_mimiciv": DATASET_ROOT / "benchmark" / "mimiciv" / "ground_truth.jsonl",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_cases(dataset: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load all cases from a dataset JSONL file."""
    path = DATASET_PATHS.get(dataset)
    if path is None:
        raise ValueError(f"Unknown dataset: {dataset!r}. Valid: {list(DATASET_PATHS)}")
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


def _load_stage4():
    """Stage 4 is not built. Returns (orchestrate, run_smoke_test) when it
    exists, (None, None) otherwise.

    This module used to import src.orchestration.stage4.entry unconditionally
    at the top of run_case(), so EVERY case died with ModuleNotFoundError
    before Stage 1 even started. Guarded so the harness can score Stages 1-3
    today and pick Stage 4 up automatically once it lands.
    """
    try:
        # type: ignore -- these modules do not exist yet by design; the whole
        # point of this function is to tolerate that.
        from src.orchestration.stage4.entry import (  # type: ignore[import-not-found]
            orchestrate as stage4,
        )
        from src.pipeline.stage4.smoke_test import (  # type: ignore[import-not-found]
            run_smoke_test,
        )

        return stage4, run_smoke_test
    except ImportError:
        return None, None


async def run_case(
    case: Dict[str, Any],
    model: Optional[str],
    ablation_config: Any,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], List[str]]:
    """
    Run the pipeline on a single case.

    Returns (s1_output, s2_output, s3_output, s4_result, logs).
    Any stage that fails returns None for that output and all downstream outputs.
    s4_result is always None until Stage 4 exists.
    """
    from src.orchestration.stage1.entry import orchestrate as stage1
    from src.orchestration.stage2.entry import orchestrate as stage2
    from src.orchestration.stage3.entry import orchestrate as stage3

    stage4, _ = _load_stage4()

    nl = extract_nl(case)
    logs: List[str] = []

    # Stage 1
    try:
        s1_out, _ = await stage1(
            nl_description=nl,
            model=model,
            ablation_config=ablation_config,
        )
        logs.append(f"[Stage 1] OK ({len(s1_out.final_facts)} facts)")
    except Exception as e:
        logs.append(f"[Stage 1] FAILED: {e}")
        return None, None, None, None, logs

    # Stage 2 -- `plan` is required (fact chunking moved into Stage 1's own
    # chunker); omitting it used to raise TypeError on every case.
    try:
        s2_out, _, _registry = await stage2(
            plan=s1_out.plan,
            facts=s1_out.final_facts,
            domain=s1_out.domain,
            analytical_goal=s1_out.analytical_goal,
            nl_query=nl,
            model=model,
            ablation_config=ablation_config,
        )
        logs.append("[Stage 2] OK")
    except Exception as e:
        logs.append(f"[Stage 2] FAILED: {e}")
        return s1_out, None, None, None, logs

    global_schema = getattr(s2_out, "final_global_schema", None) or getattr(
        s2_out, "merged_schema", None
    )
    if global_schema is None:
        logs.append("[Stage 2] ERROR: no usable schema in output")
        return s1_out, s2_out, None, None, logs

    # Stage 3 -- signature is (schema, facts, shards=...), not the
    # global_schema=/all_facts=/registry= this used to pass. Shards are derived
    # internally when not supplied, so the registry is no longer threaded here.
    try:
        s3_out, _ = await stage3(
            schema=global_schema,
            facts=s1_out.final_facts,
            model=model,
            ablation_config=ablation_config,
        )
        report = s3_out.analysis_report
        logs.append(
            f"[Stage 3] OK ({s3_out.total_constraints} constraints, "
            f"{len(report.conflicts)} unresolved conflicts)"
        )
    except Exception as e:
        logs.append(f"[Stage 3] FAILED: {e}")
        return s1_out, s2_out, None, None, logs

    if stage4 is None:
        logs.append("[Stage 4] SKIPPED (not built)")
        return s1_out, s2_out, s3_out, None, logs

    try:
        s4_result, _ = await stage4(
            global_schema=global_schema,
            constraints=s3_out,
            business_facts=s1_out.final_facts,
            model=model,
            ablation_config=ablation_config,
        )
        logs.append(
            f"[Stage 4] OK (smoke={'PASSED' if s4_result.success else 'FAILED'})"
        )
    except Exception as e:
        logs.append(f"[Stage 4] FAILED: {e}")
        return s1_out, s2_out, s3_out, None, logs

    return s1_out, s2_out, s3_out, s4_result, logs


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def _schema_metrics(
    pred_schema: Any,
    gt_case: Dict[str, Any],
    facts: Any = None,
    conceptual: Any = None,
) -> Dict[str, Any]:
    """Compute schema-level metrics for one case."""
    from src.evaluation.schema_level.capacity_eval import evaluate_capacity
    from src.evaluation.schema_level.kdc_eval import evaluate_kdc_from_conceptual
    from src.evaluation.schema_level.structural_eval import evaluate_structural
    from src.util.schema_model.data_types import DataType
    from src.util.schema_model.schema import Schema, Table, Column, ForeignKey

    def _as_data_type(raw: Any) -> DataType:
        """Ground-truth JSON carries data types as free strings. Column.data_type
        is a DataType enum, so an unrecognised or missing value falls back to
        VARCHAR rather than raising -- a GT file with an odd type name should
        not abort the whole evaluation."""
        try:
            return DataType(str(raw).upper())
        except ValueError:
            return DataType.VARCHAR

    try:
        gt_raw = gt_case.get("ground_truth_schema", {})

        # Build GT Schema object + type maps for DT Acc
        gt_tables = []
        for t in gt_raw.get("tables", []):
            t_name: str = t["name"]
            default_pk = t_name.lower() + "_id"
            cols = []
            for c in t.get("columns", []):
                dt = c.get("data_type") or "VARCHAR"
                cols.append(Column(name=c["name"], data_type=_as_data_type(dt)))
            # Table's field is `primary_key: List[str]`, not `pk: str`.
            raw_pk = t.get("pk") or t.get("primary_key") or default_pk
            primary_key = [raw_pk] if isinstance(raw_pk, str) else list(raw_pk)
            gt_tables.append(
                Table(
                    name=t_name,
                    columns=cols,
                    primary_key=primary_key,
                )
            )
        gt_rels = [
            ForeignKey(
                referencing_table=r["referencing_table"],
                referencing_column=r["referencing_column"],
                referred_table=r["referred_table"],
            )
            for r in gt_raw.get("relationships", [])
        ]
        gt_schema = Schema(tables=gt_tables, relationships=gt_rels)

        # Name-blind: structural alignment for shape, provenance for capacity.
        # See docs/design/EVALUATION_METRICS.md.
        out: Dict[str, Any] = dict(evaluate_structural(pred_schema, gt_schema).as_dict())
        capacity = evaluate_capacity(pred_schema, facts or [])
        out.update(capacity.as_dict())
        out["uncovered_fact_ids"] = capacity.uncovered_fact_ids
        out["unsupported_elements"] = capacity.unsupported_elements

        # Normalisation, checked against the dependencies the pipeline itself
        # derived -- so this needs no ground truth, only internal consistency.
        if conceptual is not None:
            kdc = evaluate_kdc_from_conceptual(pred_schema, conceptual)
            out.update(kdc.as_dict())
            out["kdc_violations"] = (
                kdc.unenforced + kdc.partial_2nf + kdc.transitive_3nf
            )
        return out
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
        return {
            "mre": 1.0,
            "nll": 0.0,
            "ks": 1.0,
            "fa": 0.0,
            "n_evaluated": 0,
            "n_missing": 0,
        }
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
"structural_score": _aggregate(schema_scores, "structural_score"),
            "fk_topology_f1": _aggregate(schema_scores, "fk_topology_f1"),
            "table_structural_recall": _aggregate(
                schema_scores, "table_structural_recall"
            ),
            "column_type_agreement": _aggregate(schema_scores, "column_type_agreement"),
            "ic_f1": _aggregate(schema_scores, "ic_f1"),
            "ic_recall": _aggregate(schema_scores, "ic_recall"),
            "ic_precision": _aggregate(schema_scores, "ic_precision"),
            "kdc": _aggregate(schema_scores, "kdc"),
            "kdc_n_checked": _aggregate(schema_scores, "kdc_n_checked"),
        },
        "data": {
            "mre": _aggregate(data_scores, "mre"),
            "nll": _aggregate(data_scores, "nll"),
            "ks": _aggregate(data_scores, "ks"),
            "fa": _aggregate(data_scores, "fa"),
        },
        "smoke_pass_rate": (
            sum(1 for r in smoke_results if r is True) / len(smoke_results)
            if smoke_results
            else float("nan")
        ),
    }
    return agg


def _print_aggregate(agg: Dict[str, Any], label: str = "ScribbleDB") -> None:
    print(f"\n{'=' * 62}")
    print(f"  {label} -- Aggregate Metrics  (n={agg['n_cases']})")
    print(f"{'=' * 62}")
    s = agg["schema"]
    d = agg["data"]
    print("  Schema")
    print(f"    IC F1           : {s['ic_f1']:.3f}")
    print(f"      recall        : {s['ic_recall']:.3f}   (facts the schema can hold)")
    print(f"      precision     : {s['ic_precision']:.3f}   (structure a fact supports)")
    print(f"    Structural      : {s['structural_score']:.3f}")
    print(f"      FK topology F1: {s['fk_topology_f1']:.3f}")
    print(f"      table recall  : {s['table_structural_recall']:.3f}")
    print(f"      column types  : {s['column_type_agreement']:.3f}")
    print(f"    KDC             : {s.get('kdc', float('nan')):.3f}", end="")
    print(f"   (dependencies checked: {s.get('kdc_n_checked', 0):.1f} avg)")
    print("  Data")
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

    _, run_smoke_test = _load_stage4()

    ablation = AblationConfig(
        enable_enrichment=not args.no_enrichment,
        enable_sharding=not args.no_sharding,
        enable_logical_constraints=not args.no_logical_constraints,
    )

    model = args.model or None
    cases = load_cases(args.dataset, limit=args.limit)
    print(f"\n[Eval] Dataset: {args.dataset}  ({len(cases)} cases)")
    print(f"[Eval] Model  : {model}")
    print(
        f"[Eval] Ablation: enrichment={ablation.enable_enrichment}, "
        f"sharding={ablation.enable_sharding}, "
        f"logical_constraints={ablation.enable_logical_constraints}\n"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_tag = (
        "no_enrichment"
        if not ablation.enable_enrichment
        else "no_sharding"
        if not ablation.enable_sharding
        else "no_logical"
        if not ablation.enable_logical_constraints
        else "full"
    )
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            PROJECT_ROOT
            / "artifacts"
            / "runs"
            / "eval"
            / f"{ts}_{args.dataset}_{ablation_tag}"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    case_results: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        case_id = case.get("id", i)
        nl = extract_nl(case)
        print(f"[Case {i + 1}/{len(cases)}] id={case_id}")

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
                pred_schema = getattr(s2_out, "final_global_schema", None) or getattr(
                    s2_out, "merged_schema", None
                )
            if pred_schema is not None and case.get("ground_truth_schema"):
                result["schema_metrics"] = _schema_metrics(
                    pred_schema,
                    case,
                    getattr(s1_out, "final_facts", None),
                    getattr(s2_out, "final_conceptual_model", None),
                )
            else:
                result["schema_metrics"] = {
                    "structural_score": 0.0,
                    "fk_topology_f1": 0.0,
                    "table_structural_recall": 0.0,
                    "column_type_agreement": 0.0,
                    "ic_f1": 0.0,
                    "ic_recall": 0.0,
                    "ic_precision": 0.0,
                    "kdc": 0.0,
                }

            # Smoke test + data metrics
            smoke_passed = False
            smoke_dfs: Dict[str, Any] = {}
            if s4_result is not None:
                smoke_passed = bool(s4_result.success)
                # Re-run smoke test at full scale to collect DataFrames for metrics
                if (
                    smoke_passed
                    and run_smoke_test is not None
                    and case.get("ground_truth_distributions")
                ):
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
                result["data_metrics"] = {
                    "mre": 1.0,
                    "nll": 0.0,
                    "ks": 1.0,
                    "fa": 0.0,
                    "n_evaluated": 0,
                    "n_missing": 0,
                }

            sm = result["schema_metrics"]
            dm = result["data_metrics"]
            print(
                f"  ic_f1={sm.get('ic_f1', 0):.2f}  "
                f"struct={sm.get('structural_score', 0):.2f}  "
                f"mre={dm.get('mre', 1):.2f}  "
                f"ks={dm.get('ks', 1):.2f}  "
                f"smoke={'P' if smoke_passed else 'F'}  "
                f"({elapsed:.1f}s)"
            )

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
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to evaluate: handcrafted | rschema | "
        "benchmark_tpch | benchmark_imdb | benchmark_tpcds | benchmark_mimiciv",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of cases to evaluate (default: all)",
    )
    p.add_argument(
        "--model", type=str, default=None, help="LLM model (default: from env)"
    )
    p.add_argument(
        "--provider",
        # Sourced from the registry so this list cannot drift from what the
        # code actually supports -- it used to say openai|gemini only.
        choices=sorted(PROVIDERS),
        default=None,
        help="LLM provider (overrides PROVIDER env var)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Output directory (default: artifacts/runs/eval/{timestamp}_{dataset})",
    )
    p.add_argument("--no-enrichment", action="store_true", dest="no_enrichment")
    p.add_argument("--no-sharding", action="store_true", dest="no_sharding")
    p.add_argument(
        "--no-logical-constraints", action="store_true", dest="no_logical_constraints"
    )
    return p


def main() -> None:
    if sys.platform == "win32":
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")

    parser = _build_parser()
    args = parser.parse_args()

    if args.provider:
        os.environ["PROVIDER"] = args.provider

    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
