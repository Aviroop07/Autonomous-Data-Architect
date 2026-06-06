"""
ScribbleDB -- Full Pipeline Runner

Execute the complete 4-stage NL-to-database pipeline on a single description
or a dataset case, with optional ablation flags.

Usage examples
--------------
  # Direct NL input
  python run_pipeline.py --nl "An e-commerce platform that ..."

  # RSchema dataset (0-based index)
  python run_pipeline.py --dataset rschema --case-idx 0

  # Handcrafted dataset (by string ID field)

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")
  python run_pipeline.py --dataset handcrafted --case-id handcrafted-001

  # Benchmark case
  python run_pipeline.py --dataset benchmark_tpch --case-id tpch-001

  # Ablation: disable enrichment
  python run_pipeline.py --nl "..." --no-enrichment

  # Run only stages 1 and 2
  python run_pipeline.py --nl "..." --stages 1,2

  # Skip Stage 1, load from saved JSON
  python run_pipeline.py --from-stage2 output/runs/foo/stage1_output.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy stage imports
# ---------------------------------------------------------------------------

def _import_stage1() -> Callable[..., Any]:
    mod = importlib.import_module("src.orchestration.stage1.entry")
    return mod.orchestrate  # type: ignore[return-value]

def _import_stage2() -> Callable[..., Any]:
    mod = importlib.import_module("src.orchestration.stage2.entry")
    return mod.orchestrate  # type: ignore[return-value]

def _import_stage3() -> Callable[..., Any]:
    mod = importlib.import_module("src.orchestration.stage3.entry")
    return mod.orchestrate  # type: ignore[return-value]

def _import_stage4() -> Callable[..., Any]:
    mod = importlib.import_module("src.orchestration.stage4.entry")
    return mod.orchestrate  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

DATASET_ROOT = PROJECT_ROOT / "dataset"

DATASET_PATHS: Dict[str, Path] = {
    "rschema":           DATASET_ROOT / "RSchema" / "annotation.jsonl",
    "handcrafted":       DATASET_ROOT / "handcrafted" / "cases.jsonl",
    "benchmark_imdb":    DATASET_ROOT / "benchmark" / "imdb" / "ground_truth.jsonl",
    "benchmark_tpch":    DATASET_ROOT / "benchmark" / "tpch" / "ground_truth.jsonl",
    "benchmark_tpcds":   DATASET_ROOT / "benchmark" / "tpcds" / "ground_truth.jsonl",
    "benchmark_mimiciv": DATASET_ROOT / "benchmark" / "mimiciv" / "ground_truth.jsonl",
}


def _load_nl_from_rschema(case_idx: int) -> str:
    path = DATASET_PATHS["rschema"]
    if not path.exists():
        raise FileNotFoundError(f"RSchema dataset not found: {path}")
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == case_idx:
                data = json.loads(line)
                return str(data.get("question", data.get("nl_description", "")))
    raise IndexError(f"RSchema: case index {case_idx} out of range for {path}")


def _load_nl_from_jsonl(dataset_key: str, case_id: Optional[str], case_idx: Optional[int]) -> str:
    path = DATASET_PATHS.get(dataset_key)
    if path is None:
        raise ValueError(f"Unknown dataset: {dataset_key!r}")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if case_id is not None and str(data.get("id", "")) == str(case_id):
                return str(data.get("nl_description", data.get("question", "")))
            if case_idx is not None and i == case_idx:
                return str(data.get("nl_description", data.get("question", "")))
    raise ValueError(f"No matching case in {path} (case_id={case_id}, case_idx={case_idx})")


def resolve_nl(args: argparse.Namespace) -> str:
    if args.nl:
        return args.nl

    dataset = (args.dataset or "").lower()

    if dataset == "rschema":
        if args.case_idx is None:
            raise ValueError("--case-idx is required for --dataset rschema")
        return _load_nl_from_rschema(args.case_idx)

    if dataset in DATASET_PATHS:
        return _load_nl_from_jsonl(dataset, args.case_id, args.case_idx)

    raise ValueError("Provide --nl or --dataset with --case-id / --case-idx")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _make_run_id(args: argparse.Namespace) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.nl:
        slug = args.nl[:30].strip().replace(" ", "_").replace("/", "_")
        return f"{ts}_{slug}"
    if args.dataset:
        idx = args.case_idx if args.case_idx is not None else args.case_id
        return f"{ts}_{args.dataset}_{idx}"
    return ts


def _prepare_output_dir(args: argparse.Namespace, run_id: str) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = PROJECT_ROOT / "output" / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return dict(model) if model else {}


def _print_summary(
    run_id: str,
    nl: str,
    stages_run: List[int],
    stages_time: Dict[str, float],
    total_tokens: int,
    smoke_passed: Optional[bool],
    out_dir: Path,
) -> None:
    total_time = sum(stages_time.values())
    print("\n" + "=" * 60)
    print(f"  ScribbleDB Run: {run_id}")
    print("=" * 60)
    print(f"  NL (first 80 chars): {nl[:80]!r}")
    print(f"  Stages run:          {stages_run}")
    print(f"  Total time:          {total_time:.1f}s")
    print(f"  Total tokens:        {total_tokens}")
    if smoke_passed is not None:
        print(f"  Smoke test:          {'PASSED' if smoke_passed else 'FAILED'}")
    print(f"  Output dir:          {out_dir}")
    print("-" * 60)
    for stage_key, elapsed in stages_time.items():
        print(f"    {stage_key:<20} {elapsed:.1f}s")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

async def run_pipeline(args: argparse.Namespace) -> None:
    from src.util.ablation import AblationConfig

    stages_to_run: Set[int] = (
        {int(s.strip()) for s in args.stages.split(",")} if args.stages else {1, 2, 3, 4}
    )

    ablation = AblationConfig(
        enable_enrichment=not args.no_enrichment,
        enable_sharding=not args.no_sharding,
        enable_logical_constraints=not args.no_logical_constraints,
    )

    nl = resolve_nl(args)
    model = args.model or "gpt-4o"
    run_id = _make_run_id(args)
    out_dir = _prepare_output_dir(args, run_id)

    print(f"\n[Pipeline] Run ID  : {run_id}")
    print(f"[Pipeline] Model   : {model}")
    print(f"[Pipeline] Stages  : {sorted(stages_to_run)}")
    print(f"[Pipeline] Ablation: enrichment={ablation.enable_enrichment}, "
          f"sharding={ablation.enable_sharding}, "
          f"logical_constraints={ablation.enable_logical_constraints}")
    print(f"[Pipeline] Output  : {out_dir}\n")

    total_tokens = 0
    stages_time: Dict[str, float] = {}
    stages_run: List[int] = []

    # --- Stage 1: Fact Extraction ---
    s1_output = None

    if args.from_stage2:
        s1_path = Path(args.from_stage2)
        if not s1_path.exists():
            print(f"[Pipeline] ERROR: --from-stage2 file not found: {s1_path}")
            sys.exit(1)
        print(f"[Pipeline] Skipping Stage 1 -- loading from {s1_path}")
        from src.orchestration.stage1.models import Output as Stage1Output
        raw = json.loads(s1_path.read_text(encoding="utf-8"))
        s1_output = Stage1Output(**raw)
    elif 1 in stages_to_run:
        print("[Pipeline] --- Stage 1: Fact Extraction ---")
        t0 = time.time()
        try:
            stage1_orchestrate = _import_stage1()
            s1_output, s1_tokens = await stage1_orchestrate(
                nl_description=nl,
                model=model,
                ablation_config=ablation,
            )
            elapsed = time.time() - t0
            total_tokens += s1_tokens
            stages_time["stage1"] = elapsed
            stages_run.append(1)
            print(f"[Stage 1] Done in {elapsed:.1f}s ({s1_tokens} tokens)")
            _save_json(out_dir / "stage1_output.json", _model_to_dict(s1_output))
        except Exception as exc:
            elapsed = time.time() - t0
            stages_time["stage1"] = elapsed
            print(f"[Stage 1] FAILED after {elapsed:.1f}s: {exc}")
            _save_json(out_dir / "stage1_error.json", {"error": str(exc), "elapsed": elapsed})
            _save_run_summary(out_dir, run_id, nl, total_tokens, stages_time, stages_run, None)
            raise

    if s1_output is None and 2 in stages_to_run:
        print("[Pipeline] Stage 2 requires Stage 1 output. Skipping Stage 2+.")
        stages_to_run -= {2, 3, 4}

    # --- Stage 2: Schema Generation ---
    s2_output = None
    s2_registry = None

    if 2 in stages_to_run and s1_output is not None:
        print("[Pipeline] --- Stage 2: Schema Generation ---")
        t0 = time.time()
        try:
            stage2_orchestrate = _import_stage2()
            result = await stage2_orchestrate(
                facts=s1_output.final_facts,
                domain=s1_output.domain,
                analytical_goal=s1_output.analytical_goal,
                model=model,
                ablation_config=ablation,
            )
            if isinstance(result, tuple) and len(result) == 3:
                s2_output, s2_tokens, s2_registry = result
            elif isinstance(result, tuple) and len(result) == 2:
                s2_output, s2_tokens = result
            else:
                s2_output = result
                s2_tokens = 0

            elapsed = time.time() - t0
            total_tokens += s2_tokens
            stages_time["stage2"] = elapsed
            stages_run.append(2)
            print(f"[Stage 2] Done in {elapsed:.1f}s ({s2_tokens} tokens)")
            _save_json(out_dir / "stage2_output.json", _model_to_dict(s2_output))
        except Exception as exc:
            elapsed = time.time() - t0
            stages_time["stage2"] = elapsed
            print(f"[Stage 2] FAILED after {elapsed:.1f}s: {exc}")
            _save_json(out_dir / "stage2_error.json", {"error": str(exc), "elapsed": elapsed})
            _save_run_summary(out_dir, run_id, nl, total_tokens, stages_time, stages_run, None)
            raise

    if s2_output is None and 3 in stages_to_run:
        print("[Pipeline] Stage 3 requires Stage 2 output. Skipping Stage 3+.")
        stages_to_run -= {3, 4}

    # --- Stage 3: Constraint Modeling ---
    s3_output = None

    if 3 in stages_to_run and s2_output is not None:
        print("[Pipeline] --- Stage 3: Constraint Modeling ---")
        t0 = time.time()
        try:
            stage3_orchestrate = _import_stage3()
            global_schema = (
                getattr(s2_output, "final_global_schema", None)
                or getattr(s2_output, "merged_schema", None)
                or getattr(s2_output, "global_schema", None)
            )
            if global_schema is None:
                raise RuntimeError("Stage 2 output has no usable schema field")

            assert s1_output is not None
            kw: Dict[str, Any] = dict(
                global_schema=global_schema,
                all_facts=s1_output.final_facts,
                model=model,
                ablation_config=ablation,
            )
            if s2_registry is not None:
                kw["registry"] = s2_registry
            result = await stage3_orchestrate(**kw)

            if isinstance(result, tuple) and len(result) == 2:
                s3_output, s3_tokens = result
            else:
                s3_output = result
                s3_tokens = 0

            elapsed = time.time() - t0
            total_tokens += s3_tokens
            stages_time["stage3"] = elapsed
            stages_run.append(3)
            print(f"[Stage 3] Done in {elapsed:.1f}s ({s3_tokens} tokens)")
            _save_json(out_dir / "stage3_output.json", _model_to_dict(s3_output))
        except Exception as exc:
            elapsed = time.time() - t0
            stages_time["stage3"] = elapsed
            print(f"[Stage 3] FAILED after {elapsed:.1f}s: {exc}")
            _save_json(out_dir / "stage3_error.json", {"error": str(exc), "elapsed": elapsed})
            _save_run_summary(out_dir, run_id, nl, total_tokens, stages_time, stages_run, None)
            raise

    if s3_output is None and 4 in stages_to_run:
        print("[Pipeline] Stage 4 requires Stage 3 output. Skipping Stage 4.")
        stages_to_run -= {4}

    # --- Stage 4: Code Generation + Smoke Test ---
    smoke_passed: Optional[bool] = None

    if 4 in stages_to_run and s3_output is not None:
        print("[Pipeline] --- Stage 4: Code Generation ---")
        t0 = time.time()
        try:
            stage4_orchestrate = _import_stage4()
            global_schema = (
                getattr(s2_output, "final_global_schema", None)
                or getattr(s2_output, "merged_schema", None)
                or getattr(s2_output, "global_schema", None)
            )
            manifest = getattr(s3_output, "global_manifest", None)
            if global_schema is None:
                raise RuntimeError("No schema available from Stage 2.")
            if manifest is None:
                raise RuntimeError("No manifest available from Stage 3.")

            assert s1_output is not None
            s4_result, s4_tokens = await stage4_orchestrate(
                global_schema=global_schema,
                manifest=manifest,
                business_facts=s1_output.final_facts,
                model=model,
                ablation_config=ablation,
            )
            elapsed = time.time() - t0
            total_tokens += s4_tokens
            stages_time["stage4"] = elapsed
            stages_run.append(4)
            print(f"[Stage 4] Done in {elapsed:.1f}s ({s4_tokens} tokens)")

            _save_json(out_dir / "stage4_output.json", _model_to_dict(s4_result))

            generated_code: str = s4_result.generated_code
            code_path = out_dir / "generated_code.py"
            code_path.write_text(generated_code, encoding="utf-8")
            print(f"[Stage 4] Code saved to {code_path}")

            smoke_passed = s4_result.success
            print(f"[SmokeTest] {'PASSED' if smoke_passed else 'FAILED'}")
            for log_line in (s4_result.verification_logs or []):
                print(f"  {log_line}")

        except Exception as exc:
            elapsed = time.time() - t0
            stages_time["stage4"] = elapsed
            print(f"[Stage 4] FAILED after {elapsed:.1f}s: {exc}")
            _save_json(out_dir / "stage4_error.json", {"error": str(exc), "elapsed": elapsed})
            _save_run_summary(out_dir, run_id, nl, total_tokens, stages_time, stages_run, smoke_passed)
            raise

    _save_run_summary(out_dir, run_id, nl, total_tokens, stages_time, stages_run, smoke_passed)
    _print_summary(run_id, nl, stages_run, stages_time, total_tokens, smoke_passed, out_dir)


def _save_run_summary(
    out_dir: Path,
    run_id: str,
    nl: str,
    total_tokens: int,
    stages_time: Dict[str, float],
    stages_run: List[int],
    smoke_passed: Optional[bool],
) -> None:
    summary = {
        "run_id": run_id,
        "nl": nl,
        "total_tokens": total_tokens,
        "total_time": sum(stages_time.values()),
        "stages_time": stages_time,
        "stages_run": stages_run,
        "smoke_test_passed": smoke_passed,
    }
    _save_json(out_dir / "run_summary.json", summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ScribbleDB -- full pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--nl", type=str, default=None,
                   help="Natural language description (direct input)")
    p.add_argument("--dataset", type=str, default=None,
                   help="Dataset: rschema | handcrafted | benchmark_imdb | "
                        "benchmark_tpch | benchmark_tpcds | benchmark_mimiciv")
    p.add_argument("--case-id", type=str, default=None, dest="case_id",
                   help="Case ID string (e.g. handcrafted-001, tpch-001)")
    p.add_argument("--case-idx", type=int, default=None, dest="case_idx",
                   help="0-based line index (for rschema)")
    p.add_argument("--model", type=str, default="gpt-4o",
                   help="LLM model name (default: gpt-4o)")
    p.add_argument("--output-dir", type=str, default=None, dest="output_dir",
                   help="Output directory (default: output/runs/{timestamp})")
    p.add_argument("--no-enrichment", action="store_true", dest="no_enrichment",
                   help="Disable Stage 1 context enrichment")
    p.add_argument("--no-sharding", action="store_true", dest="no_sharding",
                   help="Disable Stage 2 fact sharding")
    p.add_argument("--no-logical-constraints", action="store_true",
                   dest="no_logical_constraints",
                   help="Disable Stage 4 logical constraint overrides")
    p.add_argument("--stages", type=str, default=None,
                   help="Comma-separated stages to run: 1,2,3,4 (default: all)")
    p.add_argument("--from-stage2", type=str, default=None, dest="from_stage2",
                   help="Skip Stage 1, load Stage 1 output JSON from this path")
    return p


def main() -> None:
    if sys.platform == "win32":
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")

    parser = _build_parser()
    args = parser.parse_args()

    if not args.nl and not args.dataset and not args.from_stage2:
        parser.error(
            "Provide one of: --nl, --dataset with --case-id/--case-idx, "
            "or --from-stage2 FILE"
        )

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
