"""src/util/ must not depend on src/pipeline/.

util/ is the shared, stage-agnostic layer. It had accumulated 12 modules
importing Schema/Table/Column from src/pipeline/stage2/models/, so it could not
be imported without pulling in Stage 2 -- a producer/owner confusion: Stage 2
PRODUCES a schema, but Stages 2/3/4, the evaluation harness and the constraint
model all speak in Schema, so the type belongs in shared code. The types now
live in src/util/schema_model/.

This is asserted mechanically because it is the kind of edge that gets
reintroduced by a single convenient import.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

UTIL = pathlib.Path("src/util")

# The one documented exception. bridge/from_cross_shard.py wraps Stage 3's four
# extraction shapes into constraint-model terms, so it necessarily names Stage 3
# types. The audit's own conclusion was that this file belongs under
# src/pipeline/stage3/ rather than in util/; until it moves, it is exempted here
# explicitly rather than silently.
ALLOWED = {pathlib.Path("src/util/constraint_model/bridge/from_cross_shard.py")}


def _util_modules() -> list[pathlib.Path]:
    return [p for p in UTIL.rglob("*.py") if "__pycache__" not in str(p)]


def test_there_are_util_modules_to_check():
    """Guards the glob -- an empty parametrisation would pass vacuously."""
    assert len(_util_modules()) > 20


@pytest.mark.parametrize(
    "path", [p for p in _util_modules() if p not in ALLOWED], ids=lambda p: p.name
)
def test_no_util_module_imports_from_pipeline(path):
    # utf-8-sig, not utf-8: a stray BOM would otherwise make ast.parse raise
    # SyntaxError and turn a layering check into a confusing encoding failure.
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "src.pipeline"
        ):
            offenders.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src.pipeline"):
                    offenders.append(f"line {node.lineno}: import {alias.name}")
    assert not offenders, (
        f"{path} imports from src/pipeline/: {offenders}. Shared types belong in "
        f"src/util/schema_model/; stage-specific logic belongs in src/pipeline/."
    )


def test_schema_model_is_importable_without_loading_stage2():
    """The actual property the move exists to deliver, checked end to end."""
    import subprocess
    import sys

    code = (
        "import sys;"
        "import src.util.schema_model;"
        "import src.util.schema_ops.patching_engine;"
        "import src.util.constraint_model.variables;"
        "leaked=[m for m in sys.modules if m.startswith('src.pipeline')];"
        "assert not leaked, leaked;"
        "print('clean')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd="."
    )
    assert "clean" in out.stdout, out.stderr[-2000:]
