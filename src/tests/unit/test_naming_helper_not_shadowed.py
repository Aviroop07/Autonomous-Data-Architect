"""The naming heuristics must have exactly one definition each.

relational_mapper.py shadowed looks_singular_noun with a weaker copy that
matched a hardcoded domain word set against the whole name instead of
tokenizing. The two disagreed on half the names tested, always in the same
direction -- the local copy called a singular noun plural -- so the mapper
rejected good relationship names for junction tables. Meanwhile the canonical
version's docstring claimed the mapper already shared it.
"""

from __future__ import annotations

import ast
import pathlib

from src.util.schema_model.schema import looks_singular_noun

SRC = pathlib.Path(__file__).resolve().parents[2]
SHARED = {"looks_singular_noun", "to_snake_case", "schema_to_prompt_text"}


def _definition_sites(func_name: str) -> list[str]:
    sites: list[str] = []
    for path in SRC.rglob("*.py"):
        if "tests" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                sites.append(f"{path.relative_to(SRC)}:{node.lineno}")
    return sites


def test_shared_naming_helpers_are_defined_exactly_once() -> None:
    for name in SHARED:
        sites = _definition_sites(name)
        assert len(sites) == 1, f"{name} has {len(sites)} definitions: {sites}"


def test_a_singular_noun_ending_in_s_is_not_called_plural() -> None:
    """Every one of these was misjudged by the copy that has been removed."""
    for name in (
        "ORDER_STATUS",
        "PATIENT_DIAGNOSIS",
        "CAMPUS",
        "ADDRESS",
        "ANALYSIS",
        "BONUS",
    ):
        assert looks_singular_noun(name), name


def test_a_genuine_plural_is_still_rejected() -> None:
    for name in ("ORDERS", "SHIPMENTS", "PATIENTS"):
        assert not looks_singular_noun(name), name


def test_the_exception_tokens_are_matched_per_token_not_whole_name() -> None:
    """Tokenizing is the whole difference between the two implementations."""
    assert looks_singular_noun("TV_SERIES")
    assert looks_singular_noun("BREAKING_NEWS")
