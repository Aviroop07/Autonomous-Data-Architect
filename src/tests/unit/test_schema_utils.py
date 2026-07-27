"""Tests for src/util/schema_ops/schema_utils.py's
generate_hierarchical_schema_description(), the utility get_agent_() uses
to build every agent's OUTPUT FORMAT prompt section.

Covers 3 real bugs found via a live Stage 3 run against DeepSeek (which
uses json_mode, with no server-side schema enforcement, unlike OpenAI's
function_calling -- the first time these bugs actually surfaced):
  1. A discriminated union's tag field (Literal["table"], etc.) rendered
     as a bare "string"/"any" -- the LLM had no way to know the required
     value and used the class name instead (e.g. "ONBaseTable" instead of
     "table"), which Pydantic then rejected as an invalid discriminator tag.
  2. A self-/mutually-referential model (RArithmetic nesting itself via
     left/right) rendered as an empty section -- Pydantic hoists such
     models into $defs with a bare top-level {"$ref": ...} pointer that
     the function didn't resolve.
  3. Fixing bug 2 without deduplication caused combinatorial re-expansion
     of repeated union members (confirmed: ~230k tokens for one real
     model) -- fixed via a shared "already described" set so each distinct
     model class is expanded in full exactly once.
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from src.util.schema_ops.schema_utils import generate_hierarchical_schema_description


class _Leaf(BaseModel):
    kind: Literal["leaf"] = "leaf"
    value: int = Field(description="A leaf value.")


class _Branch(BaseModel):
    kind: Literal["branch"] = "branch"
    left: "_NodeUnion" = Field(description="Left child.")
    right: "_NodeUnion" = Field(description="Right child.")


# Annotated + Field(discriminator=...) matches this project's real
# ON-tree/R-AST union convention (on_nodes.py/condition_nodes.py) --
# a plain Union[...] with no discriminator produces a different
# (non-tagged) "anyOf" schema shape that the discriminator-mapping fix
# doesn't apply to at all.
_NodeUnion = Annotated[Union[_Leaf, _Branch], Field(discriminator="kind")]
_Branch.model_rebuild()


class _Tree(BaseModel):
    root: _NodeUnion = Field(description="The tree root.")
    siblings: List[_NodeUnion] = Field(
        default_factory=list, description="More nodes at the top level."
    )


class _Simple(BaseModel):
    name: str = Field(description="A plain string field.")
    count: Optional[int] = Field(default=None, description="An optional int.")


class TestDiscriminatorTagRendering:
    def test_single_literal_tag_field_shows_the_actual_value(self):
        desc = generate_hierarchical_schema_description(_Leaf)
        assert "Literal['leaf']" in desc
        assert "ONBaseTable" not in desc  # sanity: no bleed from other tests

    def test_discriminated_union_field_shows_tag_to_class_mapping(self):
        desc = generate_hierarchical_schema_description(_Tree)
        assert "{kind: 'leaf'} -> _Leaf" in desc
        assert "{kind: 'branch'} -> _Branch" in desc

    def test_plain_fields_render_without_regression(self):
        desc = generate_hierarchical_schema_description(_Simple)
        assert "**name**" in desc
        assert "**count**" in desc


class TestSelfReferentialModel:
    def test_self_referential_union_member_is_not_empty(self):
        desc = generate_hierarchical_schema_description(_Branch)
        assert "**left**" in desc
        assert "**right**" in desc
        assert "Literal['branch']" in desc

    def test_self_referential_union_member_does_not_infinite_loop(self):
        # Must terminate at all -- a regression here would hang or raise
        # RecursionError instead of failing an assertion.
        desc = generate_hierarchical_schema_description(_Tree)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_cycle_is_reported_as_a_pointer_not_re_expanded(self):
        desc = generate_hierarchical_schema_description(_Branch)
        assert "already fully described" in desc


class TestDeduplicationKeepsOutputBounded:
    def test_repeated_non_cyclic_union_member_expanded_only_once(self):
        desc = generate_hierarchical_schema_description(_Tree)
        # _Leaf appears as a union member of both `root` and `siblings`,
        # and again nested inside every _Branch -- it must be fully
        # expanded (showing its own fields) exactly once.
        full_leaf_expansions = desc.count("A leaf value.")
        assert full_leaf_expansions == 1

    def test_output_size_stays_small_for_a_moderately_deep_recursive_schema(self):
        desc = generate_hierarchical_schema_description(_Tree)
        # Without dedup this would blow up combinatorially with tree depth;
        # bounded confirms the fix actually caps re-expansion.
        assert len(desc) < 5000
