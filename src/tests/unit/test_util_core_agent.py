"""Tests for src/util/core/agent.py's function_calling schema-explosion
fix (_build_ref_preserving_tool + StructuredAgent's rewired chain).

Root cause: LangChain's with_structured_output(method="function_calling")
converts a Pydantic model via convert_to_openai_tool() -> dereference_refs(),
which breaks true cycles but does NOT memoize repeated non-cyclic $ref
occurrences -- a schema with a recursive type reused across many sibling
fields (this project's RExprUnion/RPredicate, reused throughout
UnifiedExtractionOutput) explodes combinatorially (measured live: 33,585
chars via model_json_schema() -> 21,393,132 chars post-dereference, a 637x
blowup that exceeds every real provider's context window).

Fix: hand-build the tool dict from Pydantic's own $ref-preserving
model_json_schema() and pass it as an already-built dict to bind_tools() --
convert_to_openai_tool() passes a dict tool with type="function" straight
through unchanged, bypassing dereference_refs() entirely. Verified live
against a real OpenAI-compatible endpoint (OpenRouter) that $defs/$ref
inside a tool's parameters is accepted at the wire level.
"""

from __future__ import annotations

import json
from typing import List, Optional
from unittest.mock import MagicMock

from pydantic import BaseModel, Field

from src.pipeline.stage3.models.cross_shard import UnifiedExtractionOutput
from src.util.core import agent as agent_module
from src.util.core.agent import StructuredAgent, _build_ref_preserving_tool, get_agent_


class _Node(BaseModel):
    """A small recursive-and-reused model mirroring the real bug's shape:
    self-referential (children), and reused across 3 sibling fields."""

    name: str
    children: List["_Node"] = Field(default_factory=list)


class _Container(BaseModel):
    left: _Node
    right: _Node
    extra: Optional[_Node] = None


class TestBuildRefPreservingTool:
    def test_shape_matches_openai_tool_envelope(self):
        tool = _build_ref_preserving_tool(_Container)
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "_Container"
        assert "parameters" in tool["function"]

    def test_preserves_defs_and_refs_no_explosion(self):
        tool = _build_ref_preserving_tool(_Container)
        params = tool["function"]["parameters"]
        assert "$defs" in params
        # A recursive+reused model must stay small via $ref, not explode.
        assert len(json.dumps(tool)) < 5_000

    def test_real_unified_extraction_output_stays_small(self):
        """The actual motivating case -- proves the fix on the real,
        previously-exploding model, not just a synthetic example."""
        tool = _build_ref_preserving_tool(UnifiedExtractionOutput)
        size = len(json.dumps(tool))
        # Pydantic's own schema was 33,585 chars; LangChain's dereferenced
        # version was 21,393,132 chars. Should be close to the former.
        assert size < 100_000

    def test_no_title_or_description_leak_into_parameters(self):
        tool = _build_ref_preserving_tool(_Container)
        params = tool["function"]["parameters"]
        assert "title" not in params


class TestStructuredAgentFunctionCallingBypassesLangChainConversion:
    def test_bind_tools_receives_our_dict_not_the_pydantic_class(self):
        mock_llm = MagicMock()
        mock_bound = MagicMock()
        mock_llm.bind_tools.return_value = mock_bound

        StructuredAgent(
            system_prompt="test",
            llm=mock_llm,
            output_structure=_Container,
            method="function_calling",
        )

        assert mock_llm.bind_tools.called
        (tools_arg,), kwargs = mock_llm.bind_tools.call_args
        assert len(tools_arg) == 1
        assert isinstance(tools_arg[0], dict)
        assert tools_arg[0]["type"] == "function"
        assert tools_arg[0]["function"]["name"] == "_Container"
        assert kwargs.get("tool_choice") == "_Container"
        assert kwargs.get("parallel_tool_calls") is False
        # with_structured_output must NEVER be called for function_calling --
        # that's the whole point, it's what triggers the dereference bug.
        mock_llm.with_structured_output.assert_not_called()

    def test_json_mode_is_unchanged_regression_guard(self):
        mock_llm = MagicMock()
        mock_chain = MagicMock()
        mock_llm.with_structured_output.return_value = mock_chain

        agent = StructuredAgent(
            system_prompt="test",
            llm=mock_llm,
            output_structure=_Container,
            method="json_mode",
        )

        mock_llm.with_structured_output.assert_called_once_with(
            _Container, include_raw=True, method="json_mode"
        )
        mock_llm.bind_tools.assert_not_called()
        assert agent.chain is mock_chain


class TestGetAgentOutputFormatIsConditional:
    """The ## OUTPUT FORMAT prose block is only needed for json_mode (no
    server-side schema enforcement there) -- function_calling providers
    already get the real schema via the tool definition, so appending the
    same structure again as prose is pure duplication."""

    def _patch_provider(self, monkeypatch, provider: str):
        monkeypatch.setattr(
            agent_module,
            "_detect_provider",
            lambda: (provider, "fake-key", None, "fake-model"),
        )
        mock_llm = MagicMock()
        monkeypatch.setattr(agent_module, "_build_llm", lambda *a, **k: mock_llm)
        return mock_llm

    def test_json_mode_provider_includes_output_format(self, monkeypatch):
        self._patch_provider(monkeypatch, "deepseek")
        agent = get_agent_(
            system_prompt="base prompt",
            output_structure=_Container,
            name="test",
        )
        assert "## OUTPUT FORMAT" in agent.system_prompt
        assert "base prompt" in agent.system_prompt

    def test_function_calling_provider_excludes_output_format(self, monkeypatch):
        self._patch_provider(monkeypatch, "openai")
        agent = get_agent_(
            system_prompt="base prompt",
            output_structure=_Container,
            name="test",
        )
        assert "## OUTPUT FORMAT" not in agent.system_prompt
        assert agent.system_prompt == "base prompt"
