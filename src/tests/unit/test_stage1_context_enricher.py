from __future__ import annotations

from typing import Any, Dict

from src.pipeline.stage1.agents.context_enricher.agent import (
    PROMPT_PATH,
    build_context_enricher_agent,
    get_context_enricher_tools,
)
from src.pipeline.stage1.models.rephrased_nl import FactList


class DummyAgent:
    async def ainvoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return input_dict


def test_context_enricher_exposes_openai_web_search_tool():
    tools = get_context_enricher_tools()
    assert tools == [{"type": "web_search_preview", "search_context_size": "medium"}]


def test_context_enricher_agent_requests_web_search_and_responses_api():
    captured: Dict[str, Any] = {}

    def fake_agent_factory(**kwargs: Any) -> DummyAgent:
        captured.update(kwargs)
        return DummyAgent()

    agent = build_context_enricher_agent(
        system_prompt="test prompt",
        model="gpt-4o-test",
        agent_factory=fake_agent_factory,
    )

    assert isinstance(agent, DummyAgent)
    assert captured["system_prompt"] == "test prompt"
    assert captured["output_structure"] is FactList
    assert captured["tools"] == get_context_enricher_tools()
    assert captured["model"] == "gpt-4o-test"
    assert captured["name"] == "domain_specialist"
    assert captured["use_responses_api"] is True


def test_context_enricher_prompt_requires_web_retrieval():
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    assert "Use web retrieval" in prompt
    assert "web search tool" in prompt
    assert "grounding" in prompt
