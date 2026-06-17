from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from src.pipeline.stage1.agents.context_enricher.agent import (
    PROMPT_PATH,
    build_context_enricher_agent,
    get_context_enricher_tools,
)
from src.pipeline.stage1.models.rephrased_nl import FactList


class DummyAgent:
    async def ainvoke(self, input_payload: object) -> object:
        return input_payload


class CapturedFactoryArgs(BaseModel):
    system_prompt: str
    output_structure: type
    tools: list[object]
    model: Optional[str]
    name: str
    use_responses_api: bool


def test_context_enricher_exposes_openai_web_search_tool():
    tools = get_context_enricher_tools()
    assert tools == [{"type": "web_search_preview", "search_context_size": "medium"}]


def test_context_enricher_agent_requests_web_search_and_responses_api():
    captured: list[CapturedFactoryArgs] = []

    def fake_agent_factory(
        system_prompt: str,
        output_structure: type,
        tools: list[object],
        model: Optional[str],
        name: str,
        use_responses_api: bool,
    ) -> DummyAgent:
        captured.append(CapturedFactoryArgs(
            system_prompt=system_prompt,
            output_structure=output_structure,
            tools=tools,
            model=model,
            name=name,
            use_responses_api=use_responses_api,
        ))
        return DummyAgent()

    agent = build_context_enricher_agent(
        system_prompt="test prompt",
        model="gpt-4o-test",
        agent_factory=fake_agent_factory,
    )

    assert isinstance(agent, DummyAgent)
    assert len(captured) == 1
    assert captured[0].system_prompt == "test prompt"
    assert captured[0].output_structure is FactList
    assert captured[0].tools == get_context_enricher_tools()
    assert captured[0].model == "gpt-4o-test"
    assert captured[0].name == "domain_specialist"
    assert captured[0].use_responses_api is True


def test_context_enricher_prompt_requires_web_retrieval():
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    assert "Use web retrieval" in prompt
    assert "web search tool" in prompt
    assert "high-value external facts" in prompt


def test_context_enricher_prompt_bans_generic_schema_advice():
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    assert "Do not emit facts starting with \"Schema Guideline:\"" in prompt
    assert "use primary keys" in prompt
    assert "use foreign keys" in prompt
    assert "normalize tables" in prompt
    assert "Do not repeat anything already explicitly stated" in prompt


def test_context_enricher_prompt_says_knobs_are_deterministic():
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    assert "The deterministic compiler emits the knob set" in prompt
