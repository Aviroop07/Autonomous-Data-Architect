from contextvars import ContextVar, Token
from pathlib import Path
from typing import List, Optional, Sequence

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class LLMMessageTrace(BaseModel):
    sequence: int
    agent_name: str
    output_structure_name: str
    input_messages: List[str] = Field(default_factory=list)
    returned_messages: List[str] = Field(default_factory=list)
    token_usage: int = 0
    parsed_response_type: str = ""


class LLMTraceCollector(BaseModel):
    traces: List[LLMMessageTrace] = Field(default_factory=list)

    def add_trace(
        self,
        agent_name: str,
        output_structure_name: str,
        input_messages: Sequence[BaseMessage],
        returned_messages: Sequence[BaseMessage],
        token_usage: int,
        parsed_response_type: str,
    ) -> None:
        self.traces.append(
            LLMMessageTrace(
                sequence=len(self.traces) + 1,
                agent_name=agent_name,
                output_structure_name=output_structure_name,
                input_messages=[
                    format_trace_message(message) for message in input_messages
                ],
                returned_messages=[
                    format_trace_message(message) for message in returned_messages
                ],
                token_usage=token_usage,
                parsed_response_type=parsed_response_type,
            )
        )

    def write_artifacts(self, output_dir: Path, prefix: str = "llm") -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / f"{prefix}_message_trace.json"
        trace_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

        message_dir = output_dir / f"{prefix}_messages"
        message_dir.mkdir(parents=True, exist_ok=True)
        for trace in self.traces:
            for idx, message in enumerate(trace.input_messages, start=1):
                (
                    message_dir
                    / f"{trace.sequence:02d}_{trace.agent_name}_input_{idx}.txt"
                ).write_text(message, encoding="utf-8")
            for idx, message in enumerate(trace.returned_messages, start=1):
                (
                    message_dir
                    / f"{trace.sequence:02d}_{trace.agent_name}_returned_{idx}.txt"
                ).write_text(message, encoding="utf-8")

        with (output_dir / f"{prefix}_message_trace_summary.tsv").open(
            "w", encoding="utf-8"
        ) as handle:
            handle.write(
                "sequence\tagent\ttoken_usage\toutput_structure\tparsed_response_type\tinput_chars\treturned_chars\n"
            )
            for trace in self.traces:
                input_chars = sum(len(message) for message in trace.input_messages)
                returned_chars = sum(
                    len(message) for message in trace.returned_messages
                )
                handle.write(
                    f"{trace.sequence}\t{trace.agent_name}\t{trace.token_usage}\t"
                    f"{trace.output_structure_name}\t{trace.parsed_response_type}\t"
                    f"{input_chars}\t{returned_chars}\n"
                )


_ACTIVE_TRACE_COLLECTOR: ContextVar[Optional[LLMTraceCollector]] = ContextVar(
    "active_llm_trace_collector",
    default=None,
)


def activate_trace_collector(
    collector: LLMTraceCollector,
) -> Token[Optional[LLMTraceCollector]]:
    return _ACTIVE_TRACE_COLLECTOR.set(collector)


def reset_trace_collector(token: Token[Optional[LLMTraceCollector]]) -> None:
    _ACTIVE_TRACE_COLLECTOR.reset(token)


def get_active_trace_collector() -> Optional[LLMTraceCollector]:
    return _ACTIVE_TRACE_COLLECTOR.get()


def format_trace_message(message: object) -> str:
    if isinstance(message, BaseMessage):
        return f"[{message.__class__.__name__}] {message.content}"
    return str(message)
