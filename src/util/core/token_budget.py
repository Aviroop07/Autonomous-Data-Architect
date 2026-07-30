"""The token-budget assumptions shared by every module that sizes a prompt.

These three numbers were duplicated across three modules -- budget_chunker.py,
sharding_ilp.py and orchestration/stage1/entry.py -- and their agreement was
maintained by COMMENT rather than by code. budget_chunker.py said of its divisor:

    "The same crude divisor sharding_ilp.py uses for its own budget arithmetic
    -- deliberately consistent with it rather than independently 'better', so
    the two budget calculations cannot silently disagree."

That is the right intent and the wrong mechanism: nothing stopped one copy from
being tuned. Two stages computing a budget from silently different constants is
the kind of divergence this project has already paid for elsewhere (the shadowed
looks_singular_noun, which disagreed with its canonical twin on half the names
tested).

All three are MODEL-DEPENDENT -- the divisor is a property of the tokenizer, the
overhead of the prompt as that tokenizer counts it -- so they belong in one place
for the same reason they belong in
docs/design/MODEL_DEPENDENT_CONSTANTS.md: a model swap should mean editing one
value, not hunting three.
"""

from __future__ import annotations

# Characters per token. Crude on purpose: an exact count would need the target
# model's own tokenizer, and every consumer here is sizing a budget with a wide
# safety margin rather than counting to the limit.
CHARS_PER_TOKEN = 4.0

# What a prompt costs before any payload: system prompt, schema-so-far, and the
# output-format block.
DEFAULT_PROMPT_OVERHEAD_TOKENS = 6000

# Fraction of a context window treated as usable. Leaves room for the response
# and for the divisor above being approximate.
DEFAULT_CONTEXT_SAFETY_MARGIN = 0.6


def estimate_tokens(text: str) -> int:
    """Rough token count for a string, by the shared divisor."""
    return int(len(text) / CHARS_PER_TOKEN)
