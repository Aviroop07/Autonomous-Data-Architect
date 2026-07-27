"""The project's identifier-casing rules, in one place.

ScribbleDB has exactly two identifier conventions (see CLAUDE.md's naming
table): table names are UPPER_SNAKE_CASE and singular; column names, aliases
and derived-column names are lower_snake_case. Both rules were previously
re-implemented as a private regex pair in four separate modules
(pipeline/stage2/models/schema.py, pipeline/stage3/models/on_nodes.py,
util/constraint_model/relation/nodes.py, util/constraint_model/condition/
cohesive.py), which meant a change to what counts as a legal identifier had
four places to miss.

`is_valid_alias` is deliberately its own name even though it is currently
identical to `is_lower_snake`: an alias and a column are different things that
happen to share a rule today, and callers reading `is_valid_alias(x)` should
not have to know that.
"""

from __future__ import annotations

import re

UPPER_SNAKE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
LOWER_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def is_upper_snake(name: str) -> bool:
    """True for a legal table name: UPPER_SNAKE_CASE, leading letter."""
    return bool(UPPER_SNAKE_RE.fullmatch(name))


def is_lower_snake(name: str) -> bool:
    """True for a legal column name: lower_snake_case, leading letter."""
    return bool(LOWER_SNAKE_RE.fullmatch(name))


def is_valid_alias(name: str) -> bool:
    """True for a legal relation/aggregate alias (same rule as a column)."""
    return bool(LOWER_SNAKE_RE.fullmatch(name))
