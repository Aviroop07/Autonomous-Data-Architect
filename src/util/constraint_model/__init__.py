"""The Relation/Condition/Constraint object model -- Stage 3's single source
of truth for what a constraint IS.

Design reference: docs/design/RELATION_CONDITION_CONSTRAINT_DESIGN.md, which
the modules here cite by section number.

This package is no longer "built and validated in isolation", as this
docstring used to claim: the Stage 3 extraction models (pipeline/stage3/
models/) now use `condition/` directly instead of carrying their own copy of
the predicate/expression taxonomy -- which is why condition_nodes.py no longer
exists. The `bridge/` subpackage translates only what genuinely still differs:
the ON tree's node shapes, plus the four constraint-level wrappers.
"""
