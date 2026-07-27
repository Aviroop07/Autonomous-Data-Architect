"""Standalone Relation/Condition/Constraint object model.

Design reference: experiments/RELATION_CONDITION_CONSTRAINT_DESIGN.md
(project root). This package is built and validated in isolation --
nothing in src/pipeline/stage3 imports from it yet, and it does not import
from src/pipeline/stage3 either. It supersedes on_nodes.py/
condition_nodes.py/grain.py eventually, not yet.
"""
