from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.pipeline.stage3.models.sql_models import BinaryOperator, OperandValue


ExpressionKind = Literal[
    "column",
    "literal",
    "add",
    "sub",
    "mul",
    "div",
    "case_when",
    "min",
    "max",
    "unsupported",
]

SolverTier = Literal[
    "linear",
    "piecewise_linear",
    "big_m_gate",
    "groundable_product",
    "nonlinear_product",
    "unsupported",
]


class ExpressionNode(BaseModel):
    kind: ExpressionKind = Field(description="Expression node kind.")
    text: str = Field(description="Original normalized expression text for this node.")
    table_name: Optional[str] = Field(default=None, description="Optional table qualifier for column refs.")
    column_name: Optional[str] = Field(default=None, description="Column name for column refs.")
    literal_value: Optional[OperandValue] = Field(default=None, description="Literal value for literal nodes.")
    left: Optional[ExpressionNode] = Field(default=None, description="Left child for binary arithmetic nodes.")
    right: Optional[ExpressionNode] = Field(default=None, description="Right child for binary arithmetic nodes.")
    arguments: List[ExpressionNode] = Field(default_factory=list, description="Function arguments for MIN/MAX nodes.")
    case_branches: List[CaseBranch] = Field(default_factory=list, description="CASE WHEN branches.")
    else_result: Optional[ExpressionNode] = Field(default=None, description="CASE ELSE result expression.")


class PredicateNode(BaseModel):
    left: ExpressionNode = Field(description="Left side of a predicate.")
    operator: BinaryOperator = Field(description="Predicate operator.")
    right: ExpressionNode = Field(description="Right side of a predicate.")


class CaseBranch(BaseModel):
    condition: PredicateNode = Field(description="CASE branch condition.")
    result: ExpressionNode = Field(description="CASE branch result expression.")


class ExpressionClassification(BaseModel):
    expression: ExpressionNode = Field(description="Parsed expression tree.")
    solver_tier: SolverTier = Field(description="Lowest currently required solver tier.")
    variable_factor_count: int = Field(description="Number of non-literal factors in multiplicative chains.")
    features: List[str] = Field(default_factory=list, description="Detected expression features.")
    unsupported_reasons: List[str] = Field(default_factory=list, description="Reasons this expression exceeds supported v1 solving.")


ExpressionNode.model_rebuild()
PredicateNode.model_rebuild()
CaseBranch.model_rebuild()
