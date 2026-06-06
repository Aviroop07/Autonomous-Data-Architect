"""Unit tests for Stage 3 distribution models.

Deterministic, offline. Covers NumericRange, NormalDist, LogNormalDist,
CategoricalDist and UnivariateDist._validate against a schema fixture.
"""
from __future__ import annotations

import pytest

from src.pipeline.stage3.models.distributions import (
    NumericRange,
    NormalDist,
    LogNormalDist,
    PoissonDist,
    ZipfDist,
    CategoricalDist,
    UnivariateDist,
)
from src.tests.fixtures import sample_data


# --------------------------------------------------------------------------- #
# NumericRange
# --------------------------------------------------------------------------- #

def test_numeric_range_valid_min_lt_max():
    r = NumericRange(min=0.0, max=100.0)
    assert r._validate() == []


def test_numeric_range_equal_min_max_is_ok():
    r = NumericRange(min=5.0, max=5.0)
    assert r._validate() == []


def test_numeric_range_min_gt_max_error():
    r = NumericRange(min=10.0, max=1.0)
    errors = r._validate()
    assert len(errors) == 1
    assert "min" in errors[0].lower() and "max" in errors[0].lower()


def test_numeric_range_none_bounds_is_ok():
    r = NumericRange(min=None, max=None)
    assert r._validate() == []


def test_numeric_range_only_min_is_ok():
    assert NumericRange(min=5.0, max=None)._validate() == []


def test_numeric_range_only_max_is_ok():
    assert NumericRange(min=None, max=50.0)._validate() == []


# --------------------------------------------------------------------------- #
# NormalDist
# --------------------------------------------------------------------------- #

def test_normal_dist_valid():
    d = NormalDist(mean=50.0, variance=25.0)
    assert d._validate() == []


def test_normal_dist_zero_variance_ok():
    d = NormalDist(mean=0.0, variance=0.0)
    assert d._validate() == []


def test_normal_dist_negative_variance_error():
    d = NormalDist(mean=50.0, variance=-1.0)
    errors = d._validate()
    assert len(errors) == 1
    assert "variance" in errors[0].lower()


# --------------------------------------------------------------------------- #
# LogNormalDist
# --------------------------------------------------------------------------- #

def test_lognormal_dist_valid():
    d = LogNormalDist(mean=3.5, variance=1.2)
    assert d._validate() == []


def test_lognormal_dist_negative_variance_error():
    d = LogNormalDist(mean=3.5, variance=-0.5)
    errors = d._validate()
    assert len(errors) == 1
    assert "variance" in errors[0].lower()


# --------------------------------------------------------------------------- #
# PoissonDist / ZipfDist (no validation failures expected)
# --------------------------------------------------------------------------- #

def test_poisson_always_valid():
    assert PoissonDist(lam=3.0)._validate() == []


def test_zipf_always_valid():
    assert ZipfDist(a=2.0)._validate() == []


# --------------------------------------------------------------------------- #
# CategoricalDist
# --------------------------------------------------------------------------- #

def test_categorical_valid():
    d = CategoricalDist(
        values={"A", "B", "C"},
        weights={"A": 0.5, "B": 0.3, "C": 0.2},
    )
    assert d._validate() == []


def test_categorical_weights_dont_sum_to_one_error():
    d = CategoricalDist(
        values={"A", "B"},
        weights={"A": 0.6, "B": 0.6},
    )
    errors = d._validate()
    assert any("sum" in e.lower() or "1" in e for e in errors)


def test_categorical_mismatched_keys_error():
    d = CategoricalDist(
        values={"A", "B"},
        weights={"A": 0.5, "C": 0.5},  # "C" not in values, "B" missing
    )
    errors = d._validate()
    assert len(errors) >= 1


# --------------------------------------------------------------------------- #
# UnivariateDist._validate
# --------------------------------------------------------------------------- #

@pytest.fixture
def schema():
    return sample_data.fintech_schema()


def test_univariate_valid(schema):
    ud = UnivariateDist(
        table_name="USER",
        column_name="credit_score",
        distribution=NormalDist(mean=700.0, variance=10000.0),
    )
    assert ud._validate(schema) == []


def test_univariate_invalid_table(schema):
    ud = UnivariateDist(
        table_name="GHOST_TABLE",
        column_name="any_col",
        distribution=NormalDist(mean=0.0, variance=1.0),
    )
    errors = ud._validate(schema)
    assert any("not found" in e for e in errors)


def test_univariate_invalid_column(schema):
    ud = UnivariateDist(
        table_name="USER",
        column_name="nonexistent_col",
        distribution=NormalDist(mean=0.0, variance=1.0),
    )
    errors = ud._validate(schema)
    assert any("not found" in e for e in errors)


def test_univariate_pk_column_rejected(schema):
    ud = UnivariateDist(
        table_name="USER",
        column_name="user_id",  # this is the PK
        distribution=NormalDist(mean=1.0, variance=1.0),
    )
    errors = ud._validate(schema)
    assert any("primary key" in e.lower() or "pk" in e.lower() for e in errors)


def test_univariate_type_mismatch_numeric_dist_on_varchar(schema):
    # earnings_information is FLOAT -- ok; if we used a VARCHAR column with NormalDist
    # We can construct a schema with a VARCHAR column for this
    from src.pipeline.stage2.models.schema import Schema, Table, Column
    test_schema = Schema(tables=[
        Table(
            name="TEST",
            pk="test_id",
            columns=[
                Column(name="test_id", data_type="INTEGER"),
                Column(name="description", data_type="VARCHAR"),
            ],
        )
    ])
    ud = UnivariateDist(
        table_name="TEST",
        column_name="description",
        distribution=NormalDist(mean=0.0, variance=1.0),
    )
    errors = ud._validate(test_schema)
    assert any("type mismatch" in e.lower() for e in errors)
