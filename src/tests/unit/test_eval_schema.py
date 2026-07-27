"""Unit tests for schema-level evaluation helpers.

Deterministic parts only -- get_lcs_length and calculate_f1. The full
SchemaEvaluator requires sentence_transformers (heavy ML model download);
those tests are skipped unless running in an environment that has it.

The LCS and F1 functions are pure Python and require no ML or network.
"""

from __future__ import annotations

import pytest

# Skip the entire module if sentence_transformers is not installed.
pytest.importorskip("sentence_transformers")
pytest.importorskip("nltk")

from src.evaluation.schema_level.schema_eval import SchemaEvaluator, coarsen_dt  # noqa: E402


@pytest.fixture(scope="module")
def evaluator():
    return SchemaEvaluator()


# --------------------------------------------------------------------------- #
# coarsen_dt
# --------------------------------------------------------------------------- #


def test_coarsen_int_variants():
    assert coarsen_dt("INT") == "NUMERIC"
    assert coarsen_dt("INTEGER") == "NUMERIC"
    assert coarsen_dt("BIGINT") == "NUMERIC"
    assert coarsen_dt("FLOAT") == "NUMERIC"
    assert coarsen_dt("DECIMAL") == "NUMERIC"


def test_coarsen_text_variants():
    assert coarsen_dt("VARCHAR") == "TEXT"
    assert coarsen_dt("TEXT") == "TEXT"
    assert coarsen_dt("CHAR") == "TEXT"


def test_coarsen_datetime_variants():
    assert coarsen_dt("DATE") == "DATETIME"
    assert coarsen_dt("TIMESTAMP") == "DATETIME"
    assert coarsen_dt("DATETIME") == "DATETIME"


def test_coarsen_bool():
    assert coarsen_dt("BOOLEAN") == "BOOL"
    assert coarsen_dt("BOOL") == "BOOL"


def test_coarsen_unknown_defaults_to_text():
    assert coarsen_dt("JSONB") == "TEXT"


def test_coarsen_none_defaults_to_text():
    assert coarsen_dt(None) == "TEXT"


def test_coarsen_strips_length_modifier():
    assert coarsen_dt("VARCHAR(255)") == "TEXT"
    assert coarsen_dt("DECIMAL(10,2)") == "NUMERIC"


def test_coarsen_case_insensitive():
    assert coarsen_dt("varchar") == "TEXT"
    assert coarsen_dt("Integer") == "NUMERIC"


# --------------------------------------------------------------------------- #
# get_lcs_length (pure Python DP)
# --------------------------------------------------------------------------- #


def test_lcs_empty_strings(evaluator):
    assert evaluator.get_lcs_length("", "abc") == 0
    assert evaluator.get_lcs_length("abc", "") == 0
    assert evaluator.get_lcs_length("", "") == 0


def test_lcs_identical_strings(evaluator):
    s = "customer"
    assert evaluator.get_lcs_length(s, s) == len(s)


def test_lcs_no_common_substring(evaluator):
    assert evaluator.get_lcs_length("abc", "xyz") == 0


def test_lcs_partial_overlap(evaluator):
    # "credit" and "credit_score" share "credit" (length 6)
    assert evaluator.get_lcs_length("credit", "credit_score") == 6


def test_lcs_is_longest_common_substring_not_subsequence(evaluator):
    # "abcde" vs "ace" -- LCS *substring* is at most 1 (no 2-char run of consecutive chars in common)
    # Actually "a", "c", "e" are each len-1 matches. Max is 1.
    result = evaluator.get_lcs_length("abcde", "ace")
    assert result == 1


# --------------------------------------------------------------------------- #
# calculate_f1
# --------------------------------------------------------------------------- #


def test_f1_both_empty_returns_one(evaluator):
    f1, acc = evaluator.calculate_f1(set(), set(), lambda p, g: p == g)
    assert f1 == 1.0
    assert acc == 1.0


def test_f1_pred_empty_returns_zero(evaluator):
    f1, acc = evaluator.calculate_f1(set(), {"a", "b"}, lambda p, g: p == g)
    assert f1 == 0.0
    assert acc == 0.0


def test_f1_gt_empty_returns_zero(evaluator):
    f1, acc = evaluator.calculate_f1({"a"}, set(), lambda p, g: p == g)
    assert f1 == 0.0
    assert acc == 0.0


def test_f1_perfect_match_returns_one(evaluator):
    items = {"customer", "order", "product"}
    f1, acc = evaluator.calculate_f1(items, items, lambda p, g: p == g)
    assert f1 == 1.0
    assert acc == 1.0


def test_f1_no_match_returns_zero(evaluator):
    f1, acc = evaluator.calculate_f1({"a", "b"}, {"c", "d"}, lambda p, g: p == g)
    assert f1 == 0.0


def test_f1_half_match_computes_correctly(evaluator):
    pred = {"customer", "order"}
    gt = {"customer", "product"}
    f1, _ = evaluator.calculate_f1(pred, gt, lambda p, g: p == g)
    # precision = 1/2, recall = 1/2 -> f1 = 0.5
    assert f1 == pytest.approx(0.5)


def test_f1_acc_is_one_only_when_f1_is_one(evaluator):
    pred = {"a", "b"}
    gt = {"a", "b", "c"}
    f1, acc = evaluator.calculate_f1(pred, gt, lambda p, g: p == g)
    assert f1 < 1.0
    assert acc == 0.0


# --------------------------------------------------------------------------- #
# max_bipartite_matching -- determinism and optimality
#
# calculate_f1 used to scan greedily for the FIRST match while iterating Python
# sets. That was order-dependent, and because str hashing is randomised per
# process the reported F1 changed between runs on identical input: measured at
# 0.50 on 9 of 20 PYTHONHASHSEED values and 1.00 on the other 11.
# --------------------------------------------------------------------------- #


# `interest_rate` matches both ground-truth names; `rate` matches only one of
# them. Greedy strands `rate` whenever `interest_rate` is visited first. These
# are real names from dataset/handcrafted with the real matcher's verdicts.
_BLOCKING = {
    ("interest_rate", "interest_rate"),
    ("interest_rate", "loan_amount"),
    ("rate", "interest_rate"),
}


def _blocking_match(p, g):
    return (p, g) in _BLOCKING


def test_matching_is_independent_of_input_order():
    import itertools

    from src.evaluation.schema_level.schema_eval import max_bipartite_matching

    sizes = set()
    for pred in itertools.permutations(["interest_rate", "rate"]):
        for gt in itertools.permutations(["interest_rate", "loan_amount"]):
            sizes.add(len(max_bipartite_matching(pred, gt, _blocking_match)))
    assert sizes == {2}, f"order-dependent: got {sizes}"


def test_matching_finds_the_maximum_not_merely_a_maximal_matching():
    """Greedy returns 1 here whenever it visits `interest_rate` first; the
    maximum matching is 2."""
    from src.evaluation.schema_level.schema_eval import max_bipartite_matching

    m = max_bipartite_matching(
        ["interest_rate", "rate"], ["interest_rate", "loan_amount"], _blocking_match
    )
    assert len(m) == 2


def test_f1_is_order_independent_through_calculate_f1(evaluator):
    f1_a, _ = evaluator.calculate_f1(
        {"interest_rate", "rate"}, {"interest_rate", "loan_amount"}, _blocking_match
    )
    f1_b, _ = evaluator.calculate_f1(
        {"rate", "interest_rate"}, {"loan_amount", "interest_rate"}, _blocking_match
    )
    assert f1_a == f1_b == pytest.approx(1.0)


def test_matching_handles_thousands_of_names_without_recursion_error():
    """The textbook recursive Kuhn's would blow the stack well before this.
    A chain where each pred matches its own gt and the next one is the worst
    case for augmenting-path depth."""
    from src.evaluation.schema_level.schema_eval import max_bipartite_matching

    n = 3000
    names = [f"c{i:05d}" for i in range(n)]
    allowed = {(names[i], names[i]) for i in range(n)}
    allowed |= {(names[i], names[i + 1]) for i in range(n - 1)}
    m = max_bipartite_matching(names, names, lambda p, g: (p, g) in allowed)
    assert len(m) == n


def test_matching_degenerate_ends():
    from src.evaluation.schema_level.schema_eval import max_bipartite_matching

    eq = lambda p, g: p == g  # noqa: E731
    assert max_bipartite_matching([], [], eq) == {}
    assert max_bipartite_matching([], ["a"], eq) == {}
    assert max_bipartite_matching(["a"], [], eq) == {}
    assert max_bipartite_matching(["a"], ["a"], eq) == {"a": "a"}


def test_no_pred_name_is_matched_to_two_gt_names():
    """A matching, not a covering: one prediction cannot satisfy two
    ground-truth columns even when it is similar to both."""
    from src.evaluation.schema_level.schema_eval import max_bipartite_matching

    m = max_bipartite_matching(
        ["interest_rate"], ["interest_rate", "loan_amount"], _blocking_match
    )
    assert len(m) == 1
    assert len(set(m.values())) == 1
