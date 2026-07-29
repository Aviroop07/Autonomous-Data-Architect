"""ScribbleDB -- deep, heuristic audit of a benchmark cases file.

Separate from validate_dataset.py on purpose. That script is a HARD GATE: every
check is exact, and a failure means the case is definitely malformed. This one
asks the fuzzier questions a gate cannot -- is the ground truth internally
consistent, is it actually entailed by the prose, do the cases look like they
were written by different people -- and reports SUSPICIONS. A finding here needs
a human to judge; some will be false positives by design.

Both are needed. A case can satisfy every structural rule and still be a bad
benchmark case: a distribution that contradicts its own range constraint, a
lognormal authored in linear space, a table nobody mentions in the prose, or
fifteen descriptions that open with the same sentence.

Usage
-----
  python audit_dataset.py
  python audit_dataset.py --cases path/to.jsonl
  python audit_dataset.py --only contradictions   # one section
  python audit_dataset.py --verbose               # every finding, not a sample
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).parent
DEFAULT_CASES = PROJECT_ROOT / "dataset" / "handcrafted" / "cases.jsonl"

NUMERIC_TYPES = {"INTEGER", "FLOAT", "DECIMAL"}
COUNT_FAMILIES = {"poisson", "zipf"}

# A log-space mean outside this band almost certainly means the author wrote a
# linear-space value: exp(25) is already 7.2e10, and exp(-15) is 3e-7.
LOGNORMAL_MEAN_BAND = (-15.0, 25.0)


class Report:
    def __init__(self) -> None:
        self.sections: Dict[str, List[str]] = defaultdict(list)

    def add(self, section: str, message: str) -> None:
        self.sections[section].append(message)

    def total(self) -> int:
        return sum(len(v) for v in self.sections.values())


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        sys.exit(f"{path} does not exist")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def schema_index(
    case: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any], Set[str]]:
    """Return ({table: {col: type}}, {table: pk-as-list}, {"TABLE.col" of FKs})."""
    tables: Dict[str, Dict[str, str]] = {}
    pks: Dict[str, Any] = {}
    fk_cols: Set[str] = set()
    schema = case.get("ground_truth_schema") or {}
    for t in schema.get("tables") or []:
        name = t.get("name")
        if not isinstance(name, str):
            continue
        cols: Dict[str, str] = {}
        for c in t.get("columns") or []:
            if not isinstance(c, dict):
                continue
            cname = c.get("name")
            if not isinstance(cname, str):
                continue
            ctype = c.get("data_type")
            cols[cname] = ctype if isinstance(ctype, str) else ""
        tables[name] = cols
        pk = t.get("pk")
        pks[name] = pk if isinstance(pk, list) else [pk]
    for r in schema.get("relationships") or []:
        if isinstance(r, dict):
            fk_cols.add(f"{r.get('referencing_table')}.{r.get('referencing_column')}")
    return tables, pks, fk_cols


def family_of(spec: Any) -> Optional[str]:
    if not isinstance(spec, dict):
        return None
    fam = spec.get("distribution", spec.get("family"))
    return fam if isinstance(fam, str) else None


def dist_support(spec: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Plausible (low, high) value range implied by a distribution, or None."""
    fam = family_of(spec)
    p = spec.get("params") or {}
    try:
        if fam == "uniform":
            return float(p["min"]), float(p["max"])
        if fam == "normal":
            m, s = float(p["mean"]), float(p["std"])
            return m - 3.0 * s, m + 3.0 * s
        if fam == "lognormal":
            m, v = float(p["mean"]), float(p["variance"])
            s = math.sqrt(v)
            return math.exp(m - 3.0 * s), math.exp(m + 3.0 * s)
        if fam == "poisson":
            lam = float(p["lambda"])
            return 0.0, lam + 4.0 * math.sqrt(max(lam, 1e-9))
        if fam == "exponential":
            lam = float(p["lambda"])
            return 0.0, 6.0 / max(lam, 1e-9)
        if fam == "categorical":
            w = p.get("weights") or {}
            nums = []
            for k in w:
                try:
                    nums.append(float(k))
                except TypeError, ValueError:
                    return None
            return (min(nums), max(nums)) if nums else None
    except KeyError, TypeError, ValueError:
        return None
    return None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_distribution_sanity(case: Dict[str, Any], rep: Report) -> None:
    cid = case.get("id")
    tables, pks, fk_cols = schema_index(case)
    for key, spec in (case.get("ground_truth_distributions") or {}).items():
        if "." not in key:
            continue
        tname, cname = key.split(".", 1)
        ctype = (tables.get(tname) or {}).get(cname)
        fam = family_of(spec)
        p = (spec or {}).get("params") or {}

        if cname in (pks.get(tname) or []):
            rep.add(
                "distribution-on-key",
                f"case {cid}: {key} is the PRIMARY KEY of {tname}; a marginal "
                "distribution over unique identifiers is not meaningful",
            )
        if key in fk_cols and fam not in ("zipf", "categorical"):
            # zipf/categorical over FK values is legitimate: it models WHICH
            # parent row gets referenced (product popularity, merchant mix).
            # Any other family on a key column is a different matter.
            rep.add(
                "distribution-on-key",
                f"case {cid}: {key} is a FOREIGN KEY carrying a {fam} distribution; "
                "a non-skew family over key values is suspect",
            )

        if fam == "lognormal":
            try:
                m = float(p["mean"])
                if not (LOGNORMAL_MEAN_BAND[0] <= m <= LOGNORMAL_MEAN_BAND[1]):
                    rep.add(
                        "lognormal-linear-space",
                        f"case {cid}: {key} lognormal mean={m:g} is outside the "
                        f"plausible LOG-space band {LOGNORMAL_MEAN_BAND}; this looks "
                        f"like a linear-space value (median would be exp({m:g}))",
                    )
            except KeyError, TypeError, ValueError:
                pass

        if fam in COUNT_FAMILIES and ctype and ctype not in NUMERIC_TYPES:
            rep.add(
                "family-type-mismatch",
                f"case {cid}: {key} is {fam} (a count distribution) but the column "
                f"is {ctype}",
            )

        if fam == "categorical" and ctype == "BOOLEAN":
            n = len((p.get("weights") or {}))
            if n > 2:
                rep.add(
                    "family-type-mismatch",
                    f"case {cid}: {key} is BOOLEAN but its categorical has {n} categories",
                )


def check_contradictions(case: Dict[str, Any], rep: Report) -> None:
    """Distribution support versus range constraints on the same column."""
    cid = case.get("id")
    dists = case.get("ground_truth_distributions") or {}

    # Collect unconditional range constraints per TABLE.column.
    ranges: Dict[str, List[Tuple[Optional[float], Optional[float]]]] = defaultdict(list)
    for con in case.get("ground_truth_constraints") or []:
        if not isinstance(con, dict) or con.get("type") != "range":
            continue
        if con.get("condition") is not None:
            continue  # conditional bounds need not hold marginally
        key = f"{con.get('table')}.{con.get('column')}"
        lo = con.get("min")
        hi = con.get("max")
        try:
            ranges[key].append(
                (None if lo is None else float(lo), None if hi is None else float(hi))
            )
        except TypeError, ValueError:
            continue

    for key, bounds in ranges.items():
        # Two unconditional ranges on one column that cannot both hold.
        for i in range(len(bounds)):
            for j in range(i + 1, len(bounds)):
                lo1, hi1 = bounds[i]
                lo2, hi2 = bounds[j]
                lo = (
                    max(x for x in (lo1, lo2) if x is not None)
                    if (lo1 or lo2)
                    else None
                )
                hi = (
                    min(x for x in (hi1, hi2) if x is not None)
                    if (hi1 or hi2)
                    else None
                )
                if lo is not None and hi is not None and lo > hi:
                    rep.add(
                        "contradictions",
                        f"case {cid}: {key} has two unconditional ranges that cannot "
                        f"both hold: [{lo1},{hi1}] and [{lo2},{hi2}]",
                    )

        spec = dists.get(key)
        if not isinstance(spec, dict):
            continue
        support = dist_support(spec)
        if support is None:
            continue
        d_lo, d_hi = support
        for lo, hi in bounds:
            if hi is not None and d_lo > hi:
                rep.add(
                    "contradictions",
                    f"case {cid}: {key} {family_of(spec)} sits mostly above its own "
                    f"range max ({hi}); plausible support starts near {d_lo:.3g}",
                )
            if lo is not None and d_hi < lo:
                rep.add(
                    "contradictions",
                    f"case {cid}: {key} {family_of(spec)} sits mostly below its own "
                    f"range min ({lo}); plausible support ends near {d_hi:.3g}",
                )
            # A distribution far wider than its constraint means most sampled
            # values would be rejected -- worth a look, not necessarily wrong.
            if lo is not None and hi is not None and hi > lo:
                width = hi - lo
                if d_hi - d_lo > 8.0 * width:
                    rep.add(
                        "distribution-vs-range-width",
                        f"case {cid}: {key} {family_of(spec)} spans "
                        f"~{d_hi - d_lo:.3g} but its range allows only {width:.3g}",
                    )


_WORD = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Crude singularisation, enough to match a table name against prose.

    Table names are mandated SINGULAR while prose naturally says the plural, so
    comparing raw tokens reported almost every table as unmentioned -- 87 false
    positives on the first run, including PATIENT in a clinic case whose text is
    full of "patients". Not lemmatisation, just the three endings that matter.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> Set[str]:
    raw = _WORD.findall(text.lower())
    return set(raw) | {_stem(w) for w in raw}


def check_entailment_proxy(case: Dict[str, Any], rep: Report) -> None:
    """Does the prose even mention the things the ground truth claims?

    A proxy, not a proof: a table can be legitimately implied without its name
    appearing. But a table whose every name token is absent from the prose is
    worth reading by hand.
    """
    cid = case.get("id")
    nl = case.get("nl_description") or ""
    toks = _tokens(nl)
    tables, _pks, _fks = schema_index(case)

    unmentioned = []
    for tname in tables:
        parts = [_stem(p) for p in tname.lower().split("_") if len(p) > 2]
        if parts and not any(p in toks for p in parts):
            unmentioned.append(tname)
    if unmentioned:
        rep.add(
            "unmentioned-tables",
            f"case {cid}: no token of these table names appears in the prose: "
            f"{', '.join(sorted(unmentioned))}",
        )

    # Columns carrying a distribution are always a claim about the text.
    for key in case.get("ground_truth_distributions") or {}:
        if "." not in key:
            continue
        _t, cname = key.split(".", 1)
        parts = [_stem(p) for p in cname.lower().split("_") if len(p) > 3]
        if parts and not any(p in toks for p in parts):
            rep.add(
                "unmentioned-dist-columns",
                f"case {cid}: {key} carries a distribution but no token of the "
                "column name appears in the prose",
            )


def check_prose_diversity(cases: List[Dict[str, Any]], rep: Report) -> None:
    """Fifteen authors writing to one brief tend to converge on a formula."""
    openings: Counter = Counter()
    for c in cases:
        nl = (c.get("nl_description") or "").strip()
        first = " ".join(_WORD.findall(nl.lower())[:6])
        if first:
            openings[first] += 1
    for phrase, n in openings.most_common():
        if n > 1:
            rep.add(
                "repeated-openings",
                f"{n} cases open with the same six words: {phrase!r}",
            )

    # First-sentence shape: an identical opening verb across many cases.
    first_words: Counter = Counter()
    for c in cases:
        w = _WORD.findall((c.get("nl_description") or "").lower())
        if w:
            first_words[w[0]] += 1
    for word, n in first_words.most_common(5):
        if n >= max(8, len(cases) // 12):
            rep.add(
                "repeated-openings",
                f"{n} of {len(cases)} cases start with the word {word!r}",
            )


def check_cross_column_usage(cases: List[Dict[str, Any]], rep: Report) -> None:
    """How much of the constraint mass actually compares two columns?"""
    literal = cross = arith = 0

    def walk(node: Any) -> None:
        nonlocal literal, cross, arith
        if not isinstance(node, dict):
            return
        if node.get("type") in ("and", "or"):
            for sub in node.get("conditions") or []:
                walk(sub)
            return
        if node.get("type") == "not":
            walk(node.get("condition"))
            return
        if "rhs_column" in node:
            cross += 1
        elif "rhs_expr" in node:
            arith += 1
        elif "value" in node:
            literal += 1

    for c in cases:
        for con in c.get("ground_truth_constraints") or []:
            if not isinstance(con, dict):
                continue
            if con.get("condition") is not None:
                walk(con["condition"])
            if con.get("result") is not None:
                walk(con["result"])
            if con.get("type") == "range":
                literal += 1  # min/max are literal bounds by construction

    total = literal + cross + arith
    if total:
        pct = 100.0 * (cross + arith) / total
        rep.add(
            "cross-column-share",
            f"{cross} column-to-column and {arith} arithmetic comparisons out of "
            f"{total} leaves ({pct:.1f}%); the contract asks for roughly half",
        )


def check_pipeline_stress(cases: List[Dict[str, Any]], rep: Report) -> None:
    """Coverage of pipeline features, not of the JSON schema."""
    sizes = []
    with_composite_pk = 0
    with_self_ref = 0
    max_fk_depth = 0

    for c in cases:
        tables, pks, _fk = schema_index(c)
        sizes.append(len(tables))
        if any(isinstance(p, list) and len(p) > 1 for p in pks.values()):
            with_composite_pk += 1
        rels = (c.get("ground_truth_schema") or {}).get("relationships") or []
        if any(
            isinstance(r, dict)
            and r.get("referencing_table") == r.get("referred_table")
            for r in rels
        ):
            with_self_ref += 1
        # crude longest FK chain
        edges = defaultdict(set)
        for r in rels:
            if isinstance(r, dict):
                edges[r.get("referencing_table")].add(r.get("referred_table"))

        def depth(node: str, seen: Set[str]) -> int:
            if node in seen:
                return 0
            seen = seen | {node}
            return 1 + max((depth(n, seen) for n in edges.get(node, ())), default=0)

        if edges:
            max_fk_depth = max(max_fk_depth, max(depth(n, set()) for n in edges))

    n = len(cases)
    rep.add(
        "pipeline-stress",
        f"tables per case: min={min(sizes)} max={max(sizes)} mean={sum(sizes) / n:.1f}",
    )
    over_18 = sum(1 for s in sizes if s > 18)
    rep.add(
        "pipeline-stress",
        f"cases exceeding max_tables_per_shard=18: {over_18} -- Stage 3's "
        f"cross-shard path is {'exercised' if over_18 else 'NEVER exercised'}",
    )
    rep.add(
        "pipeline-stress", f"cases with a composite primary key: {with_composite_pk}"
    )
    rep.add(
        "pipeline-stress",
        f"cases with a self-referencing relationship: {with_self_ref}",
    )
    rep.add("pipeline-stress", f"longest foreign-key chain anywhere: {max_fk_depth}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

SECTION_ORDER = [
    "contradictions",
    "lognormal-linear-space",
    "distribution-on-key",
    "family-type-mismatch",
    "distribution-vs-range-width",
    "unmentioned-tables",
    "unmentioned-dist-columns",
    "repeated-openings",
    "cross-column-share",
    "pipeline-stress",
]

SAMPLE = 8


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--only", type=str, default=None, help="run one section only")
    ap.add_argument("--verbose", action="store_true", help="show every finding")
    args = ap.parse_args(argv)

    cases = load(args.cases)
    rep = Report()

    for case in cases:
        check_distribution_sanity(case, rep)
        check_contradictions(case, rep)
        check_entailment_proxy(case, rep)
    check_prose_diversity(cases, rep)
    check_cross_column_usage(cases, rep)
    check_pipeline_stress(cases, rep)

    print(f"{args.cases}: {len(cases)} case(s)")
    for section in SECTION_ORDER:
        items = rep.sections.get(section) or []
        if args.only and section != args.only:
            continue
        if not items:
            continue
        print(f"\n[{section}] {len(items)} finding(s)")
        shown = items if args.verbose else items[:SAMPLE]
        for it in shown:
            print(f"  - {it}")
        if len(items) > len(shown):
            print(f"  ... and {len(items) - len(shown)} more (use --verbose)")

    print(f"\n{rep.total()} finding(s) total. These are SUSPICIONS, not errors --")
    print("validate_dataset.py is the gate; this needs a human read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
