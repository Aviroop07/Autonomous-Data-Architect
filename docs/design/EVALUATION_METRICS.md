# ScribbleDB evaluation metrics

This is the whole metric set. There is no secondary "legacy" table, and the
name-matching metrics that used to be here are gone, not demoted.

## Why the previous set was replaced

`Table F1`, `Attr F1`, `PK Acc`, `FK Acc`, `DT Acc` were all keyed on names.
Table F1 matched table names; Attr F1 matched column names; PK, FK and DT
accuracy were computed only over pairs that had already matched by name. So the
most arbitrary property of a schema decided every number, and a fuzzy matcher had
to carry the weight.

That matcher cannot do it. Measured on its own implementation over 12 true
synonym pairs and 12 genuinely distinct pairs:

| | cosine range |
|---|---|
| true synonyms | 0.235 - 0.839 |
| unrelated pairs | 0.216 - 0.608 |

The ranges overlap across most of their width, so no threshold separates them. At
the 0.6 in use, `DOCTOR`/`NURSE` were merged (0.608) while the real synonyms
`MEMBER`/`PATRON` (0.505) and `PRODUCT`/`STYLE` (0.382) were counted as misses.
Tuning the threshold trades one error for the other; it cannot remove both.

Two further defects, both demonstrated on live runs rather than argued:

- **Structure was barely measured.** The FK metric ran only over tables that had
  already matched by name, so a wrong foreign-key topology could score well.
- **Valid re-normalisation was punished.** One hospital run produced 12 tables
  and another 11, with *identical* 100% fact coverage -- a junction decomposed
  differently. Name-set F1 read that as a regression. It is not one.

A schema is not a bag of strings. Two schemas are equivalent when they can
represent the same set of database states -- relative information capacity -- and
that is the frame these metrics work in.

---

## The schema metrics

All four are name-independent: renaming every table and column in a schema cannot
change any of them.

**Implementation status.** IC-R and IC-P are implemented and wired into
`run_evaluation.py` (`capacity_eval.py`). RSC is partially implemented as the
structural score in `structural_eval.py` -- table alignment and FK topology are
live, but reversed-versus-missing are not yet counted separately and optionality
is not checked. KDC is implemented (`kdc_eval.py`) and wired in. Nothing here is
reported as a number until it is actually computed.

### 1. Information Capacity Recall (IC-R)

**Question:** can the predicted schema hold every fact the specification states?

Each fact extracted from the NL is checked for a home in the schema: a table, a
column of an appropriate type, or a foreign key that realises it. A fact with no
home is capacity that was lost.

This subsumes what table and attribute recall were reaching for, and is immune to
both naming and normalisation -- a junction split two ways still carries the same
facts either way.

Type appropriateness belongs to this question in principle -- money stored as
`FLOAT` cannot represent the stated amounts exactly, so it is a capacity failure
rather than a cosmetic one, which is what `DT Acc` was gesturing at without the
justification. In the current implementation it is measured instead by
`column_type_agreement` in the structural metric, over coarse type families.
Judging a type against what a fact *needs* requires reading the fact's semantics
and is not done yet; a keyword rule for 'money' would be exactly the brittle
kind of check this project forbids.

### 2. Information Capacity Precision (IC-P)

**Question:** is there structure in the schema that nothing in the specification
asks for?

Tables, columns and foreign keys with no supporting fact are hallucinations. A
schema that invents an audit trail nobody mentioned is not better than one that
does not, and IC-R alone would reward it.

`IC-F1` is the harmonic mean, and is the closest thing here to a single headline
number for schema content.

### 3. Key and Dependency Correctness (KDC)

**Question:** do the declared keys actually determine the rows, and is every
functional dependency enforced by the key structure?

It needs dependencies, and it does NOT need them from the ground truth: Stage 2
already derives `functional_dependencies` into the conceptual model, and they now
survive the merge. So this asks a self-consistency question --
*the extractor asserted these dependencies from the text; did the mapper then lay
out keys such that they actually hold?* -- which is the same property that makes
IC-Recall usable on any spec someone writes rather than only on authored cases.
It is only possible at all because the merge stopped discarding FDs; before that
fix the count was zero and this metric would have been vacuous.

This is normalisation, checked deterministically rather than by eye:

- every stated FD's determinant is a key or a superkey of the table holding it;
- no non-key attribute depends on part of a composite key (no partial
  dependency, i.e. 2NF);
- no non-key attribute depends on another non-key attribute (no transitive
  dependency, i.e. 3NF);
- a table with no key at all is reported.

A dependency spanning two tables is reported but NOT counted as a defect: after
decomposition those are normal, and penalising them would punish the schema for
being normalised. A dependency naming a table that did not survive mapping is
also not counted here -- that is a capacity failure, which IC reports, and
counting it twice would double-penalise one defect.

`n_checked` is always reported alongside, so a vacuous 1.0 on a schema with no
dependencies is never mistaken for an earned one.

A veteran weights this above everything else on this list, because a partial
dependency is what produces update anomalies in production, long after the schema
review passed.

### 4. Referential Structure Correctness (RSC)

**Question:** is every relationship the specification describes realised as a
foreign key, pointing the right way, with the right optionality?

Scored over a name-blind alignment of tables (structural role plus, once the
dataset carries them, the NL spans each element cites). Three failures counted
separately, because they are not equally bad:

- **missing** -- the relationship is not represented at all;
- **reversed** -- the FK is on the wrong side, which changes the cardinality the
  schema can express and is a genuine modelling error;
- **mis-optional** -- nullability contradicts stated mandatory or optional
  participation.

---

## The data metrics

These need Stage 4. They are the strongest claims the system makes, and they are
entirely name-free because they run against generated rows.

**Implementation status.** DF is implemented for the column-level distances;
CSR is not implemented at all. Both still need Stage 4 to produce rows before they
can run on real output.

Within DF, what exists: MRE, NLL, and a single `distance` reported alongside the
`distance_kind` used -- Kolmogorov-Smirnov for continuous and ordered discrete
families, total variation for categoricals. Nominal categoricals ARE now scored,
which was previously impossible. What does not exist: `poisson` and `zipf` still
use a discrete KS (the supremum of |F_emp - F_gt| over the observed support)
rather than TVD. They are ordered, so KS is at least meaningful there, but the
design below calls for TVD across all discrete families and that part is not
built.

### 5. Constraint Satisfaction Rate (CSR)

Fraction of the specification's constraints that actually hold in the generated
instance. Reported separately for range, cross-column and conditional
constraints, because a system that satisfies simple bounds while violating every
cross-column invariant should not look the same as one that does the reverse.

### 6. Distribution Fidelity (DF)

Per column, how close the generated values are to the stated distribution:
Kolmogorov-Smirnov distance for continuous families, total variation distance for
categoricals.

KS over a nominal variable is not merely awkward, it is undefined: KS is a
supremum over a CUMULATIVE distribution, and accumulating requires an ordering
that nominal labels do not have. Choosing to sort BRONZE below SILVER is arbitrary
and changes the statistic. TVD needs no ordering:

    TVD(p, q) = (1/2) * sum over the union of labels of |p(k) - q(k)|

It lands in [0, 1] and reads the same direction as KS, so both report through one
`distance` field with a `distance_kind` beside it. The aggregate also reports a
count per kind, so a mean over mixed kinds is never presented as homogeneous.

This unlocked the 132 nominal categoricals in the benchmark -- about a quarter of
all its distributions -- which previously scored worst-case because
`float("BRONZE")` raised and the failure became a number.

Columns are paired through the schema alignment, never by name.

`FA` has been removed from the code as well as from this list: it was defined
as `1 - KS`, so it carried no information KS did not, and reporting both
invited reading one number as two.

---

## No thresholds

Nothing in this suite has a similarity cutoff. That is a design property, not an
accident, and a unit test enforces it by scanning the metric modules for one.

The set it replaced turned on a 0.6 cosine threshold that could not separate
synonyms from unrelated words at ANY value, because the two classes overlap. So
every metric here is built from set arithmetic, key containment, or a similarity
that is *summed* rather than *compared*:

| metric | why no cutoff is needed |
|---|---|
| IC-Recall / IC-Precision | set intersection over fact IDs |
| FK topology P/R/F1 | set intersection over translated edges |
| column type agreement | multiset Jaccard, reported as a value |
| table structural recall | SOFT -- each table contributes its own similarity |
| KDC | key containment: `X ⊇ K`, `X ⊂ K`, `X ∩ K = ∅` |

Table structural recall used to have a 0.5 floor deciding whether a pairing
"counted". Two changes removed the need for it: a predicted table with no
provenance is excluded from alignment outright, and recall became soft. A weak
pairing now contributes little on its own terms rather than being classified by a
constant.

The five structural WEIGHTS (3, 3, 2, 2, 1 over out-degree, in-degree, PK arity,
type multiset, column count) are the only tuned numbers left. They are not
thresholds -- no decision flips on crossing them -- but they are chosen, and this
is the honest place to say so.

## What is deliberately not here

- **No name matching anywhere.** Not as a primary signal, not as a tiebreak.
- **No single composite score across the four.** They fail in qualitatively
  different ways and a weighted sum hides which one occurred. `IC-F1` covers
  content; the structural score sits beside it, not folded in. (`structural_score`
  is itself a product of topology and column agreement -- a deliberate exception,
  because topology alone is gameable: reversing the only FK in a two-table schema
  makes it its own mirror image and topology then scores a perfect 1.0.)
- **No accuracy-style all-or-nothing metrics.** `Table Acc` and `Attr Acc` were
  1.0 only on a perfect set match, which is why they read 0.000 while F1 was
  0.75. A metric that is almost always zero measures nothing.
