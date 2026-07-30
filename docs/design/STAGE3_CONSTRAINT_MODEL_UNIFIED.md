# Stage 3 Constraint Model: Current State, Gap Taxonomy, and Extension Plan

**Status: DESIGN-ONLY. Not validated against a prototype. No code written.**
This is the master planning document for "how robust and inclusive can
Stage 3's constraint-representation model become, without breaking the
math." It formalizes the EXISTING model precisely (Section 1), catalogs
everything found across this research arc that the model cannot express or
mishandles today (Section 2), and proposes a concrete, mathematically-
justified extension for each gap (Section 3) rather than treating them as
one undifferentiated pile.

The multivariate/cross-column-distribution piece (correlation, conditional
independence, copulas) is deliberately NOT re-derived here -- it already
has its own rigorous treatment at `experiments/
MULTIVARIATE_CONSTRAINT_DESIGN.md`, referenced from Section 3.6. This doc
is about everything else: what the CURRENT single-variable DOF model is,
precisely, and how to widen it soundly.

---

## 1. The existing model, precisely

### 1.1 The mathematical object

Every fact Stage 3 extracts today ultimately becomes one of two things:

- A **Variable** (`src/util/algorithms/dof_graph.py:30`): a single scalar
  unknown -- a distribution parameter, a table row count, an FK fan-out
  mean. Never a whole distribution, never a whole NL fact. Carries optional
  `lower_bound`/`upper_bound` as **domain metadata**, explicitly NOT
  consumed by the classification algorithm (`dof_graph.py`'s docstring:
  "classify() ignores these fields entirely").
- A **Constraint** (`dof_graph.py:68`): one equation, naming the set of
  Variables it touches (`variables: list[str]`, non-empty).

These form a **bipartite graph** G = (Variables, Constraints, Edges), an
edge existing wherever a Constraint's `variables` list names a Variable
(`dof_graph.py:134-141`, `DOFGraph._build_graph`).

**The one hard rule carried over from the original design** (restated in
both `dof_graph.py`'s module docstring and `baselines/experiments/
STAGE3_PHASE2_DESIGN.md` section 2): if a single NL fact states two
independent parameter values ("mean 8, std 2"), it becomes TWO Constraints,
one per parameter -- never one Constraint touching both Variables. One
equation over two unknowns is mathematically underdetermined. A genuine
multi-variable relationship (`subtotal = quantity * unit_price`) legitimately
needs one Constraint touching multiple Variables -- the model cannot
distinguish these two cases for a caller; getting it right is the
extraction agent's/converter's responsibility, not something `DOFGraph`
itself can validate.

### 1.2 The classification algorithm

`DOFGraph.classify()` (`dof_graph.py:143-170`) delegates to
`pyomo.contrib.incidence_analysis.dulmage_mendelsohn`, which computes a
**maximum bipartite matching** between Constraints and Variables and
decomposes the graph into the classical Dulmage-Mendelsohn partition. This
is a **structural rank** question -- "how many Variables can be
simultaneously pinned by SOME assignment of Constraints to Variables,
ignoring the actual numeric content of each equation" -- not a numeric
solve. It is exactly the same technique used for DAE (differential-
algebraic equation) index reduction in process-flowsheeting software (e.g.
Pantelides' algorithm), cited in `STAGE3_PHASE2_DESIGN.md` section 2.

The three output categories, precisely as implemented:

- **`square_variables`**: Variables matched by the maximum matching --
  structurally pinned by exactly one Constraint (`col_part.square`).
- **`loose_variables`**: the union of unmatched AND "underconstrained"
  Variables (`col_part.unmatched | col_part.underconstrained`,
  `dof_graph.py:147`) -- genuinely free, the target for an LLM probe to
  Stage 4 (never filled in by Stage 3 itself, per the project's stated
  `stage3_stage4_division_of_labor` division).
- **`overconstrained_blocks`**: NOT a per-node verdict -- a
  **connected-component** classification (`dof_graph.py:158-164`). A
  Constraint that DID match can still belong to an overconstrained block
  alongside the Constraint that lost the matching contest over the same
  Variable (`row_part.unmatched`) -- `classify()` explicitly folds this in
  (`dof_graph.py:148-156`) because reporting only the "losing" Constraint
  in isolation would hide which OTHER Constraint it's actually contending
  with. This block-level discipline is called out as a real, previously-
  made mistake in the module's own docstring (point 2) -- worth preserving
  exactly when extending the model, not incidental.

### 1.3 Grain-scoping: what makes a Variable name well-formed

The raw `DOFGraph`/`Variable`/`Constraint` triplet above has no notion of
"which population/row-level entity does this parameter describe" -- that
is supplied entirely by the layer above it,
`src/pipeline/stage3/middleware/constraint_graph.py`'s `RichVariable`
(`constraint_graph.py:688-717`) and `Grain`
(`src/pipeline/stage3/models/grain.py:87-109`).

- **`Grain`**: the canonical (base table + PK, FK-edge-multiset,
  optional aggregate signature, `narrowed` flag) identity of a joined ON
  tree, produced by `canonicalize()` under the deliberate restriction that
  every join must be a real foreign-key-to-primary-key relationship
  (`grain.py`'s module docstring). This is what makes "the average order
  total per customer" and "the average order total overall" DIFFERENT
  Variables even though both mention `ORDER.total` -- their Grains differ.
- **`RichVariable.flat_name()`** (`constraint_graph.py:702-717`) collapses
  a `(Grain, VariableKind, name, branch)` tuple into the flat string
  identity `DOFGraph` actually operates on -- Grain-scoped AND
  branch-scoped (Q4 conditional-fork-aware), so two Variables with the
  same short name but different population never accidentally unify.
- **`Grain.is_comparable_with(population_sensitive=...)`**
  (`grain.py:195-218`): the population-identity discipline. For
  population-SENSITIVE kinds (`COLUMN_DISTRIBUTION_PARAM`,
  `MOMENT_TARGET`, `TABLE_CARDINALITY`, `CORRELATION`) two Variables are
  only comparable at an EXACT narrowed/edge match; `COLUMN_RANGE`/
  `DERIVED_COLUMN` correctly keep the looser subset rule (a hard
  per-row bound on a superset genuinely does bind every subset).

### 1.4 The value-level pre-filter: `confirmed_conflicts`

`build_and_classify()` (`constraint_graph.py:786-857`) sits between
`RichVariable` construction and `DOFGraph`. Before handing anything to
`DOFGraph.classify()`, it deduplicates `RichVariable`s sharing a
`flat_name()`, MERGING their bounds/categories (`_merge_rich_bounds()`,
`constraint_graph.py:745-778`) rather than keeping only the first-seen
one. If the merge produces an empty interval or disjoint category sets,
that flat name is pulled into `confirmed_conflicts` BEFORE reaching
`DOFGraph` at all -- a PROVABLE value contradiction (two facts stating
incompatible exact numbers) is real infeasibility regardless of what the
purely structural degree-count would say. This is distinguished sharply
from `overconstrained_blocks`, which DOF flags from constraint-to-variable
RATIO alone and cannot tell "two facts agreeing" (harmless redundancy)
apart from "two facts disagreeing" (a real bug) -- that numeric
distinction is exactly what `confirmed_conflicts` is for.

### 1.5 The healing loop

Both `confirmed_conflicts` (value-level) and `overconstrained_blocks`
(structural) surface, via `analyze_cross_shard_constraints`'s
`Stage3AnalysisReport`, as `_ConflictItem`s in
`src/orchestration/stage3/entry.py`'s Tier 2/Tier 3 reconciliation loop
(`_reconciliation_loop`, `entry.py`). Each is traced back to its source NL
facts (via `variable_fact_map`) and sent to the `conflict_reconciler` agent,
which classifies it MISEXTRACTION (re-extract with guidance) /
FALSE_POSITIVE (drop, record in `dismissed_conflicts`) /
GENUINE_CONTRADICTION (keep). **Any extension proposed in Section 3 that
produces its own notion of "square/loose/overconstrained" should feed this
SAME healing loop** rather than inventing a parallel one -- it already
handles the deterministic-classification-plus-LLM-reconciliation pattern
generally, not specifically for today's Variable/Constraint shape.

### 1.6 What this model is FOR, precisely

Stepping back: every piece above exists to answer ONE question --
*"for a Quantity that some generative parameter of the database depends on
(a distribution's mean, a table's row count, a fan-out's average), is it
uniquely pinned by the stated facts, genuinely free, or over-determined?"*
This is a **parameter identifiability** question. It is emphatically NOT
about whether two columns' realized VALUES on the same row are mutually
consistent (Section 2.1 below), and it is NOT about the SHAPE of a joint
distribution across two variables (Section 3.6 / the multivariate doc).
Conflating these is the single most common way a "clever" extension could
quietly corrupt the model -- every proposal in Section 3 is checked against
this boundary explicitly.

---

## 2. Gap taxonomy: two categories, not one pile

### 2.1 Category A -- facts that are NOT parameter-identifiability facts at all

These were never going to fit the DOF graph, no matter how it's extended,
because they are a different mathematical object:

| Fact kind | Example | Why it's not a DOF Variable |
|---|---|---|
| Per-row cross-column comparison | `end_date > start_date` | Constrains the JOINT SUPPORT of one row's value tuple across rows -- not a statement about a distribution parameter's value. |
| Uniqueness | "SSN must be unique" | A property of the realized column across ALL rows, not a single scalar parameter (see 3.1 for a reduction that DOES fit, though). |
| NOT NULL | "email must never be null" | A presence/support constraint per row (see 3.2 for a reduction that DOES fit). |
| Correlation / conditional independence | "quantity and total are correlated" | A joint-DISTRIBUTION-shape parameter (see `MULTIVARIATE_CONSTRAINT_DESIGN.md`), a genuinely different mathematical layer, already designed separately. |

Confirmed via code trace this research arc (`constraint_graph.py`'s
`_range_constraint_to_rich`, `constraint_graph.py:1015-1055`): a per-row
cross-column comparison with 2+ `RColumnRef`s makes `extract_columns()`
return != 1 column, so the function silently returns `([], [])` --
**not data loss to Stage 4** (the raw `Constraint` still reaches
`Stage3Output.logic_constraints`/`structural_constraints` via `entry.py`'s
unconditional merge -- verified directly) but a genuine hole in Stage 3's
OWN conflict-detection coverage for this fact family.

### 2.2 Category B -- facts that ARE parameter-identifiability facts, but the current CODE mishandles them (bugs, not missing math)

| Gap | Where | Confirmed behavior |
|---|---|---|
| Group-scoped `ONAggregate` structural constraint misrouted | `constraint_graph.py`'s `_convert_cross_shard_constraints`, structural loop | Routes by `extract_tables()` table-count (1 table -> cardinality, else -> range); an `ONAggregate` re-rooted by `canonicalize()` to a PARENT table still reports only the un-re-rooted child table via `extract_tables()`, so a real per-group aggregate bound (e.g. `SUM(total) GROUP BY customer_id <= 10000`) is silently treated as a bogus row-count bound on the wrong table. |
| `RExists`/`RNotExists` never converted | `condition_nodes.py`'s `_collect_columns_pred` has no branch for them; downstream `_range_constraint_to_rich`/`_cardinality_to_rich` | Currently INERT (no prompt instructs agents to emit them) but a live landmine -- would silently drop or fabricate a bogus unbounded `row_count` Variable if ever emitted. |
| `RComparison.op == "~"` | `condition_nodes.py` declares it, nothing consumes it | Dead/aspirational, zero risk, pure cleanup. |
| Composite FK unsupported | `grain.py`'s `_FKRef`, `schema.py`'s `ForeignKey.referencing_column` (singular, all the way down) | Fails LOUDLY (`CanonicalizationFailure`), not silently -- but blocks a genuinely common pattern (composite natural keys). |

---

## 3. Proposed extensions, per gap -- each checked against Section 1.6's boundary

### 3.1 Uniqueness: reduce to an ordinary pinning equation, not a new primitive

**The clever part**: "SSN is unique" is algebraically
`COUNT(DISTINCT ssn) == COUNT(*)` for the same table. Both sides are
ALREADY DOF-shaped Quantities -- `COUNT(*)` is exactly today's
`TABLE_CARDINALITY` row_count Variable; `COUNT(DISTINCT ssn)` needs one
addition: an 8th `ONAggregate.fn` value, `"COUNT_DISTINCT"`
(`on_nodes.py`'s `_VALID_AGG_FNS` currently has 6). Once that exists,
uniqueness becomes an ORDINARY `Constraint` pinning
`RAggregateRef(alias="distinct_ssn")` equal to the table's own row_count
Variable, via `RComparison(op="=", ...)` -- no new `VariableKind`, no new
feasibility test, no touch to `DOFGraph` at all. This is the single
highest-leverage, lowest-risk addition in this whole document: **one new
enum value plus a well-defined conversion path**, reusing 100% of existing
machinery.

Caveat (carried from `MULTIVARIATE_CONSTRAINT_DESIGN.md` section 5.2's
discipline, restated because it applies here too): this is a fact about a
column's REALIZED values across rows, still fundamentally a per-row/
per-dataset aggregate equality -- it's being ADMITTED into the DOF graph
because it happens to reduce to a clean equation between two existing
Quantity shapes, not because "uniqueness in general" is secretly a DOF
concept. Composite-column uniqueness (`(a, b)` together unique) needs
`COUNT_DISTINCT` to accept a column LIST, a small, contained widening of
`ONAggregate.column` (currently a single string) -- flagged, not designed
in detail here.

### 3.2 NOT NULL: reuse the presence-rate variable already built

**No new primitive needed at all.** `grain.py`'s `accessible_columns()`
(`grain.py:148-153`) ALREADY synthesizes a `"{col}.is_present"` column for
every nullable FK column on the own table, built earlier this session
specifically for the narrowing-fix work (a nullable FK's presence rate as
a genuine DOF variable, per the `stage3_stage4_division_of_labor` project
memory). A NOT NULL fact on a nullable column is exactly the statement
"this column's presence rate equals 1.0" -- `RComparison(op="=",
left=RColumnRef(name="email.is_present"), right=RLiteral(value=1.0))`,
using the EXACT mechanism already wired in for nullable FKs. The only gap:
this synthetic column is currently only generated for nullable FK columns
specifically (per `grain.py`'s narrowing logic) -- widening it to ANY
nullable column (not just FKs) is the actual, small, contained change
needed. Zero new math, zero new `VariableKind`, reuses an already-built
and already-tested mechanism.

### 3.3 Fix the aggregation-routing bug (Section 2.2) -- a routing decision, not new math

The fix is precise: in `_convert_cross_shard_constraints`'s structural
loop, check `grain_result.agg_signature is not None` BEFORE falling back
to the `extract_tables()`-count heuristic. When it IS an aggregate, the
Constraint should be converted via a NEW small function (`_aggregate_
range_to_rich`, not yet written) that reads the aggregate's `RAggregateRef`
alias directly as the Variable's identity -- confirmed necessary because
`extract_columns()` (`condition_nodes.py:433-439`) only ever collects
`RColumnRef` nodes, never `RAggregateRef`, so routing an aggregate-
referencing condition through the EXISTING `_range_constraint_to_rich`
would silently return zero columns regardless of the routing fix. This is
purely mechanical -- the Grain/DOF math is unaffected; it is a missing
CASE in a dispatch function, not a missing THEOREM.

### 3.4 `RExists`/`RNotExists`: recommend removal, not repair

Both common real-world uses of "exists" reduce cleanly to primitives
ALREADY built and ALREADY DOF-integrated:

- "Every LINE_ITEM must reference an existing PRODUCT" (child-references-
  parent completeness) is exactly what a non-nullable FK already
  guarantees by construction under the FK-PK canonicalization model; if
  the FK is nullable, it's exactly the presence-rate Variable (3.2).
- "Every CUSTOMER must have placed at least one ORDER" (parent-must-have-
  child) is exactly `ONFanout`'s `min_fanout >= 1` -- already fully
  representable and DOF-integrated (`fanout_constraint_to_graph_nodes`,
  the old pathway's equivalent; the live `_cardinality_to_rich`/`ONFanout`
  pairing on the cross_shard side).

Since both realistic use cases are already better-served by existing,
DOF-integrated primitives, and `RExists`/`RNotExists` are currently unused
by any prompt (confirmed this research arc), the recommendation is to
**remove them from the `RPredicate` union** rather than build a conversion
path for a predicate shape that adds no expressiveness the model doesn't
already have more precisely. This is a net simplification, not a feature
addition -- worth flagging explicitly since it runs against the "increase
the repertoire" framing of this whole research arc: sometimes the correct
answer to "can we represent X" is "we already do, more precisely, under a
different name."

### 3.5 Category A (per-row cross-column facts): a genuinely new, but small and well-precedented, layer

Per Section 2.1, `end_date > start_date` and similar per-row comparisons
are NOT DOF Variables. The proposed treatment: a lightweight, SEPARATE
**difference-constraint network**, one per Grain, over the same-row
columns referenced by Category-A facts. This is structurally identical to
a **Simple Temporal Network** (STN) in AI planning/scheduling (Dechter,
Meiri & Pearl 1991) -- each fact `X - Y <= c` (or `X > Y`, `X != Y` after
suitable encoding) becomes an edge in a weighted graph; joint
satisfiability is decided by a **negative-cycle check** (Bellman-Ford),
a well-established, deterministic, polynomial-time algorithm entirely
distinct from Dulmage-Mendelsohn.

**Caveat -- flagged deliberately, not glossed over**: the Dechter/Meiri/
Pearl STN citation above is stated from general background knowledge, NOT
independently verified by a dedicated research pass this session (unlike
every claim in `MULTIVARIATE_CONSTRAINT_DESIGN.md`, which WAS freshly
grounded via three parallel research agents). Before implementing this
layer, the STN/negative-cycle claim should get the same verification
treatment -- confirm the exact theorem statement and citation, and confirm
the encoding of `!=`/strict inequalities into a difference-constraint
graph (which typically needs an epsilon-perturbation trick, itself worth
double-checking) before relying on it.

Design implication if confirmed: this is Stage 3's "Layer 2," running
ALONGSIDE (never merged into) the DOF graph -- its own square/loose/
overconstrained-shaped output (a negative cycle = `confirmed_conflict`,
everything else either fully determined by the transitive closure or
genuinely unconstrained) feeding the SAME `conflict_reconciler` healing
loop per Section 1.5's reuse principle.

### 3.6 Correlation / conditional independence / multivariate distributions

Already fully designed -- see `experiments/
MULTIVARIATE_CONSTRAINT_DESIGN.md` in full. Summary for cross-reference
only: correlation and CI facts are entries in one shared partially-
specified symmetric matrix (correlation matrix Sigma / precision matrix
Omega = Sigma^-1); joint feasibility is decided by chordality of the
fact-specified-entries graph plus Grone-Johnson-Sa-Wolkowicz's 1984
positive-definite-completion theorem -- a THIRD deterministic feasibility
test, alongside Dulmage-Mendelsohn (Section 1.2) and the STN negative-
cycle check (Section 3.5), all three feeding the same `conflict_reconciler`
loop.

### 3.7 Composite FK support -- correctly deferred, not attempted here

Confirmed this research arc: the restriction is not contained to the
ON-tree layer alone -- `schema.py`'s `ForeignKey.referencing_column` is a
single string all the way down, and `grain.py`'s `_FKRef`/`_SchemaView.
_fk_by_child` are keyed by a single `(table, column)` pair. Generalizing
requires touching the Stage 2 schema model itself
(`referencing_columns: List[str]`), which is a real, deliberate, separate
piece of work -- correctly out of scope for "clever, minimal, non-breaking"
extension and should get its own dedicated design pass (the edge/Grain
CONCEPT generalizes cleanly -- one hop, wider key -- but touching the
schema model is a larger-blast-radius change per the project's own
graduated-proof convention for costly-reversible work).

### 3.8 Format/regex constraints -- no new insight, restated for completeness

No predicate node exists for pattern-matching (`condition_nodes.py`'s
`RPredicate` union has no LIKE/regex shape), and none of the research this
arc did suggests a natural, safe reduction the way uniqueness (3.1) and
NOT NULL (3.2) had one. Left as an explicitly open, undesigned gap --
logic_extractor's prompt already tells agents not to fabricate an
approximation for this.

---

## 4. Prioritized roadmap

| Tier | Item | Blast radius | Touches DOF/Grain core? |
|---|---|---|---|
| 0 | Delete the dead `~` `RComparison` op | Trivial | No |
| 1 | Uniqueness via `COUNT_DISTINCT` (3.1) | 1 new enum value + 1 conversion path | No |
| 1 | NOT NULL via presence-rate widening (3.2) | Widen an existing synthesized-column rule | No |
| 1 | Remove `RExists`/`RNotExists` (3.4) | Delete unused code | No |
| 2 | Fix aggregation-routing bug (3.3) | New dispatch branch + 1 new conversion function | No (routing fix only) |
| 2 | STN-style per-row layer for cross-column facts (3.5) | New, small, separate module -- needs its own citation-verification pass first | No (parallel layer) |
| 3 | Correlation/CI/copula layer (3.6) | See `MULTIVARIATE_CONSTRAINT_DESIGN.md` | No (parallel layer) |
| 3 | Composite FK support (3.7) | Stage 2 schema model change | Yes -- deliberately deferred |
| Unscoped | Format/regex (3.8) | No safe reduction found | N/A |

Everything in Tier 0-2 is achievable without touching `DOFGraph` or
`Grain`'s core logic at all -- either it reduces to an existing Quantity
shape (3.1, 3.2), is a routing/dispatch fix (3.3), is a deletion (3.4), or
is an entirely parallel, independently-classified layer feeding the same
healing loop (3.5, 3.6). Tier 3 items are the only ones that touch the
actual mathematical core, and both are correctly scoped as separate,
dedicated design efforts rather than folded into this pass.
