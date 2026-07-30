# The Relation / Condition / Constraint Model: Consolidated Design

**Status: DESIGN-ONLY. Not validated against a prototype. No code written.**
This consolidates ~30 turns of live design conversation into one coherent
reference. It supersedes (eventually -- not yet, nothing is wired in or
deleted) `src/pipeline/stage3/models/on_nodes.py`, `condition_nodes.py`, and
`src/pipeline/stage3/models/grain.py`. It will live in a new package,
`src/util/constraint_model/`, since Stage 4 needs this representation too,
not just Stage 3 (see Section 10 for the exact folder layout).

**Relationship to the other two design docs in this folder** -- read
together, not duplicated:
- `STAGE3_CONSTRAINT_MODEL_UNIFIED.md` formalizes the CURRENT, live
  Dulmage-Mendelsohn DOF model precisely and catalogs its gaps. This doc is
  the actual replacement design for the representation layer that feeds
  that DOF model (and successors to it).
- `MULTIVARIATE_CONSTRAINT_DESIGN.md` derived the copula/chordal-completion
  machinery for cross-column dependence. This doc adopts that machinery
  wholesale (Section 8.2) and extends it with the polychoric/polyserial
  mechanism for categorical/mixed dependence (Section 8.2.1), closing a gap
  that doc explicitly left open (its Section 5.1).

---

## 1. Scope and non-goals

The `Relation` algebra (Section 3) is a deliberately **closed** set:
`BaseTable`, `Join`, `Aggregate`, `Filter`, `Project`, `Fanout`. Explicitly
excluded, decided deliberately rather than by omission:

- **UNION/INTERSECT/EXCEPT** -- no realistic constraint fact needs them;
  every case considered reduces to existing primitives (e.g. a compound
  `Filter` condition).
- **Correlated subqueries** -- same reasoning, plus a structural cost: a
  correlated subquery's inner scope depends on the outer scope, which
  breaks the bottom-up, each-operand-validated-independently model
  (Section 4) outright.
- **Window functions** -- real expressiveness value (rank/percentile/
  sequential facts) but the largest lift, needing its own validation math.
  Treated as a named, deliberate non-goal, same tier as composite FKs and
  format/regex predicates (also excluded -- see `STAGE3_CONSTRAINT_MODEL_
  UNIFIED.md` Section 3.7-3.8 for why). The one concrete case motivating
  window functions (sequential/state-machine facts) is instead covered by
  a dedicated `StateSequence` term (Section 9), not general `LAG`/`RANK`.

---

## 2. The core objects, at a glance

```
Constraint = Relation (the "on") + Condition (the assertion) + fact_references + severity
```

- **`Relation`**: defines WHERE in the schema a constraint's columns live --
  a recursive relational-algebra expression (Section 3).
- **`Condition`**: the assertion itself -- either an ordinary boolean
  predicate tree (Section 7), or one of three **cohesive** statistical/
  sequential terms that can never nest inside ordinary logical connectives
  (Section 8-9): `Distributed`, `Correlated`, `StateSequence`.
- **`Constraint`**: ties the two together, plus bookkeeping. No family/
  category tag (Section 11.1) -- the object's own type is self-describing.

---

## 3. The `Relation` algebra

### 3.1 Node types

- `BaseTable(name)` -- a single schema table.
- `Join(left, right, on)` -- FK-PK-restricted (Section 1 non-goals aside,
  this restriction is KEPT deliberately -- see Section 5 for why it matters).
- `Aggregate(source, fn, column, group_by, alias, fn_param?)` -- `fn_param`
  is new, needed for `PERCENTILE(p)` (Section 3.3).
- `Filter(source, condition)` -- new node type (didn't exist in the old
  model at all).
- `Project(source, columns)` -- new node type; each entry is either a bare
  column reference (optionally renamed via alias) or a computed arithmetic
  expression (alias REQUIRED for computed entries).
- `Fanout(parent_table, child_table, fk_column)` -- unchanged from today,
  guarantees every parent row is counted, including zero-child parents.

### 3.2 Hybrid representation and the homogenization rule

Any node may be expressed as a raw SQL string instead of a structured
object, recursively, at any level of nesting. **Homogenization rule**: a
base table expressed as SQL must be `SELECT * FROM X`, never a bare table
name -- every SQL-string form is a complete, valid SELECT statement, so the
parser's input grammar is uniform (no special-cased shorthand). Bidirectional
conversion is required: parse SQL -> object, and serialize object -> SQL,
both round-tripping to a semantically equivalent form. A parsed SQL string
that uses an out-of-scope feature (Section 1) must fail parsing with a
specific reason, not silently produce a partial object.

### 3.3 The locked aggregate-function catalogue

Extends `ONAggregate.fn` beyond today's `SUM/COUNT/AVG/MAX/MIN/MEDIAN`:

| Function | Purpose | Type requirement | Notes |
|---|---|---|---|
| `SUM`, `AVG` | total / mean | numeric | unchanged |
| `MAX`, `MIN` | extremes | orderable | unchanged |
| `MEDIAN` | 50th percentile | orderable | literally `PERCENTILE(50)`, kept as a named special case |
| `COUNT` | row count | any | unchanged |
| `COUNT_DISTINCT` | distinct-value count | any | reduces uniqueness to an ordinary pinning equation: `COUNT_DISTINCT(col) == row_count` -- no new predicate needed at all |
| `STDDEV` / `VARIANCE` | spread, without claiming a full distribution shape | numeric | new |
| `PERCENTILE(p)` | rank statistic | orderable | new; `p` lives in a new `fn_param` field on `Aggregate` itself, the pinned threshold value still lives on `Condition` via ordinary `Comparison`+`AggregateRef`, same pattern as every other aggregate fact |
| `MODE` | most frequent value | any | new |

Explicitly excluded: `RANGE` (derivable from `MAX`-`MIN`, no new primitive
needed), skewness/kurtosis (deferred, out of scope unless a real case
surfaces). NULL-skipping semantics (`AVG`/`SUM` ignore NULLs, `COUNT(col)`
vs. `COUNT(*)`) are explicitly Stage 4's concern, not Stage 3 validation's.

### 3.4 Aliasing

Every `Relation` node gets an OPTIONAL alias. Supplying one (when needed to
disambiguate a self-join or a repeated table) is the **LLM's
responsibility** -- the validator's only job is to detect a collision (the
same table appearing twice, or genuine ambiguity) with no alias present,
and reject/force a retry. The validator never invents an alias itself.

---

## 4. Validation model: bottom-up effective-schema synthesis

Validation is a **type-checker-style pass**: each operator's validity
requires its operand(s) already valid, and it synthesizes an **effective
schema** for its parent, recursively, bottom-up. An effective schema
carries:

- **Columns**: name -> (type, nullable).
- **Effective primary key**: the column set uniquely identifying a row of
  THIS resulting relation (not necessarily the root table's own PK).
- **Remaining foreign keys**: which of the original FKs are still present
  as real, joinable columns from here.
- **Column-level FK/PK provenance** (Section 4.1): lineage back to the
  real, schema-declared FK/PK each column traces to, if any.
- **A row-count variable** (Section 4.2).

### 4.1 Provenance -- validating joins over arbitrarily-derived relations

Every column carries a provenance tag: which real, schema-declared FK or
PK it traces back to, if any. This propagates through every operator:

- `BaseTable`: provenance is itself (source of truth).
- `Join`: both operands' provenance passes through unchanged.
- `Filter`: passes through unchanged (no columns touched).
- `Project`: a straight passthrough/rename keeps its source's provenance;
  a computed (arithmetic) column has none -- it's a new derived value, not
  a reference to a real key. Renaming is NOT a new node -- just a `Project`
  entry that's a bare `ColumnRef` with an alias (Section 3.1), and it does
  NOT count as "dropping" the PK if the renamed column was part of it.
- `Aggregate`: `group_by` columns keep provenance if they're untouched
  passthroughs; the aggregate's own alias has none.

**Why this matters**: a `Join` between two relations, however derived
(not just base tables), is validated by checking whether one side's
column provenance resolves to "real FK to X" and the other's resolves to
"X's real PK," for the same X -- checked against the schema's declared FK
list, not by comparing surface-level column names. This generalizes
`grain.py`'s current join-check (which only really works for shallow joins
directly over base tables) to arbitrary-depth derived subqueries. Worked
example: `Project(ORDER_ITEM, [id, order_id, product_id])` joined to a
base `ORDER` table on `order_id = id` validates because `order_id`'s
provenance says "real FK to `ORDER.id`" and `id`'s provenance says "is
`ORDER`'s real PK" -- even though the left side is a derived subquery three
operations removed from any base table.

### 4.2 Row-count tracking, per operator

Every `Relation` node carries its own row-count variable, connected to its
operand(s) by a structural equation:

- **`BaseTable`**: its own free/fact-pinnable variable (as `TABLE_
  CARDINALITY` works today).
- **`Join`**: `row_count(join) = row_count(child)` **exactly** -- a fixed
  identity, coefficient 1. This is guaranteed by the FK-PK-restricted +
  LEFT JOIN model (Section 5): the join can never fan out (parent PK is
  unique) or drop rows (LEFT JOIN preserves every child row), so there's a
  strict 1:1 correspondence between child rows and joined-result rows.
- **`Project`**: `= row_count(source)` exactly (doesn't touch rows).
- **`Fanout`**: `= row_count(parent)` exactly (guarantees every parent row
  survives, by definition).
- **`Filter`**: `row_count(filtered) = row_count(source) x selectivity_
  factor` -- see Section 4.3.
- **`Aggregate`**: its own NEW free variable (the count of distinct
  `group_by` combinations actually present in the source) -- NOT a
  deterministic function of the source's row count alone, since it depends
  on the actual distribution of values. Naturally bounded `[1, row_count
  (source)]` (domain metadata, like `Variable.lower_bound`/`upper_bound`
  already work). Connects directly to `COUNT_DISTINCT` (Section 3.3) --
  the same underlying quantity.

**These row-count/selectivity identities are auto-generated from Relation
STRUCTURE, not from any NL fact.** This means the object model needs a way
to represent fact-independent structural equations (no `fact_references`,
or an explicit tag distinguishing them) -- REPRESENTATION deferred to a
later, lower-stakes decision, but the underlying math is fully settled
here. `conflict_reconciler` needs different handling for a conflict among
these: there's no fact to blame for "misextraction," so a conflict here
signals a genuinely different problem -- a structural/data inconsistency,
not a misread fact.

### 4.3 The mandatory selectivity-factor variable

Every `Filter` node **always** mints a selectivity-factor variable (the
fraction of source rows passing the filter) -- confirmed mandatory, not
lazy/on-demand. Pinned if some fact states it directly, otherwise a
genuine free/loose DOF variable for Stage 4 to resolve.

**Unifies with the existing nullable-FK presence-rate variable**
(`{col}.is_present`, built earlier for the narrowing-fix work): presence-
rate IS just the selectivity factor of the implicit `col IS NOT NULL`
filter -- one mechanism, not two.

### 4.4 Nullability

**Nullable-FK joins use LEFT JOIN semantics**: every child row is
preserved exactly once, including rows where the FK is null; the parent's
columns become nullable in the joined result. This is a deliberate pivot
from the old model's inner-join-style population-shrinking framing (see
Section 5 for why the population size is actually unchanged by this).

**`Filter` narrows downstream nullability**, using the full three-valued-
logic version (not just a literal `IS NOT NULL` check): narrowing
propagates through `RAnd`, a single `RComparison`/`RBetween`/`RInSet`/
`RNotInSet`, and `RNot` (negating an unknown stays unknown, still
excluded) -- but explicitly NOT through `ROr` in general (a row can
survive via one disjunct alone, so OR tells you nothing about another
disjunct's referenced column's nullness, unless every branch independently
requires the same non-null, which isn't worth checking).

**This narrowing is specifically a `Filter`/`Relation`-side concept**, not
a general property of `Condition`'s own logical connectives (Section 7.3).

### 4.5 `HAVING` and `Project`'s hard rules

`HAVING` is modeled as `Filter(source=Aggregate(...), ...)` -- no separate
node type. A filter-after-aggregate may reference the aggregate's alias/
`group_by` columns, never a raw pre-aggregation column that didn't
survive; a filter-before-aggregate may not reference an aggregate alias
that doesn't exist yet.

`Project` can **never** drop primary-key columns -- a hard validation
error, not a flagged-but-legal state. Columns it drops are genuinely
inaccessible to anything above it in the tree (this is what makes "column
dropping" real rather than cosmetic).

---

## 5. Population identity (why something Grain-shaped survives)

Initial hope: with proper PK/provenance/row-count tracking, the separate
`Grain`/`narrowed`-flag machinery might be fully retirable. **This turned
out to be wrong**, via two worked counter-examples:

1. `Aggregate(GROUP BY customer_id over ORDER)`, re-rooted to `CUSTOMER`,
   vs. the raw `BaseTable(CUSTOMER)` -- both report identical `(table, PK)`
   but describe different populations ("customers with >=1 order" vs "all
   customers"), because a `GROUP BY` can only ever produce a row for a
   value that actually appears in the source.
2. `BaseTable(CUSTOMER)` vs. `Filter(BaseTable(CUSTOMER), region='US')` --
   IDENTICAL schema (Filter doesn't touch columns), still different
   populations.

**Resolution**: the real invariant is the full OPERATION HISTORY a
relation was derived through -- joins, aggregates, AND filters (the old
`Grain` never tracked filters, since `Filter` didn't exist as a node
before this redesign). So a narrower, Filter-aware successor to `Grain`
survives, used for:

- Deciding whether two `Distributed`/`Correlated` facts about "the same"
  quantity are actually comparable/mergeable (`is_comparable_with
  (population_sensitive=True)`, reused unchanged in spirit).
- The cross-fact reconciliation mechanism below.

**Companion rule**: `Aggregate`'s re-rooted result is never allowed to
claim the parent table's own full identity -- only `Fanout` can, since only
`Fanout` actually guarantees every parent row survives.

---

## 6. Cross-fact reconciliation across related-but-different populations

### 6.1 Same population: direct algebra

If two facts about the same quantity are stated over the same population,
checking consistency is exact linear algebra: any elliptical distribution
(Gaussian, Student-t with `nu > 2`) has `rho_ij = Sigma_ij / sqrt(Sigma_ii
* Sigma_jj)` -- extract the implied correlation from a stated covariance/
scale matrix, compare directly against another stated value for the same
entry. A **precondition conflict**, simpler and distinct from a numeric
mismatch: Student-t's correlation is only defined for `nu > 2` -- stating
both a Pearson correlation AND a Student-t family with `nu <= 2` for the
same pair is a deterministic conflict because the quantity doesn't exist
for the stated family, not because two numbers disagree.

### 6.2 Related-but-different populations: the law of total covariance

For a fact over the whole population vs. another over a filtered subset,
rather than refusing to compare them or defaulting to "always flag":

```
Cov_overall(X,Y) = sum_g p_g * Cov_g(X,Y)  +  sum_g p_g * (mu_g^X - mu^X)(mu_g^Y - mu^Y)
```

(law of total variance is the same identity with X=Y; law of total
expectation, `mu_overall = p*mu_subset + (1-p)*mu_complement`, is the
mean-only special case).

**Algorithm**: treat the complement group's statistics (and/or the
selectivity factor itself, if unpinned -- Section 4.3) as unknowns. Solve
the system for what they would have to be for both facts to hold
simultaneously. Check whether the IMPLIED values are themselves
mathematically valid: variance >= 0, correlation in `[-1,1]`, Cauchy-
Schwarz (`|Cov(X,Y)| <= sqrt(Var(X)*Var(Y))`). **Only flag a conflict when
the required values come out impossible** -- a genuine, provable proof
that no database could satisfy both facts together. If valid (or
underdetermined because the selectivity/proportion isn't known), there is
no conflict.

This is a real constraint-satisfaction question specifically because the
selectivity factor is now a tracked, pinnable-or-free DOF variable
(Section 4.3) -- "does any valid assignment of the free variables satisfy
both facts and the validity constraints."

**Two honest limits**: needs the subgroup's population share to fully pin
down (without it, only Cauchy-Schwarz-style bounds are checkable, valid
for any proportion); only proves/disproves MOMENT-level consistency
(means/variances/covariances), not full distributional-shape consistency
-- deliberately out of scope, same tier as the other deferred items.

---

## 7. The `Condition` grammar

### 7.1 Expression layer

`RLiteral`, `RColumnRef` (unqualified, resolved against whatever the
enclosing `Relation` makes accessible), `RArithmetic` (+,-,*,/),
`RAggregateRef` (references an `Aggregate`'s alias by name -- must match a
REAL alias declared in the `Relation` tree, not merely checked for
non-emptiness).

**Condition needs its own bottom-up type-inference pass**, mirroring
`Relation`'s schema synthesis: every expression node synthesizes its own
result type from its operands' already-inferred types. `RAggregateRef`'s
type comes from the aggregate's own inferred result type in the `Relation`
schema (`COUNT`->integer, `AVG`->float, `MAX`/`MIN`->source column's type,
`PERCENTILE`->source column's type).

### 7.2 Predicate layer

`RComparison`, `RAnd`/`ROr` (2+ operands), `RNot`, `RBetween`, `RInSet`/
`RNotInSet`, `RIfThen`.

**`RExists`/`RNotExists` recommended for removal entirely.** Both
realistic use cases already reduce more precisely to existing primitives:
"child references an existing parent" is guaranteed by a non-nullable FK
by construction (or is exactly the presence-rate variable if nullable,
Section 4.3); "parent has >=1 child" is exactly `Fanout`'s `min_fanout >=
1`. Keeping them adds no expressiveness the model doesn't already have
more precisely, and they're a live landmine otherwise (would silently
misclassify or drop if ever emitted).

**The `~` `RComparison` operator is dead.** Originally flagged as unused;
briefly considered resurrecting it for "distributionally pinned to"
semantics, but `Distributed` (Section 8.1) fully absorbed that use case as
its own cohesive term. `~` has no remaining use -- delete it.

**Type-compatibility rules for comparisons**: cross-family (string vs.
numeric) rejected outright; numeric-numeric cross-subtype (int vs. float)
allowed via ordinary promotion; boolean-numeric comparison deliberately
NOT allowed (no implicit bool-as-0/1).

### 7.3 Nullability-narrowing scope, clarified

The three-valued-logic narrowing (Section 4.4) is specifically a `Filter`/
`Relation`-side concept. An `RIfThen`/`RAnd`/`ROr` appearing as a
`Constraint`'s own top-level assertion isn't filtering any rows -- it's
just asserting a fact that holds or doesn't over whatever population the
`Relation` already produced. Narrowing only applies when a `Condition`
sits inside a `Filter` node.

### 7.4 Cohesive terms reuse ordinary column resolution

`Distributed`/`Correlated`/`StateSequence`'s column references all resolve
via the exact same accessible-columns mechanism as ordinary `RColumnRef`,
including being allowed to reference an aggregate's alias (e.g.
correlating "average order total per customer" with "customer age" needs
an `RAggregateRef`, not just raw base columns) -- one unified resolution
path, no bespoke per-term mechanism.

---

## 8. The cohesive terms: `Distributed` and `Correlated`

**Shared rule (Section 9.3 has the third term, `StateSequence`, and the
full standalone-only rule)**: these terms are cohesive because their
parameters only mean something jointly -- unlike a bare moment fact
(Section 8.3), which decomposes cleanly into independent primitives.

### 8.1 `Distributed`

`family` + `parameters`, kept as one term (not decomposed into per-
parameter pins). Parameters can be PARTIAL -- a fact can state just the
family, or some-but-not-all parameters; missing ones become free/loose DOF
variables for Stage 4 (matching `VariableProbe`).

- **Column-type compatibility** is now precise, via the synthesized
  `Relation` schema's real column types (`CATEGORICAL` needs string/enum,
  `GAUSSIAN`/`LOG_NORMAL`/`BETA`/`UNIFORM` need numeric, `POISSON` needs
  integer/count-like).
- **Population/conditional scoping**: the old `if_condition` field +
  `ForkKeyRegistry` mechanism is FULLY OBSOLETE, replaced entirely by
  ordinary `Filter` + the selectivity-factor variable. If separate facts
  state every branch of a categorical fork key, their selectivities sum to
  1 ONLY when exhaustiveness over the column's full domain is confirmable
  (e.g. cross-referenced against a `Distributed`/`CATEGORICAL` fact
  enumerating the full category list); otherwise each branch's selectivity
  stays an independent free variable in `[0,1]`.
- **Only one univariate distribution per (column, table/intermediate
  relation)** -- but different distributions across a filtered vs.
  unfiltered version of the SAME column remain fine (different
  populations, Section 5).
- **Simplifying assumption (explicit, load-bearing)**: the source NL is
  assumed self-consistent -- no genuine contradictions, only possibly
  missing information. So "two facts assert different families for the
  same column+grain" is not a new conflict category needing new machinery
  -- it's the existing `confirmed_conflicts` equality check extended to
  cover the `family` attribute too, expected to resolve via MISEXTRACTION
  almost always rather than GENUINE_CONTRADICTION.
- **Cross-fact consistency at related grains** reuses Section 6.2's
  machinery via each family's own mean/variance formula (Gaussian's mean
  is its own parameter; Poisson's is `lambda`; Uniform's is `(min+max)/2`;
  Beta's is `alpha/(alpha+beta)`; **`LOG_NORMAL` needs care** -- its stated
  parameters are the mean/std of the UNDERLYING NORMAL, not the log-
  normal's own actual mean, so extraction must use `exp(mu + sigma^2/2)`,
  not the raw parameter; `CATEGORICAL` has no scalar mean for non-ordinal
  labels, so this cross-check is skipped for it, not forced).

### 8.2 `Correlated`

Generalized to an **arbitrary-arity column list** (not just 2) --
`Correlated(columns=[...], family=..., parameters=...)`, binary
correlation is just the `len(columns)==2` special case. Motivated by facts
naming a joint family across 3+ columns with ONE shared parameter (e.g.
multivariate Student-t with one shared degrees-of-freedom) -- forcing this
into pairwise decomposition would lose the sharing.

Parameters can be partial too (same rule as `Distributed`) -- missing
entries propagate to Stage 4 as free variables.

**"Must all N columns share a common grain?"** resolved as automatic-by-
construction WITHIN one `Correlated` term (its columns are only ever
whatever its own `Relation` already exposes -- `Condition` only sees what
its `Relation` produced). The real question was cross-fact merging of
SEPARATE `Correlated`/`Distributed` terms at possibly-different
populations -- resolved via `is_comparable_with` reuse (Section 5) for
same/comparable grains, and Section 6.2's law-of-total-covariance
machinery for related-but-different ones.

The underlying dependence-structure catalogue (from `MULTIVARIATE_
CONSTRAINT_DESIGN.md`, adopted wholesale): Gaussian (correlation matrix,
PSD required, checked via Cholesky), Student-t (correlation matrix + one
shared `nu > 0`), Clayton/Gumbel/Frank (Archimedean, asymmetric tail
behavior, exact parameter domains in that doc's Section 4), with vine
copulas (pair-copula cascade, validity = the purely combinatorial
"proximity condition") as the upgrade path for asymmetric/tail-dependent
facts beyond the Gaussian default.

#### 8.2.1 Categorical and mixed dependence -- polychoric/polyserial correlation

This closes the gap `MULTIVARIATE_CONSTRAINT_DESIGN.md` Section 5.1
flagged (discrete marginals break Sklar's copula-uniqueness guarantee) but
left undesigned. `Correlated` supports numeric-numeric, categorical-
categorical, AND mixed dependence, via well-established statistics:

- **Numeric-numeric**: ordinary Pearson/copula correlation (Section 8.2
  above) -- a numeric column is trivially its own latent variable.
- **Categorical-categorical**: **polychoric correlation** -- the
  correlation between the LATENT continuous variables presumed to underlie
  two categorical/ordinal columns, estimated via a bivariate-normal latent
  scale with threshold cut-points reproducing the observed category
  boundaries.
- **Numeric-categorical (mixed)**: **polyserial correlation** -- the same
  idea, one side already continuous, one side latent-thresholded.

This is ONE mechanism, not three: every pairwise entry, regardless of
pair-type, is still an ordinary correlation value in `[-1,1]` in the
shared latent space, so the chordality + PD-completion feasibility
machinery (`MULTIVARIATE_CONSTRAINT_DESIGN.md` Section 2.1, Grone-Johnson-
Sa-Wolkowicz 1984) applies completely unchanged regardless of pair-type.

**Two new requirements this creates**:
1. A categorical column participating in `Correlated` needs a known,
   finite, CONFIRMABLE category set (needed to define the latent-threshold
   cut-points at all) -- same exhaustiveness discipline as selectivity
   branches (Section 8.1).
2. Each participating categorical column needs its own latent-threshold
   cut-points tracked as new free/pinnable DOF variables.

### 8.3 `MomentTarget` is explicitly NOT a dedicated node

A bare, family-less statistical claim ("the average is $150") is ALWAYS
fully decomposed into an ordinary `Aggregate` (on the `Relation` side) +
`Comparison` (on the `Condition` side, via `RAggregateRef`). This is
different from `Distributed` specifically because a bare moment fact is a
single, independent, atomic value with no cross-parameter coupling.

---

## 9. The cohesive terms: `StateSequence`

### 9.1 Motivation and shape

A genuinely NEW, fourth constraint category -- not a DOF parameter pin, not
a joint/correlation fact, not even the per-row two-column comparison layer
-- for sequential/state-machine facts across MULTIPLE ROWS of the same
entity, ordered in time. Motivating example: "status must follow
ready -> packed -> shipped -> out_for_delivery -> delivered" over an
event-log-style table.

Fields: `partition_by` (grouping key, e.g. `order_id`), `order_by` (the
sequencing expression -- accepts ANY arithmetic expression, not just a
bare column, reusing `Project`'s expression grammar, e.g. `updated_at -
created_at`), `sequence_column` (the categorical column being tracked --
**must be discrete/categorical type**, a continuous column can't be "the
state"), and transitions -- supporting BOTH positive (allowed) and
negative (forbidden) assertions, explicitly so the full state machine can
emerge from many small independent facts rather than needing one fact to
state the whole picture (matching the atomic-fact-extraction philosophy
used everywhere else in this pipeline).

Chosen deliberately over general window-function support (Section 1) --
every realistic sequential/ranking case in this domain reduces to a
narrow, purpose-built term instead; a sibling `ConsecutiveRowConstraint`/
`RankConstraint` is the anticipated next one if a real case surfaces.

### 9.2 Cross-fact consistency algorithm

1. **Group facts by triple**: `(partition_by, order_by, sequence_column)`
   -- only facts sharing a triple are ever compared (same Grain-style
   scoping discipline as everywhere else).
2. **Union each fact's transitions** into one graph, split into ALLOWED
   and FORBIDDEN edges.
3. **Conflict type 1 (direct contradiction)**: an edge asserted both
   allowed and forbidden across different facts.
4. **Conflict type 2 (cycle formation)**: a cycle in the merged ALLOWED-
   transitions graph, standard DFS-based cycle detection -- but ONLY
   flagged when some fact explicitly marks the sequence as a strict/
   acyclic lifecycle. **Cycles are ALLOWED by default** (to avoid false
   positives on legitimately cyclic real flows like returns/reprocessing).
5. **No conflict** -> the union becomes the actual state machine handed to
   Stage 4, with any no-outgoing-edge state treated as a legitimate
   terminal state, not an error.

### 9.3 The standalone-only rule (applies to all three cohesive terms)

`Distributed`/`Correlated`/`StateSequence` can NEVER nest inside `RAnd`/
`ROr`/`RIfThen` with ordinary boolean predicates OR each other -- each is
always the SOLE top-level assertion of its own `Constraint`. Resolved via
a worked example: "for US orders, total ~ Gaussian(...)" is NOT expressed
by ANDing a comparison with a `Distributed` term (that doesn't type-check
-- `Distributed` is a population-level claim, not a per-row boolean); it's
expressed by scoping the population via `Filter` on the `Relation` side,
with `Distributed` left as the sole, standalone condition. Combining two
cohesive facts ("total ~ Gaussian AND quantity ~ Poisson") is simply two
separate `Constraint` objects, never one `Constraint` with an AND of two
cohesive terms.

---

## 10. Module layout

```
src/util/constraint_model/
├── __init__.py
├── relation/                  # the ON / relational algebra (Section 3-6)
│   ├── nodes.py                 # BaseTable, Join, Aggregate, Filter, Project, Fanout
│   ├── schema.py                 # bottom-up effective-schema synthesis
│   ├── sql_bridge.py              # bidirectional SQL<->object (sqlglot-backed)
│   └── validate.py                # FK-PK join checks, PK-never-dropped, alias collision
├── condition/                 # the Condition grammar (Section 7-9)
│   ├── expressions.py            # Literal/ColumnRef/Arithmetic/AggregateRef + type-inference
│   ├── predicates.py              # Comparison/And/Or/Not/Between/InSet/NotInSet/IfThen
│   ├── cohesive.py                # Distributed/Correlated/StateSequence (standalone-only)
│   └── validate.py                # column resolution, type-compatibility, nullability
├── constraint.py               # Constraint = Relation + Condition + metadata + severity
├── population.py               # the Grain-successor (Section 5): join+aggregate+filter history
└── variables.py                 # bridge layer: row-count/selectivity/latent-threshold
                                  # variables, feeding the EXISTING util/algorithms/dof_graph.py
```

No imports FROM this module into existing Stage 3 code yet -- built and
validated in isolation first. `util/algorithms/dof_graph.py`'s Dulmage-
Mendelsohn machinery is reused as-is, not reimplemented; `variables.py` is
a thin bridge producing `Variable`/`Constraint` objects for it, mirroring
how `constraint_graph.py` bridges `cross_shard.py` shapes into it today.

---

## 11. The `Constraint` object

### 11.1 No category/family tags

`statistical`/`structural`/`logic` tags are dropped entirely. The object's
own type (`Distributed` vs. `Correlated` vs. `StateSequence` vs. an
ordinary `Condition` shape) is already self-describing -- a separate string
tag was pure redundant bookkeeping tied to the OLD 3-agent extraction
split (Section 12).

### 11.2 Severity: hard by default, soft only as a reconciliation outcome

Every `Constraint` is `hard` by default -- no fact claims to be approximate
at extraction time. `soft` is ONLY ever assigned as an OUTCOME of
reconciliation: when a genuine, provable conflict between two hard facts
is found, the reconciler can downgrade (rather than leave a permanently-
unresolved `GENUINE_CONTRADICTION`) so both can coexist as approximate/
best-effort guidance.

- **Which fact(s) get downgraded is an LLM judgment call**, not a fixed
  rule -- explicitly captured in the reconciliation output, not left
  implicit.
- **`SOFTEN` is a 4th `ReconciliationVerdict` option**
  (`MISEXTRACTION`/`FALSE_POSITIVE`/`GENUINE_CONTRADICTION`/`SOFTEN`) --
  one widened enum, one call, not a separate follow-up step.
- **Softenability is gated by a fixed, deterministic, code-level rule
  based on constraint KIND**, not per-instance LLM judgment:
  - **Softenable** (genuine "approximately true" reading exists):
    `Distributed`, `Correlated`, plain aggregate-based moment facts.
  - **Never softenable** (binary integrity properties, or not even
    fact-derived at all -- "approximate" is incoherent): `StateSequence`
    transitions, uniqueness, NOT NULL, and the auto-generated fact-
    independent structural equations (Section 4.2).

---

## 12. Open, connected question: extraction architecture (not decided here)

Now that the representation is unified, should Stage 3's 3 separate
extraction agents (statistical/structural/logic) be combined into ONE
unified extractor? Default plan: build ONE unified agent, but treat this
as PROVISIONAL -- test it empirically against complicated real examples
once this object model and its validation machinery actually exist, rather
than deciding abstractly now (same build-then-validate discipline used
elsewhere in this project). Not resolved in this doc; flagged so it isn't
lost.

---

## 13. Explicitly open / deferred items (honest inventory, not glossed over)

- **Fact-independent structural equations' representation** (Section 4.2)
  -- the math is settled, the exact object-model encoding (no
  `fact_references`? an explicit tag?) is deliberately deferred.
  `conflict_reconciler`'s handling of a conflict among them is unresolved.
- **Non-chordal Dependency Matrix patterns** (`MULTIVARIATE_CONSTRAINT_
  DESIGN.md` Section 7) -- Grone et al.'s completion guarantee only covers
  the chordal case; non-chordal partial specifications need an explicit
  semidefinite feasibility solve or a documented "not supported" boundary.
- **Mixed continuous/categorical CI consistency in the fully general
  sense** -- Section 8.2.1's polychoric/polyserial mechanism closes the
  *correlation* gap for discrete marginals, but `MULTIVARIATE_CONSTRAINT_
  DESIGN.md` Section 5.3's separate finding (general CI-implication is
  undecidable, decidable only for bounded-cardinality variables) is a
  different, still-open problem for categorical CONDITIONAL INDEPENDENCE
  facts specifically, not resolved by this doc.
- **What Stage 4 does with a loose Dependency Matrix entry** -- hand it a
  probe, or use the max-determinant completion as a principled default?
  Not resolved (`MULTIVARIATE_CONSTRAINT_DESIGN.md` Section 7).
- **Whether vine-copula edges interact with conditional distributions** --
  not analyzed; also now needs rephrasing given `if_condition`/`ForkKey
  Registry` is fully obsolete (Section 8.1) -- a conditional `Correlated`
  fact would be scoped via `Filter` exactly like a conditional `Distributed`
  fact, but the interaction with vine-copula tree structure specifically
  hasn't been worked through.
- **Format/regex constraints, composite FKs** -- explicitly deferred non-
  goals (`STAGE3_CONSTRAINT_MODEL_UNIFIED.md` Section 3.7-3.8), unchanged
  by this doc.
- **Project's rename-drops-old-name default** (Section 4.1) -- stated as
  the working assumption (matches ordinary SQL projection semantics), not
  separately, explicitly re-confirmed as its own decision.
