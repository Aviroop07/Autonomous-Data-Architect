# Stage 3 Phase 2 -- Constraint Modeling: Architecture

Supersedes the constraint-extraction half of STAGE3_CONSTRAINT_TAXONOMY.md /
STAGE3_AGENT_TOPOLOGY.md / STAGE3_DETERMINISTIC_COMPILER.md (2026-07-07) with
the DOF-graph feasibility/loose-end-detection layer researched and verified
in STAGE3_DOF_OPEN_QUESTIONS.md. This doc is the synthesis: what to actually
build. Grounded in a real prototype against brainstorm_report.md's 89 facts,
not just synthetic examples -- see "Validated vs. design-only" at the end.

Phase 1 (schema sharding: `src/util/algorithms/sharding_ilp.py`,
`src/pipeline/stage3/models/shard.py`) is untouched by this doc and feeds
into it: each shard produces a local variable-constraint subgraph, merged
per the shard-merge architecture below.

## 1. Constraint taxonomy v2

Seven kinds of fact, corresponding to seven kinds of graph participation
(six graphed or graph-adjacent, one deliberately excluded). This replaces
the old taxonomy's flatter 5-class split (which didn't distinguish moment
targets from row formulas, or give conditionals a structural home).

| Kind | Example | Graph role |
|---|---|---|
| **Distribution parameter pin** | "mean 8, std 2" | One constraint-node PER parameter (never one node touching both -- see 2. below), edge to that parameter's variable-node. |
| **Moment/calibration target** | "order totals average $150" | One constraint-node touching every parameter its closed-form moment expression depends on (e.g. `{mu_qty, mu_price, N_LI_mean}`). Needs a NEW model (`MomentTarget`, section 4.1) plus a derivation-chain walk -- see 4. |
| **Structural bound** | "1 to 8 line items", "never negative" | NOT a graph edge. Metadata attached to the relevant variable-node (a domain bound), checked in a separate pass. |
| **Row-level derived formula** | `order_total = SUM(line_item.subtotal)`, `subtotal = quantity * unit_price` | NOT in this graph at all -- handled by the deterministic compiler's dependency-ordered execution. Modeled by `AggregationConstraint` (SUM/AVG/MAX/MIN of one descendant column) and unconditional `CrossColumnLogic` (`if_condition is None`) respectively. Section 4's moment-target walk READS these facts (to resolve a derived column's expected value) without graphing them itself. |
| **Conditional/decision-tree** | shipping_cost depends on loyalty_tier | Forks the graph: one independent subgraph per branch, keyed by the upstream categorical variable -- see 5. |
| **Cross-table parametrized distribution** | unit_price centered on the referenced product's base_unit_cost | **Not modeled by this design.** Real hard case (Q3 research), deliberately out of scope for this pass -- see "Non-goals." |
| **Cross-column correlation (D7)** | "older customers tend to spend more" | **NOT a DOF concept -- deliberately excluded from this graph entirely.** Correlation is a joint-distribution shape parameter, not a variable/equation. Modeled by `ColumnCorrelation` (section 4.5); consumed directly by Stage 4 generation via an already-researched-and-validated mechanism (topo-ordered conditional Gaussian copula), not built in this pass. |

## 2. The graph itself

- **Nodes**: one per distribution parameter (`mu_X`, `sigma_X`, ...), one per
  table row-count (`N_T`), one per FK fan-out mean (`F_{P,C}`) -- always
  scalar, never one node representing a whole distribution or a whole NL
  fact. This is Q1's finding, confirmed unanimous in the DAE-structural-
  analysis literature and now hands-on verified.
- **The one hard-earned modeling rule**: if a single NL fact states two
  independent parameter values (e.g. "mean 8 AND std 2"), it becomes **two
  separate constraint-nodes**, one per parameter -- never one constraint-node
  with edges to both. A single equation over two unknowns is mathematically
  underdetermined; conflating "one fact" with "one equation" was a real bug
  caught in the first (failed) run of `pyomo_dm_experiment.py`. Q1's "multi-
  variable edges are standard" finding still holds, but only for genuine
  multi-variable relationships (`subtotal = quantity * unit_price`), not for
  independent facts merely co-stated in one sentence.
- **Edges**: constraint-node to variable-node, wherever the constraint's
  expression references that parameter.
- **Tool**: `pyomo.contrib.incidence_analysis.dulmage_mendelsohn.dulmage_mendelsohn(graph_or_matrix, top_nodes=constraint_node_list)`
  -- note the import path is one level deeper than the module name suggests
  (`pyomo.contrib.incidence_analysis.dulmage_mendelsohn` is the submodule;
  the callable is inside it, same name). Accepts a bare `networkx.Graph` or
  `scipy.sparse.coo_matrix` directly -- confirmed no measurable performance
  difference between the two (0.9x), so use whichever is more convenient to
  build (networkx, for readability and provenance-tagging via node
  attributes).
- **Output**: `RowPartition`/`ColPartition` namedtuples, each with
  `unmatched`, `underconstrained`, `square`, `overconstrained` fields.
  `square` = exactly-determined. `unmatched`/`underconstrained` on the
  variable side = loose ends needing an LLM probe (section 6).
  `overconstrained` is a **block-level** classification -- when flagged,
  treat the whole connected component (all its rows AND columns) as one
  unit needing a numeric check, not a single "bad" constraint. This was the
  second real correction earned from the toy experiment: the matched row
  within an overconstrained block still gets labeled as part of that block,
  not as cleanly "square".

## 3. Inequalities: hybrid handling (Q2, Approach D)

- Single-variable bounds (`sigma >= 0`, `N_LI in [1, 8]`) are **node
  metadata**, not graph edges -- attached to the variable-node, checked in a
  feasibility pass after matching, never consumed as part of the DOF count.
- Multi-variable inequalities (`refund_amount <= order_total`) need interval/
  bounds propagation as a separate preprocessing pass. **Not implemented in
  this design pass** -- flagged as a residual item (fact 85's constraint in
  the real prototype was deliberately left unmodeled for exactly this
  reason).
- Two-pass feasibility overall (Q7): structural pass (the DM decomposition
  itself, cheap, run on every shard and every merge) then a numeric pass
  (only on subgraphs the structural pass actually flags as overconstrained
  -- evaluate the equations to distinguish harmless redundancy from a real
  contradiction). Never run the numeric pass universally; it's the expensive
  one.
- **LLM probing IS in scope here, but only for confirmed contradictions,
  never for free variables (corrected 2026-07-11, see the
  stage3_stage4_division_of_labor memory note).** Determining true
  infeasibility is explicitly Stage 3's own job, not something to hand off
  -- when the numeric pass confirms an overconstrained block is a genuine
  contradiction (not harmless redundancy), Stage 3 probes an LLM to
  reconcile it (most likely by re-examining the conflicting facts' source
  NL text -- was this a misextraction, or a genuine ambiguity that needs
  surfacing?), mirroring the existing Deterministic-Validation-+-LLM-Retry
  pattern used elsewhere in this codebase. This is narrower than, and
  different in kind from, the free-variable probe in section 6: a free
  variable gets reported to Stage 4 untouched; a confirmed contradiction
  gets Stage 3's own LLM-assisted reconciliation attempt before anything
  is reported onward. Not yet designed in detail (which agent, what
  prompt, what happens if reconciliation itself fails) or implemented --
  section 3's numeric pass itself remains entirely unbuilt (see Non-goals).

## 4. Moment/calibration targets (Q3)

**Scope correction from the first pass of this section (2026-07-11): the
original framing said "the dispatcher that decides which path applies, given
an arbitrary aggregation fact." That undersold the actual gap.
`AggregationConstraint` (e.g. `order_total = SUM(line_item.subtotal)`, fact
28) is NOT a Q3 problem at all -- re-reading its own fields against taxonomy
v2's row-level-derived-formula bucket (section 1) confirms it's a
deterministic, per-row formula belonging to the compiler
(STAGE3_DETERMINISTIC_COMPILER.md), unrelated to this graph. The real gap:
`constraints.py` has no model at all for a fact like "order totals average
around $150" (fact 30) -- it's not a `DistributionConstraint` (no stated
family, just a bare population mean) and not an `AggregationConstraint`
(that describes a row formula, not a population statistic). A new model is
needed before any dispatcher has well-typed input to consume.**

### 4.1 New model: `MomentTarget`

```python
class MomentTarget(ConstraintBase):
    table_name: str
    column_name: str
    statistic: Literal["MEAN"]  # scope: MEAN only for this pass -- see 4.4
    target_value: float
```

Lives in `StatisticalManifest` as a new `moment_targets: list[MomentTarget]`
field. `column_name` names the column the statistic describes -- which is
very often itself a *derived* column (like `order_total`), not a base column
with its own distribution.

### 4.2 The resolution algorithm: walk the derivation chain

Given a `MomentTarget` on `column_name`, resolve `E[column_name]` by
recursively walking backward through however that column is *defined*,
until hitting base columns with their own pinned parameters. Two things
define a column's value, and the walk alternates between them:

1. **It's the target of an `AggregationConstraint`** (`parent_column`
   matches): apply Wald's identity, `E[parent_column] = E[N] * E[descendant_column]`,
   where `E[N]` comes from the `FanoutConstraint` whose `(parent_table,
   child_table)` matches `(AggregationConstraint.parent_table,
   AggregationConstraint.descendant_table)`. If more than one `FanoutConstraint`
   matches (an ambiguous multi-path schema), **bail out** -- flag as
   unhandled rather than guess which path is intended.
2. **It's the LHS of an unconditional `CrossColumnLogic` fact**
   (`if_condition is None`, `then_enforcement` is an assignment like
   `"subtotal = quantity * unit_price"`): parse `then_enforcement` with
   `sqlglot` (`sqlglot.parse_one(expr)` gives an `EQ` node; `.expression` is
   the RHS operator tree -- confirmed hands-on, parses this exact shape
   cleanly into `Mul(Column, Column)`). Recognize only two RHS shapes for
   this pass: a product of two columns (`E[XY] = E[X]E[Y]`, assumed
   independent) or a sum of two columns (`E[X+Y] = E[X]+E[Y]`, always valid,
   no independence assumption needed). Anything else -- three-plus operands,
   division, `CASE`, a function call -- **bail out**.
   `CrossColumnLogic` is genuinely dual-purpose here: an instance with
   `if_condition is None` is this section's territory (an unconditional row
   formula the walk needs to read); an instance WITH `if_condition` set is
   section 5's territory (Q4's conditional branching). The model doesn't
   need to change -- only how each instance is routed.
3. **Base case**: `column_name` isn't defined by either of the above --
   it's a base column with its own `DistributionConstraint` (already a
   pinned Variable from section 2's adapter, e.g. `LINE_ITEM.quantity.lam`)
   or it's genuinely unconstrained (a free Variable, handled the same as any
   other loose end).

The walk terminates in one of two ways: a fully resolved symbolic expression
in terms of already-graphed Variables (emit ONE `Constraint` touching all of
them, edge-per-Variable, matching section 2's rules) -- or a bail-out at any
step, in which case the whole `MomentTarget` is unhandled for this pass (see
4.4), not partially resolved.

**Hand-traced against the real data (not yet run as code):** fact 30 (`E[order_total]=150`)
-> `AggregationConstraint` (fact 28, `SUM` over the `ORDER->LINE_ITEM` fanout,
fact 27) -> `E[order_total] = E[N_LI] * E[subtotal]` -> `subtotal` is itself
an unconditional `CrossColumnLogic` (fact 39, `subtotal = quantity *
unit_price`) -> product of two base columns, each already pinned
(`LINE_ITEM.quantity.lam` from fact 34, `LINE_ITEM.unit_price.mean` from fact
38) -> final equation: `N_LI_mean * lam_qty * mean_price = 150`, touching
exactly `{N_LI_mean, LINE_ITEM.quantity.lam, LINE_ITEM.unit_price.mean}` --
**the same equation `C5_moment_target` in the prototype was hand-built to
be**, now reconstructed by a general algorithm instead of by hand. Not yet
run as actual code -- next increment.

### 4.3 Independence is assumed, not verified

Independence is an explicit, stated simplifying assumption for every
product/sum step in the derivation walk -- not something checked. As of
2026-07-11, `ColumnCorrelation` (section 4.5) DOES let a fact state "these
two columns are correlated," but the walk itself doesn't consume it yet:
adding a covariance term to `_resolve_mean`'s product/sum steps when a
matching `ColumnCorrelation` exists is still a real, un-designed extension
(still closed-form, per the original Q3 research). Flagged as a known
limitation, narrower than before -- the missing fact type is no longer the
gap, the walk's lack of a covariance-aware combination rule is.

### 4.4 Bail-out trigger list -- Stage 3 tags, Stage 4 calibrates (corrected scope, 2026-07-11)

**Originally framed as "Stage 3 builds an MSM-based calibration loop."
Per the same probe-not-fill principle as section 6, that's Stage 4's job
-- MSM/ABC-style calibration requires actually running Stage 4's
generation code repeatedly, which is a generation-time concern by
definition.** Stage 3's job on a bail-out is narrower: tag the
`MomentTarget` as unresolved-needs-calibration (already partially done --
`constraint_manifest_to_graph_nodes()` logs and drops it) and hand it to
Stage 4 as a probe, the same as any other loose end, rather than Stage 3
attempting the calibration itself.

The walk bails out on exactly these conditions, not a vague "anything
complicated":
- `AggregationConstraint.operation` is `MAX` or `MIN` (no general closed
  form exists, per Q3's research).
- Ambiguous fanout resolution (multiple `FanoutConstraint` matches).
- A `CrossColumnLogic` RHS with more than 2 operands, division, or any
  non-arithmetic SQL construct `sqlglot` doesn't classify as `Add`/`Mul`.
- `MomentTarget.statistic` other than `MEAN` (variance/median targets are
  out of scope entirely for this pass, not just deferred to Stage 4).

None of these four cases is implemented in this pass -- they're the
explicit, enumerable boundary of what IS implemented, so a future increment
knows exactly what it's picking up.

### 4.5 Cross-column correlation (D7) -- scope decision, 2026-07-11

**Section 10 of the old taxonomy (`TAXONOMY_V2_DRAFT.md`) declared D7 out
of scope, but that recommendation was explicitly deadline-driven ("Twelve
days out," the original SIGMOD cutoff). The user's own Round 4 directive
on this exact item said the opposite -- "we have to do it for sure," must
be solved fundamentally, not deferred (`.setup-interview.md`). With the
SIGMOD deadline dropped in favor of VLDB's longer runway, that tension
resolves: D7 is back in scope.** The other Section-10 future-work items
(general query cardinality, exact-ratio enforcement, multi-row DC repair,
semantic-realism lexicons, polymorphic FKs) do NOT have the same explicit
override and remain deferred -- confirmed with the user rather than
assumed.

**Not a DOF concept.** Correlation is a joint-distribution shape
parameter -- it doesn't pin or free a variable the way a mean or row-count
does, so it's excluded from this graph entirely, in the same bucket as
`UniqueConstraint`/`FormatConstraint`.

**The mechanism is already researched and validated, not a new unknown.**
`experiments/taxonomy_research/OPENCODE_CORRELATIONS.md` (2026-07-06)
surveyed Gaussian copulas, vine copulas, Iman-Conover, conditional chains,
and NORTA, and recommended a topo-ordered conditional Gaussian copula
(continuous columns) + CPTs (categorical) as primary, because it composes
correctly with hard bounds and FK ordering where the alternatives break
them. `CORRELATION_VALIDATION_REPORT.md` then empirically validated it:
1,240 trials, 220/220 achievable targets passed, calibration bias <0.01 at
n>=10000, verdict "READY for Stage 4 implementation" with two caveats:
calibration is mandatory (raw copula has large systematic bias), and
achievability must be checked at planning time (a bound like `Y <= X`
imposes a correlation floor no algorithm can go below).

**What this pass adds:** just the fact model, `ColumnCorrelation`
(`table_name_a/column_name_a`, `table_name_b/column_name_b`, `direction`,
qualitative `strength`) in `StatisticalManifest.correlations`, deliberately
excluded from the DOF graph adapter. The actual copula sampling mechanism
is Stage 4 generation-time work -- Stage 4 is currently kept only as a
reference (`parameter_agent`), not being rebuilt in this session, so
implementing the sampler itself is out of scope for this increment. The
derivation walk's independence assumption (section 4.3) is a related,
narrower gap: consuming a `ColumnCorrelation` fact to add a covariance term
in `_resolve_mean` is still undesigned.

## 5. Conditional/decision-tree branches (Q4)

**Forked subgraph, not a mixture-distribution node** -- precedented by
multi-mode DAE structural analysis and conditional/dynamic CSP literature,
both of which analyze each mode/branch as an independent local subgraph
rather than folding branches into one node's equation (which would inject a
bilinear moment term into every conditional column, compounding directly on
section 4's already-hard problem).

Concretely: `shipping_cost`'s Platinum/non-Platinum split becomes two
independent subgraphs, one per branch, each with its own local variables
(`mu_ship_platinum` -- trivially a constant 0, no free parameter in our real
example -- and `mu_ship_nonplat`/`sigma_ship_nonplat`, which the real
prototype validated). Recombination (e.g. a downstream moment constraint
that needs the *overall* population mean across both branches) is a
separately-scoped step, not baked into the per-branch matching.

### 5.1 Fork-Key Registry (Deduplication and Shard-Merge)

**How it works:**
A new ForkKeyRegistry pass runs over the ConstraintManifest before graph construction.
1. **Discover Forks:** Scan all DistributionConstraint and CrossColumnLogic instances for an if_condition. (Note: DistributionBase needs an if_condition: Optional[str] = None field added for statistical conditional branches).
2. **Parse Fork Key:** Use sqlglot to parse the if_condition (e.g., CUSTOMER.loyalty_tier = 'Platinum'). Extract the Column as the ForkKey (CUSTOMER.loyalty_tier) and the branch literals (['Platinum']).
3. **Register Exhaustive Branches:** Look up the ForkKey in manifest.statistical.distributions. It must be a CategoricalDistribution. Extract its categories and register the ForkKey with its exhaustive branches in the registry.
4. **Merge Reconcile:** Because shards use the same ForkKey string representation, unioning them across shards naturally deduplicates the branches (a single loyalty_tier entry in the registry tracks all 4 tier categories).

### 5.2 Variable Expansion

Instead of creating entirely separate DOFGraph objects per branch, the independent subgraphs are embedded into the single global DOFGraph using namespace suffixes on the Variables.
- If a constraint lacks an if_condition, its nodes are created normally (shared).
- If a constraint has an if_condition, the registry maps that condition to its matching branches (e.g. != 'Platinum' maps to Bronze, Silver, Gold).
- The graph adapter creates independent Variable nodes for *each* matching branch via a suffix: ORDER.shipping_cost.mean|CUSTOMER.loyalty_tier=Bronze.
- CrossColumnLogic (Platinum = 0) targets only the |...=Platinum variable. GaussianDistribution targets the other three. This isolates the subgraphs structurally without cross-talk, exactly matching the DAE multi-mode requirement.

## 6. Probing loose ends to Stage 4 (Q6) -- CORRECTED SCOPE, 2026-07-11

**This section originally designed a Stage-3-resident LLM agent that
guesses plausible values for free (`loose`) variables. That's the wrong
job for Stage 3.** Explicit user correction: "Stage 3 is NOT supposed to
fill up gaps for the flexible variables, it is supposed to acknowledge the
degree of freedoms and probe them to Stage 4. Stage 4 will decide how to
handle the variables (most probably we will generate parameterized code)."
See the memory note `stage3_stage4_division_of_labor` for the full
principle. The DOF graph's `loose` classification already does the
acknowledging half correctly (section 2) -- what was missing wasn't an
LLM agent, it was a clean way to hand that list to Stage 4.

**Corrected design: Stage 3 emits a probe list, not a filled-in value.**
For every `loose` variable in the final merged graph, `ConstraintManifest`
(or a new output alongside it) needs to carry: the variable's name, its
known bounds (if any, from section 3's metadata), and its provenance
(which facts touch it, even if none pin it). No LLM call happens in Stage
3 for this. Stage 4 -- most likely via generating parameterized code
rather than hardcoding a guessed value -- decides what to do with each
probe. Stage 4's existing `parameter_agent`
(`src/pipeline/stage4/agents/parameter_agent/`, retained as a conceptual
reference within this document's framing, not as an actively maintained
code path) already does LLM-based numeric estimation with
`EvidenceStore` web-search grounding -- if an LLM-filled value is ever
needed, that's where it belongs, not a new Stage 3 agent.

**Implemented (2026-07-11):** `src/pipeline/stage3/models/probe.py` --
`VariableProbe` (variable_name, bounds, fact_references), `MomentTargetProbe`
(the original bailed MomentTarget's table/column/statistic/target_value,
so whatever consumes it has the stated target, not just "unresolved"),
and `Stage3AnalysisReport` bundling `square_variables` (informational),
`loose_variable_probes`, `unresolved_moment_target_probes`, and
`overconstrained_blocks` (with an `is_feasible` convenience property).
`analyze_constraint_manifest()` in `constraint_graph.py` is the single
entry point: builds the graph, classifies it, and packages the result --
Stage 3's job ends here. `constraint_manifest_to_graph_nodes()` now
returns the list of bailed MomentTargets as a third value (previously
print()-and-dropped with no structured signal) so `analyze_constraint_manifest`
can wrap them into probes. 4 new tests, including the real fact-chain
(28/30/34/38/39) confirming a fully-resolved manifest produces zero probes.
Full suite: 404/404.

**Still not designed:** whether Stage 4's `parameter_agent` needs any
changes to consume a `Stage3AnalysisReport`'s probes instead of whatever
input it currently expects -- explicitly out of scope for this pass per
user instruction (Stage 4 is not being touched this session), deferred to whenever
this is picked up as an increment.

## 7. Shard-then-merge (Q5)

Full recompute of the DM decomposition on every merge -- no incremental
matching machinery. Confirmed via literature search that no incremental
Dulmage-Mendelsohn algorithm exists at all, and via hands-on measurement
that full recomputation is fast enough to not matter (13ms at a realistic
balanced shape; the real prototype's actual shape, 9 constraints/20
variables/11 edges, is far smaller still and trivially fast). The
originally-estimated "sub-millisecond at ~1000 variables" claim was wrong as
stated (604ms measured at that literal, but unrealistically over-determined,
shape) -- the corrected, shape-aware conclusion still supports full
recompute, just not with the original number.

### 7.1 Fact allocation and cross-shard stubs (2026-07-12)

**Corrected a wrong assumption found during extraction-agent research:**
Phase 1's ILP sharder (`sharding_ilp.py`) does NOT guarantee that two
tables connected by a fact land in the same shard. FK closure (its
lines 48-50) is a soft reward term in the objective, competing against
`max_tables_per_shard` and other capacity penalties -- once capacity
binds, a fact can legitimately end up with no single shard containing
all its referenced tables. Cross-shard fact handling is therefore the
normal case for extraction, not a rare edge case.

`fact_allocation.py`'s `allocate_facts_to_shards()` was fixed accordingly:

- **Orphan recovery is now table-mention-aware.** A fact the registry
  never associated with any table (a genuine orphan) is first checked via
  a deterministic keyword-membership scan (`find_mentioned_tables()`) --
  does its raw text name a known table, or a simple natural-language
  variant (`LINE_ITEM` -> "line_item"/"line item"/"line items")? If so, it
  routes to the shard covering the MOST of its mentioned tables. Pure
  fact-to-fact text similarity (the original, only mechanism) is now a
  last resort for facts mentioning no known table at all -- it has zero
  notion of table identity, which made it a poor fit for exactly the
  cross-shard case this fix targets.
- **Every shard's allocation now reports `stub_tables`.** For each fact
  allocated to a shard (via any of the three paths), any table it
  mentions that ISN'T in the shard's own table set is collected as a stub
  requirement -- this is the schema-only (table + column names, no data)
  context the shard's future extraction agent will need injected into
  its prompt to correctly reference a foreign table (e.g. emit
  `AggregationConstraint(descendant_table="LINE_ITEM", ...)` from a
  shard that only has ORDER). Return type changed from a raw
  `list[list[int]]` to `list[ShardFactAllocation]` (`fact_ids` +
  `stub_tables`) to carry this.

**Not yet built:** the actual stub CONTENT construction (pulling table +
column names from the merged schema for each `stub_tables` entry) and
wiring this into an extraction prompt -- both belong to the
not-yet-designed extraction-agent orchestration, not this allocation
step. 6 new tests, full suite 410/410.

## 8. Pipeline shape (putting it together)

```
per shard:
  3-agent extraction (Statistical / Structural+Aggregate / Logic -- reuse
  the AgentLoop graph machinery in src/util/orchestration/loop_types.py,
  which is what Stage 1/2 actually use)
    -> constraint facts, tagged by kind (taxonomy v2, section 1)
    -> build local variable-constraint graph (section 2)
    -> structural feasibility pass (Pyomo DM) on the local graph

merge (mirrors Phase 1's shard-merge):
  union local graphs -> full recompute (section 7)
    -> numeric pass ONLY on flagged overconstrained blocks (section 3)
       -> harmless redundancy: resolved, no further action
       -> CONFIRMED contradiction: Stage 3 probes an LLM itself to
          reconcile it (re-examine the conflicting facts' source text --
          misextraction vs. genuine ambiguity), per the corrected
          division of labor -- this is the one place Stage 3 calls an
          LLM, since determining true infeasibility is its own job
    -> moment-target resolution: closed-form where it applies (Q3's walk),
       else a MomentTargetProbe (section 4.4, corrected scope)
    -> conditional branches resolved as forked subgraphs, recombined only
       where a downstream target needs the combination (section 5)

Stage 3's output (analyze_constraint_manifest(), implemented 2026-07-11):
    -> square_variables (informational -- what got determined)
    -> loose_variable_probes + unresolved_moment_target_probes -- reported
       AS-IS to Stage 4, never filled in by Stage 3 (section 6, corrected
       scope)
    -> overconstrained_blocks / is_feasible -- Stage 3's own feasibility
       verdict, after the LLM-reconciliation attempt above

output: ConstraintManifest (extend the kept constraints.py's existing
StatisticalManifest/StructuralManifest/LogicManifest shape with the new
moment-target, correlation, and forked-branch representations) plus a
Stage3AnalysisReport (src/pipeline/stage3/models/probe.py)
```

## Non-goals for this pass (explicitly out of scope, not overlooked)

- Cross-table parametrized distributions (unit_price centered on the joined
  product's base_unit_cost) -- Q3's hard case, no closed-form derivation
  attempted here, not modeled in the real prototype either.
- Multi-variable inequality propagation (`refund_amount <= order_total`) --
  noted in section 3, not implemented.
- ~~The fork-key registry's actual implementation (section 5)~~ --
  **implemented (2026-07-11)**: `fork_registry.py` (`ForkKeyRegistry`,
  `parse_if_condition` via `sqlglot`, EQ/NEQ/IN operators), wired into
  `constraint_manifest_to_graph_nodes()`; branch-suffixed variable
  expansion for both conditional distributions and conditional
  `CrossColumnLogic`. Committed as 0a60c28.
- ~~The `MomentTarget` model and the derivation-chain walk (section 4)~~ --
  **implemented (2026-07-11)**: `MomentTarget` in `constraints.py`,
  `moment_target_to_graph_nodes()` + the recursive `_resolve_mean()` walk in
  `constraint_graph.py`, wired into `constraint_manifest_to_graph_nodes()`.
  `sqlglot` is now a real dependency (`pyproject.toml`, installed into
  `venv`). Tests in `test_stage3_moment_target.py` reproduce the exact
  hand-traced equation from the real fact chain (28/30/34/38/39) as
  executable code, plus cover all four section 4.4 bail-out triggers and
  the AVG/free-variable edge cases. 39/39 passing, 393/393 full suite green.
- The MSM-style fallback itself (section 4.4's four bail-out triggers) --
  still entirely unbuilt, only its trigger conditions are now enumerated.
- Materialization of derived aggregate columns (`order_total`,
  `insured_value`, `total_weight`) as real schema columns -- resolved as a
  design decision in the earlier chat discussion (auto-materialize,
  mirroring Stage 2's `wire_orphan_fk_columns()` pattern) but not
  implemented; this belongs to the deterministic compiler, not this graph.
- The `ColumnCorrelation` (D7) sampling mechanism itself (section 4.5) --
  the fact model is implemented (2026-07-11) and deliberately excluded from
  this graph, but the actual topo-ordered conditional Gaussian copula
  sampler is Stage 4 generation-time work, out of scope while Stage 4 is
  kept only as a reference. The derivation walk also doesn't consume
  `ColumnCorrelation` facts yet (section 4.3's covariance-term gap).
- The other Section-10 future-work items (general query/projection
  cardinality, exact-ratio soft-constraint enforcement, multi-row DC repair,
  semantic-realism lexicons, polymorphic FKs) -- confirmed with the user
  (2026-07-11) as still deferred, unlike D7; none had an explicit override.

## Validated vs. design-only

**Hands-on verified, on real data, not assumed:**
- The graph construction convention (parameter-level nodes, split
  multi-parameter facts) -- section 2.
- Pyomo's `dulmage_mendelsohn` actually works as designed, correct import
  path, correct interpretation of its output -- section 2.
- The moment-target mechanism actually resolves a real calibration target
  (`N_LI_mean`) via pure graph elimination -- section 4's closed-form path,
  concretely, not just in theory. As of 2026-07-11, this is now an
  executable, tested code path (`moment_target_to_graph_nodes()`), not just
  a hand trace.
- Performance is a non-issue at the real schema's actual shape/scale --
  section 7.
- Inequality-as-metadata and the two-pass feasibility split are consistent
  with everything measured so far, though the multi-variable propagation
  half was deliberately not exercised -- section 3.

**Implemented and tested (2026-07-11):**
- The probe-list output for bailed MomentTargets and loose DOF-graph
  variables (sections 4.4 and 6, corrected scope) --
  `src/pipeline/stage3/models/probe.py` + `analyze_constraint_manifest()`.
  Stage 3's job ends at tagging + reporting, never calibrating or
  guessing. 4 tests, including a real-fact-chain case confirming zero
  probes when everything resolves.

**Researched and reasoned through, but not yet built or stress-tested:**
- Whether/how Stage 4's existing `parameter_agent` needs to change to
  consume a `Stage3AnalysisReport`'s probes (section 6) -- not designed,
  and explicitly out of scope while Stage 4 isn't being touched.
- The `ColumnCorrelation` (D7) sampler (section 4.5) -- researched AND
  empirically validated (1,240 trials, `CORRELATION_VALIDATION_REPORT.md`,
  "READY for Stage 4 implementation"), but not built here; the fact model
  itself is built and excluded from this graph.
- Cross-table parametrized distributions and multi-variable inequalities --
  deliberately deferred, not designed at all (see Non-goals).
