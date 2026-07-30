# Stage 3 Phase 2 -- Constraint Feasibility & Loose-End Detection: Research Questions -- All RESOLVED (archived 2026-07-12)

Companion to STAGE3_CONSTRAINT_TAXONOMY.md / STAGE3_AGENT_TOPOLOGY.md /
STAGE3_DETERMINISTIC_COMPILER.md. Those docs specify the 5 constraint classes
and the 3-agent extraction pipeline; this doc concerns the layer above that --
how Stage 3 determines, given an extracted constraint set, (a) whether it is
jointly feasible (per-shard and globally) and (b) which variables are still
"loose" (free/underdetermined) and need an LLM probe, analogous to Stage 2's
merge-conflict probing.

**Research pass completed 2026-07-11**: all 8 questions researched (Q1/Q2/Q5
via OpenCode, Q3/Q4/Q6/Q8 via background Agent subagents, run in parallel).
Findings below; a few claims are flagged as not yet independently verified
(see each question). Q7 was folded into the synthesis rather than researched
separately -- it's an engineering-depth scoping call, not a literature
question.

**Implementation status (2026-07-12):** Q3 (moment propagation) and Q4
(forked subgraphs for decision-tree branches) are not just researched but
fully implemented and tested in tracked code (see STAGE3_PHASE2_DESIGN.md
sections 4-5, and the committed code in `src/pipeline/stage3/`).

See baselines/experiments/STAGE3_PHASE2_DESIGN.md for the current spec.

## Guiding framework -- CONFIRMED as the right analogue, with a concrete
## implementation path

Structural/degrees-of-freedom analysis of a bipartite variable-constraint
graph, via Dulmage-Mendelsohn decomposition. Research confirmed this is not
just "the closest analogue" -- it's a live, current, off-the-shelf-supported
technique (see Q8), and the literature (Murota, Pantelides, Pryce) is
unambiguous about the base convention (see Q1). The genuinely open part
remains applying it to a system that mixes deterministic equalities,
inequalities, probabilistic/distributional assignments, soft population-level
moment targets, and discrete conditional (decision-tree) branching -- Q2-Q6
below are exactly that extension work, and two of them (Q3, Q4) turned out to
be more tractable than expected, with real literature precedent.

## Open Questions

### Q1. Variable granularity -- RESOLVED

**Answer: parameter-level granularity.** Each distribution parameter (mu_X,
sigma_X, N_T, F_{P,C}, ...) is its own scalar variable-node. This is not a
judgment call -- the DAE structural-analysis literature is unanimous:
Murota's foundational bipartite-incidence formulation, Pantelides' algorithm,
Pryce's Sigma-method, the flattened Modelica DAE representation, and Pyomo's
`IncidenceGraphInterface` all operate on individual scalar variables, never
on composite objects. A single fact pinning two parameters at once ("mean 8,
std 2") is just an ordinary multi-variable edge -- one constraint-node
touching two variable-nodes -- which is the standard case in these graphs,
not a complication. This granularity is also what makes **partial pinning**
representable at all: a fact that gives only the mean leaves sigma as a
separate, genuinely free node, which a column-level or object-level
granularity couldn't express.

### Q2. Inequalities -- multiple viable approaches found, one recommended

Classical DOF/matching analysis is equality-only; a one-sided bound doesn't
"pin" a value the same way. Research surfaced **four real approaches with
different tradeoffs**, not a single answer:

- **(A) Exclude from matching, post-check.** Equality-only graph for the
  matching; inequalities checked separately once equality-determined values
  exist. Pyomo's own `IncidenceGraphInterface` has an `include_inequality`
  flag (default True, but commonly set False for structural-singularity
  debugging per its own documentation/tutorials) -- this is the precedented
  convention in practice. *Not independently re-verified against a live
  Pyomo install in this repo's venv (Pyomo wasn't installed when checked) --
  Q8 did independently confirm Pyomo's `dulmage_mendelsohn` module is real
  and current, which lends the broader `incidence_analysis` package
  credibility, but the specific `include_inequality` flag claim should be
  spot-checked once Pyomo is actually installed.* Weakness: can't detect an
  empty feasible set (e.g. `x>=5` and `x<=3`) until the post-check runs.
- **(B) Domain-bound metadata + interval propagation.** Single-variable
  inequalities decorate the variable node (a bound, not a matching edge);
  multi-variable inequalities get AC-3-style bounds propagation as a
  preprocessing pass. Precedented in numeric CSP (Lhomme 1993, Benhamou et
  al. 1995) and explicitly stated as a definition in the geometric
  constraint-solving literature ("inequality constraints do not affect DOF;
  they restrict the domain of the solution search" -- Gershon et al.).
  Composes well, catches `x>=5 and x<=3` during propagation itself.
- **(C) Include in matching with active/inactive semantics.** Standard in
  NLP/optimization DOF counting (active inequalities count like equalities;
  inactive ones don't) and in DAE complementarity conditions. Rejected as
  the general-purpose default: determining which inequalities *could* be
  active is NP-hard in general (2^k combinatorial assignments), and Lacroix/
  Mahjoub/Martin (2009) proved structural analysis for *conditional* DAE
  systems (equations that change form) is NP-complete -- directly relevant
  since our decision-tree constraints (Q4) are exactly this kind of
  conditional system. Overkill for what are mostly simple domain bounds
  (`sigma>=0`, `N>=1`) in our case.
- **(D) Hybrid of A+B (RECOMMENDED).** Single-variable bounds as node
  metadata; multi-variable inequalities via interval propagation; the
  equality-only graph handles matching/DOF; a feasibility pass afterward
  verifies bounds and gives under-determined variables their propagated
  feasible range (a genuinely useful answer to "what's loose," not just
  "loose, no further info").

**Recommendation: (D).** Best-supported combination across numeric CSP,
geometric constraint solving, and optimization literature; handles both our
constraint shapes (mostly single-variable, occasionally joint); polynomial
time throughout.

### Q3. Moment/calibration constraints -- RESOLVED at the theory level, needs two implementation paths

**Closed-form path exists and is well-understood, with precise
preconditions.** For `S_N = SUM_{i=1}^{N} X_i`:
- Mean: **Wald's identity**, `E[S_N] = E[N]*E[X]` -- exactly the hand-derived
  `E[order_total] = E[N_LI]*E[qty]*E[price]` formula from the earlier chat
  discussion. Requires only independence of N from the X_i's and of the
  X_i's from each other; no distributional-family assumption needed.
- Variance: **Blackwell-Girshick equation**, `Var(S_N) = E[N]*Var(X) +
  Var(N)*E[X]^2` -- same independence preconditions, no harder to compute
  than the mean, worth deriving alongside it if a std-dev target is ever
  needed.
- Products of independent terms (e.g. `quantity*unit_price`): `E[XY] =
  E[X]E[Y]` holds cleanly under independence.
- **What breaks it**: correlated terms still have a closed form but need a
  covariance parameter (adds a graph node, doesn't blow up the framework);
  count correlated with the terms breaks Wald's identity with no general
  replacement; nonlinear aggregations (MAX/MIN/percentile) have **no general
  closed form at all**, even for independent inputs -- only bounds
  (Bertsimas-style moment-problem bounds) or asymptotic approximations
  (extreme-value theory) exist.

**Fallback path for everything else: Method of Simulated Moments (MSM) or
Approximate Bayesian Computation (ABC)** -- both are the "simulate forward,
compare to target moments, adjust free parameters, repeat" loop, differing
mainly in whether you want a single point estimate (MSM, simpler, matches
our need for one value to drive Stage 4) or a posterior distribution over
plausible values (ABC, heavier). Neither requires adopting a full
probabilistic-programming dependency (Stan/PyMC) -- this is a narrower
"N free parameters, M stated moments" problem than what those tools target.

**Confirmed empirically: none of the four vendored baseline tools
(benerator, datafiller, snowfakery, synth) have any aggregate/moment
calibration capability** -- verified via direct file/line citations
(`baselines/DATA_GENERATION_BASELINES.md` lines 38-208), not assumed. All
four require hand-tuning every parameter by trial and error outside the
tool; none derives a parameter from a stated population target. This
confirms there's no implementation shortcut to borrow here -- Stage 3 is
ahead of all four baselines on this specific problem.

**Recommendation: build both paths behind one interface.** Attempt the
closed-form derivation first (pattern-match the aggregation's structure
against the independence preconditions); if it succeeds, emit it as an
ordinary equation-edge into the Q1/Q2 DOF graph. If it fails (nonlinear op,
declared correlation, or a branching mixture per Q4), register the moment
target as a calibration objective consumed by an MSM-style outer loop
against Stage 4's actual generation code -- a different consumer of the same
fact, not a second implementation of the same math. No general algorithm
exists for automatically detecting which case applies from an arbitrary
graph shape -- this pattern-matching step is real design/implementation
work, not a research gap with a citable answer.

### Q4. Decision-tree branches -- RESOLVED, with a real (not coin-flip) recommendation

**Recommendation: forked subgraph (per-branch, separately analyzed,
recombined only when needed), not a mixture-distribution node.** This is
evidence-based, not a default pick: the two literatures that actually had to
make conditionals *structurally checkable* -- multi-mode DAE structural
analysis (HSCC 2017, MDPI Electronics 2022) and conditional/dynamic CSP
(Mittal & Falkenhainer; IJCAI 2007) -- both converge on per-mode/per-branch
separate structural analysis, with cross-branch recombination pushed out as
a distinct, separately-scoped problem. Neither folds branches into one
node's equation.

**Why this matters concretely for us**: a mixture node's moment is `E[X] =
SUM w_i * mean_i` -- bilinear in the mixture weight and the per-branch
parameters. That bilinear term would inject itself into the DOF matching for
*every* conditional column, directly compounding Q3's moment-derivation
problem (already the hardest one). The forked representation keeps every
individual matching sub-problem exactly as simple as the unconditional
case -- plain scalar equality edges, no bilinear terms -- at the cost of
running more (structurally identical, cheap) matching problems instead of
one complicated one.

**The real, unresolved cost of the forked approach: a fork-key registry.**
Two comparison points expose this concretely: (a) if multiple downstream
columns branch on the *same* upstream categorical (e.g. shipping_cost,
discount_rate, and delivery_sla all keyed off loyalty_tier), forks must be
deduplicated by upstream key across all of them, or you get a combinatorial
blow-up from independently re-forking; (b) at shard-merge time, the existing
Gale-Shapley union machinery only extends cleanly to per-branch merging
(Platinum-with-Platinum, non-Platinum-with-non-Platinum) if both shards fork
on the *same* key -- if they don't, mixture-style weighted recombination
re-enters as a fallback anyway. Neither of these is solved by the
recommendation itself; both are real engineering work still needed
(effectively a small "which upstream variable does this branch key on"
index, checked at both column-fan-out time and shard-merge time).

**Confirmed empirically: none of the four vendored baseline tools
represent conditionals as a checkable graph/distribution construct either.**
All four (including Benerator, whose actual Java source
(`ConditionalComponentBuilder.java`) does have a formal conditional
primitive) implement branching as **per-row imperative scripting**
(ternaries, Jinja `if`/`else`) evaluated at generation time, never as a
declarative node subject to any feasibility analysis. This means Stage 3's
ambition here (a conditional that survives as an analyzable graph node, not
just an executable script) has no precedent to borrow from among mainstream
synthetic-data tools -- confirmed absence, not an oversight in the search.

### Q5. Shard-local vs. global feasibility, incrementally -- RESOLVED: full recompute is fine, but the "sub-millisecond" number was WRONG

**No incremental Dulmage-Mendelsohn algorithm exists in the literature at
all** -- confirmed absence, not an oversight (the DM partition depends on
global alternating-path reachability from unmatched vertices, which a
single edge insertion can rewrite anywhere in the graph; this is equivalent
to dynamic transitive closure, a known-hard dynamic-graph problem).
Incremental *matching* (not the fuller decomposition) exists but only in a
restrictive model (one side fixed, vertices arrive with their full edge set
at once -- Bosek et al., FOCS 2014) that doesn't match our merge pattern
(union of two complete, arbitrary bipartite graphs); the general "both sides
dynamic" setting has no known sub-quadratic exact algorithm, with a
conditional lower bound (via the OMv conjecture) suggesting none is coming.

**CORRECTION (2026-07-11, empirically measured, not estimated):** the
original "sub-millisecond at ~1,000 variables/~3,000 edges" claim was wrong
-- it was a theoretical complexity estimate, never actually run. Hands-on
testing (`experiments/stage3_dof/pyomo_dm_scale_test.py` +
`pyomo_dm_scipy_vs_networkx.py`) found real `dulmage_mendelsohn()` runtime is
**dominated by graph SHAPE (constraint-to-variable ratio), not raw node
count**, and it's nowhere near sub-millisecond at the originally-estimated
scale:
- At the original estimate's literal shape (1000 vars, 3000 *constraints* --
  a 3:1 over-determined ratio): **604ms**, not sub-ms. This ratio also
  turned out to be an unrealistic thing to test in the first place --
  3x more constraints than variables makes a graph pathologically
  over-determined by construction, not representative of our actual domain
  (per the 2026-07-11 10:32 walkthrough, real specs leave FAR more free
  parameters than they explicitly pin -- the opposite skew).
- At a more realistic, roughly balanced shape (1000 vars, 1000 constraints):
  **13ms**. At an under-determined shape (1000 vars, 500 constraints):
  **4.9ms**.
- Input representation (bare `networkx.Graph` vs `scipy.sparse.coo_matrix`)
  made no measurable difference (0.9x) -- ruled out as the cause.
- At 5x/20x the original scale with the pathological 3:1 ratio: 19 seconds
  and 9+ minutes respectively -- confirms shape, not just size, matters, and
  that a genuinely pathological or very large over-determined graph would be
  a real problem, not a theoretical one.

**Revised recommendation: still full recompute per merge, but verify the
REAL shape/scale first, don't assume.** Tens of milliseconds at a realistic,
roughly-balanced shape is trivially fine for an offline, per-shard/per-merge
feasibility check (not a hot loop, and small next to the LLM calls
elsewhere in this pipeline that take seconds each) -- Q5's ultimate
architectural conclusion (no incremental machinery needed) still holds. But
it holds for the *realistic* shape, not the originally-claimed number, and
the actual shape/scale of OUR real domain is unconfirmed until measured
against real data (see the Synthesis section's "real prototype" step) --
don't treat "it'll be fine" as settled independent of what that prototype
actually shows.

### Q6. LLM probing for continuous loose ends -- RESOLVED, with an existing analogue to build on

**Stage 2's actual merge-conflict probe (the adjudicator agent) is a pure
discrete-choice classifier** -- `ResolutionAction` is a flat model with one
`ActionType` enum (6 members) plus type-specific optional fields, validated
only for structural completeness (right fields present), never for numeric
plausibility. It never does open-ended numeric estimation at all --
confirmed via direct reading of `conceptual_merger.py` and the adjudicator's
23-line prompt.

**The closer, already-working analogue is Stage 4's `parameter_agent`, not
Stage 1's context enricher.** `derive_parameters()` already fills exactly
the kind of gaps Stage 3 will need to fill: row counts, fan-out averages,
per-column null sparsity. Its prompt already handles three of our four gap
cases without being asked to: it states a default/prior for the
fully-unconstrained case, explicitly instructs "use the stated value" when a
fact partially pins the parameter, and gives worked few-shot domain
exemplars (interest rates, credit scores, age) for the wholly-open
distribution-parameter case, with an explicit ban on the degenerate
LLM-default answer (`mean=0, std=1`).

**External research confirms cases (a)-(d) genuinely differ, but don't need
four separate templates -- one flexible template with conditional sections
is the right shape**, which is roughly what `parameter_agent`'s prompt
already does informally. Numeric elicitation is a different problem class
from discrete choice: raw LLM self-reported confidence is well-documented as
poorly calibrated, and chain-of-thought reasoning improves quantitative
*reasoning* but not *calibration* -- a caution against over-engineering a
heavy reasoning scaffold for this probe. Partially-pinned gaps (case c) are
meaningfully lower-risk than fully-free ones (a/b/d) since they can reason
relative to a given anchor rather than needing external grounding.

**What Stage 3 needs beyond what's already there:** (1) a plausibility/
sanity layer beyond structural-completeness checking -- current `_validate()`
methods (e.g. `GaussianDistribution`) check `std_dev>0` but nothing checks
it's not absurd relative to the mean; (2) `parameter_agent`'s few-shot
exemplar list is currently hardcoded to a handful of domains (interest
rates, credit scores...) -- worth reframing as "reason from comparable
real-world quantities" per this project's no-brittle-fixes convention,
rather than a fixed lookup table that only covers what it happens to
enumerate; (3) the actual missing wiring: Stage 1's context enricher already
has a web-search/evidence-fetch mechanism (`EvidenceStore`) that's never been
connected to a numeric-output agent -- this is exactly the missing piece for
gap type (a)/(b), where the LLM has no stated prior and would otherwise be
guessing from parametric memory alone.

**Note:** Stage 3's old constraint agents/models and `entry.py` were deleted
as part of the rewrite documented in STAGE3_PHASE2_DESIGN.md. Stage 3 is now
implemented via `src/pipeline/stage3/middleware/constraint_graph.py` and
related modules.

### Q7. Structural vs. numeric feasibility -- scoping decision, not researched separately

Not dispatched as a separate research track (it's an engineering-depth call,
not a literature question) but resolved by synthesis of Q2/Q3/Q4's findings:
do **both**, but as two distinct passes with very different cost profiles.
Structural-only (matching/DM decomposition, sub-millisecond per Q5) should
run on every shard and every merge, always. Numeric verification -- actually
evaluating equations to confirm an over-determined subgraph is harmlessly
redundant rather than contradictory -- is real work (potentially involving
Q3's MSM-style simulation for anything that isn't closed-form) and should
run only on subgraphs the structural pass actually flags as over-determined,
not universally. This mirrors DAE theory's own distinction between
structural and numerical singularity, and keeps the expensive check scoped
to where it's actually needed.

### Q8. Tooling survey -- RESOLVED, strong actionable answer

**Use `pyomo.contrib.incidence_analysis.dulmage_mendelsohn()` directly.**
Confirmed (not assumed): this function is real, current, and does exactly
what's needed -- accepts a `networkx.Graph` or `scipy.sparse.coo_matrix`
directly (no Pyomo modeling-language object required), returns
`RowPartition`/`ColPartition` namedtuples with `unmatched`,
`underconstrained`, `square` (exactly-determined), and `overconstrained`
fields -- precisely the four-way classification this whole design needs, no
adaptation layer beyond mapping those field names onto ours.

Verified directly in this repo's venv: `networkx` (3.6.1) has
`hopcroft_karp_matching`/`maximum_matching` but **no** Dulmage-Mendelsohn
function of its own; `scipy.sparse.csgraph` has the matching primitive
(`maximum_bipartite_matching`, the same operation MATLAB's `dmperm` performs
internally) but likewise no full decomposition. Pyomo is the only package
found (checked against CasADi too, which has the capability but a stale,
possibly-broken tutorial and a much heavier dependency footprint for one
function) that ships the complete decomposition, not just the matching step.

**Python 3.14 compatibility confirmed via PyPI metadata**: Pyomo 6.10.1
(released 2026-06-04) ships compiled wheels for `cp314` and `cp314t`
(free-threaded build) across platforms -- addresses this project's standing
3.14-compatibility caution directly. Pyomo's core install has **zero
mandatory dependencies** beyond `Python>=3.10`; networkx/scipy are only
listed under an optional extra, and both are already hard dependencies here,
so adding Pyomo adds exactly one new package, not a dependency cascade.

**HANDS-ON VERIFIED (2026-07-11, experiments/stage3_dof/pyomo_dm_experiment.py).**
`pip install pyomo` (via `uv pip install --python venv/Scripts/python.exe
pyomo` -- plain `uv pip install pyomo` silently created/used a stray `.venv`
one directory up, since this project's venv is named `venv` not `.venv`; had
to target the interpreter explicitly). Confirmed Pyomo 6.10.1 installed and
importable in the project's actual venv. One real API correction versus the
doc-only research: `dulmage_mendelsohn` is not directly callable from
`pyomo.contrib.incidence_analysis` -- that name resolves to the *submodule*;
the callable is one level deeper, at
`pyomo.contrib.incidence_analysis.dulmage_mendelsohn.dulmage_mendelsohn`. A
toy 5-variable/5-constraint graph (mirroring our real domain: a pinned
Gaussian mean+std, a moment-derived pin, a deliberately free variable, and a
deliberately conflicting redundant pair) was built directly with
`networkx.Graph` and passed to the function with zero Pyomo model object
involved, exactly as planned. All 6 correctness checks passed on the second
run.

**Two real methodological corrections earned from the first (failed) run,
not just a rerun after a typo fix:**
1. A single NL fact stating two independent parameter values (e.g. "mean 8,
   std 2") must become **two separate scalar constraint-nodes**, not one
   constraint-node with edges to both parameters. The first attempt modeled
   it as one multi-edge node and the tool correctly reported it as
   underdetermined (1 equation, 2 unknowns) -- a real bug in the *test's*
   modeling, not the tool. This refines Q1's finding: "multi-variable edges
   are standard" is true for genuine multi-variable relationships (e.g.
   `subtotal = quantity * unit_price`), but does NOT apply to independent
   facts that merely happen to be stated in the same sentence.
2. The `overconstrained`/`square` classification is a **structural-block
   classification** (which connected component of the incidence graph a
   node belongs to), not a per-node "this one is bad, that one is fine"
   verdict. In the deliberately-conflicting test case, the matched row,
   the unmatched row, AND the variable itself were ALL classified together
   as part of one overconstrained block -- confirming a real feasibility
   check needs to flag and numerically inspect the whole block together
   (per Q7), not expect the tool to point at a single culprit constraint.

## Real-data prototype (2026-07-11, experiments/stage3_dof/real_data_prototype.py)

Built the actual bipartite graph from brainstorm_report.md's real facts (not
another synthetic toy) -- 20 parameter-level variables, 9 constraints, 11
edges. Result: all 9 constraints matched cleanly (no conflicts, expected for
a clean spec), 9/20 variables exactly-determined, 11/20 genuinely free --
matching the qualitative "far more free parameters than pinned ones" finding
from the earlier fact-by-fact walkthrough. **The decisive test passed**:
`N_LI_mean` (the line-items-per-order mean) resolved to exactly-determined
purely through graph elimination via the moment-target equation touching
`{mu_qty, mu_price, N_LI_mean}` -- computationally confirming the hand-derived
`E[N_LI]~=1.5` calculation actually works end-to-end through the mechanism,
not just as manual arithmetic. Graph this size is trivially fast (far below
even the 4.9ms measured for a much larger synthetic under-determined case).
**Known gap**: this real schema's only decision-tree branch (`shipping_cost`
on `loyalty_tier`) has a trivial Platinum side (`=0`, no free parameter), so
Q4's harder fork-key-registry case (both branches having free parameters
needing separate resolution) remains researched-but-unprototyped -- not
disproven, just not yet exercised against real data.

## Synthesis: what's ready to prototype vs. what's still open engineering work

**Ready to prototype now, with a clear, well-supported design:**
- Q1 (parameter-level scalar nodes) -- settled convention, no further
  research needed.
- Q8 (Pyomo's `dulmage_mendelsohn`) -- hands-on verified, real API path and
  two modeling corrections documented above. Ready to use.
- Q5 (full recompute per merge) -- architectural conclusion settled (no
  incremental machinery needed), but the performance claim was corrected
  via hands-on measurement (13-127ms depending on shape, not sub-ms) -- the
  conclusion holds for realistic graph shapes but the real prototype's
  actual shape/scale should still be checked, not assumed.
- Q2 (hybrid inequality handling, Approach D) -- settled design, though the
  Pyomo `include_inequality` specifics need a live spot-check once installed.
- Q7 (two-pass structural-then-numeric feasibility, numeric pass scoped to
  flagged subgraphs only) -- settled scoping.

**Real design decisions made, but with genuine remaining engineering work
before they're implementable:**
- Q3 (moment propagation): need to actually design the "does a closed form
  apply" pattern-matcher, and the MSM-style calibration loop against Stage 4's
  generation code, plus a decision on how deeply to invest in the fallback
  path relative to the deadline.
- Q4 (forked subgraphs for decision-tree branches): need to actually design
  the fork-key registry that deduplicates branches across columns and across
  shard-merge boundaries -- this is the one piece of the recommendation with
  no literature precedent to lean on, since it's specific to our shard-merge
  architecture.
- Q6 (LLM gap-filling): need to design the plausibility-sanity validators, and
  wire Stage 1's existing `EvidenceStore` web-search mechanism to a new
  numeric-output agent -- both concrete, bounded implementation tasks, not
  further research.

**Unrelated but blocking, discovered during Q6's research, needs a decision
before any of the above can be wired into working code:** Stage 3's old
constraint agents/models are deleted in the uncommitted working tree while
`entry.py` still imports two of them -- Stage 3 currently doesn't run.
