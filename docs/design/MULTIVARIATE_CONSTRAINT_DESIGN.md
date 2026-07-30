# Multivariate / Cross-Column Constraint Modeling: Research Synthesis

**Status: DESIGN-ONLY. Not validated against a prototype. No code written.**
Started 2026-07-13, following the user's explicit "increase the constraint
repertoire without breaking the math" ask, then "dont code yet, keep
researching... a mathematical model which can operate on these vast
families of constraints in a deterministic way." This doc exists because
the research thread ran long enough across PROGRESS.md log entries that it
needed a single, citable home -- PROGRESS.md's Log stays append-only and
summary-shaped; this is the actual technical record.

Everything here extends, and does not replace, Stage 3's existing
constraint model (`src/pipeline/stage3/models/`, `src/pipeline/stage3/
middleware/constraint_graph.py`). See `baselines/experiments/
STAGE3_PHASE2_DESIGN.md` for that model's own synthesis (Dulmage-Mendelsohn
DOF classification over per-parameter Variables/Constraints). This doc is
about everything that model deliberately does NOT cover: joint/cross-column
distributions.

---

## 0. The problem in one sentence

Stage 3 can extract and validate facts about ONE column at a time (a
distribution, a moment target, a range) via a Dulmage-Mendelsohn-based
degrees-of-freedom (DOF) graph. It has no principled way to extract or
validate facts that state a RELATIONSHIP between two or more columns'
*distributions* (correlation, conditional independence, joint dependence
shape) -- as opposed to a relationship between two columns' *realized
values* (e.g. `end_date > start_date`, which is a separate, per-row
concern, see Section 6). The goal: a rigorous, deterministic way to
represent and validate this class of fact, without touching or risking the
existing DOF machinery's correctness.

---

## 1. The organizing principle: two different mathematical questions

Tracing `src/util/algorithms/dof_graph.py`'s actual semantics: Dulmage-
Mendelsohn decomposition answers *"in a bipartite graph of named Quantities
and Equations that pin them, is each Quantity uniquely determined, free, or
over-determined?"* -- a **structural identifiability** question (the same
one used in DAE index reduction, e.g. Pantelides' algorithm, cited in
STAGE3_PHASE2_DESIGN.md).

A cross-column fact like "quantity and total are correlated" is not asking
that question -- no Equation pins a Quantity here. It is stating something
about the SHAPE of the joint distribution. This is a fundamentally
different mathematical object, and forcing it into the DOF graph (as an
early audit found `_range_constraint_to_rich` implicitly attempting, see
Section 6) either silently drops it or misclassifies it.

The resolution used throughout this doc: **Sklar's theorem** gives the
exact, principled decomposition. For an n-dimensional CDF F with marginals
F_1, ..., F_n, there exists a copula C such that

  F(x_1, ..., x_n) = C(F_1(x_1), ..., F_n(x_n))

-- i.e., ANY joint distribution factors into (a) its marginals, already
handled by `DistributionConstraint`/the DOF graph, and (b) a copula C
capturing the dependence structure alone, entirely separate machinery.
(Sklar's Theorem, restated in Nelsen's *An Introduction to Copulas* and
summarized at [MathWorld](https://mathworld.wolfram.com/SklarsTheorem.html);
also [Lecture 12: Copula, U. Washington](https://faculty.washington.edu/yenchic/21Sp_stat542/Lec12_copula.pdf).)

**Caveat that shapes the whole design (Section 5.1)**: uniqueness of C is
only guaranteed when every F_i is *continuous*. For discrete or mixed
marginals -- i.e. most DB categorical/count columns -- C is unique only on
the Cartesian product of the marginals' *ranges*, Ran(F_1) x ... x Ran(F_n),
not the full unit cube. Multiple copulas fit the data equally well outside
observed support points. This is not a corner case for a database; it is
the common case.

---

## 2. The unifying object: one partially-specified symmetric matrix

The central finding of this research: **a correlation fact and a
conditional-independence fact are entries in the same mathematical object.**

For a multivariate Gaussian (the natural default copula -- see Section 4),
the dependence structure is fully captured by an n x n correlation matrix
Sigma. Its inverse, the precision matrix Omega = Sigma^-1, has the exact
property:

  X_i _|_ X_j | (all other variables)  <=>  Omega_ij = 0

This is the defining fact of Gaussian graphical models (confirmed across
multiple sources, e.g. the covariance-selection literature summarized at
[Banerjee/El Ghaoui/d'Aspremont, covariance selection ML estimation](https://www.seas.ucla.edu/~vandenbe/publications/covsel1.pdf)).
So:

- A **correlation fact** ("quantity and total are positively correlated,
  rho=0.6") pins an entry of Sigma.
- A **conditional-independence fact** ("given diagnosis, treatment_cost is
  independent of patient_age") pins an entry of Omega to exactly zero.

Both live in one shared **Dependency Matrix** object: for a set of columns
at a given Grain (see Section 3), a symmetric matrix whose entries are each
FIXED (a fact pinned Sigma_ij or Omega_ij), FREE (unstated), or -- for
Omega -- CONSTRAINED-ZERO (a stated CI fact).

### 2.1 The deterministic feasibility test

This reframing is what makes the whole family tractable: checking whether
a partially-specified Dependency Matrix is jointly realizable is exactly
the **positive-definite (PD) matrix completion problem**, and it has an
exact, closed-form, deterministic answer:

> **Theorem (Grone, Johnson, Sa, Wolkowicz, 1984).** Let G be the graph
> whose edges are the entries of a partial symmetric matrix that ARE
> specified (every principal submatrix corresponding to a clique of G is
> itself PD). A positive-definite completion of the matrix exists **if and
> only if G is chordal** (every cycle of length >= 4 has a chord). When it
> exists, a completion of MAXIMUM DETERMINANT is unique and computable in
> closed form, clique by clique.

(Summarized against the covariance-selection/ML literature above; also
[The positive definite completion problem revisited](https://www.sciencedirect.com/science/article/pii/S0024379508001791);
practical PSD-projection tooling: Higham's alternating-projections algorithm,
["Computing the nearest correlation matrix", N. Higham, 2002](https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/),
implemented as `Matrix::nearPD` in R; Higham & Strabic 2016 extend this to
hold specified entries fixed while completing the rest.)

This gives Stage 3's DOF-graph trichotomy (square / loose / overconstrained)
an EXACT analogue for the dependency layer:

| DOF-graph concept (existing) | Dependency-matrix concept (this design) |
|---|---|
| Dulmage-Mendelsohn structural rank | Chordality of the specified-entries graph |
| `square_variables` | Entries forced by the chordal completion |
| `loose_variable_probes` | Entries with no forced value -> same `VariableProbe` shape, handed to Stage 4 |
| `overconstrained_blocks` / `confirmed_conflicts` | Cholesky/eigenvalue failure on a non-chordal or contradictory sub-block |
| `conflict_reconciler` LLM healing loop | **Reused unchanged** -- same MISEXTRACTION / FALSE_POSITIVE / GENUINE_CONTRADICTION verdicts |

**Practical validity check**: attempt a Cholesky factorization of each
fully-specified principal submatrix (per clique, once chordal); failure at
any pivot means non-PD -- i.e. a provable contradiction, feeding the exact
same `confirmed_conflict` -> `conflict_reconciler` pathway already built.
Cholesky is preferred over eigendecomposition for this check: both are
O(n^3), but Cholesky is empirically substantially faster (a subset of the
computation eigendecomposition performs) --
[Numerical Matrix Decomposition, arXiv:2107.02579](https://arxiv.org/pdf/2107.02579).

Non-chordal specified-entry patterns are the genuinely harder case (no
closed-form completion guaranteed even if every specified submatrix is
individually PD) -- Section 7 scopes this as a documented limitation, not
silently ignored.

---

## 3. Grain-scoping: reusing population identity, not inventing it

Exactly like today's `MOMENT_TARGET`/`TABLE_CARDINALITY` variables, a
correlation or CI fact is only well-defined at a specific **population**
(Grain, per `src/pipeline/stage3/models/grain.py`). `Corr(quantity, total)`
computed over the whole ORDER table is a DIFFERENT quantity than the same
correlation computed over a narrowed subset (e.g. only orders reachable via
a nullable FK join) -- this is precisely the "population-sensitive"
distinction `Grain.is_comparable_with(population_sensitive=True)` already
encodes for distribution parameters.

**Design decision**: every Dependency Matrix entry is indexed by
`(column_i, column_j, Grain)`, not just `(column_i, column_j)`. Two facts
about the same column pair at incomparable Grains are different matrix
entries and must never be merged or compared directly -- reuses the
existing `is_comparable_with`/`narrowed` machinery verbatim, no new
population-identity logic needed.

`Grain.common_edges()` -- built earlier this session, flagged as "purely
descriptive, never wired into any check" -- is exactly where it gets used:
for a CROSS-TABLE correlation (e.g. between a CUSTOMER-level column and a
per-customer ORDER aggregate), determining the shared grain both columns
are jointly defined at is a `common_edges`-shaped question.

---

## 4. Copula family catalog (the default: Gaussian)

The Gaussian copula (Section 2) should be the DEFAULT representation --
matching the overwhelming majority of "X and Y are correlated" facts, and
the only family whose full n-dimensional specification is exactly the
Dependency Matrix of Section 2 with no extra machinery. Confirmed parameter
domains for the alternative families (needed for Section 4.1's asymmetric/
tail-dependence upgrade path):

| Family | Parameter(s) | Valid domain | Dependence shape |
|---|---|---|---|
| Gaussian | correlation matrix Sigma | Sigma symmetric PSD | Symmetric, **zero tail dependence** (asymptotic independence in both tails except degenerate +-1 limit) |
| Student-t | Sigma + degrees of freedom nu | Sigma PSD, nu > 0 | Symmetric, **nonzero tail dependence both tails** (stronger as nu decreases) |
| Clayton (Archimedean) | theta | theta in (0, infinity), bivariate case theta in (-1, infinity)\{0} | Asymmetric, **lower-tail dependence only** |
| Gumbel (Archimedean) | theta | theta in [1, infinity), theta=1 is independence | Asymmetric, **upper-tail dependence only** |
| Frank (Archimedean) | theta | theta in R \ {0} | Symmetric, **no tail dependence** |

(Sources: [MetricGate Clayton Copula docs](https://metricgate.com/docs/clayton-copula/);
[MetricGate Gumbel Copula docs](https://metricgate.com/docs/gumbel-copula/);
[R `copula` package family docs](https://rdrr.io/rforge/copula/man/copFamilies.html);
[Vose Software: Archimedean copulas](https://www.vosesoftware.com/riskwiki/Archimedeancopulas-theClaytonFrankandGumbel.php).)

A standard bivariate Archimedean copula needs only one scalar theta
regardless of n when applied exchangeably, but a general n-dimensional
Archimedean extension (non-exchangeable) is far more restrictive than
Gaussian/Student-t's full n(n-1)/2-parameter matrix -- this is exactly why
vine copulas (Section 4.1) exist: to scale bivariate (possibly non-Gaussian)
building blocks to n dimensions without that restriction.

### 4.1 Vine copulas: the upgrade path for asymmetric/tail-dependent facts

When a fact states something a symmetric Gaussian relationship cannot
capture (e.g. "large orders disproportionately correlate with returns" --
an asymmetric, tail-heavy relationship), vine copulas (Bedford & Cooke,
2001/2002) generalize the same edge-count structure using non-Gaussian
bivariate building blocks:

- An n-dimensional vine decomposes the joint copula density into a cascade
  of **d(d-1)/2 bivariate (conditional) copula densities**, arranged as
  edges across a nested sequence of trees T_1, ..., T_{n-1}.
- **Proximity condition** (the vine's validity criterion): two nodes in
  tree T_{i+1} may be joined by an edge only if the corresponding edges in
  T_i share a common node. This is **purely combinatorial** -- a property
  of the tree sequence's node/edge incidence, checkable in polynomial time,
  with NO statistical inference involved.
- R-vines satisfy only the proximity condition (general case); **C-vines**
  (each tree is a star) and **D-vines** (each tree is a path) are more
  restrictive special cases.
- Confirmed: a full n-dimensional vine requires exactly **n(n-1)/2**
  pair-copulas -- matching the Gaussian correlation matrix's edge count
  exactly. Each edge gets its own family + parameter-domain check (the
  table above) -- also closed-form, also deterministic.

(Sources: [Matrix and graph representations of vine copula structures, arXiv:2205.04783](https://arxiv.org/pdf/2205.04783);
[Pair-copula Bayesian networks, arXiv:1211.5620](https://arxiv.org/pdf/1211.5620);
[R-vine Models for Spatial Time Series, arXiv:1403.3500](https://arxiv.org/pdf/1403.3500).)

**Design decision**: Gaussian/precision-matrix stays the default extraction
target; vines are an explicit escalation only for facts that name a
specific asymmetric/tail-dependence shape. The proximity condition is a
second, independent deterministic validity check alongside chordality --
NOT a replacement for it (a vine still needs each of its bivariate edges
individually parameter-valid, same table above).

---

## 5. Three scoping boundaries the research surfaced (must not be skipped)

These are not footnotes -- each one, if ignored, would make the model
silently wrong for realistic database facts.

### 5.1 Discrete/categorical columns break Sklar's uniqueness guarantee

(Restated from Section 1.) For non-continuous marginals, the copula is only
unique on the product of the marginals' ranges. **Design implication**: a
categorical or count column must NOT be plugged directly into the Gaussian-
copula/precision-matrix machinery as if it had a well-defined copula the
same way a continuous column does. The standard practical treatment (used
commercially for ordinal/mixed data) is a **latent-continuous threshold
representation** -- model the discrete column as a thresholded draw from an
underlying continuous latent variable, and let the copula/precision-matrix
machinery operate on the latent layer. This needs to be explicit in any
implementation, not silently assumed away.

### 5.2 The intersection axiom requires strict positivity -- conflicts with derived columns

The Pearl-Paz graphoid axioms governing how CI statements compose
(symmetry, decomposition, weak union, contraction) hold for ANY probability
distribution. The fifth, **intersection** --

  X _|_ Y | (Z union W)  AND  X _|_ W | (Z union Y)  =>  X _|_ (Y union W) | Z

-- additionally requires **strictly positive** densities, and can fail for
distributions with deterministic/functional structure.
([Pearl/Paz semi-graphoid framework](https://webspace.maths.qmul.ac.uk/a.fink/slides_Osnabrueck_2017.pdf);
[UCLA Stats 212 CI notes](http://www.stat.ucla.edu/~zhou/courses/Stats212_CI.pdf).)

Database derived columns (`total = price * quantity`) are EXACTLY this
deterministic-structure case. **Design decision (hard boundary, not a
tuning knob)**: statistical (in)dependence checking (this doc's whole
apparatus) and exact functional-dependency checking are two different
mathematical objects and must never share a consistency check. Derived-
column facts stay on Stage 3's EXISTING cycle-detection machinery
(`src/pipeline/stage3/middleware/cycles.py`, `detect_derived_cycles`) --
the Dependency Matrix only ever holds genuinely statistical (non-
deterministic) relationships. A column that is the target of a
`DerivedColumnConstraint` should never simultaneously appear as a node in
the Dependency Matrix.

### 5.3 General CI-consistency is undecidable -- but decidable for bounded cardinality

The general problem "does this finite set of stated CI facts admit SOME
probability distribution" has **no finite complete axiomatization**
(Studeny) and is in fact **undecidable** in general, via a reduction from
the periodic tiling/domino problem
([arXiv:1408.2030](https://arxiv.org/pdf/1408.2030),
[arXiv:2205.11461](https://arxiv.org/pdf/2205.11461)). The semi-graphoid
axioms (Section 5.2) are provably INCOMPLETE -- they do not derive every
valid CI implication.

**However**: if variable cardinalities are bounded, the implication problem
IS decidable (a finite closure check over all consistent joint tables).
Database categorical columns are typically small-cardinality -- this gives
a legitimate, principled, TRACTABLE path specifically for CI facts between
categorical columns, distinct from the chordal/PD-completion test used for
continuous correlation facts (Section 2.1).

**Design decision**: the model needs (at least) two separate, independently
tractable feasibility checks, not one universal algorithm --
1. **Continuous/numeric correlation & CI facts** -> chordality +
   PD-completion (Section 2.1).
2. **Categorical CI facts** -> bounded-cardinality finite closure check
   (decidable, but a genuinely different algorithm).
Mixed continuous/categorical facts are the hard remaining case -- not
resolved by this research pass, flagged as an open question (Section 7).

---

## 6. Why this is a SEPARATE layer from Stage 3's existing per-row conflict gaps

An adjacent, previously-flagged finding (audit of `constraint_graph.py`,
same research session) is worth stating precisely so it is not confused
with this doc's subject: `_range_constraint_to_rich` silently returns
`([], [])` for ANY condition where `extract_columns()` doesn't yield
exactly one column -- this affects genuine two-column PER-ROW facts like
`end_date > start_date` (the "temporal ordering" example already shipped
in `logic_extractor/prompt.md`). Traced: this is NOT data loss (the raw
`Constraint` still reaches `Stage3Output.logic_constraints` via `entry.py`'s
unconditional merge) -- it is a hole in Stage 3's OWN conflict-detection
coverage for that fact.

This is a DIFFERENT mathematical object than anything in this doc: `X > Y`
for every row is a constraint on the JOINT SUPPORT of a single row's value
tuple (a feasibility/domain question), not a statement about the shape of
X and Y's marginal or joint DISTRIBUTION across rows (what correlation/CI
facts describe). Per Section 5.2's boundary discipline, these should NOT
be merged into the Dependency Matrix either. They belong to their own,
much simpler, "per-row satisfiability" layer (interval propagation /
direct algebraic contradiction detection, e.g. "end > start" and
"start > end" on the same grain is a syntactic contradiction) -- separate
follow-up work, out of scope for this doc, noted here only so the two
research threads are not conflated.

---

## 7. Open questions / explicitly NOT designed here

- **Mixed continuous/categorical CI consistency** (Section 5.3) -- no
  unified algorithm found; the two tractable sub-cases don't obviously
  compose.
- **Non-chordal Dependency Matrix patterns** -- Grone et al.'s guarantee
  only covers the chordal case. A non-chordal partial specification needs
  an explicit semidefinite feasibility solve (or a documented "not
  supported yet" boundary) rather than silently assuming completability.
- **What Stage 4 does with a `loose` Dependency Matrix entry** -- the
  DOF-graph analogy suggests handing it to Stage 4 as a probe (matching
  `VariableProbe`), but the maximum-determinant completion (Grone et al.)
  is ALSO a principled default value Stage 4 could use instead of an
  arbitrary/independence default -- worth a decision, not resolved here.
- **Whether/how vine-copula edges interact with the fork-key (Q4
  conditional distribution) machinery already built** -- a correlation
  or CI fact stated conditionally ("for Platinum customers, quantity and
  total are correlated") would need its own Grain/branch scoping on top
  of everything above; not analyzed in this pass.
- **Format/regex constraints and composite FKs** -- separate, unrelated
  gaps from an earlier audit round (see PROGRESS.md's 2026-07-13 14:29 UTC
  and 16:57 UTC entries); not part of this doc's subject and not
  re-analyzed here.

---

## 8. Prior art -- and where this differs

No existing system found does **pre-generation joint-satisfiability
verification of a declared multivariate spec** (as opposed to fitting from
a real dataset, or generating-then-repairing/rejecting):

- **SDV `GaussianCopulaSynthesizer`**: fits marginals + the Gaussian copula
  correlation matrix FROM a real dataset -- no declarative-specification
  API exists. SDV's separate `Constraint` system supports `reject_sampling`/
  `transform` strategies but has NO proactive joint-satisfiability check
  ([GitHub issue #541](https://github.com/sdv-dev/SDV/issues/541) requests
  exactly this, open/unresolved as of the research date).
  ([DataCebo constraints blog](https://datacebo.com/blog/eng-sdv-constraints/);
  [GaussianCopulaSynthesizer docs](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/gaussiancopulasynthesizer).)
- **PrivBayes**: builds a Bayesian-network dependency structure by scoring
  correlations computed FROM real data (then privatizing the scores) --
  never asks whether a specification is internally consistent, because the
  structure is estimated, not authored.
  ([PrivBayes, ACM TODS](https://dl.acm.org/doi/10.1145/3134428).)
- **ProgSyn / Disjunctive Refinement Layer**: both require a real base
  dataset and use generate-then-repair/reject against declared constraints,
  not pre-generation verification.
- **TVineSynth (AISTATS 2025)**: the closest analog -- uses TRUNCATED
  C-vine copulas for tabular synthetic data, but for privacy/utility
  trade-off (truncating weak dependencies), not specification-consistency
  checking. Confirms vine copulas are current (2025), applied research for
  this exact tabular-synthesis domain.
  ([arXiv:2503.15972](https://arxiv.org/abs/2503.15972).)
- **@RISK (Palisade)**: the cleanest real-world precedent for
  independently-authored marginals + a separately-specified correlation
  matrix/copula (mirroring Sklar's decomposition) -- but a Monte Carlo
  risk-simulation tool, not a database/record generator, and its
  correlation-matrix input is validated only at the linear-algebra level
  (PSD-ness), not the richer CI/vine layer this doc proposes.
  ([@RISK Correlation Matrix docs](https://help.palisade.com/v8/en/@RISK/1-Define/5-Correlation/Define-Correlation-Matrix.htm).)

**The gap this design fills**: a deterministic chordality/PD-completion
feasibility check performed BEFORE generation, fed by atomic extracted
facts (matching Stage 1/3's existing extraction philosophy) rather than a
fitted real dataset -- this appears to be genuinely novel relative to the
literature surveyed, not a re-implementation of an existing known method.
Worth stating explicitly in the paper if this gets built.

---

## References

Full source list from the research (three parallel research passes,
2026-07-13; see PROGRESS.md's 16:57 UTC log entry for the dispatch record):

- Sklar's Theorem: [MathWorld](https://mathworld.wolfram.com/SklarsTheorem.html); [Nelsen survey, U. Torino](https://www.matematica.unito.it/didattica/att/d11d.7507.file.pdf); [Lecture 12: Copula, U. Washington](https://faculty.washington.edu/yenchic/21Sp_stat542/Lec12_copula.pdf)
- Copula families: [MetricGate Clayton](https://metricgate.com/docs/clayton-copula/); [MetricGate Gumbel](https://metricgate.com/docs/gumbel-copula/); [R `copula` package](https://rdrr.io/rforge/copula/man/copFamilies.html); [Vose Software Archimedean copulas](https://www.vosesoftware.com/riskwiki/Archimedeancopulas-theClaytonFrankandGumbel.php)
- PSD checking: [Numerical Matrix Decomposition, arXiv:2107.02579](https://arxiv.org/pdf/2107.02579); [MATLAB chol() discussion](https://www.mathworks.com/matlabcentral/answers/225177-chol-gives-error-for-a-barely-positive-definite-matrix); [Practical Topics in Optimization, arXiv:2503.05882](https://arxiv.org/pdf/2503.05882)
- Vine copulas: [Matrix/graph representations, arXiv:2205.04783](https://arxiv.org/pdf/2205.04783); [Pair-copula Bayesian networks, arXiv:1211.5620](https://arxiv.org/pdf/1211.5620); [R-vine Models for Spatial Time Series, arXiv:1403.3500](https://arxiv.org/pdf/1403.3500)
- Matrix completion / partial specification: [Dependence control via max copula entropy, arXiv:2012.14759](https://arxiv.org/html/2012.14759v5); [Positive definite completion problem revisited](https://www.sciencedirect.com/science/article/pii/S0024379508001791); [CP-matrix completion, arXiv:1305.0632](https://arxiv.org/pdf/1305.0632); [PD completion for DAGs, arXiv:1201.0310](https://arxiv.org/pdf/1201.0310); [Covariance selection, Banerjee/El Ghaoui/d'Aspremont](https://www.seas.ucla.edu/~vandenbe/publications/covsel1.pdf)
- Nearest-correlation-matrix tooling: [Higham, "Computing the nearest correlation matrix"](https://nhigham.com/2013/02/13/the-nearest-correlation-matrix/); [R `Matrix::nearPD`](https://www.rdocumentation.org/packages/Matrix/versions/1.7-4/topics/nearPD); [Completing correlation matrices, arXiv:2111.12640](https://ar5iv.labs.arxiv.org/html/2111.12640); [Explicit solutions, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5882745/)
- Graphoid axioms / CI theory: [Pearl-Paz semi-graphoid slides](https://webspace.maths.qmul.ac.uk/a.fink/slides_Osnabrueck_2017.pdf); [UCLA Stats 212 CI notes](http://www.stat.ucla.edu/~zhou/courses/Stats212_CI.pdf); [d-Separation: From Theorems to Algorithms, arXiv:1304.1505](https://arxiv.org/abs/1304.1505); [CI implication lattice-theoretic paper, arXiv:1408.2030](https://arxiv.org/pdf/1408.2030); [Undecidability of CI implication, arXiv:2205.11461](https://arxiv.org/pdf/2205.11461); [CMU Undirected Graphical Models notes](https://www.stat.cmu.edu/~larry/=stat700/UG.pdf)
- Prior art: [DataCebo SDV constraints blog](https://datacebo.com/blog/eng-sdv-constraints/); [SDV GitHub issue #541](https://github.com/sdv-dev/SDV/issues/541); [GaussianCopulaSynthesizer docs](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/gaussiancopulasynthesizer); [CopulaGANSynthesizer docs](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/copulagansynthesizer); [ProgSyn](https://www.researchgate.net/publication/372234413_Programmable_Synthetic_Tabular_Data_Generation); [Survey of Synthetic Tabular Data Generation, arXiv:2504.16506](https://arxiv.org/html/2504.16506v2); [Beyond convexity, arXiv:2502.18237](https://arxiv.org/pdf/2502.18237); [PrivBayes, ACM TODS](https://dl.acm.org/doi/10.1145/3134428); [PrivBayes original, DIMACS](https://dimacs.rutgers.edu/~graham/pubs/papers/PrivBayes.pdf); [TVineSynth, arXiv:2503.15972](https://arxiv.org/abs/2503.15972); [@RISK Correlation Matrix docs](https://help.palisade.com/v8/en/@RISK/1-Define/5-Correlation/Define-Correlation-Matrix.htm); [@RISK product page](https://www.palisade.com/risk/monte_carlo_simulation.asp)
