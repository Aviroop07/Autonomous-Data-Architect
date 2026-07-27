from typing import List, Dict, Set, Tuple, Any, FrozenSet, Optional
from ortools.sat.python import cp_model
from collections import defaultdict
from pydantic import BaseModel, Field
import itertools
import concurrent.futures
import logging
import time

logger = logging.getLogger(__name__)

# Per-solve wall-clock cap handed to CP-SAT. Measured on a real 12-table /
# 14-FK / 52-fact schema: EVERY solve in the 216-config sweep ran to this cap
# without proving optimality, so the sweep's runtime is grid_size * this /
# pool_size rather than anything data-dependent. Hoisted from a literal so the
# cost is stated in one place and can be logged before the sweep starts.
_SOLVER_MAX_SECONDS = 30.0


class ILPSharder:
    def __init__(
        self,
        max_shards: int = 10,
        max_tables_per_shard: int = 8,
        w_size: float = 10,
        w_shard: float = 20,
        w_cohesion: float = 30,
        w_facts: float = 10,
    ):
        self.max_shards = max_shards
        self.max_tables_per_shard = max_tables_per_shard
        # w_size: penalize TOTAL footprint (tables+columns) summed across
        #   every active shard -- this is the real token-cost proxy across
        #   all of Phase 1's parallel LLM calls. Deliberately a sum, not a
        #   makespan/max: a max only bounds the single worst shard and
        #   does not grow at all once every active shard is already at
        #   capacity (the exact condition a fully FK-connected schema
        #   forces), which let a solver duplicate the whole schema across
        #   every available shard slot for zero additional size cost.
        # w_shard: penalize the number of active shards (orchestration
        #   overhead -- one parallel LLM call per active shard).
        # w_cohesion: reward column pairs that co-occur in the same fact
        #   being placed in the same shard, capped at most once per pair
        #   regardless of how many shards contain that pair (see SO1).
        # w_facts: penalize the total number of facts TOUCHING any shard
        #   (r_fk, HC10) -- the real per-shard LLM-context-load proxy,
        #   distinct from w_cohesion (which only rewards tables landing
        #   TOGETHER, saying nothing about how much total fact-driven
        #   context accumulates on any one shard).
        # All four weights are normalized by their own achievable scale
        # before being applied (see shard_schema()'s objective section) --
        # a raw weight of "1" means the same relative influence on every
        # term regardless of its natural magnitude.
        self.w_size = w_size
        self.w_shard = w_shard
        self.w_cohesion = w_cohesion
        self.w_facts = w_facts

    def shard_schema(
        self,
        tables: List[str],
        columns_by_table: Dict[str, List[str]],
        pks_by_table: Dict[str, List[str]],
        fks: List[Tuple[str, str, str, str]],
        facts: Dict[str, List[Tuple[str, str]]],
    ):
        model = cp_model.CpModel()
        K = range(self.max_shards)

        # --- Decision variables ---
        y_k = {}  # shard k active
        y_tk = {}  # table t in shard k
        x_ck = {}  # column (t,c) in shard k

        for k in K:
            y_k[k] = model.NewBoolVar(f"y_{k}")
            for t in tables:
                y_tk[(t, k)] = model.NewBoolVar(f"y_{t}_{k}")
                for c in columns_by_table.get(t, []):
                    x_ck[(t, c, k)] = model.NewBoolVar(f"x_{t}_{c}_{k}")

        # =====================================================================
        # HARD CONSTRAINTS
        # =====================================================================

        # HC1: Table in shard -> shard active; Column in shard -> table in shard;
        #      Table in shard -> PK in shard
        for k in K:
            for t in tables:
                model.Add(y_tk[(t, k)] <= y_k[k])
                for c in columns_by_table.get(t, []):
                    model.Add(x_ck[(t, c, k)] <= y_tk[(t, k)])
                for p in pks_by_table.get(t, []):
                    model.Add(y_tk[(t, k)] <= x_ck[(t, p, k)])

        # HC2: FK closure -- FK column in shard -> parent table + parent PK in shard
        for k in K:
            for t_fk, c_fk, t_ref, c_ref in fks:
                model.Add(x_ck[(t_fk, c_fk, k)] <= y_tk[(t_ref, k)])
                model.Add(x_ck[(t_fk, c_fk, k)] <= x_ck[(t_ref, c_ref, k)])

        # HC3: Every table in at least one shard
        for t in tables:
            model.Add(sum(y_tk[(t, k)] for k in K) >= 1)

        # HC4: Every column in at least one shard
        for t in tables:
            for c in columns_by_table.get(t, []):
                model.Add(sum(x_ck[(t, c, k)] for k in K) >= 1)

        # HC5: Shard capacity
        M = model.NewIntVar(0, len(tables), "M")
        for k in K:
            model.Add(sum(y_tk[(t, k)] for t in tables) <= M)
            model.Add(sum(y_tk[(t, k)] for t in tables) <= self.max_tables_per_shard)

        # HC6: Symmetry breaking
        for k in range(self.max_shards - 1):
            model.Add(y_k[k] >= y_k[k + 1])

        # HC7: Fact containment -- h_fk[f,k]=1 iff all fact f's columns in shard k
        # HC7a: facts with empty or fully-unmatched column lists get h=0 forced --
        # they must never count toward HC8/HC9 for free (ISSUES.md #4 fix).
        # HC10 (same pass): r_fk[f,k]=1 whenever ANY matched column lands in
        # shard k -- a broader "touches" notion than h_fk's "fully contained,"
        # feeding the facts-touched soft penalty (SO3) below: in the real
        # downstream pipeline (fact_allocation.py's stub-table mechanism), a
        # shard sees a fact's context the moment it holds ANY of that fact's
        # tables, not only when it holds ALL of them. Only a lower-bound
        # constraint is needed for r_fk since it's about to be PENALIZED in
        # the objective -- the solver never sets it above what's forced.
        valid_fact_ids: Set[str] = set()
        h_fk = {}
        r_fk = {}
        for f_id, f_cols in facts.items():
            matched = [
                (t, c)
                for t, c in f_cols
                if (t, c, k) in x_ck or any((t, c, k2) in x_ck for k2 in K)
            ]
            if not matched:
                # No valid columns -- this fact can never genuinely be contained anywhere
                for k in K:
                    h = model.NewBoolVar(f"h_{f_id}_{k}")
                    h_fk[(f_id, k)] = h
                    model.Add(h == 0)
                continue
            valid_fact_ids.add(f_id)
            for k in K:
                h = model.NewBoolVar(f"h_{f_id}_{k}")
                h_fk[(f_id, k)] = h
                r = model.NewBoolVar(f"r_{f_id}_{k}")
                r_fk[(f_id, k)] = r
                for t, c in matched:
                    if (t, c, k) in x_ck:
                        model.Add(h <= x_ck[(t, c, k)])
                        model.Add(r >= x_ck[(t, c, k)])

        # HC8: Every fact fully contained in >=1 shard (only valid facts)
        for f_id in valid_fact_ids:
            model.Add(sum(h_fk[(f_id, k)] for k in K) >= 1)

        # HC9: Non-emptiness -- every active shard has >=1 valid fact
        for k in K:
            model.Add(sum(h_fk[(f_id, k)] for f_id in valid_fact_ids) >= y_k[k])

        # =====================================================================
        # SOFT OBJECTIVES
        # =====================================================================

        # SO1: Column co-occurrence reward (semantic affinity proxy -- the
        #      "attribute affinity matrix" of classical vertical-fragmentation
        #      theory, Ozsu/Valduriez). If two COLUMNS from different tables
        #      co-occur in a fact, reward putting them in the same shard --
        #      capped at MOST ONCE per column pair, regardless of how many
        #      shards happen to contain that pair (see the any_together
        #      rationale below). Deliberately COLUMN-granularity, not table-
        #      granularity: rewarding at the table level (the original
        #      formulation) ignores which specific columns a fact actually
        #      needs, so once ANY column-level reason pulled a table into a
        #      shard, the reward gave no further signal about which of that
        #      table's OTHER columns belong there too -- x_ck's per-column
        #      variables already support genuine per-table projections
        #      (different column subsets in different shards, PK constant
        #      everywhere, matching vertical fragmentation's own invariant),
        #      but nothing was using that granularity for cohesion until now.
        #      any_together[(col1,col2)] models "co-located somewhere" as a
        #      genuine OR: the model always wants it as large as feasible (it
        #      strictly helps the objective), so the solver sets it to 1
        #      whenever at least one shard co-locates the pair and 0
        #      otherwise -- never more than 1, regardless of duplicate count
        #      (the same fix applied at table granularity earlier this
        #      session, now correctly applied where the real affinity data
        #      -- facts[f_id]'s (table,column) pairs -- actually lives).
        col_pairs: Set[Tuple[Tuple[str, str], Tuple[str, str]]] = set()
        co_occurrence: Dict[Tuple[Tuple[str, str], Tuple[str, str]], int] = defaultdict(
            int
        )
        for f_cols in facts.values():
            unique_cols = list({(t, c) for t, c in f_cols})
            for i in range(len(unique_cols)):
                for j in range(i + 1, len(unique_cols)):
                    col1, col2 = unique_cols[i], unique_cols[j]
                    if col1[0] == col2[0]:
                        continue  # same table -- not a cross-table cohesion signal
                    pair = (col1, col2) if col1 < col2 else (col2, col1)
                    co_occurrence[pair] += 1
                    col_pairs.add(pair)

        co_occur_vars = {}
        for col1, col2 in co_occurrence:
            for k in K:
                if (*col1, k) not in x_ck or (*col2, k) not in x_ck:
                    continue
                c_var = model.NewBoolVar(
                    f"co_{col1[0]}_{col1[1]}_{col2[0]}_{col2[1]}_{k}"
                )
                co_occur_vars[(col1, col2, k)] = c_var
                model.Add(c_var <= x_ck[(*col1, k)])
                model.Add(c_var <= x_ck[(*col2, k)])

        any_together_vars = {}
        for col1, col2 in co_occurrence:
            any_var = model.NewBoolVar(
                f"any_together_{col1[0]}_{col1[1]}_{col2[0]}_{col2[1]}"
            )
            any_together_vars[(col1, col2)] = any_var
            model.Add(
                any_var
                <= sum(
                    co_occur_vars[(col1, col2, k)]
                    for k in K
                    if (col1, col2, k) in co_occur_vars
                )
            )

        reward_cooccur = sum(
            co_occurrence[(col1, col2)] * any_together_vars[(col1, col2)]
            for (col1, col2) in any_together_vars
        )

        # SO2: Minimize TOTAL footprint (tables + columns), summed across
        #      every active shard -- this is the real proxy for total
        #      tokens spent across all of Phase 1's parallel LLM calls.
        #      Deliberately a sum, not a makespan/max: a max only bounds
        #      the single worst active shard and stops growing entirely
        #      once every active shard is already at its capacity ceiling
        #      (exactly what a fully FK-connected schema forces via HC2),
        #      which meant duplicate shards cost nothing in this term --
        #      the other half of the bug above. max_tables_per_shard (HC5)
        #      remains the HARD per-shard context-budget cap; this term is
        #      the separate, SOFT concern of minimizing actual spend given
        #      that cap, not guaranteeing it.
        total_size = sum(y_tk[(t, k)] for t in tables for k in K) + sum(
            x_ck[(t, c, k)]
            for t in tables
            for c in columns_by_table.get(t, [])
            for k in K
        )

        # SO3: Minimize the total number of facts TOUCHING any shard
        #      (sum of r_fk, HC10) -- the real proxy for per-shard LLM
        #      context load. Confirmed live to matter: achieving perfect
        #      cohesion coverage by duplicating hub tables into every shard
        #      that wants them costs nothing under SO1/SO2 alone, but each
        #      such duplicate shard ends up seeing every fact that touches
        #      any of its tables -- measured on a real schema, 3 of 4
        #      shards each carrying the ENTIRE cross-table fact burden.
        #      This term prices that directly, distinct from SO1 (which
        #      only rewards tables landing together, never penalizes how
        #      much total fact-context piles onto one shard).
        total_facts_touched = sum(r_fk.values())

        # --- Objective normalization ---
        # Each term is normalized by its own achievable scale, computed
        # directly from THIS instance's schema/fact data (never a
        # hardcoded constant), so a raw weight of "1" means the same
        # relative influence on every term regardless of its natural
        # magnitude -- without this, a term whose natural range is 0-500
        # would dominate one whose natural range is 0-10 at equal raw
        # weight values, which isn't a meaningful trade-off at all.
        # Scaled by PRECISION and rounded to an integer since CP-SAT's
        # objective requires integer coefficients.
        PRECISION = 1000
        size_scale = max(
            1, len(tables) + sum(len(columns_by_table.get(t, [])) for t in tables)
        )
        cohesion_scale = max(1, sum(co_occurrence.values()))
        facts_scale = max(1, len({f_id for (f_id, _k) in r_fk}))
        shard_scale = max(1, self.max_shards)

        def _scaled(raw_weight: float, scale: int) -> int:
            return max(1, round(PRECISION * raw_weight / scale))

        # --- Objective ---
        model.Minimize(
            _scaled(self.w_shard, shard_scale) * sum(y_k[k] for k in K)
            + _scaled(self.w_size, size_scale) * total_size
            + _scaled(self.w_facts, facts_scale) * total_facts_touched
            - _scaled(self.w_cohesion, cohesion_scale) * reward_cooccur
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = _SOLVER_MAX_SECONDS
        solver.parameters.num_search_workers = 1

        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            shards = []
            shard_facts = []
            for k in K:
                if solver.Value(y_k[k]):
                    shard_tables = {}
                    for t in tables:
                        if solver.Value(y_tk[(t, k)]):
                            proj = [
                                c
                                for c in columns_by_table.get(t, [])
                                if solver.Value(x_ck[(t, c, k)])
                            ]
                            shard_tables[t] = proj
                    if shard_tables:
                        s_facts = [
                            f for f in facts.keys() if solver.Value(h_fk[(f, k)])
                        ]
                        shards.append(shard_tables)
                        shard_facts.append(s_facts)
            return shards, shard_facts
        else:
            return None, None


class GridRun(BaseModel):
    hyperparameters: Dict[str, Any] = Field(
        description="The hyperparameter coordinates for this run"
    )
    # list of dict mapping table to list of columns
    shards: List[Dict[str, List[str]]] = Field(
        description="The resulting sharding structure"
    )
    shard_facts: List[List[str]] = Field(
        description="The source fact IDs driving each shard"
    )

    @property
    def normalized_structure(self) -> FrozenSet[FrozenSet[str]]:
        # Normalize the structural output for equality checking (just using table projections stringified)
        shard_sets = []
        for s in self.shards:
            s_set = []
            for t, cols in s.items():
                s_set.append(f"{t}({','.join(sorted(cols))})")
            shard_sets.append(frozenset(s_set))
        return frozenset(shard_sets)


class PlateauResult(BaseModel):
    structure: List[Dict[str, List[str]]] = Field(description="The sharding structure")
    shard_facts: List[List[str]] = Field(
        description="The fact provenance mapping for the structure"
    )
    configs: List[Dict[str, Any]] = Field(description="Hyperparameter configs")
    frequency: int = Field(description="Number of times this structure appeared")


class StabilitySelector:
    def __init__(self) -> None:
        self.runs: List[GridRun] = []

    def add_result(
        self,
        hyperparameters: Dict[str, Any],
        shards: List[Dict[str, List[str]]],
        shard_facts: List[List[str]],
    ) -> None:
        self.runs.append(
            GridRun(
                hyperparameters=hyperparameters, shards=shards, shard_facts=shard_facts
            )
        )

    def find_most_stable(self) -> Optional[PlateauResult]:
        if not self.runs:
            return None
        plateaus = self.get_all_plateaus()
        return plateaus[0] if plateaus else None

    def get_all_plateaus(self) -> List[PlateauResult]:
        structure_to_configs: Dict[FrozenSet[FrozenSet[str]], List[Dict[str, Any]]] = (
            defaultdict(list)
        )
        structure_to_original: Dict[
            FrozenSet[FrozenSet[str]],
            Tuple[List[Dict[str, List[str]]], List[List[str]]],
        ] = {}

        for run in self.runs:
            norm = run.normalized_structure
            structure_to_configs[norm].append(run.hyperparameters)
            if norm not in structure_to_original:
                structure_to_original[norm] = (run.shards, run.shard_facts)

        plateaus: List[PlateauResult] = []
        for norm, configs in structure_to_configs.items():
            shards, facts = structure_to_original[norm]
            plateaus.append(
                PlateauResult(
                    structure=shards,
                    shard_facts=facts,
                    configs=configs,
                    frequency=len(configs),
                )
            )

        plateaus.sort(key=lambda x: x.frequency, reverse=True)
        return plateaus


class SearchSpaceSeeder:
    """Provides the weight-sweep grid for run_stability_sweep(). Needs NO
    schema/fact statistics: every objective term is already normalized by
    its own per-instance scale inside shard_schema() (size by whole-schema
    footprint, cohesion by max achievable reward, facts-touched by valid
    fact count, shard-count by max_shards), so a single universal
    geometric range works consistently across any schema size -- the old
    schema-size-scaled step derivation was a workaround for UNNORMALIZED
    terms and is no longer needed now that the terms themselves are
    normalized.

    The range is geometric (log-spaced, spanning ~300x from 0.1 to 30),
    not linear: a narrow linear band like [0.5, 1, 2, 4] can only ever
    explore ratios within an 8x spread, which isn't enough to discover
    "this factor should dominate" or "this factor barely matters"
    regimes -- exactly the regimes that matter for finding a genuinely
    different trade-off than the default.

    w_shard is deliberately EXCLUDED from the swept grid -- fixed at 1 as
    the sweep's numeraire in generate_grid(). Scaling every weight in a
    weighted-sum objective by the same positive constant never changes
    which solution is optimal; only the RATIOS between weights matter. So
    sweeping w_shard independently would only re-explore the same ratios
    already covered by sweeping the other three, at 6x the compute cost --
    fixing it removes a redundant degree of freedom for free.
    """

    _GEOMETRIC_RANGE = [0.1, 0.3, 1, 3, 10, 30]

    def weight_ranges(self) -> Dict[str, List[float]]:
        return {
            "w_size": list(self._GEOMETRIC_RANGE),
            "w_cohesion": list(self._GEOMETRIC_RANGE),
            "w_facts": list(self._GEOMETRIC_RANGE),
        }

    def generate_grid(self) -> List[Dict[str, Any]]:
        ranges = self.weight_ranges()
        grid = []
        for combo in itertools.product(*ranges.values()):
            conf = dict(zip(ranges.keys(), combo))
            conf["w_shard"] = 1
            grid.append(conf)
        return grid


def solve_instance(
    conf, max_shards, max_tables_per_shard, tables, cols, pks, fks, facts
):
    sharder = ILPSharder(
        max_shards=max_shards, max_tables_per_shard=max_tables_per_shard, **conf
    )
    shards, shard_facts = sharder.shard_schema(tables, cols, pks, fks, facts)
    return conf, shards, shard_facts


def run_stability_sweep(
    tables: List[str],
    columns_by_table: Dict[str, List[str]],
    pks_by_table: Dict[str, List[str]],
    fks: List[Tuple[str, str, str, str]],
    facts: Dict[str, List[Tuple[str, str]]],
    *,
    max_shards: int,
    max_tables_per_shard: int,
) -> Optional[PlateauResult]:
    """Sweeps every weight combination in SearchSpaceSeeder's grid -- 3
    swept dimensions (w_size/w_cohesion/w_facts, each a geometric range
    spanning 0.1-30) with w_shard fixed at 1 as the numeraire, 216
    combinations total -- and returns the structure that recurs most
    often across them (StabilitySelector's plateau). Weight choice only
    affects the SOFT objective, never the hard constraints, so
    feasibility is identical across every combination in the grid -- this
    can never be "flaky" (some combos feasible, others not) for a fixed
    max_shards/max_tables_per_shard pair.

    max_shards/max_tables_per_shard are REQUIRED, caller-derived values
    (see _derive_max_shards/_derive_max_tables_per_shard) -- this function
    itself never guesses or hardcodes either."""
    seeder = SearchSpaceSeeder()
    grid = seeder.generate_grid()
    selector = StabilitySelector()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                solve_instance,
                conf,
                max_shards,
                max_tables_per_shard,
                tables,
                columns_by_table,
                pks_by_table,
                fks,
                facts,
            )
            for conf in grid
        ]
        for future in concurrent.futures.as_completed(futures):
            conf, shards, shard_facts = future.result()
            if shards:
                selector.add_result(conf, shards, shard_facts)

    return selector.find_most_stable()


# ---------------------------------------------------------------------------
# Automatic entry point -- no manually-set hyperparameters anywhere.
# Callers pass only the schema/fact data plus which provider/model will
# consume the resulting shards; every ILP hyperparameter is derived.
# ---------------------------------------------------------------------------


def _estimate_prompt_tokens_per_table(columns_by_table: Dict[str, List[str]]) -> float:
    """Rough per-table prompt-token cost estimate from the primitive
    schema shape alone (table name + column names/count) -- deliberately
    does NOT depend on the actual prompt-rendering code in pipeline/
    (this module stays pipeline-independent, consistent with its
    existing design). ~10 chars for the table header line, ~35 chars per
    column line (name + type + nullability annotation), /4 chars-per-
    token as a standard rough estimate."""
    if not columns_by_table:
        return 40.0
    avg_cols = sum(len(cols) for cols in columns_by_table.values()) / len(
        columns_by_table
    )
    return (10 + avg_cols * 35) / 4.0


def _derive_max_shards(
    tables: List[str], facts: Dict[str, List[Tuple[str, str]]]
) -> int:
    """A PROVABLE safe ceiling, not an estimate: CP-SAT can never need
    more ACTIVE shards than the smaller of (a) the table count -- every
    active shard needs at least one table -- or (b) the number of facts
    with at least one resolvable column pair -- HC9 already requires
    every active shard to contain at least one such fact. The existing
    w_shard soft objective already minimizes down from this ceiling to
    whatever count is actually optimal; this bound only has to be safe,
    never tight."""
    valid_fact_count = sum(1 for pairs in facts.values() if pairs)
    if valid_fact_count == 0:
        return max(1, len(tables))
    return max(1, min(len(tables), valid_fact_count))


def _derive_max_tables_per_shard(
    tables: List[str],
    columns_by_table: Dict[str, List[str]],
    *,
    provider: str,
    model: str,
    api_key: str,
    fixed_prompt_overhead_tokens: int,
    context_window_safety_margin: float,
) -> int:
    """Derives a real token-budget ceiling from the target model's actual
    context window (get_context_window() -- always a live lookup, never
    a maintained constant) minus the caller's fixed prompt overhead and a
    safety margin, divided by the estimated marginal cost of one more
    table. Degrades to "no meaningful cap" (len(tables)) rather than
    raising if the budget or per-table estimate come out non-positive --
    a missing cap is safe (the ILP's own capacity constraint just becomes
    non-binding), unlike guessing a wrong one."""
    from src.util.core.context_window import get_context_window

    context_window = get_context_window(provider, model, api_key=api_key)
    budget_tokens = (
        context_window * context_window_safety_margin - fixed_prompt_overhead_tokens
    )
    per_table_tokens = _estimate_prompt_tokens_per_table(columns_by_table)
    if budget_tokens <= 0 or per_table_tokens <= 0:
        return max(1, len(tables))
    return max(1, min(len(tables), int(budget_tokens // per_table_tokens)))


def shard_schema_auto(
    tables: List[str],
    columns_by_table: Dict[str, List[str]],
    pks_by_table: Dict[str, List[str]],
    fks: List[Tuple[str, str, str, str]],
    facts: Dict[str, List[Tuple[str, str]]],
    *,
    provider: str,
    model: str,
    api_key: str = "",
    fixed_prompt_overhead_tokens: int = 6000,
    context_window_safety_margin: float = 0.6,
):
    """The one entry point live pipeline code should call -- no
    max_shards, max_tables_per_shard, or weight parameter is ever exposed
    here. Every ILP hyperparameter is derived from the schema, the fact
    set, and the target provider/model's real context window:

      - max_shards: a provable ceiling (see _derive_max_shards), never a
        guess.
      - max_tables_per_shard: a real token-budget calculation against
        the target model's live-queried context window (see
        _derive_max_tables_per_shard / get_context_window).
      - soft-objective weights: NOT a single guessed point -- actually
        swept via run_stability_sweep() across every combination in
        SearchSpaceSeeder's grid (3 swept dimensions -- w_size/
        w_cohesion/w_facts, each a normalized-scale-independent geometric
        range -- with w_shard fixed at 1 as the numeraire; 216
        combinations total, still solved exhaustively as a routine step).
        Weight choice never affects feasibility (only the hard
        constraints do, and those don't depend on weights), so this
        picks the STRUCTURE that recurs most often across the swept
        range -- a genuinely more robust notion of "automatic" than any
        single derived point could be.

    Returns the same (shards, shard_facts) shape as ILPSharder.shard_schema().
    """
    max_shards = _derive_max_shards(tables, facts)
    max_tables_per_shard = _derive_max_tables_per_shard(
        tables,
        columns_by_table,
        provider=provider,
        model=model,
        api_key=api_key,
        fixed_prompt_overhead_tokens=fixed_prompt_overhead_tokens,
        context_window_safety_margin=context_window_safety_margin,
    )
    grid_size = len(SearchSpaceSeeder().generate_grid())
    # This step emitted NOTHING before, and on a real run it took 8m07s of
    # wall clock between two log lines -- indistinguishable from a hang while
    # watching a live run. Every solve is capped at 30s (see shard_schema), so
    # the worst case is knowable up front; say so.
    logger.info(
        "[sharder] %d tables, %d FKs, %d facts | derived max_shards=%d, "
        "max_tables_per_shard=%d (model=%s) | sweeping %d weight configs, "
        "each capped at %.0fs -- worst case ~%.0fs of CPU across the pool",
        len(tables),
        len(fks),
        len(facts),
        max_shards,
        max_tables_per_shard,
        model,
        grid_size,
        _SOLVER_MAX_SECONDS,
        grid_size * _SOLVER_MAX_SECONDS,
    )
    started = time.monotonic()
    plateau = run_stability_sweep(
        tables,
        columns_by_table,
        pks_by_table,
        fks,
        facts,
        max_shards=max_shards,
        max_tables_per_shard=max_tables_per_shard,
    )
    elapsed = time.monotonic() - started
    if plateau is None:
        logger.warning(
            "[sharder] sweep found no feasible structure after %.1fs.", elapsed
        )
        return None, None
    logger.info(
        "[sharder] sweep done in %.1fs -> %d shard(s): %s",
        elapsed,
        len(plateau.structure),
        [sorted(s.keys()) for s in plateau.structure],
    )
    return plateau.structure, plateau.shard_facts
