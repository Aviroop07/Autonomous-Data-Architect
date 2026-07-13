import collections
from typing import List, Dict, Set, Tuple, Any, FrozenSet, Optional
from ortools.sat.python import cp_model
from collections import defaultdict
from pydantic import BaseModel, Field
import itertools
import concurrent.futures


class ILPSharder:
    def __init__(
        self,
        max_shards: int = 10,
        max_tables_per_shard: int = 8,
        w_cap: int = 10,
        w_shard: int = 20,
        w_fact: int = 50,
        w_fk: int = 30,
        w_col: int = 5,
    ):
        self.max_shards = max_shards
        self.max_tables_per_shard = max_tables_per_shard
        # w_cap: penalize max shard size (tables + columns)
        # w_shard: penalize #active shards
        # w_fk: reward table co-occurrence in same shard (semantic similarity)
        # w_fact, w_col: legacy params, unused (fact containment is hard constraint)
        self.w_cap = w_cap
        self.w_shard = w_shard
        self.w_fact = w_fact
        self.w_fk = w_fk
        self.w_col = w_col

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
        valid_fact_ids: Set[str] = set()
        h_fk = {}
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
                for t, c in matched:
                    if (t, c, k) in x_ck:
                        model.Add(h <= x_ck[(t, c, k)])

        # HC8: Every fact fully contained in >=1 shard (only valid facts)
        for f_id in valid_fact_ids:
            model.Add(sum(h_fk[(f_id, k)] for k in K) >= 1)

        # HC9: Non-emptiness -- every active shard has >=1 valid fact
        for k in K:
            model.Add(sum(h_fk[(f_id, k)] for f_id in valid_fact_ids) >= y_k[k])

        # =====================================================================
        # SOFT OBJECTIVES
        # =====================================================================

        # SO1: Table co-occurrence reward (semantic similarity proxy)
        #      If two tables co-occur in facts, reward putting them in same shard.
        table_pairs: Set[Tuple[str, str]] = set()
        co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        for f_cols in facts.values():
            f_tables = list({t for t, _ in f_cols})
            for i in range(len(f_tables)):
                for j in range(i + 1, len(f_tables)):
                    pair = tuple(sorted((f_tables[i], f_tables[j])))
                    co_occurrence[pair] += 1
                    table_pairs.add(pair)

        co_occur_vars = {}
        for (t1, t2), weight in co_occurrence.items():
            for k in K:
                c_var = model.NewBoolVar(f"co_{t1}_{t2}_{k}")
                co_occur_vars[(t1, t2, k)] = c_var
                model.Add(c_var <= y_tk[(t1, k)])
                model.Add(c_var <= y_tk[(t2, k)])

        reward_cooccur = sum(
            co_occurrence[(t1, t2)] * co_occur_vars[(t1, t2, k)]
            for (t1, t2, k) in co_occur_vars
        )

        # SO2: Minimize max shard size (tables + columns) via makespan variable
        max_size = model.NewIntVar(
            0,
            len(tables) + sum(len(cs) for cs in columns_by_table.values()),
            "max_shard_size",
        )
        for k in K:
            shard_size_k = sum(y_tk[(t, k)] for t in tables) + sum(
                x_ck[(t, c, k)] for t in tables for c in columns_by_table.get(t, [])
            )
            model.Add(max_size >= shard_size_k)

        # --- Objective ---
        w_shard = self.w_shard  # penalize #shards
        w_size = self.w_cap  # penalize max shard size (reuse w_cap)
        w_cooccur = self.w_fk  # reward co-located tables (reuse w_fk)

        model.Minimize(
            w_shard * sum(y_k[k] for k in K)
            + w_size * max_size
            - w_cooccur * reward_cooccur
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
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
    def __init__(
        self,
        tables: List[str],
        facts: Dict[str, List[Tuple[str, str]]],
        fks: List[Tuple[str, str, str, str]],
    ):
        self.num_tables = len(tables)
        self.num_facts = len(facts)
        self.num_fks = len(fks)
        self.fact_density = self.num_facts / max(1, self.num_tables)
        self.fk_density = self.num_fks / max(1, self.num_tables)

    def generate_grid(self) -> List[Dict[str, Any]]:
        cap_step = max(1, self.num_tables // 4)
        w_cap_range = list(range(1, cap_step * 4 + 1, cap_step))

        shard_step = max(1, self.num_tables // 2)
        w_shard_range = list(range(1, shard_step * 4 + 1, shard_step))

        fact_step = max(1, int(self.fact_density * 10))
        w_fact_range = list(range(fact_step, fact_step * 4 + 1, fact_step))

        fk_step = max(1, int(self.fk_density * 10))
        w_fk_range = list(range(fk_step, fk_step * 4 + 1, fk_step))

        col_step = max(1, int(self.fact_density * 5))
        w_col_range = list(range(1, col_step * 3 + 1, col_step))

        grid = []
        for c, s, fa, fk, col in itertools.product(
            w_cap_range, w_shard_range, w_fact_range, w_fk_range, w_col_range
        ):
            grid.append(
                {"w_cap": c, "w_shard": s, "w_fact": fa, "w_fk": fk, "w_col": col}
            )
        return grid


def solve_instance(conf, tables, cols, pks, fks, facts):
    sharder = ILPSharder(max_shards=5, max_tables_per_shard=8, **conf)
    shards, shard_facts = sharder.shard_schema(tables, cols, pks, fks, facts)
    return conf, shards, shard_facts


def run_stability_sweep(
    tables: List[str],
    columns_by_table: Dict[str, List[str]],
    pks_by_table: Dict[str, List[str]],
    fks: List[Tuple[str, str, str, str]],
    facts: Dict[str, List[Tuple[str, str]]],
) -> Optional[PlateauResult]:
    seeder = SearchSpaceSeeder(tables, facts, fks)
    grid = seeder.generate_grid()
    selector = StabilitySelector()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                solve_instance, conf, tables, columns_by_table, pks_by_table, fks, facts
            )
            for conf in grid
        ]
        for future in concurrent.futures.as_completed(futures):
            conf, shards, shard_facts = future.result()
            if shards:
                selector.add_result(conf, shards, shard_facts)

    return selector.find_most_stable()
