# ScribbleDB docs

Everything here is untracked (`.gitignore` excludes `*.md` wholesale) -- these
are working documents, not part of the reproducibility artifact. The artifact's
own front door is the root `README.md`.

These used to be scattered across `experiments/`, `baselines/`,
`baselines/experiments/` and the repo root. Two of them were load-bearing from
those scratch locations: `RELATION_CONDITION_CONSTRAINT_DESIGN.md` is cited by
section number throughout `src/util/constraint_model/`, and
`STAGE3_PHASE2_DESIGN.md` is what `CLAUDE.md` calls the authoritative Stage 3
architecture. Neither belonged in a directory whose stated purpose is
throwaway scratch.

## design/

| File | What it is |
|---|---|
| `RELATION_CONDITION_CONSTRAINT_DESIGN.md` | The Relation/Condition/Constraint object model. `src/util/constraint_model/` cites this by section number -- read it before changing any node shape. |
| `STAGE3_PHASE2_DESIGN.md` | Stage 3's overall architecture. |
| `STAGE3_CONSTRAINT_MODEL_UNIFIED.md` | The unified constraint-model design. |
| `MULTIVARIATE_CONSTRAINT_DESIGN.md` | Correlation / joint-dependence modelling. |
| `STAGE3_DOF_OPEN_QUESTIONS.md` | Unresolved degrees-of-freedom questions. |

## eval/

| File | What it is |
|---|---|
| `EVAL_PLAN_STAGE1-3.md` | Evaluation methodology for Stages 1-3. |
| `EVAL_PLAN_STAGE4.md` | Full-pipeline evaluation plan: Stage 4 experiments, cross-cutting ablations, real-DB grounding. Assumes Stage 4 built (it isn't yet) -- a target spec, not a runnable plan. |
| `GPU_COST_PLAN.md` | Self-hosted inference cost analysis and measured benchmarks. |
| `DATA_GENERATION_BASELINES.md` | Survey of the data-generation baselines. |

## adr/

Architecture decision records. One per irreversible call, newest wins.

## Other

- `ISSUES.md` -- a dated (2026-07-13) Stage 3 findings document. Predates the
  Stage 3 rewrite; treat its file/line references as historical.
- `teaching/` -- course material, unrelated to the pipeline.

## Not here, on purpose

`CLAUDE.md`, `AGENTS.md`, `PROGRESS.md`, `PROGRESS_ARCHIVE.md`,
`GENERAL_KNOWLEDGE.md` and `specs/` stay at the repo root. They are agent-harness
working memory and the tooling expects them there.
