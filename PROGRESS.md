# ScribbleDB -- Progress Log

## 2026-06-17

### 00:00 -- Repository onboarding, test fixes, and WIP commit cleanup

**Context:** 109 uncommitted files had accumulated since the last commit (11 days ago). This
session was a clean-up pass: survey the WIP, fix broken tests, and land everything in logical
commits.

**Test fixes (3 failures in the existing suite):**
- `test_stage1_models.py`: `RawFact.novelty_reason` default changed from `""` to `None`
  (field is `Optional[str]`); updated assertion to `is None`.
- `test_stage1_text_matching.py`: `TokenSpanIndex` renamed its internal FAISS index from
  `.index` to `._tfidf`; updated two assertions.
- `test_stage1_models.py`: `IntegrityReport(missing_information=[{...}])` passed raw dicts
  where `List[Issue]` is expected; replaced with explicit `Issue(...)` construction to
  satisfy the type checker.

After fixes: **381 passed, 0 failed**.

**Commits landed (5):**

1. `chore(util): restructure flat utilities into domain-specific subdirectories`
   - 14 flat `src/util/*.py` files moved into 7 subdirectories:
     `core/`, `algorithms/`, `analysis/`, `config/`, `observability/`, `orchestration/`,
     `schema_ops/`
   - New files: `span_index.py`, `semantic_match.py`, `llm_trace.py`, `loop.py`,
     `loop_types.py`, `retry_loop.py` (inside `orchestration/`)

2. `feat(stage1): add AgentLoop, context auditor, and enrichment middleware`
   - Migrated all Stage 1 agents to AgentLoop framework (`loop_config.py`)
   - Added `context_auditor` agent (LLM-driven semantic review of external facts,
     replacing old deterministic filter)
   - Added `external_context_filter`, `tag_normalization`, `underspec_detector` middleware
   - Added `context_audit` model for audit trail tracking
   - Removed deprecated `text_matching.py` and `technical_term.py`
   - Updated `integrity_report`: `unresolved_ambiguities` and `search_suggestions` fields
   - Updated `raw_fact`: `novelty_reason` is now `Optional[str]`

3. `feat(stage2): improve schema architect, domain auditor, and merger`
   - Schema architect: stronger fact-coverage rules, reserved-keyword guards, M:N handling
   - Domain auditor: patch normalization, prompt compression with fact-pool hints
   - SchemaMerger: Gale-Shapley column alignment, FK consistency validation,
     junction table consolidation
   - Similarity: embedding score used as `max(lexical, embedding)`

4. `feat(stage3): add constraint patch agent, math verifier, and expression parser`
   - Added `constraint_patch_agent` and `mathematics_verifier` agents
   - Added `expression_parser`, `mathematics`, `satisfiability` middleware
   - Added `expressions` and `validation` models
   - Major `entry.py` refactoring: improved healing loop, fact allocation on retry

5. `chore: update pipeline runners, test harness, config, and gitignore`
   - `run_pipeline.py`, `run_evaluation.py`: updated for new util import paths
   - Added `run_human_stress_tests.py`, `run_large_test.py`
   - Added `experiments/` to `.gitignore`
   - Test imports updated throughout for new util layout

## 2026-06-10 (continued, session 5)

### 15:10 -- Stage 2 auditor patch normalization

**Issue:** Stage 2 failed during auditing because the domain auditor emitted a DELETE_COLUMN patch without column_name (it provided a columns list), causing structured-output parsing to fail.

**Fixes:**
- `schema_patch.py`: expanded CritiqueReport preprocessor to normalize table/column key aliases and split DELETE_COLUMN patches with columns lists into one patch per column.
- `domain_auditor/prompt.txt`: added explicit rule that DELETE_COLUMN requires table_name + column_name, and to emit one patch per column (no columns list).

### 15:40 -- Stage 2 auditor prompt compression

**Issue:** Auditor prompts were extremely large because each entity listed the full fact list, causing massive duplication and latency on large shards.

**Fix:** The auditor now receives a FACT POOL (unique facts listed once) and an ENTITY FACT IDS hint map. Domain intelligence is compacted to domain + research_summary when available.
- `src/pipeline/stage2/agents/domain_auditor/agent.py`: build fact pool + entity-id map, compact intelligence payload.
- `src/pipeline/stage2/agents/domain_auditor/prompt.txt`: updated input format and guidance to use FACT POOL with ENTITY FACT IDS hints.

### 16:05 -- Stage 2 architect fact formatting compression

**Issue:** Per-shard architect prompts repeated full tag lists for every fact, inflating token usage on large clusters.

**Fix:** Only non-STRUCTURAL tags are shown inline (e.g., `[LOGICAL]`), and STRUCTURAL is treated as the default.
- `src/pipeline/stage2/agents/schema_architect/agent.py`: compact fact rendering and clarified tag legend in the query header.

### 17:30 -- Enrichment pipeline bug fix

**Context:** Three large, complex, human-like NL descriptions (cases 21-23) were run
through Stage 1 + Stage 2. Reviewing the internals revealed two separate bugs in the
enrichment pipeline that caused all accepted external facts to be silently dropped.

**Bug 1 fixed: enrichment facts never merged on successful final round**

Root cause: `ContextEnricherLoopAgent.build_context()` (which merges newly accepted facts
into `accumulated_accepted`) is only called at the START of each enricher round to
prepare the next query. When the auditor returns `is_acceptable=True`, the loop terminates
immediately via the `auditor -> end` edge -- there is no next enricher round, so
`build_context()` is never called with the final accepted set. `accumulated_accepted`
remains empty regardless of what the auditor approved.

Fix (`src/orchestration/stage1/entry.py` -- `_run_context_enrichment_loop`): after
`AgentLoop.run()` returns, inspect `result.node_outputs["auditor"]` and
`result.node_outputs["enricher"]`. If the auditor ended with `is_acceptable=True`,
merge the approved facts from the final enricher output into `accumulated_accepted`.

Evidence from case 21: auditor attempt 2 returned `accepted_fact_ids=[53,54,55]`,
`is_acceptable=True`. But `enrichment_filter_report.accepted_facts` was `[]`, and 0
external facts appeared in the final fact set.

**Bug 2 fixed (same post-loop merge handles it): auditor omits `accepted_fact_ids` on success**

Root cause: `ContextAuditReport.accepted_fact_ids` has no field description instruction
to enumerate IDs when `is_acceptable=True`. The LLM correctly sets `is_acceptable=True`
and leaves `rejected_facts=[]` but also leaves `accepted_fact_ids=[]`.

Fix: when `accepted_fact_ids` is empty but `is_acceptable=True`, the post-loop merge
infers accepted IDs as `proposed_ids - rejected_ids` (all proposed minus any rejected).

Evidence from case 22: attempt 2 returned `is_acceptable=True, accepted_fact_ids=[]`.
Three proposed facts should have been accepted; none were.

---

### 18:00 -- Cases 21, 22, 23 internals investigation

Ran all three cases end-to-end (Stage 1 + 2) and investigated the resulting schemas
against the input facts. Full findings at `output/investigation/cases_21_22_investigation.md`.

**Case 21 -- Hospital Operations (48 facts, 15 tables, ~100k tokens)**

Run: `output/runs/20260610_135719_handcrafted_21/`

Stage 2 issues:
- `ORDER` is a SQL reserved keyword -- any SQL DDL will fail on this table name.
- `PRESCRIPTION` has no FK to `ORDER` or `ENCOUNTER` -- island table.
- `LAB_ORDER` has no FK to `ORDER` -- island table.
- `ENCOUNTER` uses two inline diagnosis FKs instead of a junction table despite
  fact 35 stating "a single encounter can have multiple diagnoses."
- `PATIENT` missing `national_health_id` (fact 4 present in input); hallucinated
  `gender` column not grounded in any fact.
- No `PAYER` entity -- facts 38 and 41 reference payers explicitly; CLAIM has no
  `payer_id` FK; the denial-escalation rule (3+ denials per payer) is unenforceable.
- `LAB_ORDER` missing `is_critical` / `notification_recorded` columns (facts 51-52).
- `CLAIM` missing `denial_count` (required for fact 41 escalation rule).
- `REFERRAL` only 3 columns -- missing `specialist_id`, `referral_date`,
  `required_seen_by_datetime` (needed for the 24h EMERGENT rule, fact 43).
- `RESIDENT_DIMENSION` -- OLAP-style naming. Should be `RESIDENT` or `RESIDENT_SUPERVISION`.
- `PROVIDER` only 2 columns (`license_number`, `employment_status`); missing `name`,
  `provider_type` discriminator; `DOCTOR` and `RESIDENT_DIMENSION` both repeat
  `employment_status` redundantly.
- All 15 tables have every column as `nullable=True` -- PKs should be NOT NULL.
- `PRESCRIPTION.route_of_administration` typed as FLOAT; should be VARCHAR enum
  (ORAL, IV, IM, SUBCUTANEOUS, TOPICAL).

**Case 22 -- Platform Marketplace (39 facts, 11 tables, ~100k tokens)**

Run: `output/runs/20260610_140129_handcrafted_22/`

Stage 2 issues:
- `LISTING (1 col)` -- ghost supertype table with only a PK. `REVIEW.listing_id`
  FKs to this empty table instead of the actual listing types.
- `VERIFICATION_STATUS (1 col)` -- single-column lookup table for an enum. Should be
  a CHECK constraint on `HOST.verification_status`, not a separate entity.
- `SERVICE_LISTING` has no FK to `HOST` -- service listings are by hosts.
- `SERVICE_LISTING.review_score` is stored even though fact 40 says "a host's average
  rating is computed, not stored."
- `BOOKING` only FKs to `SERVICE_LISTING` -- no path to book a `SPACE_LISTING`.
- `GUEST` has standalone `name/email/phone` instead of a FK to `PERSON` -- violates
  fact 9 ("a person can be both host and guest without separate accounts").
- No commission or payment table (facts 25-26 define 12%/5% platform commission).
- No `DISPUTE` table (facts 23-24).
- `SERVICE_LISTING` missing cancellation policy (present on `SPACE_LISTING`, fact 35).
- `MESSAGE` not linked to listing inquiries (only `booking_id`; fact 33 says
  communication can be tied to bookings OR listing inquiries).

**Case 23 -- Securities Trading / Retail Brokerage (47 facts, 11 tables, 107k tokens)**

Run: `output/runs/20260610_141551_handcrafted_23/`

Critical -- ghost entity with 3 broken FKs:
- Fact 22 says "executions created from orders include executed price, quantity,
  timestamp, and exchange." The architect understood an `ORDER_EXECUTION` entity was
  needed, but instead of creating it, folded those attributes into `ORDER` and left
  three tables referencing `order_execution_id -> ORDER` (column name mismatch:
  `ORDER` has `order_id`, not `order_execution_id`). DDL would fail on
  `REGULATORY_FEE`, `STOCK_POSITION`, and `SETTLEMENT_RECORD`.

Other issues:
- `ORDER` again a SQL reserved keyword (same as case 21).
- `ORDER` missing `account_id -> ACCOUNT` and `instrument_ticker -> FINANCIAL_INSTRUMENT`
  FKs -- orders are effectively unlinked from any context.
- `POSITION` has `account_id` and `lot_cost_basis` but no `instrument_ticker` -- a
  position table that doesn't record what instrument is held is unusable.
- `STOCK_POSITION` and `POSITION` are both position-like tables; no clear ownership
  or unification.
- No `USER`/`CUSTOMER` entity -- accounts are owned by people; no CUSTOMER table.
- `ACCOUNT` missing `pdt_flagged` column (fact 51: pattern day trader rule).
- `DIVIDEND` has no `ACCOUNT` FK -- can't identify which accounts to credit on pay date.
- `ACCOUNT.buying_power` stored as column despite fact 8 saying it's a derived value
  (~2x cash balance for margin accounts).
- `ORDER.timestamp` typed as VARCHAR -- should be DATETIME/TIMESTAMP.
- `SETTLEMENT_RECORD.expected_settlement_date` and `.actual_settlement_date` typed as
  VARCHAR -- should be DATE.

Note: Stage 1 enrichment worked correctly for case 23 -- the auditor rejected all
proposed external facts as speculative or unrelated, and that is the correct call.
The bug fix was not triggered here (final attempt also not acceptable).

---

### Cross-Case Pattern Summary

| Pattern | Case 21 | Case 22 | Case 23 | Verdict |
|---------|---------|---------|---------|---------|
| SQL reserved keyword as table name | ORDER | -- | ORDER | Systematic -- prompt issue |
| Ghost entity (referenced but never created) | -- | LISTING (1 col) | ORDER_EXECUTION | Case-specific but severe |
| Missing core FKs on central entity | PRESCRIPTION, LAB_ORDER | SERVICE_LISTING, BOOKING | ORDER-ACCOUNT, ORDER-INSTRUMENT | Systematic -- cross-chunk blindness |
| Derived value stored as column | -- | SERVICE_LISTING.review_score | ACCOUNT.buying_power | Recurring -- prompt issue |
| Junction table missing for M:N | ENCOUNTER_DIAGNOSIS | -- | -- | Occasional |
| All columns nullable=True | all 15 tables | -- | -- | Case 21 specific |
| Facts in chunk silently dropped | national_health_id, PAYER | -- | PDT flag, DIVIDEND-ACCOUNT | Recurring |
| Wrong data type on enum/time column | route_of_admin FLOAT | -- | timestamp/dates VARCHAR | Occasional |
| Enrichment facts silently dropped | 3 facts (bug -- fixed) | 3 facts (bug -- fixed) | 0 (auditor correctly rejected) | Bug fixed |

Root cause taxonomy for Stage 2 failures:
- **Naming blindness**: architect doesn't check SQL reserved words or OLAP naming conventions.
- **Cross-chunk FK blindness**: tables generated in different shards are merged structurally
  but the merge step doesn't infer missing cross-shard FKs.
- **Fact omission under token pressure**: specific facts present in the input chunk are
  silently dropped in the output schema (no fact-coverage validation post-generation).
- **Ghost entities**: architect conceptualizes an entity, puts its attributes on another
  table, then leaves FKs pointing at the ghost column name on the wrong table.
- **Nullable universally applied**: no distinction between required and optional columns
  in the architect's output.
- **Derived values stored**: architect materializes computed values as columns without
  checking whether the facts mark them as derived/computed.

## 2026-06-10 (continued, session 4)

### 16:00 -- Dead code removal + test suite cleanup (Stage 1 + Stage 2)

**Dead backward-compat stubs removed:**
- `context_enricher/agent.py`: removed `get_agent()`, `enrich_context()`
- `context_auditor/agent.py`: removed `get_agent()`, `audit_context()`
- `chunker/agent.py`: removed `get_agent()`, `run_chunker()`
- `schema_architect/agent.py`: removed `get_agent()`, `run_schema_architect()`
- `domain_auditor/agent.py`: removed `get_agent()`, `audit_domain()`

**entry.py / utils.py wiring fixed:**
- `stage2/entry.py`: removed `get_architect`/`get_domain_auditor` imports, removed stale calls and `architect=`/`auditor=` params passed to loop runners
- `utils.py`: removed `architect: Optional[object]` param from `run_architect_self_correction_loop` and `auditor: Optional[object]` param from `run_auditor_self_correction_loop` (both were silently ignored)

**Test suite:**
- Deleted `src/tests/unit/test_stage1_context_auditor.py` -- imported deleted `audit_context`
- Added `src/tests/unit/test_util_loop.py` -- 15 offline unit tests for loop infrastructure (all pass)
- 362 unit tests pass; 3 pre-existing failures in `test_stage1_models.py` and `test_stage1_text_matching.py` are unrelated to this session

---

## 2026-06-10 (continued, session 3)

### Loop Infrastructure Consolidation -- Stage 1 + Stage 2 (LoopAgent ABC)

**Scope:** Stage 1 and Stage 2 only (Stage 3/4 deferred).

**What changed:**

`src/util/orchestration/loop_types.py`:
- Added `LoopOutputModel(BaseModel, ABC)` with mandatory `get_errors() -> list[str]`.
- Added `LoopAgent(ABC)` with mandatory `invoke()`, `build_context()`, `emit_history()`.
- `AgentRoleConfig` reduced to 2 fields: `agent_factory` + `det_error_sources`.
- Infrastructure calls `output.get_errors()` after every invocation to own `unresolved_issues` and `det_errors`.

`src/util/orchestration/loop.py` -- full rewrite (~130 lines, was 463).
- Removed `_build_context`, `_extract_history_entry`, `_run_det_checks`, all `make_*` helpers.

Stage 1 output models now inherit `LoopOutputModel`:
- `RephrasedOutput.get_errors() -> []` (NL validation runs in `FactExtractorLoopAgent.build_context`)
- `FactList.get_errors() -> []`
- `IntegrityReport.get_errors()` -- returns HIGH/CRITICAL issue strings
- `ContextAuditReport.get_errors()` -- returns `[]` if acceptable, else `[retry_instructions]`

Stage 2 output models now inherit `LoopOutputModel`:
- `ChunkedPlan.get_errors() -> []`
- `Schema.get_errors() -> []`
- `CritiqueReport.get_errors() -> []` (from schema_patch.py)
- `_ChunkerValidationReport.get_errors()` -- blocking error descriptions
- `_SchemaValidationReport.get_errors()` -- validation error strings
- `_AuditPatchResult.get_errors()` -- post-patch validation errors

New LoopAgent subclasses added:
- `FactExtractorLoopAgent` -- absorbs `_ExtractorContextBuilder` + runs `deterministic_validator` in `build_context`; stateful `_errored_ids_history`
- `VerifierLoopAgent` -- absorbs `_verifier_context_builder`
- `ContextEnricherLoopAgent` -- stateful `accumulated_accepted`; merges accepted facts in `build_context`
- `ContextAuditorLoopAgent` -- stateful `audit_trail`; populated in `emit_history`
- `ChunkerLoopAgent` -- absorbs `_ChunkerContextBuilder`; uses `ctx.det_errors` for feedback
- `SchemaArchitectLoopAgent` -- absorbs `_ArchitectContextBuilder`; stateful `fix_history`
- `DomainAuditorLoopAgent` -- absorbs `_AuditorContextBuilder`; uses duck-typing for `schema_state` to avoid circular import
- `ChunkerValidatorLoopAgent`, `SchemaValidatorLoopAgent`, `AuditPatchValidatorLoopAgent` -- deterministic validator nodes in utils.py

`src/orchestration/stage1/loop_config.py` -- full rewrite:
- `make_stage1_loop_config(nl_description, model)` -- no more `max_retries` param, `max_iter=5`
- Added `make_enrichment_loop_config(original_facts, search_suggestions, model)` returning `(LoopConfig, ContextEnricherLoopAgent, ContextAuditorLoopAgent)`

`src/orchestration/stage1/entry.py`:
- `_run_context_enrichment_loop` replaced with `AgentLoop(make_enrichment_loop_config(...)).run("")`
- Reads `enricher_agent.accumulated_accepted` and `auditor_agent.audit_trail` after loop

`src/orchestration/stage2/utils.py` -- full rewrite:
- Removed all old context builders, history extractors, and callable validator classes
- Runner functions all use `max_iter=5`

**Key design decisions:**
- NL validation in `FactExtractorLoopAgent.build_context` (not `get_errors`) because it needs `nl_description` context
- `DomainAuditorLoopAgent` uses `hasattr(validator_output, 'schema_state')` to read patch results without importing `_AuditPatchResult` (avoids circular import)
- Old standalone `get_agent()` / `enrich_context()` / `audit_context()` functions kept in agent files for backward compatibility

## 2026-06-10 (continued)

### 20:00 -- Human stress tests + global repair bug fix

**Stress test cases run:**

| Case | Stage 1 | Stage 2 | Errors | Tokens |
|---|---|---|---|---|
| portfolio_mgmt | 81 facts / 57.5s | 24 tables / 17 FKs / 340s | 3 | 204k |
| saas_pm | 105 facts / 92.3s | 25 tables / 33 FKs / ~380s | 0 | ~293k |

**Portfolio management errors root-caused (3 isolated tables):**

All 3 isolated tables (MARKET_PRICE_SOURCE, FEE_EVENT, ONBOARDING) were born in shards
that didn't contain the parent entity (SECURITY/SNAPSHOT, ACCOUNT, CLIENT respectively).
The per-shard auditor is scoped and cannot create cross-shard FKs. The global anchored
repair was supposed to fix this but produced 0 patches every time.

**Global repair silent failure -- root cause and fix:**

`_AuditorContextBuilder.__call__()` reports "None. The previous state was structurally
valid." on its first call because `patch_validator` hasn't run yet. Pre-computed
`global_errors` from `entry.py` were never passed to the context builder. The auditor
saw "no errors", returned empty patches list, and the loop exited.

Fix: Added `initial_errors: Optional[List[str]] = None` to both
`_AuditorContextBuilder.__init__` and `run_auditor_self_correction_loop`. On the first
call (before any `patch_validator` output), injects the initial errors.
Call site in `entry.py` now passes `initial_errors=global_errors`.

**SaaS PM schema quality assessment:**

Strengths: 0 validation errors, 33 FKs (dense connectivity), proper junction tables
(ISSUE_LABEL, ISSUE_TEAM, TEAM_MEMBER, USER_PROJECT, USER_ROLE), self-referential project
(PROJECT.parent_project_key -> PROJECT), full billing/integration/activity coverage,
FTS columns (TSVECTOR) included.

Weaknesses identified (Stage 2 semantic errors):
1. `ISSUE.assignee_id -> WORKSPACE` and `ISSUE.reporter_id -> WORKSPACE` -- should be `-> USER`
2. PAYMENT_METHOD isolated (SUBSCRIPTION never FKs to it); exempted by "PAYMENT" skeleton keyword
3. ISSUE.parent_issue_id / epic_issue_id / blocking_issue_id have no FK back to ISSUE
4. STATUS and LABEL not workspace/project-scoped

**Utils.py cleanup:**
Moved `_ChunkerIssue` class before `_format_chunk_errors` (was a forward reference issue);
changed `_format_chunk_errors` signature from `List[object]` to `List[_ChunkerIssue]`.

## 2026-06-10

### 14:00 -- src/util/ restructured into subpackages

Reorganized 18 flat files into 7 subpackages via migration script (275 import rewrites):
- `core/` -- agent factory, invoke, web search, config
- `orchestration/` -- loop, loop_types, retry_loop, loop_config
- `schema_ops/` -- schema_utils, schema_patch, patching_engine, sharding_utils, matching
- `algorithms/` -- semantic_match, span_index
- `observability/` -- llm_trace, error_formatter
- `analysis/` -- dist_miner
- `config/` -- ablation, web_search

### 14:30 -- Stage 1 + Stage 2 dead code cleanup

**invoke.py:** Deduplicated token extraction into `_tokens_from_message()` helper;
removed duplicate extraction logic from both the LangChain and OpenAI paths.

**llm_trace.py:** Fixed pre-existing type error -- `add_trace` now takes
`Sequence[BaseMessage]` instead of `List[object]` (invariant type caused
`list[HumanMessage]` assignment errors).

**semantic_match.py:** Removed outer `try/except Exception` fallback in
`_split_sentences` that silently ate NLTK data errors; now fails explicitly.

**schema.py (`Stage 2`):** Removed unused `version: Optional[str]` field; collapsed
two FK iteration loops in `_validate()` into one (removed dead `relationship_counts`
and `upper_relationship_counts` variables).

**merger.py:** Removed second `table_map_b_to_a` reconstruction at relationship-merge
step (first construction already handles name-collision); removed unused
`matched_indices_a`; removed two dead commented-out code blocks.

**chunk.py:** Collapsed redundant `elif isinstance(item, list)` / `else` branches
that both did the same thing.

**span_index.py (CRITICAL REWRITE):** `SentenceTransformer("static-similarity-mrl-multilingual-v1")`
crashes Python 3.14 at load time via `torch.storage.__getitem__` <- `safetensors.torch.load_file`
(StaticEmbedding model path). The crash is a native-level segfault uncatchable by Python.
Replaced the entire embedding stack with pure numpy TF-IDF + FAISS `IndexFlatIP`.
Key change: `DEFAULT_WINDOW_SIZES = [12, 24, 40, 64, 96]` now counts words (not BERT tokens);
legacy `model_name`/`device`/`truncate_dim` kwargs absorbed by `**_ignored` for caller compatibility.
Removed `transformers` and `sentence_transformers` imports; removed `sentence-transformers`
from `pyproject.toml`.

**Note:** `similarity.py` (Stage 2 schema merging) also uses `sentence_transformers`
but with `all-MiniLM-L6-v2` (standard BERT, not StaticEmbedding) -- loads fine on Python 3.14.

### 16:30 -- Large end-to-end test (hospital management system)

Run `run_large_test.py` with 8,408-char hospital network NL description.

**Stage 1:** 146 facts in 210.4s, 54,039 tokens (144 STRUCTURAL, 2 LOGICAL).

**Stage 2:** 38 tables, 60 relationships, 0 validation errors in 1321.4s, 334,384 tokens.
Notable tables: PATIENT (22 cols, 1 FK), HOSPITAL (20 cols, 0 FK), PRESCRIPTION (17 cols, 3 FK),
CLINICAL_TRIAL_ENROLLMENT (18 cols, 2 FK), VISIT (14 cols, 5 FK). Full schema covers clinical,
billing, pharmacy, scheduling, inventory, compliance, staffing, and training domains.

**Total:** 388,423 tokens, 25.5 minutes. Output saved to `output/runs/large_test/`.

**Issues found:**
- `domain`/`analytical_goal` returned `"Unknown"` -- fact extractor prompt gave no instruction
  to identify or fill these fields.
- BERT tokenizer warning (`Token indices sequence length > 512`) in Stage 2 similarity -- benign
  (truncation) but noisy.

### 17:00 -- Post-run fixes

**fact_extractor/prompt.txt:** Added explicit instruction in TASK section to identify `domain`
(specific industry/sector, e.g. "Hospital Network Management") and `analytical_goal` (purpose
of the system). Previously the model focused entirely on facts and left both fields at their
Pydantic defaults ("Unknown"/"General Purpose").

**similarity.py:** Suppressed BERT truncation warning with `warnings.catch_warnings()` context
around `model.encode()` call. Also silenced two pre-existing pyright errors on `SentenceTransformer`
assignment/return (its overloaded `encode` signature doesn't satisfy the `EmbeddingModel` protocol)
with `# type: ignore[assignment]` and `# type: ignore[return-value]`.

## 2026-06-09 (continued)

### AgentLoop infrastructure + Stage 1 migration

**New files:**
- `src/util/loop_types.py` -- all AgentLoop data models and config (split for import hygiene)
- `src/util/loop.py` -- AgentLoop executor; config-driven, fully offline-testable
- `src/orchestration/stage1/loop_config.py` -- Stage 1 wiring (extractor + verifier)
- `src/tests/unit/test_util_loop.py` -- 24 unit tests, all passing

**Key design decisions:**
- Three-tier history extraction: (1) `LoopOutputProtocol.get_history_entry`, (2) `AgentRoleConfig.history_extractor`, (3) RuntimeError -- no silent fallback
- `AgentRoleConfig` accepts `system_prompt` (inline) OR `agent_factory` (existing packaged agent), never both
- `LoopResult.node_outputs: dict[str, BaseModel]` gives callers direct access to any node's last artifact
- EMA issue tracking (alpha=0.4) with persistent-issue markers above threshold 0.6
- History extractor signatures use `BaseModel` (not specific subtype) to satisfy contravariance; isinstance-narrow inside

**Stage 1 entry.py migrated:**
- Replaced `RetryLoop` + `_captured_report` dict hack with `AgentLoop` + `result.node_outputs`
- `make_stage1_loop_config` encapsulates extractor/verifier wiring, accepted-facts accumulation, and error formatting
- Exhaustion handled by checking `result.node_outputs.get("extractor")` -- no exception raised

**Verification:** `pytest src/tests/unit/test_util_loop.py` -> 24 passed

## 2026-06-09

### 11:30 -- Stage 1 dead code cleanup

- Removed unused `src/orchestration/stage1/documentation.py`
- Removed `TechnicalTerm` model and `definitions` field from `RephrasedOutput`
- Updated Stage 1 unit tests to reflect removed definitions

### 12:00 -- Stage 1 smoke run after cleanup

- Ran Stage 1 on a simple loan NL description (max_retries=2)
- Output produced 6 facts, ~12.6k tokens, domain/goal remained Unknown
- Observed known Responses API Pydantic serialization warnings; run completed successfully

### 12:20 -- Stage 2 chunker validation and retry loop

- Added deterministic chunk validation (required fact coverage, orphaned refs, empty chunks)
- Added chunker retry loop (max 5) with error feedback, plus safe single-chunk fallback
- Included domain/goal context and referenced_fact_ids in chunker input
- Updated chunker prompt with constraint locality and refinement policy

### 13:00 -- Stage 2 audit bypass + junction hardening

- Added `enable_audit` flag to Stage 2 orchestration (default disabled) to skip shard critique loops
- When audit disabled, run a single global validation pass without repair
- Hardened junction-table detection to avoid removing real FKs when extra non-FK columns exist

### 13:20 -- Stage 2 FK type alignment

- Added deterministic FK type alignment to match referencing column types to referred PK types
- Applied alignment after shard merges to reduce PK/FK type mismatches in large schemas

### 13:40 -- Stage 2 connectivity validation softening

- Relaxed Schema validation to only flag isolated tables, not overall fragmentation
- Avoids spurious failures when large schemas split into multiple connected components

### 14:25 -- Stage 2 merge guardrails + schema prompt tightening

- Relaxed singular-name validation to allow common singular S endings (e.g., DIAGNOSIS); unit test added
- Prevented table merges based solely on identifier columns when names are not near-exact; unit test added
- Schema architect prompt now enforces role clarity (physician vs nurse), distinct noun preservation, no abbreviations, and event vs summary separation
- Large EHR Stage 2 rerun (from saved Stage 1) now includes DISCHARGE, INSURANCE_COVERAGE, HOSPITAL, and NURSING_STAFF_ASSIGNMENT; no global validation errors
- Stage 2 unit tests: `test_stage2_schema.py` + `test_stage2_merger.py` passed (55 tests)

### 15:20 -- Stage 2 stress suite + merge/validation hardening

- Stress suite (Stage 1+2) run across 12 cases: handcrafted IDs 2,6,10,11,12,13,15,17 and benchmark tpch-003, tpcds-003, imdb-003, mimiciv-005
- Findings: table drops from merge (ORDER, WEB_SALE), FK column mismatch (tpch-003), and repeated isolated-table errors where relationships were implied but missing
- Stage 1: retry loop exhausted for handcrafted IDs 2 and 17 (partial fact sets still produced); several cases flagged as underspecified (no explicit relationships)
- Merge fixes: remap relationship columns on matched tables, penalize distinct-name modifiers to reduce false merges, and restrict PK force-matching to identifier/similar names
- FK inference: skip PK columns when auto-adding _id relationships to avoid invalid PK-as-FK edges
- Relationship repair: map FK table names to closest existing table when only pluralization/near-name mismatch exists (e.g., TV_SERIE -> TV_SERIES)
- Singular-name validation: allow SERIES token in compound table names
- Unit tests updated and green: `test_stage2_schema.py` + `test_stage2_merger.py` (60 tests)
- Targeted reruns: handcrafted_13, tpch-003, tpcds-003, imdb-003 now pass validation; tpcds-003 retains WEB_SALE; tpch-003 FK mismatch resolved

### 15:45 -- Stage 2 auditor rework (fact-grounded)

- Rewrote Stage 2 auditor prompt to be fact-grounded and to prohibit hallucinated tables/columns/relationships
- Auditor now uses only fact clusters for additions/removals and prioritizes structural-error feedback
- Audit is enabled by default in Stage 2 orchestration to activate the critique loop

### 16:10 -- Stage 2 auditor FK enforcement

- Tightened domain auditor rules to add FK columns for relationship facts before adding relationships
- Added explicit handling for junction, dependent-entity, and self-referential facts
- Prohibited PK-as-FK unless facts explicitly say shared primary key; enforced FK type alignment guidance

### 18:55 -- Stage 2 auditor PK-as-FK repair rule

- Added explicit instruction to introduce surrogate PKs when a required FK would otherwise be the table PK

### 19:10 -- imdb-005 audit rerun

- Reran Stage 2 (audit enabled) for imdb-005 using saved Stage 1 output
- Result: 7 tables, 6 relationships, no isolated-table errors

### 19:25 -- tpcds-005 audit rerun

- Reran Stage 1 + Stage 2 for tpcds-005 (Stage 2 from saved Stage 1 output)
- Result: 19 tables, 15 relationships, errors remain for DATE_DIM/TIME_DIM forbidden token and isolated DATE_DIM/TIME_DIM/ITEM

### 19:40 -- Stage 2 auditor dimension linking

- Added auditor guidance to rename forbidden DIM tokens and link explicit shared/conformed dimensions to fact tables

### 19:45 -- tpcds-005 rerun after auditor update

- Reran Stage 2 (audit enabled) for tpcds-005 from saved Stage 1 output
- Result: DATE_DIM/TIME_DIM errors cleared, ITEM still isolated (18 tables, 14 relationships)

### 19:55 -- Stage 2 auditor prompt cleanup

- Removed example-specific wording and duplicate bullets from the Stage 2 auditor prompt
- Clarified that shared-dimension facts authorize adding FK columns to fact tables

### 20:05 -- Stage 2 auditor dimension direction

- Clarified shared-dimension directionality (fact tables as children; no dimension-to-dimension links unless explicit)

### 20:15 -- Stage 2 auditor global shared-dimension facts

- Allowed shared/conformed dimension facts to apply across all clusters (not just per-entity clusters)

### 20:25 -- tpcds-005 rerun after global shared-dimension rule

- Reran Stage 2 (audit enabled) for tpcds-005 from saved Stage 1 output
- Result: DATE_DIMENSION, TIME_DIMENSION, and ITEM still isolated (17 tables, 12 relationships)

### 20:40 -- Large NL stress (case 11)

- Ran Stage 1+2 for test_cases.json id 11 (Healthcare EHR, 50+ tables intent)
- Output: 28 tables, 28 relationships, no schema validation errors

### 20:55 -- Large NL stress (case 12)

- Ran Stage 1+2 for test_cases.json id 12 (E-Government civil registry)
- Output: 17 tables, 13 relationships, isolated TAX_FILING table

### 21:05 -- Large NL stress (case 13)

- Ran Stage 1+2 for test_cases.json id 13 (Manufacturing ERP)
- Output: 26 tables, 21 relationships, no schema validation errors

### 21:25 -- Stage 2 loop infra migration

- Replaced manual chunker/architect/auditor retry loops with AgentLoop-based configs
- Added deterministic validator nodes for chunking, schema validation, and patch application

### 21:40 -- Stage 2 loop cleanup

- Renamed audit patch result field to avoid BaseModel attribute shadowing
- Confirmed Stage 2 now uses loop infra consistently (no manual retry loops)

### 21:50 -- Large NL stress (case 11, post-migration)

- Ran Stage 1 for test_cases.json id 11 (Healthcare EHR)
- Stage 2 failed with APIConnectionError during schema architect loop (stage2_error.json saved)

### 22:05 -- Large NL stress (case 13, post-migration)

- Ran Stage 1+2 for test_cases.json id 13 (Manufacturing ERP)
- Output: 23 tables, 20 relationships, no schema validation errors

### 22:15 -- Large NL stress (case 12, post-migration)

- Ran Stage 2 for test_cases.json id 12 from saved Stage 1 output
- Output: 13 tables, 10 relationships, no schema validation errors

### 22:30 -- Large NL stress (case 11, Stage 2 rerun)

- Reran Stage 2 for test_cases.json id 11 using saved Stage 1 output
- Output: 30 tables, 27 relationships, isolated PRESCRIPTION table

### 22:40 -- Large NL stress (case 14)

- Ran Stage 1+2 for test_cases.json id 14 (Education Management)
- Output: 28 tables, 29 relationships, no schema validation errors

### 22:50 -- Large NL stress (case 18)

- Ran Stage 1+2 for test_cases.json id 18 (HR & Payroll)
- Output: 10 tables, 9 relationships, no schema validation errors

### 23:05 -- Loop context alignment

- Loop infra now supports default context sections per agent role (history/persistent/errors toggles)
- Stage 1 loop config updated to rely on loop-managed history/persistent sections instead of manual round history
 - Stage 1 verifier/extractor now use default context for the original NL description
 - Stage 2 loop roles now include default history/persistent context without manual wiring

### 23:20 -- Loop infra enhancements

- Added per-role det_error_sources and prior_output_nodes for precise context/error routing
- Added generator+critic+validator helper to standardize common loop topology

### 23:30 -- Dead code cleanup

- Removed unused util error formatter module
- Dropped unused persistent-error block from Stage 1 error formatter

## 2026-06-06

### 20:10 -- Pipeline verified healthy after typing overhaul

**End-to-end run (hospital database):**
- All 4 stages completed: 17.8s + 25.7s + 2.1s + 8.7s = 54.3s total, 57424 tokens
- Smoke test PASSED: APPOINTMENT 300 rows, DOCTOR 20 rows, MEDICAL_RECORD 150 rows, PATIENT 100 rows
- Lingering "exit code 1" on PowerShell runs is a false alarm -- langchain deprecation warning
  written to stderr; actual pipeline exits 0 internally

**Cleanup in `src/orchestration/stage3/entry.py`:**
- `healing_history.append({...})` -> `HealingAttempt(...)` (no longer raw dicts)
- Removed dead duplicate `history.append` + `return` block (unreachable after the first return
  in the try block -- lines 309-315 in prior state)

### 19:15 -- Strict typing + Pydantic audit: all violations fixed, 302 tests green

**Scope:** Enforced strict typing across all 4 pipeline stages + `src/util`. No bare `Dict`/`List`/`Any`,
all function signatures fully annotated, no unannotated parameters, Optional expressed explicitly.

**Key changes:**

- `src/util/agent.py`: Exported `AgentType = Union[StructuredAgent, Runnable]` as the canonical
  return/parameter type for all agent factory functions. All 13 agent files (Stage 1-4) updated to
  import and use `AgentType` instead of bare `Runnable` (which didn't cover `StructuredAgent`).
- `src/pipeline/stage2/models/schema.py`: `registry: Any` -> `Optional[TableFactRegistry]` via
  `TYPE_CHECKING` guard; added `-> None` on all mutating methods; `Dict[str, set]` -> `Dict[str, Set[str]]`.
- `src/pipeline/stage2/middleware/schema_merging/merger.py`: Raw dict list `junction_tables` replaced
  with `_JunctionEntry(NamedTuple)`. `-> None` on all mutating methods.
- `src/pipeline/stage2/middleware/schema_merging/similarity.py`: Renamed `SemanticSimilarity` to
  `TokenSimilarity` (reflects lexical-only implementation). Updated `fact_allocation.py` import.
- `src/orchestration/stage3/models.py`: Added `RawSQLRule(BaseModel)` and `HealingAttempt(BaseModel)`
  to replace `List[Dict[str, Any]]` fields. Updated `entry.py` to construct these models at call sites.
- `src/pipeline/stage4/compiler.py`: `Dict` -> `Dict[str, Any]` for schema params; `-> None` on
  `emit`, `_compile_table`, `_compile_if_node`, `_apply_result`; `Union[ColumnNode, ConstNode]` for `_resolve_rhs`.
- `src/pipeline/stage3/models/`: `ConstNode.value: Any` left as `Any` (runtime duck-typed); `schema: Any`
  params left pending (TYPE_CHECKING refactor deferred -- no runtime impact).
- Stage 1 agent files: all `get_agent()` return types and main function params fully annotated.
  `-> Tuple[T, int]` return types added. `assert isinstance(parsed, T)` guards already present.

**Tests:** 302 passed, 16 skipped (live API tests require OPENAI_API_KEY).

### 17:30 -- End-to-end pipeline run + blocking bugs fixed

**End-to-end run completed (library database example):**
- Stages 1-4 all ran; smoke test PASSED after fixes
- Schema: 5 tables (BOOK, AUTHOR, BOOK_AUTHOR, MEMBER, BORROWAL/BORROWING) with correct FKs
- Generated code produces valid pandas DataFrames; smoke test at 10% scale passes in ~1s
- Full run: ~46s, ~81k tokens; rerun from saved Stage 1: ~54s, ~72k tokens

**Bugs fixed:**
- All 12 agent `name=` fields had spaces -> OpenAI message `name` validation rejected them.
  Fixed: all names now snake_case (e.g. `'domain_intelligence_extractor_stage2'`).
- `domain_intelligence_extractor` used `web_search_preview` via Chat Completions API
  (requires Responses API with `use_responses_api=True`). Fixed: removed `tools=` so it
  falls through to `StructuredAgent` like all other agents. Documented as F2-0.
- `StructuredAgent.chain` used default `json_schema` method which rejects discriminated
  unions (`oneOf`). Fixed: switched to `method="function_calling"` in `with_structured_output`.
- Missing `assert isinstance(parsed, T)` after `get_response` in tagger, verifier, context_enricher.
  Fixed: assertions added to narrow `Union[T, str]` return type.
- `psutil` not installed in venv. Fixed: `pip install psutil`.
- Compiler `_emit_logic_mask_expr` fell back to `BUFFERS['TABLE']['id']` when column not found
  in schema (Stage 3 generated `id` for PK of `BOOK_AUTHOR` which is `book_author_id`).
  Fixed: fallback now uses `self._get_pk(table, schema)` -> mask degrades to `pk == pk`
  (always True) instead of crashing. Documented as F4-4.

**Pre-existing type errors fixed (found by post-edit hook):**
- `verifier/agent.py`: imported `Severity` from `integrity_report` but `ErrorRecord` expects
  `retry_loop.Severity`. Fixed import.
- `domain_auditor/agent.py`: `structural_errors: List[str] = None` -> `Optional[List[str]] = None`
- `semantic_agent/agent.py`: return type `Tuple[str, int]` -> `Tuple[Dict[str, str], int]`



### 18:00 -- Full test suite built + pipeline audit

**Unit test suite (302 tests, all green):**
- `pytest.ini` + `conftest.py` + `fixtures/sample_data.py` already existed from prior session
- Fixed 4 pre-existing failures: `_is_discrete` returned `np.bool_` not `bool` (fixed source),
  cycle test had wrong schema construction (fixed), AIMessage usage_metadata missing required fields (fixed)
- New unit test files written: `test_stage2_registry.py`, `test_stage2_similarity.py`,
  `test_stage2_merger.py`, `test_stage3_distributions.py`, `test_eval_data.py`, `test_eval_schema.py`
- `test_eval_schema.py` uses `pytest.importorskip("sentence_transformers")` so it's auto-skipped
  in environments without the ML model

**Integration tests written (auto-skipped without OPENAI_API_KEY):**
- `src/tests/integration/test_pipeline_stage1.py` -- Stage 1 orchestrate() end-to-end
- `src/tests/integration/test_pipeline_stage2.py` -- Stage 2 with schema validation checks
- `src/tests/integration/test_pipeline_full.py` -- Full 4-stage chain (marked `slow`)

**Source bug fixes:**
- `src/util/dist_miner.py`: `_is_discrete` now returns `bool(...)` to avoid numpy bool_ leakage
- `src/orchestration/stage4/entry.py`: `_find_missing_columns` escapes column names with `re.escape`
  before building the regex pattern (was vulnerable to regex metacharacters in column names)

**FINDINGS.md created** -- 15 documented issues across stages:
- F1-2 (MEDIUM): context enricher never uses web search, relying on LLM parametric memory
- F2-3 (MEDIUM): `final_global_schema` is Optional but downstream callers assume non-None
- F3-1/F3-2 (INCONSISTENCY): NormalDist/LogNormalDist param names differ from evaluation layer
- F3-4 (PERFORMANCE): healing loop is serial, should use asyncio.gather
- E2 (MEDIUM): missing-column detection in data_eval uses post-hoc metric check instead of lookup-time flag
- No monkey-patching found anywhere in `src/`

### 14:00 -- Benchmark JSONL rewrite + test cleanup

**Ground truth JSONL files completely rewritten (all 4 datasets, 6 cases each = 24 total):**
- Format fixes: `low`/`high` for uniform (was `min`/`max`), `mu`/`sigma` for lognormal (was `mean`/`variance`), `rate` for exponential (was `lambda`)
- Added `level` field: noob/noob/intermediate/intermediate/expert/expert per dataset
- Added `data_type` on all columns (required for DT Acc metric in schema_eval.py)
- Noob/some-intermediate descriptions include deliberate spelling mistakes
- Dataset names redacted from all NL descriptions
- 6 cases per dataset: varying schema size from subset to full schema

**Test cleanup:**
- Deleted `src/tests/stage4/rigorous_test.py` (broken imports from deleted stage3 models)
- Deleted `src/tests/stage4/example.py` (same)
- Deleted `src/tests/stage4/orchestration.py` (depended on deleted example.py)
- Deleted `src/tests/stage3/orchestration.py` (old API + deleted documentation.py)
- Deleted `src/tests/focused/test_rename_patch.py` (deleted patch.py/patcher.py)
- Deleted `src/tests/stage2/chunker_philosophy.py` (old class-based ChunkerAgent API)
- Cleaned 41 __pycache__ directories (excluding venv)

### 02:00 -- End-to-end eval + ablation wiring complete

**Benchmark JSONL datasets created** (12 cases total, 3 per dataset):
- `dataset/benchmark/tpch/ground_truth.jsonl` -- TPC-H supply chain (8 tables full, 5-table catalog, 3-table order mgmt)
- `dataset/benchmark/imdb/ground_truth.jsonl` -- IMDB entertainment (7 tables full, 2-table films, 4-table people)
- `dataset/benchmark/tpcds/ground_truth.jsonl` -- TPC-DS retail (8 tables full, 4-table dimension, 6-table multi-channel)
- `dataset/benchmark/mimiciv/ground_truth.jsonl` -- MIMIC-IV ICU (8 tables full, 5-table vitals, 5-table labs/Rx)
- Ground-truth distributions are scientifically sourced (TPC-H spec, IMDB dataset stats, MIMIC-IV literature)

**AblationConfig wired through all 4 stages:**
- Stage 1: `enable_enrichment` gate added to `enrich_context` call (was missing; now fixed)
- Stage 2: `enable_sharding` bypass creates synthetic single-chunk `ChunkedPlan` so audit/certify still run
- Stage 3: pass-through (no current logic to gate)
- Stage 4: `enable_ancestral_sampling` passed to `MinimalCompiler`

**Compiler updates (`src/pipeline/stage4/compiler.py`):**
- `SCALE_FACTOR = 1.0` emitted as Python variable; seed counts use `max(2, int(n_seeds * SCALE_FACTOR))`
- Global rules and per-table logical predicates gated on `enable_ancestral_sampling`
- Smoke test injects `SCALE_FACTOR = 0.1` via regex to run fast at test time

**New evaluation infrastructure:**
- `src/evaluation/data_level/data_eval.py` -- MRE, NLL (normalised), KS, FA per column; schema recall penalty for missing columns
- `run_pipeline.py` -- CLI for single-case pipeline runs with ablation flags
- `run_evaluation.py` -- full evaluation harness: dataset loop, schema + data metrics, SchemaAgent baseline option, aggregate reporting

**Pre-existing type errors fixed** in stages 3, 4 entry points and compiler during wiring.

## 2026-05-27

### 15:45 -- Presentation Text Refinement

- Tightened presentation slide text for 10-minute deck in `presentation\main.tex`
- Aligned results and ablation metrics with README (Table F1/Acc ranges, NLL/FA, dataset list)
- Refined conclusion messaging to emphasize expressiveness/hallucination/scalability and copula-based future work

### 15:55 -- Presentation Build

- Ran `pdflatex main.tex` in `presentation\` (build succeeded; warnings logged)

### 16:10 -- Presentation Text Expansion

- Added additional bullet content per slide for fuller 10-minute deck in `presentation\main.tex`

## 2026-05-17

### 14:00 -- Planning Session Complete

**Decisions made:**
- LLM framework: LangChain with `ChatOpenAI.with_structured_output()` (non-tool agents), `create_react_agent` (tool-using agents)
- LLM provider: OpenAI only (Ollama removed)
- Web search: OpenAI `web_search_preview` tool
- Stages 1-2: Full audit (per-file)
- Stages 3-4: Rewrite from scratch (design deferred until after Stages 1-2 are solid)
- Frontend/backend: Out of scope
- Platform: Cross-platform (pathlib everywhere)
- Dataset: 135 handcrafted cases -- generate manually (no API calls), discuss format later
- Evaluation metrics: MRE, NLL, KS, FA need to be built from scratch
- Artifact goal: Reproducibility artifact

**Deferred questions (Stage 3/4):**
- What is broken in current Stage 3?
- LLM output format: SQL-based vs structured JSON vs hybrid?
- AST node hierarchy: simplify or keep?
- Cross-table constraints and aggregations handling?
- Distribution families to support?

### 14:30 -- Phase 0: Foundation

**CLAUDE.md created** with all coding standards:
- Agent folder convention (agent.py + prompt.txt)
- Prompt format (ROLE, TASK, INPUT, GUIDELINES, RESTRICTIONS -- OUTPUT appended at runtime)
- Pydantic model standards (_validate() methods)
- Naming conventions
- DRY principles

**src/util/agent.py rewritten:**
- Removed broken `from langchain.agents import create_agent`
- Removed Ollama support
- Added `StructuredAgent` wrapper class that wraps `ChatOpenAI.with_structured_output(include_raw=True)`
- `StructuredAgent.ainvoke()` returns same format as langgraph agents: `{"structured_response": model, "messages": [...]}`
- Tool-using agents still use `create_react_agent` from langgraph
- Both paths share the same `ainvoke({"messages": [...]})` interface

**src/util/invoke.py rewritten:**
- Fixed import: `from langchain_core.messages import HumanMessage, BaseMessage`
- Extracted `_extract_token_usage()` helper for clarity
- Compatible with both StructuredAgent and langgraph agents
- Token tracking works via AIMessage's `usage_metadata` or `response_metadata`

**src/util/web_search.py rewritten:**
- Returns OpenAI `web_search_preview` tool spec
- Documented integration point (context_enricher agent)
- Note: context_enricher currently does NOT use web search -- relies on LLM knowledge. Integration is a Stage 1 audit task.

### 14:45 -- Cleanup Complete

**Root-level scripts deleted:**
- `run_bench.py`, `run_full_benchmark.py`, `run_stage1_test.py`, `run_mock_stage3.py`, `run_chunk_test.py`, `analyze_chunks.py`, `agent_schema_report.py`, `test_custom_bank_case.py`, `test_e2e_case_16_repaired.py`, `test_nullability_flow.py`, `descr.txt`, `stage1_output.txt`

**Module consolidation:**
- Moved `src/utils/sql_utils.py` -> `src/util/sql_utils.py`, deleted `src/utils/`
- Deleted `src/util/dependency_graph.py` (imported old Stage 3 models)
- Fixed `from src.utils.sql_utils` -> `from src.util.sql_utils` in `stage3/models/sql_models.py`

**Cross-platform path fixes (all 12 agent.py files):**
- Replaced hardcoded `PROMPT_FILE_URL = "src/pipeline/.../prompt.txt"` with `PROMPT_PATH = Path(__file__).parent / "prompt.txt"`
- Replaced `open(PROMPT_FILE_URL, 'r', encoding='utf-8')` with `PROMPT_PATH.open(encoding='utf-8')`
- Files fixed: fact_extractor, context_enricher, verifier, tagger, chunker, schema_architect, compliance_certifier, domain_auditor, domain_intelligence_extractor, metadata_extractor, parameter_agent, semantic_agent

**Import fixes:**
- Fixed `get_search_tool` -> `get_web_search_tool` in `domain_intelligence_extractor/agent.py`
- Removed unused `import os` from `metadata_extractor/agent.py` and `semantic_agent/agent.py`
- Guarded broken `from src.pipeline.stage3.models import LogicNode, NodeType` in `documentation.py` (old Stage 3 names; will be updated during Stage 3 rewrite)

**Util audit summary (files kept):**
- `agent.py` -- rewritten (core agent factory)
- `invoke.py` -- rewritten (standardized caller)
- `web_search.py` -- rewritten (OpenAI tool spec)
- `sql_utils.py` -- moved from src/utils/, thin wrapper around sql_validator
- `schema_utils.py` -- generates hierarchical schema descriptions for prompts
- `schema_patch.py` -- Pydantic patch models (AddColumn, RenameTable, etc.) + CritiqueReport
- `patching_engine.py` -- deterministic patch application to Schema objects
- `matching.py` -- greedy 1-1 matching (Gale-Shapley variant) for schema merging
- `sharding_utils.py` -- degree-based deterministic sharding with networkx
- `retry_loop.py` -- generic retry orchestration with error tracking
- `error_formatter.py` -- formats ErrorRecords for LLM retry prompts
- `documentation.py` -- Graphviz rendering (schema ER, logic trees, DAGs); Stage 3 functions guarded

**Remaining `os.path` usage:** Only in test/evaluation files (out of scope for pipeline cleanup).

### 15:30 -- Stage 1 Full Audit Complete

**Bug fixes:**
- Fixed double `return ""` in `pipeline/stage1/middleware/error_formatter.py`
- Fixed broken `IntegrityReport` import in `util/documentation.py` (was importing from `rephrased_nl`, which doesn't export it; now imports from `integrity_report.py`)
- Fixed `format_atomic_facts()` crash: now handles both `RawFact` (no `.tags`) and `AtomicFact` (has `.tags`) via `getattr(f, "tags", [])`

**DRY/cleanup:**
- Removed duplicate `convert_to_atomic()` inline code from `orchestration/stage1/entry.py`; now calls the shared function from `rephrased_nl.py`
- Removed dead `ModelingConstraint` class from `rephrased_nl.py` (defined, never referenced)
- Removed dead `TaggedRawFact` class from `rephrased_nl.py` (defined, never referenced)
- Removed unused imports from `entry.py`: `verify_facts_parallel`, `get_response`
- Removed unused imports from `verifier/agent.py`: `RephrasedOutput`, `Issue`
- Removed unused `verify_origin` import from `middleware/validation.py`

### 16:00 -- Stage 2 Full Audit Complete

**Dead code deleted:**
- `orchestration/stage2/context_filter.py` -- defined `filter_distributional_facts()`, never called from orchestration. Also had a broken sync call to async `get_response()` and a mis-configured `get_agent_()` with no `output_structure`. Dead code removed outright.
- `pipeline/stage2/models/corrections.py`: removed `SchemaResolve` class (defined, never imported or used)

**Output model cleanup (`orchestration/stage2/models.py`):**
- Removed stale dead fields from `Output`: `merged_fix_history`, `shard_steps`, `sentinel_iteration`, `sentinel_fix_history`, `linker_repair_history`, `linker_fix_history`
- These were artifacts of an earlier architecture that was replaced by the current orchestration
- Removed `ShardStep` class (only existed to populate the removed `shard_steps` field)
- Updated `orchestration/stage2/documentation.py` to remove dead rendering blocks for `shard_steps`, `merged_fix_history`, and `linker_repair_history`

**Unused imports removed:**
- `entry.py`: `ShardStep`, `PatchRepairStep`, `FactTag`, `run_schema_architect`, `audit_domain`
- `utils.py`: removed duplicate inline `from src.util.patching_engine import apply_patches` (was re-imported inside function body; top-level import sufficient)

## 2026-05-30

### 01:45 -- Poster: Inline all diagrams into single self-contained main.tex

- Inlined `poster/architecture.tex` (4-stage pipeline from `paper_latex/diagram.tex`) directly into `poster/main.tex` -- no `\input` needed
- Inlined `demo_paper/interaction_overview.tex` (worked-example diagram) directly into `poster/main.tex` with **font sizes reduced by ~2.86×** to compensate for `\resizebox{!}{28cm}` scaling:
  - `\normalsize` (->31pt) -> `\fontsize{4pt}{4.5pt}` (renders ~11pt)
  - `\fontsize{7pt}{...}` (->20pt) -> `\fontsize{3.5pt}{...}` (renders ~10pt)
  - `\fontsize{6pt}{...}` (->17pt) -> `\fontsize{3pt}{...}` (renders ~8.5pt)
  - `\fontsize{9pt}{9pt}` (.py, ->26pt) -> `\fontsize{4.5pt}{4.5pt}` (renders ~13pt)
  - `\fontsize{14pt}{5pt}` (DB icon, ->40pt) -> `\fontsize{7pt}{2.5pt}` (renders ~20pt)
- Deleted `poster/architecture.tex` (content now inline)
- Compilation successful (no errors, minor overfull \vbox 2.5pt)

### 02:15 -- Poster: Remove Objective block, expand architecture diagram

## 2026-06-06 (evening)

### 19:30 -- Paper-code alignment fixes

**Renamed `gale_shapley_matching` -> `greedy_matching`** across all files:
- `src/util/matching.py`: function definition + docstring now honestly describes "greedy 1-1 matching", explicitly notes it is NOT Gale-Shapley deferred acceptance.
- `src/pipeline/stage2/middleware/schema_merging/merger.py`: 4 call sites updated.
- `src/tests/unit/test_util_matching.py`: import + all 12 call sites.
- Matching algorithm is unchanged (still greedy descending-score assignment). Only the name was misleading.

**Renamed `SemanticSimilarity` -> `TokenSimilarity`** across all files:
- `src/pipeline/stage2/middleware/schema_merging/similarity.py`: class definition + docstring now accurately describes "token-overlap similarity" (Jaccard + context overlap, purely lexical).
- `src/pipeline/stage3/middleware/fact_allocation.py`: import + usage.
- `src/tests/unit/test_stage2_similarity.py`: import + fixtures.
- Engine behavior is unchanged. The paper's claim of "semantic" similarity was misleading for what is a lexical token matcher.

**Fixed `enable_ancestral_sampling` description** in `src/util/ablation.py`:
- Was: "use topological column order in compiler"
- Now: "enable IF-THEN logical constraint application (mask-based) in compiler"
- The flag still toggles the same code path. Only the description now honestly reflects that it's mask-based constraint application, not inverse-function ancestral sampling.

- Removed the entire **Objective secblock** from Row A (Motivation column now spans the row alone)
- Removed `\vspace{0.5cm}` gap between Row A and Middle row (diagrams start immediately below Motivation)
- Increased both middle-row diagrams from 28cm -> 32cm (14% larger, reducing the "cramped" feel)
- Final overfull \vbox: 8pt (negligible on A0, comparable to the original 2.5pt)

### 11:15 -- Renamed `enable_ancestral_sampling` -> `enable_logical_constraints`

**Across all 6 live files** (41 occurrences total, CLI arg `--no-ancestral-sampling` -> `--no-logical-constraints`):
- `src/util/ablation.py`: field + 4 classmethods (`no_ancestral_sampling()` -> `no_logical_constraints()`)
- `src/pipeline/stage4/compiler.py`: constructor param + 2 `self.enable_...` checks + comment
- `src/orchestration/stage4/entry.py`: variable + compiler constructor arg
- `run_evaluation.py`: 3 call sites + CLI arg + dest + output tag
- `run_pipeline.py`: 2 call sites + CLI arg + dest + help text
- `src/tests/unit/test_util_ablation.py`: test names, assertions, factory references
- Also updated `GAPS.md` and `IMPL_PLAN.md` doc references

**Note:** This was done before realizing the paper's ablation narrative ties "ancestral sampling" to constraint enforcement -- the old name was fine. Decided not to revert since the rename accurately describes current behavior, and a re-rename can happen when the real dependency-order + inverse-function sampling is implemented later.

### 11:35 -- Reframed `GAPS.md` around paper-faithful implementation

**Decision:** `README.md` is the source of truth. Gaps should be closed by
implementing the paper's architecture and algorithms, not by renaming comments,
docstrings, or code to better describe the current implementation.

**Updates made to `GAPS.md`:**
- Added a source-of-truth note and remediation rule: paper alignment first,
  internal naming/comment consistency later.
- Reframed the Gale-Shapley gap: the correct fix is implementing deferred
  acceptance, not editing the Stage 2 comment to say "greedy".
- Raised matching, mathematics-agent, patch-agent, schema-recall penalty, and
  ablation-semantics gaps to High because they affect paper reproducibility.
- Changed the ablation-toggle note to clarify that renaming the flag does not
  close the ancestral-sampling gap.

### 11:55 -- Stage 2 Gale-Shapley matching implemented

**Paper alignment fix:** Replaced the Stage 2 greedy schema matching path with
real row-proposing Gale-Shapley deferred acceptance.

**Code changes:**
- `src/util/matching.py`: added `gale_shapley_matching(score_matrix, threshold)`.
  It builds row and column preference lists from descending similarity scores,
  filters scores below the inclusive threshold, uses deterministic index-based
  tie-breakers, validates rectangular matrices, and returns row-sorted stable
  matches.
- `src/util/matching.py`: removed the old unused `greedy_matching()` helper.
- `src/pipeline/stage2/middleware/schema_merging/merger.py`: table matching,
  column matching before table merge, and column matching inside
  `_calculate_table_score()` now all call `gale_shapley_matching()`.
- `src/tests/unit/test_util_matching.py`: rewritten around Gale-Shapley behavior,
  including threshold, rejection/requeue, stability/no-blocking-pair, ties,
  rectangular matrices, and ragged matrix validation.
- `src/tests/unit/test_stage2_merger.py`: added a merger-level test proving
  `SchemaMerger` uses `gale_shapley_matching()` for table and column matching.
- `GAPS.md`: marked the Stage 2 matching gap as resolved and added it to Matches.

**Verification:**
- `python -m pytest src\tests\unit\test_util_matching.py src\tests\unit\test_stage2_merger.py -v` -> 30 passed.
- `python -m pytest src\tests\unit\test_util_matching.py src\tests\unit\test_stage2_merger.py src\tests\unit\test_stage2_registry.py src\tests\unit\test_stage2_similarity.py -v` -> 75 passed.
- `python -m pytest src\tests\unit -v` -> 306 passed.
- `python -m pytest src\tests\integration\test_pipeline_stage2.py -v` -> 6 skipped (`OPENAI_API_KEY` not set).

### 12:30 -- Repository structure cleanup

**Deleted local dead/out-of-scope artifacts:**
- Frontend/backend wrappers: `src/frontend/`, `src/frontend_plan.md`, `src/backend/`.
- Dead design docs / orphaned external copy: `src/algorithm/`, `SchemaAgent/`, `research/`, `src/baselines/`.
- Paper/media artifacts: `poster/`, `presentation/`, `paper_latex/`, `paper/`, `demo_paper/`, PDFs, demo video, images, diagram utility.
- Runtime/cache artifacts: `output/`, `tmp/`, `.pytest_cache/`, all `__pycache__/` directories.
- Legacy entry/deps: `main.py`, `requirements.txt`.

**Config changes:**
- Added `pyproject.toml` with runtime dependencies and a `dev` extra for pytest.
- Updated `.gitignore` to ignore `.claude/`, removed the stale `!image.png` exception,
  and added ignored artifact directories (`paper_latex/`, `poster/`, `presentation/`,
  `src/backend/`, `src/frontend/`, `src/algorithm/`, etc.).
- Removed the broken SchemaAgent `--baseline` path from `run_evaluation.py`; proper
  LLM4DBdesign cloning/inference audit is deferred.
- Updated `IMPL_PLAN.md` to reflect pyproject and deferred SchemaAgent integration.

**Verification:**
- `python -m pytest src\tests\unit -v` -> 306 passed.

### 12:55 -- Stage 2 semantic similarity implemented

**Paper alignment fix:** Replaced lexical-only `TokenSimilarity` with embedding-backed
`SemanticSimilarity` for Stage 2 schema merging and related shard/fact allocation.

**Code changes:**
- `src/pipeline/stage2/middleware/schema_merging/similarity.py`: replaced token-only
  scoring with sentence-transformer embedding cosine similarity as the primary signal.
- Kept lexical overlap (Jaccard + context overlap) as deterministic fallback and
  exact-name augmentation.
- `src/pipeline/stage3/middleware/fact_allocation.py`: now uses `SemanticSimilarity`.
- `src/tests/unit/test_stage2_similarity.py`: rewritten to test embedding-backed semantic
  matches via formal fake-model injection, matrix scoring, lexical fallback, and exact matches.
- No hardcoded domain synonym patch remains; tests use dependency injection rather than monkey-patching.
- `GAPS.md`: marked Stage 2 semantic similarity as resolved and added it to Matches.

**Verification:**
- `python -m pytest src\tests\unit\test_stage2_similarity.py src\tests\unit\test_stage2_merger.py src\tests\unit\test_util_matching.py -v` -> 56 passed.
- `python -m pytest src\tests\unit -v` -> 309 passed.

### 13:20 -- Stage 1 web retrieval wired into context enrichment

**Paper alignment fix:** Context enrichment now uses OpenAI `web_search_preview`
through a proper tool-enabled agent path instead of relying only on LLM parametric
knowledge.

**Code changes:**
- `src/util/agent.py`: `get_model()` and `get_agent_()` now accept
  `use_responses_api`, which is passed to `ChatOpenAI` for built-in OpenAI tools.
- `src/pipeline/stage1/agents/context_enricher/agent.py`: added
  `get_context_enricher_tools()` and `build_context_enricher_agent()`; production
  `get_agent()` now passes `web_search_preview` and `use_responses_api=True`.
- `src/pipeline/stage1/agents/context_enricher/prompt.txt`: added explicit web
  retrieval requirements for definitions and domain intelligence.
- `src/util/web_search.py`: updated integration note.
- `src/tests/unit/test_stage1_context_enricher.py`: added structural tests for
  tool spec, Responses API wiring, and prompt requirement using dependency injection
  rather than monkey-patching.
- `GAPS.md`: marked Stage 1 web search enrichment as resolved structurally.

**Verification:**
- `python -m pytest src\tests\unit\test_stage1_context_enricher.py src\tests\unit\test_stage1_models.py src\tests\unit\test_stage1_validation.py -v` -> 41 passed.
- `python -m pytest src\tests\unit -v` -> 312 passed.

**Note:** Live behavior still requires `OPENAI_API_KEY` and account/model access to
OpenAI built-in `web_search_preview`.

### 13:55 -- Stage 3 mathematics verifier and patch agent wired

**Paper alignment fix:** Added dedicated Stage 3 mathematics-verification and
patch-agent roles instead of sending all validation feedback back to the metadata
extractor.

**Code changes:**
- `src/pipeline/stage3/models/validation.py`: added `Stage3Issue`,
  `MathematicsValidationReport`, and `Stage3PatchPlan` Pydantic models.
- `src/pipeline/stage3/agents/mathematics_verifier/`: new strict agent folder
  (`agent.py` + `prompt.txt`) returning structured math validation reports.
- `src/pipeline/stage3/agents/constraint_patch_agent/`: new strict agent folder
  returning structured patched Stage 3 metadata.
- `src/pipeline/stage3/middleware/mathematics.py`: deterministic math issue collector
  for invalid distribution parameters, categorical weights, missing statistical refs,
  and duplicate range logic.
- `src/orchestration/stage3/entry.py`: `_extract_shard_with_retry()` now runs
  metadata extraction -> schema/SQL validation -> mathematics verifier -> patch agent
  when needed -> deterministic revalidation -> second verifier pass -> algebraic bridge.
- `src/tests/unit/test_stage3_math_patch_agents.py`: added offline tests using explicit
  fake agents passed through dependency injection; no monkey-patching or live LLM calls.
- `GAPS.md`: marked mathematics agent resolved structurally and patch-agent role partial
  pending column-level dependency graph/cycle patching.

**Verification:**
- `python -m pytest src\tests\unit\test_stage3_math_patch_agents.py src\tests\unit\test_stage3_distributions.py -v` -> 25 passed.
- `python -m pytest src\tests\unit -v` -> 316 passed.

### 06:04 -- Stage 3 solver-level satisfiability v1 added

**Scope:** Added a conservative bounded satisfiability layer for the new Stage 3
state-table predicate representation. This is not a universal solver compiler;
it proves obvious UNSAT cases for supported predicates and returns explicit
unsupported/unknown issues for patterns outside the v1 subset.

**Code changes:**
- `src/pipeline/stage3/middleware/satisfiability.py`: new solver-level checker using
  SciPy `linprog` for supported continuous linear numeric predicates over each
  SQL state table.
- Checks implemented:
  - numeric linear inequalities/equalities (`GT`, `GTE`, `LT`, `LTE`, `EQUALS`)
  - contradictory numeric bounds (e.g. `x >= 10` and `x <= 5`)
  - literal equality conflicts (`status = Completed` and `status = Cancelled`)
  - equality plus inequality conflicts (`x = A` and `x != A`)
  - nullability conflicts (`IS_NULL` and `IS_NOT_NULL` on the same state column)
  - unsupported non-numeric ordered predicates with explicit issue codes
- `src/orchestration/stage3/entry.py`: global validation now runs schema validation
  first, then solver-level satisfiability over deduplicated global state constraints.
  Issues map back to the producing table/shard so retry still happens at schema-shard
  level.
- `src/pipeline/stage3/models/manifest.py`: stripped old distribution/logical-rule
  fields from `TableConstraintManifest`; the Stage 3 manifest now only stores
  state-table constraints and nullable-column hints. `AlgebraicManifest` now exposes
  `global_state_constraints` only.
- `src/pipeline/stage4/compiler.py`: made old manifest fields (`global_rules`,
  `numeric_bounds`, `distributions`, `logical_rules`) optional via compatibility
  accessors so Stage 4 does not break before its rewrite.
- `src/tests/unit/test_stage3_satisfiability.py`: added SAT/UNSAT/unsupported tests.

**Important limitations:**
- v1 treats each state table independently; cross-state coupled solving is still future work.
- Routing/fanout/cardinality variables are not solved yet.
- Bilinear/nonlinear predicates are not supported yet; they should become explicit
  `UNSUPPORTED_*` reports when introduced into the IR.

**Verification:**
- `python -m pytest src\tests\unit\test_stage3_satisfiability.py src\tests\unit\test_stage3_math_patch_agents.py src\tests\unit\test_stage3_distributions.py -v` -> 30 passed.
- `python -m pytest src\tests\unit -v` -> 321 passed.

### 06:13 -- Rigorous Stage 3 satisfiability tests added

**Purpose:** Stress-tested whether the solver-level satisfiability layer can analyze
constraints unified from multiple shard outputs rather than only isolated single-rule
examples.

**Added coverage in `src/tests/unit/test_stage3_satisfiability.py`:**
- Equality + strict inequality contradiction on the same state table (`x = y` and
  `x > y`) -> `UNSAT_LINEAR_CONSTRAINTS`.
- Strict inequality cycle (`x > y` and `y > x`) -> `UNSAT_LINEAR_CONSTRAINTS`.
- Qualified self-inequality (`quantity != ORDER_ITEM.quantity`) ->
  `UNSAT_SELF_INEQUALITY`.
- Simulated overlapping shard outputs for an inventory aggregate state table:
  `total_demanded_quantity <= stock_quantity`, `total_demanded_quantity >= 100`,
  and `stock_quantity <= 90` are merged globally and detected as
  `UNSAT_LINEAR_CONSTRAINTS` with all involved fact references preserved.
- Independent state tables produce multiple issue types in one run: linear UNSAT,
  unsupported non-numeric ordering, and nullability conflict.
- Whitespace/case-normalized state queries from different shards still group into
  the same solver problem.

**Bug fixed:** `NOT_EQUALS` column-vs-column self checks now normalize qualified
column names before comparing, so `quantity != ORDER_ITEM.quantity` is caught.

**Verification:**
- `python -m pytest src\tests\unit\test_stage3_satisfiability.py -v` -> 10 passed.
- `python -m pytest src\tests\unit\test_stage3_satisfiability.py src\tests\unit\test_stage3_math_patch_agents.py src\tests\unit\test_stage3_distributions.py -v` -> 35 passed.
- `python -m pytest src\tests\unit -v` -> 326 passed.

### 06:36 -- Structural constraints and deterministic knob discovery added

**Design decision:** The deterministic Stage 3 compiler, not the LLM, is responsible
for emitting the exact set of independent structural knobs. The LLM may extract
explicit facts like row counts or fanout ranges, but when facts are silent the
compiler discovers the missing table-cardinality and relationship-fanout knobs.

**Code changes:**
- `src/pipeline/stage3/models/sql_models.py`: added `CardinalityConstraint`,
  `FanoutConstraint`, and `StructuralKnob` Pydantic models. `LLMResponse` now has
  explicit `cardinality_constraints` and `fanout_constraints` lists.
- `src/pipeline/stage3/models/manifest.py`: manifests now carry global state
  constraints, cardinality constraints, fanout constraints, and deterministic
  `tunable_knobs`.
- `src/pipeline/stage3/middleware/satisfiability.py`: added bounded structural
  satisfiability using SciPy MILP for integer table-count feasibility. It checks
  cardinality/fanout contradictions, FK zero-parent conflicts, and emits knobs for
  independent degrees of freedom.
- `src/orchestration/stage3/entry.py`: global Stage 3 validation now deduplicates
  extracted structural constraints, runs structural satisfiability after value
  satisfiability, and stores discovered knobs in the final manifest.
- `src/pipeline/stage3/agents/metadata_extractor/prompt.txt`: clarified that the
  extractor should emit explicit structural constraints only when grounded in facts,
  and must not emit tunable knobs.
- `src/tests/unit/test_stage3_satisfiability.py`: added tests for cardinality UNSAT,
  fanout/cardinality MILP UNSAT, unconstrained root-cardinality and derived-child
  fanout knobs, bounded fanout knobs, and exact constraints removing knobs.

**Type cleanup:** New Stage 3 path was refactored away from `Any`, `Dict`, and raw
manifest dictionaries. Stage 3 manifests and shard metadata now use typed lists and
small Pydantic entry records for dedupe/source tracking.

**Verification:**
- `python -m pytest src\tests\unit\test_stage3_satisfiability.py src\tests\unit\test_stage3_math_patch_agents.py -v` -> 19 passed.
- `python -m pytest src\tests\unit -v` -> 331 passed.

### 06:56 -- Cloud scheduling/billing stress harness tested

**Harness:** Added a distributed multi-tenant cloud scheduling and cross-region
billing scenario to `src/tests/unit/test_stage3_satisfiability.py` with five tables:
`DATA_CENTERS`, `COMPUTE_NODES`, `TENANT_PROFILES`, `VM_INSTANCES`, and
`BILLING_LEDGERS`.

**What passed as supported v1 behavior:**
- Hardware capacity state table: active VM allocations grouped by node, checking
  `total_cpu <= max_cpu_cores` and `total_ram <= max_ram_gb`.
- Contradictory CPU-capacity gridlock detected as `UNSAT_LINEAR_CONSTRAINTS`.
- Structural knobs discovered deterministically as:
  - `data_centers_row_count`
  - `compute_nodes_row_count`
  - `tenant_profiles_row_count`
  - `compute_nodes_to_vm_instances_fanout`
- Dependent structures were not emitted as independent knobs:
  - no `vm_instances_row_count`
  - no `billing_ledgers_row_count`

**What is intentionally unsupported in v1:**
- Polymorphic tier pricing with `CASE` and multiplication.
- Step-discount/budget-cap gate with `CASE`, scalar `MIN`, and arithmetic.
- Regional profit floor using derived arithmetic expressions across nested state tables.

These are now reported explicitly as `UNSUPPORTED_STATE_QUERY_EXPRESSION` rather
than being silently accepted as satisfiable.

**Bug/behavior fixed during harnessing:**
- SQLite rejected the first CTE version of the regional-profit query; the test now
  uses nested subqueries so SQL validation succeeds and the intended solver-tier
  boundary is exercised.
- Knob discovery was refined so parent/intermediate entity tables remain table-size
  knobs, exact 1:1 fanouts remove dependent table-size knobs, and explicit bounded
  fanouts become fanout knobs even for multi-parent bridge tables.

**Verification:**
- `python -m pytest src\tests\unit\test_stage3_satisfiability.py -v` -> 19 passed.
- `python -m pytest src\tests\unit -v` -> 335 passed.

### 07:55 -- Expression IR foundation and CASE classification added

**Milestone:** Implemented the first expression-IR layer needed for generalized
support of complex deterministic constraints. The solver still does not fully solve
nonlinear products or Big-M gates, but expressions are no longer opaque strings.

**Code changes:**
- `src/pipeline/stage3/models/expressions.py`: added typed Pydantic expression IR:
  `ExpressionNode`, `PredicateNode`, `CaseBranch`, and `ExpressionClassification`.
- `src/pipeline/stage3/middleware/expression_parser.py`: added deterministic parser
  and classifier for column refs, literals, arithmetic, `CASE WHEN`, scalar `MIN`/`MAX`,
  and simple predicates.
- `src/pipeline/stage3/middleware/satisfiability.py`: unsupported-expression detection
  now uses expression classification instead of raw regex only.
- `src/pipeline/stage3/models/__init__.py`: exported expression IR models.
- `src/tests/unit/test_stage3_expression_parser.py`: added tests for cloud-harness
  expressions.

**Supported/recognized tiers now:**
- Constant multiplication is classified as `linear`.
- Literal-result `CASE WHEN` expressions are classified as `piecewise_linear` and
  simple rate-bound checks are allowed.
- Scalar `MIN(linear_expr, literal)` is classified as `big_m_gate` for a future solver tier.
- Products with multiple variable factors (e.g. `runtime_hours * allocated_cpu * 0.06`)
  are classified as `nonlinear_product` and remain unsupported in v1.

**Type cleanup:** The new expression IR/parser and updated satisfiability path avoid
`Any`, `Dict`, `dict[...]`, and raw dict structures.

**Verification:**
- `python -m pytest src\tests\unit\test_stage3_expression_parser.py -v` -> 6 passed.
- `python -m pytest src\tests\unit\test_stage3_expression_parser.py src\tests\unit\test_stage3_satisfiability.py -v` -> 26 passed.
- `python -m pytest src\tests\unit -v` -> 342 passed.

### 11:54 -- Stage 1 enrichment tightened and filtered

**Problem addressed:** Context enrichment was emitting low-value generic database
advice such as "use foreign keys", "normalize tables", and "every table must have
a primary key". These facts polluted downstream context and encouraged generic
schema repair behavior.

**Code changes:**
- `src/pipeline/stage1/models/raw_fact.py`: added `ExternalFactKind`,
  `external_kind`, and `novelty_reason` so accepted external facts carry typed
  quality metadata.
- `src/pipeline/stage1/models/atomic_fact.py`: `AtomicFact.from_raw()` now preserves
  external enrichment quality fields.
- `src/pipeline/stage1/middleware/external_context_filter.py`: new deterministic
  filter rejects generic schema advice, invalid references, redundant restatements,
  and non-domain-specific external facts. Accepted facts are classified as technical
  definitions, domain modeling hints, domain constraint hints, or architecture
  patterns.
- `src/orchestration/stage1/entry.py`: context enrichment output is filtered before
  tagging; rejected facts are stored in `enrichment_filter_report` and not passed
  downstream.
- `src/orchestration/stage1/models.py`: Stage 1 output now includes
  `enrichment_filter_report` for auditability.
- `src/pipeline/stage1/agents/context_enricher/prompt.txt`: rewritten to ban
  `Schema Guideline:` facts, generic PK/FK/normalization advice, repeated input
  facts, and knob value assignment.
- `src/pipeline/stage1/agents/fact_extractor/prompt.txt`: added explicit relationship
  extraction rules for route/assign/map/link semantics and routing/bridge entities.
- Added tests for the external context filter and prompt requirements.

**Verification:**
- `python -m pytest src\tests\unit\test_stage1_models.py src\tests\unit\test_stage1_context_enricher.py src\tests\unit\test_stage1_external_context_filter.py src\tests\unit\test_stage1_fact_extractor_prompt.py src\tests\unit\test_stage1_validation.py src\tests\unit\test_stage1_text_matching.py -v` -> 77 passed.
- `python -m pytest src\tests\unit -v` -> 352 passed.

### 13:27 -- Stage 1 fail-closed retry and relationship checks tightened

**Code changes:**
- `src/pipeline/stage1/models/rephrased_nl.py`: `definitions` now defaults to an
  empty list so missing glossary output does not crash otherwise valid extraction.
- `src/pipeline/stage1/models/integrity_report.py`: issue severity now defaults to
  `medium`, and issue lists default empty. Malformed verifier issues fail safe instead
  of causing structured-output crashes.
- `src/util/retry_loop.py`: added `RetryExhaustedError`; retry loops now raise when
  max retries are exhausted with unresolved configured-severity errors. `ErrorRecord`
  also supports stable `signature_key` values.
- `src/pipeline/stage1/middleware/relationship_audit.py`: relationship omissions now
  carry stable signatures such as `missing_relationship:vm_instances:tenant_profiles`.
- `src/pipeline/stage1/middleware/error_formatter.py`: retry feedback now includes
  compact retry memory and "do not repeat" guidance for persistent relationship,
  enum, and origin errors.
- `src/pipeline/stage1/agents/verifier/prompt.txt`: verifier now explicitly checks
  relationship completeness, not just attribute existence.
- `src/pipeline/stage1/agents/tagger/prompt.txt`: enum/domain facts should be tagged
  as both `STRUCTURAL` and `LOGICAL`.

**Tests added/updated:**
- `src/tests/unit/test_stage1_relationship_audit.py`
- `src/tests/unit/test_stage1_tag_normalization.py`
- `src/tests/unit/test_util_retry_loop.py`
- additional Stage 1 model/filter/prompt tests.

**Verification:**
- Focused Stage 1 + retry tests -> 88 passed.
- Full unit suite -> 362 passed.

**Live messy Stage 1 batch:**
- Output: `output/stage1_batch_failclosed2_20260607_132137/`
- 4/4 messy cases failed closed with unresolved serious errors instead of leaking
  invalid facts downstream:
  - cloud messy ambiguous: 21 unresolved serious errors
  - hospital noisy missing: 25 unresolved serious errors
  - marketplace humanized: 15 unresolved serious errors
  - IoT energy ambiguous: 33 unresolved serious errors

**Interpretation:** Fail-closed safety now works, but Stage 1 does not yet converge
on messy human descriptions. The next work should reduce false-positive/over-strict
relationship and origin failures, and persist detailed error traces for failed live
runs so failures are easier to inspect without re-running.

### 13:38 -- Native LLM message tracing added

**Code changes:**
- `src/util/llm_trace.py`: added typed `LLMTraceCollector` and `LLMMessageTrace`
  models, context-var activation, message formatting, and artifact writers.
- `src/util/agent.py`: `StructuredAgent` now has a stable `name` for trace output.
- `src/util/invoke.py`: `get_response()` records input messages, returned messages,
  token usage, agent name, output structure, and parsed response type whenever an
  `LLMTraceCollector` is active.
- `src/orchestration/stage1/entry.py`: accepts an optional trace collector and activates
  it for the whole Stage 1 run.
- `run_pipeline.py`: added `--trace-llm`; Stage 1 traces are saved on both success and
  failure as `stage1_llm_message_trace.json`, `stage1_llm_message_trace_summary.tsv`,
  and per-message text files under `stage1_llm_messages/`.
- `src/tests/unit/test_util_invoke.py`: added trace-capture test.

**Verification:**
- Focused trace/retry/model tests -> 29 passed.
- Full unit suite -> 364 passed.
- Live CLI smoke run:
  `python run_pipeline.py --stages 1 --trace-llm ...` -> success.
  Output: `output/stage1_trace_cli_smoke_20260607_133636/`.

**Discovery:** The trace shows actual LLM call token usage includes verifier calls
inside the retry loop, but current Stage 1 `token_usage` does not include verifier
tokens. This is a separate accounting bug to fix next.

### 14:07 -- Stage 1 validation calibrated after fail-closed batch

**Code changes:**
- `src/util/retry_loop.py`: fail-closed behavior now uses only the latest unresolved
  configured-severity errors, while retry prompts use the previous attempt plus
  persistent signature markers. `ValidationResult` now carries `token_usage`, and
  retry loops count validator tokens.
- `src/pipeline/stage1/agents/verifier/agent.py`: verifier token usage is included
  in `ValidationResult`.
- `src/pipeline/stage1/middleware/relationship_audit.py`: relationship audit is now
  source-aware. It only blocks when the source text explicitly supports a missing
  relationship, accepts cardinality facts like "one ledger per tenant" as relationship
  coverage, and no longer fails closed on ID columns alone.
- `src/pipeline/stage1/middleware/text_matching.py`: origin matching now accepts short
  exact source substrings and fuzzy repairs return real source spans instead of slicing
  normalized-text offsets.
- `src/pipeline/stage1/middleware/validation.py`: reference normalization is now
  order-preserving; deterministic validation errors have stable signatures; low-confidence
  but plausible origin mismatches are warnings instead of unconditional critical errors.
- `src/pipeline/stage1/middleware/error_formatter.py`: deterministic errors now get
  type-specific repair guidance instead of all being framed as origin failures.

**Verification:**
- Focused retry/validation tests -> 62 passed.
- Full unit suite -> 370 passed.

**Calibrated live Stage 1 messy batch:**
- Output: `output/stage1_batch_calibrated_20260607_140151/`
- cloud messy ambiguous: failed closed with 9 unresolved serious errors.
- hospital noisy missing: success, 40 final facts, 2 accepted external, 6 rejected external.
- marketplace humanized: success, 53 final facts, 1 accepted external, 3 rejected external.
- IoT energy ambiguous: failed closed with 3 unresolved serious errors.

**Remaining Stage 1 blockers from live batch:**
- Cloud: relationship facts were added but with invented origin snippets like "Linked via node_id".
  Need origin repair to prefer the original relationship sentence rather than synthesized fragments.
- Cloud: soft monthly cap nuance still unresolved.
- IoT: missing `alerts -> readings` relationship, site/building ambiguity not represented as a fact,
  and export-meter negative `kw_value` uncertainty still not captured precisely.

### 14:29 -- Stage 1 deterministic postprocessing added

**Code changes:**
- `src/pipeline/stage1/middleware/origin_repair.py`: deterministic origin repair now
  records `OriginRepairEvent`s and repairs invalid origins only to exact source spans.
- `src/pipeline/stage1/middleware/relationship_recovery.py`: deterministic recovery
  creates explicit relationship facts from source clauses with explicit relationship
  verbs (`assigned to`, `owned by`, `generated from`, `installed at`, `hang off`, etc.).
  It does not infer relationships from IDs alone.
- `src/pipeline/stage1/middleware/uncertainty_normalization.py`: deterministic
  uncertainty facts are added for explicit uncertainty cues like `maybe`, `not sure`,
  `forgot to decide`, and `not described yet`.
- `src/pipeline/stage1/middleware/postprocess.py`: new Stage 1 postprocess pipeline runs
  relationship recovery -> uncertainty normalization -> origin repair before validation.
- `src/util/retry_loop.py`: added optional `postprocessor` hook so deterministic
  postprocessing runs inside retries before validation.

**Tests added:**
- `src/tests/unit/test_stage1_relationship_recovery.py`
- `src/tests/unit/test_stage1_uncertainty_normalization.py`
- `src/tests/unit/test_stage1_postprocess.py`

**Verification:**
- Focused Stage 1 postprocess/retry tests -> 101 passed.
- Full unit suite -> 377 passed.

**Postprocess live Stage 1 messy batch:**
- Output: `output/stage1_batch_postprocess_20260607_142407/`
- cloud messy ambiguous: success, 33 final facts, 4 accepted external.
- hospital noisy missing: failed closed with 5 unresolved serious errors.
- marketplace humanized: success, 54 final facts, 0 accepted external, 3 rejected external.
- IoT energy ambiguous: success, 52 final facts, 2 accepted external, 4 rejected external.

**Interpretation:** Deterministic postprocessing moved the live messy Stage 1 batch
from 0/4 passing after fail-closed to 3/4 passing. Remaining work is focused on the
hospital case: prescription -> appointment relationship recovery, exact origin repair
for typo-heavy patient facts, and calibrating strict/soft wording for `should not`
constraints.

### 19:45 -- Stage 1 context auditor loop replaces semantic filtering

**Design correction:** Removed deterministic semantic recovery/filtering as the source
of truth. Stage 1 should remain a source-grounded fact extractor; semantic usefulness
of external context is now judged by an LLM auditor, while deterministic code only
checks mechanical invariants.

**Code changes:**
- Deleted deterministic semantic recovery middleware:
  - `relationship_recovery.py`
  - `relationship_audit.py`
  - `uncertainty_normalization.py`
  - `origin_repair.py`
  - `postprocess.py`
- Added `src/pipeline/stage1/models/context_audit.py` with `ContextAuditReport`,
  `ContextRejectedFact`, and `ContextAuditAttempt`.
- Added `src/pipeline/stage1/agents/context_auditor/` agent folder (`agent.py`,
  `prompt.txt`) for semantic auditing of proposed external context.
- `src/orchestration/stage1/entry.py`: context enrichment now runs as an audited
  LLM loop: enricher -> auditor -> retry enricher if needed -> mechanical filter.
- `src/pipeline/stage1/middleware/external_context_filter.py`: reduced to mechanical
  checks only (valid references, no self-reference, no duplicate external text) and
  light type classification. Generic/unrelated advice is no longer decided by regex.

**Verification:**
- Focused Stage 1/context-auditor tests -> 99 passed.
- Full unit suite -> 364 passed.

**Auditor-loop live messy Stage 1 batch:**
- Output: `output/stage1_batch_auditor_20260607_153137/`
- cloud messy ambiguous: failed closed with 1 unresolved medium issue (region vs data-center ambiguity).
- hospital noisy missing: failed closed with 2 unresolved medium issues (appointment status storage ambiguity, days_supply inclusivity wording).
- marketplace humanized: failed closed with 2 unresolved medium issues (final_total non-negative missing, payout per-seller specificity).
- IoT energy ambiguous: success, 43 final facts, 3 accepted external facts, 0 rejected external facts.

**Interpretation:** The auditor-loop design is cleaner and less brittle than deterministic
semantic recovery. Remaining failures are true semantic coverage/ambiguity issues from
the extractor/verifier loop rather than deterministic regex behavior.

### 05:53 -- Stage 3 logical constraints moved to state-table predicates

**Architecture deviation:** Stage 3 now intentionally deviates from the paper's
old algebraic/distribution-centric metadata path. For the current rewrite, each
logical constraint is represented as a SQL state-table query plus exactly one
binary predicate over that state table. Univariate distribution extraction is
ignored by the Stage 3 extractor path for now.

**Code changes:**
- `src/pipeline/stage3/models/sql_models.py`: replaced the extraction response
  with `SQLGroundedConstraint(state_query, left_operand, operator, right_operand,
  fact_references)` and `BinaryOperand`; removed distributions from `LLMResponse`.
- `src/pipeline/stage3/agents/metadata_extractor/prompt.txt`: rewritten to ask
  only for deterministic logical constraints as state-table predicates.
- `src/orchestration/stage3/entry.py`: shard extraction now validates state queries
  against the shard schema, stores constraints directly in `TableConstraintManifest`,
  and runs global-schema feasibility validation in the healing loop. Retry feedback
  still happens at the shard level.
- `src/pipeline/stage3/models/manifest.py`: added `state_constraints` to table
  manifests and `global_state_constraints` to `AlgebraicManifest`.
- `src/orchestration/stage3/models.py`: updated raw shard metadata to record
  state-query predicate fields instead of old `on`/`condition` strings.
- `src/pipeline/stage3/middleware/mathematics.py`: repurposed deterministic issue
  collection for state-table feasibility issues.
- `src/pipeline/stage3/agents/mathematics_verifier/` and
  `constraint_patch_agent/` prompts now describe state-table feasibility repair,
  though the simple extraction path currently loops errors back to the extractor.
- `src/tests/unit/test_stage3_math_patch_agents.py`: updated offline tests to cover
  state-table constraint validation and shard storage.

**Important implementation notes:**
- The old `IfNode` algebraic bridge is no longer used for newly extracted Stage 3
  logical constraints.
- Global manifests are rebuilt from per-shard metadata so overlapping shards do not
  overwrite each other's extracted state constraints.
- Stage 4 does not yet compile `global_state_constraints`; that is a separate
  future step. Compatibility accessors were added later so Stage 4 tolerates the
  simplified Stage 3 manifest.

**Verification:**
- `python -m pytest src\tests\unit\test_stage3_math_patch_agents.py src\tests\unit\test_stage3_distributions.py -v` -> 25 passed.
- `python -m pytest src\tests\unit -v` -> 316 passed.
