# Model-Dependent Constants

Every number in this pipeline whose correct value depends on **which LLM is
running**, not on the pipeline itself. Swapping the model does not just change
output quality -- it can invalidate these, and several fail SILENTLY when wrong.

This file exists because that failure mode has already happened once. The chunk
budget was sized by the context window, which is a model property; on
gemini-2.5-flash it came out 208x larger than any real input, so chunking never
split anything, one extraction call was handed a 41-entity domain, and the result
was a 9-table schema with 86 of 121 facts unrepresented -- with **no error
anywhere**. It looked like a modelling limitation for weeks.

## The list

| Constant | Where | Value | Depends on the model because |
|---|---|---|---|
| `_FALLBACK_EXTRACTION_CAPACITY_TOKENS` | `stage1/middleware/budget_chunker.py` | 900 | How much domain one call can hold in one structured answer. **Measured on gemini-2.5-flash, 2026-07-30.** |
| `_CHARS_PER_TOKEN` | `budget_chunker.py`, `orchestration/stage1/entry.py` | 4 | Tokenizer. A different tokenizer changes every token estimate derived from it. |
| `_DEFAULT_SAFETY_MARGIN` | `budget_chunker.py` | 0.6 | Fraction of the window considered usable. |
| `_DEFAULT_PROMPT_OVERHEAD_TOKENS` | `budget_chunker.py` | 6000 | Size of the system prompt + output-format block as that model's tokenizer counts it. |
| `context_window_safety_margin`, `fixed_prompt_overhead_tokens` | `util/algorithms/sharding_ilp.py` | 0.6, 6000 | Same two quantities, for Stage 3's sharder. |
| `_NL_WINDOW_SHARE`, `NL_MAX_CHARS_FALLBACK` | `orchestration/stage1/entry.py` | 0.25, 50000 | How much raw NL is fed to extraction in one go. |
| `EXTRACTION_ROUNDS`, `ENRICHMENT_ROUNDS` | `orchestration/stage1/loop_config.py` | 3, 3 | How many audit rounds a model needs to converge. Tuned on gemini + the hospital spec. |
| `_DEFAULT_PHASE1_ROUNDS` | `orchestration/stage3/entry.py` | 3 | Same, for Stage 3's generator/checker/auditor loop. |
| `retry_count` / `max_retries` defaults | `orchestration/stage2/{entry,utils}.py` | 5, 4 | Same. |

Not model-dependent, for contrast: everything in `util/constraint_model/` and the
evaluation metrics. Those are set arithmetic and key containment over structures,
and they are deliberately threshold-free (see `EVALUATION_METRICS.md`).

## Which of these fail silently

Ranked by how hard the failure is to notice.

1. **Extraction capacity too high** -- the extractor models a fraction of the
   domain, every downstream stage succeeds on the fragment, and nothing errors.
   This is the one that already cost real time.
   *Symptom:* a large share of required facts unrepresented. Stage 2 now warns
   above one third (`_warn_if_extraction_saturated`).
   *Fix:* lower `EXTRACTION_CAPACITY_TOKENS`.

2. **Stage 3's tables-per-shard too high** -- same shape one stage later: the
   constraint generator invents columns and tables that are not in its own shard
   schema, and the deterministic checker refuses them.
   *Symptom:* constraints dropped with "not found in schema" / "ambiguous at
   grain". Seen live: 10 of 13 constraints refused on a 16-table shard.

   **Measured, and it is NOT the collapse Stage 2 had.** Same 41-table schema and
   121 facts, varying only tables-per-shard:

   | cap | shards | constraints | square | loose | tokens |
   |---|---|---|---|---|---|
   | 41 | 2 `[2,41]` | 16 | 14 | 2 | 143,526 |
   | 14 | 4 `[14,14,8,14]` | 21 | 14 | 4 | 298,584 |

   Smaller shards do help -- +31% constraints and two more loose probes -- but a
   41-table shard still yields a usable contract, where Stage 2 given the whole
   domain collapsed from 43 tables to 9. So this is an OPTIMIZATION, not a
   defect, and the default is deliberately left alone: two points at one run each
   cannot justify doubling the token cost, given this session measured Stage 1
   variance at 15-vs-121 facts on identical input. Revisit with repeats per point.
   (A cap of 8 was attempted and the ILP found no feasible sharding -- a
   harness limitation, `max_shards` was set to exactly `ceil(43/8)` with no slack
   for the co-location constraints, not a pipeline finding.)

3. **Round counts too low** -- the loop exhausts its budget with unresolved
   findings and ships whatever it has. This one at least logs, loudly, since the
   withholding work landed.

4. **`_CHARS_PER_TOKEN` wrong** -- every derived budget is off by that ratio.
   Bounded and unlikely to be catastrophic, since the margins are generous, but
   it silently skews all of the above.

## Measured: the constant was calibrated on the WEAKEST model

One er_extractor call, the SAME 121 facts from a 41-table spec, provider as the
only variable (`experiments/provider_capacity_probe.py`):

| provider | model | entities | rels | attrs | seconds | tokens |
|---|---|---:|---:|---:|---:|---:|
| cerebras | gpt-oss-120b | **38** | 49 | 144 | 82.5 | 18,967 |
| deepseek | deepseek-v4-flash | **37** | 51 | 136 | 144.2 | 32,966 |
| gemini | gemini-2.5-flash | **9** | - | - | - | - |
| openrouter | nemotron-3-super-120b:free | timeout at 420s for one call |||||
| groq | llama-3.3-70b | **400 Failed to call a function** -- schema unsupported |||||
| openai | - | probe timeout |||||

**Variance between the two capable models is LOW: 38 vs 37 entities (3%), 49 vs
51 relationships, 144 vs 136 attributes.** They agree closely on a 121-fact input
that gemini-2.5-flash reduces to 9 tables -- 4.2x fewer. So this is not general
cross-model variance; it is gemini-2.5-flash being the weak one at this task, and
the extraction-capacity constant was calibrated against it.

The outputs are real, not stubs: 28 of cerebras's 38 entity names match a
ground-truth table exactly and the other 10 are synonyms of one (AuditTrail/
AUDIT_LOG, Entry/RACE_ENTRY, Order/KIT_ORDER, Upload/RIDE_UPLOAD, ...), at 3.8
attributes per entity. Essentially the whole 41-table domain, in one call, for
19k tokens -- against the 5 chunks and 370-450k Stage 2 tokens gemini needs to
reach the same coverage.

Three conclusions:

1. **The 900-token default is pessimised to the weakest model and would HURT the
   others.** On cerebras or deepseek it splits into 4 calls what one call handles
   better, and splitting is not free -- entities separated across chunks must be
   reunified by the conceptual merger. A fixed constant is the wrong SHAPE for
   this parameter, which promotes adaptive re-chunking from refinement to fix.
2. **Portability fails before capacity does.** groq cannot run this pipeline at
   all: its tool-calling path rejects the recursive ConceptualModel schema, the
   same failure CLAUDE.md documents for vLLM and the reason `PROVIDER=vllm` routes
   to `json_mode`. Routing groq the same way is the obvious repair. Latency is the
   other wall -- openrouter's free tier exceeds 7 minutes for one call.
3. **Cost and speed do not track capability here.** cerebras reached the same
   result as deepseek in 57% of the time and 58% of the tokens.

Caveat: one call per provider, extractor only (no auditor rounds, merge or
mapping), n=1 each. The 9-vs-37/38 gap far exceeds plausible run-to-run variance
and two independent providers corroborate the high end, but the individual figures
are single observations.

## When changing model, do this

1. Set `EXTRACTION_CAPACITY_TOKENS` for the new model, or re-run the sweep:
   `experiments/forced_multichunk_stage2.py <budget>` over a large-schema case,
   and look at tables recovered plus unrepresented facts. The one robust finding
   is that a single over-large chunk collapses the schema -- start conservative.
2. Watch for the two silent symptoms above in the run log.
3. Do NOT scale capacity off the context window. That is the trap this file
   documents: gemini-2.5-flash has a ~1M-token window and still saturates near
   9-16 entities per call, so window size predicts capacity badly.

## The real fix

None of this calibration should be necessary. The pipeline already computes a
model-independent quality signal -- the share of required facts that found no
home in the schema -- and could re-chunk and retry on it, converging on the right
chunk size for whatever model is in use without any per-model constant. That is
the direction; the constants above are the interim, and the logging is what keeps
their model-dependence visible rather than hidden.
