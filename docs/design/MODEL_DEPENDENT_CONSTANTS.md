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

3. **Round counts too low** -- the loop exhausts its budget with unresolved
   findings and ships whatever it has. This one at least logs, loudly, since the
   withholding work landed.

4. **`_CHARS_PER_TOKEN` wrong** -- every derived budget is off by that ratio.
   Bounded and unlikely to be catastrophic, since the margins are generous, but
   it silently skews all of the above.

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
