# Pre-run experiment decisions

This file records operational choices made before any model result is observed.

## Frozen implementation choices

- Compression target: `0.4` retained local tokens.
- Compression algorithm: deterministic `uniform_extract`, version 1.
- JSON object keys and tool schemas are not compressed.
- List items and text units are sampled at evenly spaced positions.
- Main execution order seed: `20260914`.
- Main plan: B0 240, I-H 80, T-H 80, O 80; total 480.
- Provider failures receive at most one retry and remain in the append-only raw log.
- Statistical bootstrap unit: `pair_id`; 10,000 replicates.
- B0 binary success for McNemar: majority vote over three repetitions.
- Visible provider output is split between natural language and tool calls in
  proportion to their local-token counts. The unallocated remainder is reported.
- The frozen dataset is identified by `data/manifest.json`.
- A non-negative protocol residual is accepted up to
  `max(256 tokens, 2% of provider input)`.
- Local I/T content counting tokenizes semantic field names and values but
  excludes JSON punctuation and transport-only `type`, `role`, `id`, `call_id`,
  `status`, and opaque `encrypted_content` wrappers. Provider-private framing
  remains in protocol residual. Tool-result payloads are the exception: the
  adapter transmits them as canonical JSON strings, so T counts that exact
  serialized string, including its JSON keys and punctuation.
- The realized compression ratio must be within absolute error 0.05 of 0.4.
- Provider input usage is split into uncached, cache-read, and cache-write
  tokens. A missing cache-write sub-count is never coerced to zero when the
  selected price schedule charges cache writes.

## External values frozen on 2026-09-14 before the first smoke call

- Provider model ID: `gpt-5.6-terra`. The official model page listed the alias
  but no dated snapshot, so the alias is recorded explicitly.
- Standard price per 1M tokens: uncached input USD 2.00, cached input USD 0.20,
  cache write USD 2.50, output USD 12.00.
- Long-context price per 1M tokens: uncached input USD 4.00, cached input
  USD 0.40, cache write USD 5.00, output USD 18.00. The full request moves to
  this tier when input exceeds 272K tokens.
- Official sources:
  `https://developers.openai.com/api/docs/models/gpt-5.6-terra` and
  `https://developers.openai.com/api/docs/pricing`.
- Exact public tokenizer mapping: `gpt-5.6-terra -> o200k_base`, verified with
  `tiktoken==0.14.0`; no proxy is used.
- No provider SDK is used. The adapter uses Python's standard-library HTTP
  client and an injectable transport, so there is no SDK version to freeze.
- `temperature` is omitted and the Terra model default is used. On 2026-09-17,
  before any successful smoke response was produced, the live API returned
  HTTP 400 stating that `temperature` is unsupported for `gpt-5.6-terra`.
  The two failed transport attempts are retained separately as preflight
  diagnostics and excluded from all research analyses.
- The first successful-response diagnostic on 2026-09-17 reported 335 provider
  input tokens while the implementation counted 526 local tokens. Before any
  valid smoke run, local content accounting was corrected to exclude JSON and
  transport wrappers, as required by the frozen local-content/provider-usage
  separation. The corrected first-request local count is 320, leaving a
  non-negative 15-token protocol residual. No response content or task outcome
  was inspected to make this correction.
- A subsequent invalid second-turn diagnostic showed that provider-encrypted
  reasoning state had been counted as 1,230 local I tokens even though it is an
  opaque transport field. `encrypted_content` was therefore excluded before
  any valid smoke run; only turn number and aggregate token counts were
  inspected.
- Eight pilot rows then completed before a long structured tool result exposed
  the opposite mismatch: T omitted JSON punctuation even though the adapter
  transmits tool results as JSON strings (residual 556 versus the frozen 256
  limit). Those rows were archived as an excluded accounting pilot, and T was
  aligned with the adapter's exact canonical serialization before restarting
  the smoke plan. The amendment used aggregate token diagnostics only; pilot
  task outcomes were not used to choose the counting rule.
- One later pilot attempt exceeded the frozen residual tolerance on turn 4
  (287 versus 256) and, under the then-current implementation, stopped before a
  JSONL row was written. Before main execution, invariant failures were changed
  to append an `accounting_invalid` row with provider cost and aggregate-only
  diagnostics, mark it `analysis_valid=false`, and continue the plan. The smoke
  run-count gate now requires exactly 20 valid rows rather than merely at most
  20. The successful 20-row smoke file contains no such invalid row.
- During main execution, a standard-library read `TimeoutError` at plan ordinal
  360 exposed that only `URLError` connection failures were marked retryable.
  The failed attempt remains in the append-only raw log and is excluded from
  analysis. Read timeouts are now classified as transient and receive the same
  single frozen retry before execution resumes.
- The first resume after that timeout incorrectly treated prior
  `accounting_invalid` ordinals as unfinished and appended three replay rows
  before manual interruption. Raw append-only history was preserved. Resume and
  analysis now treat the first non-provider-error attempt as terminal for each
  ordinal; only provider errors may advance to a later canonical attempt. Thus
  the accidental replay rows are deterministically excluded without rewriting
  raw data.
- The first post-run analysis dry run revealed that condition summaries and
  McNemar counts did not apply `analysis_valid` even though CDR inputs did. No
  inferential result was accepted from that output. The common analysis entry
  point and the individual summary/McNemar functions now exclude invalid rows,
  and outputs were regenerated from the unchanged append-only raw log.

The live preflight fails closed if any frozen value or tokenizer version is
missing or differs from the installed environment.

## Ratio follow-up frozen on 2026-09-18

This stage was specified after the 0.4 main result, but before any 0.6 or 0.8
follow-up response was observed.

- Cells: I-H and T-H at retained ratios 0.6 and 0.8; 80 runs per cell, 320 total.
- Existing repeated B0 runs are reused; B0 and O are not replayed.
- Safety gate: the one-sided 95% upper bound on success harm must be at most
  5 percentage points. Accounting-invalid rows count as failures for this gate.
- Cost gate: the pair-bootstrap 95% interval for list-price saving must have a
  lower bound above zero.
- Four success comparisons use Holm correction. Cost/CDR analysis continues to
  use only accounting-valid rows, with invalid counts reported separately.
- Execution order seed: `20260918`; bootstrap unit: `pair_id`; 10,000 replicates.

## Exploratory architecture stage after the ratio follow-up

No uniform-extraction cell passed the safety gate. Tool Contract-Preserving
Compression (TCPC; internal algorithm id `contract_guard`) was therefore
created as a post-ratio-sweep exploratory architecture and is not presented as
part of the original confirmatory plan. The paper-facing name avoids collision
with unrelated systems already named ContractGuard.

- Condition: T-H at requested ratio 0.8; 80 runs; seed `20260919`.
- The adapter declares `notes` non-contractual. Every other key and value is
  preserved exactly.
- The requested ratio is a budget target, not permission to remove contract
  fields. Compression stops at the contract-preserving safety floor.
- The direct comparator is uniform extraction at T-H ratio 0.8 on the same 80
  task-language instances.

## Offline validation only

The scripted provider fixes behavior and therefore must produce CDR 0. It is
used solely to verify orchestration and accounting. Its success or cost is not a
research result and must never be merged into smoke or main JSONL files.
