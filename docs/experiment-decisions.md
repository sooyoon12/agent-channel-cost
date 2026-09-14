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

The live preflight fails closed if any frozen value or tokenizer version is
missing or differs from the installed environment.

## Offline validation only

The scripted provider fixes behavior and therefore must produce CDR 0. It is
used solely to verify orchestration and accounting. Its success or cost is not a
research result and must never be merged into smoke or main JSONL files.
