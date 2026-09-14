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

## Values that must be frozen before the first smoke call

- Exact provider model ID or snapshot.
- Official price values, price version, and source URL.
- Exact tokenizer encoding, or the explicitly justified proxy encoding.
- Installed provider SDK and tokenizer package versions.

The live preflight intentionally fails while any of these values is missing.

## Offline validation only

The scripted provider fixes behavior and therefore must produce CDR 0. It is
used solely to verify orchestration and accounting. Its success or cost is not a
research result and must never be merged into smoke or main JSONL files.
