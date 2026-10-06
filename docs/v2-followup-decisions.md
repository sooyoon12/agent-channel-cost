# V2 exploratory follow-up decisions

These decisions were frozen in code and `configs/v2_followup.yaml` before any
follow-up API call.

## Scope

- Repeat B0 for all frozen 75 pairs and both languages in the same randomized
  follow-up plan. Concurrent B0 is the primary reference; the existing 150 B0
  trajectories are retained only for drift diagnosis.
- Add uniform and head-tail cells at target ratios 0.6 and 0.9. The existing
  0.8 cells remain in the main study and are not rerun.
- Add one deployable middle baseline, `surface_ranked_extract`, at ratios 0.8,
  0.4, and 0.2.
- Treat every result as exploratory post-main evidence, not preregistered
  confirmatory evidence.

## Non-oracle surface-ranked baseline

The compressor receives only the raw Korean or English user request and the
tool observation. It does not receive the gold identifier, generator semantic
form, expected tool calls, or a list of safe-to-delete fields. It ranks
structured records by exact scalar-value overlap with the request and removes
the lowest-ranked records until reaching the token budget. Opaque identifiers,
codes, notes, descriptions, memos, and attendee arrays are excluded as ranking
signals to reduce accidental leakage.

This is a simple deployable baseline, not the proposed final adaptive
compressor. Its role is to fill the gap between position-only truncation and
privileged oracle references.

## Inference and reporting

- Resample `pair_id`, keeping Korean and English surfaces clustered.
- Report success, realized ratio, list-price cost/run, and list-price
  cost/success for every cell.
- Compare strict bilingual-pair success with concurrent B0 and report Holm
  adjustment across the seven exploratory compression cells.
- Retain invalid planned runs as failures for success reporting.
- Generate a figure contrasting change in cost/run and cost/success.

## Execution order

1. Run the offline audit and preserve its plan checksum.
2. Run the 32-call smoke plan.
3. Inspect completeness and accounting before the 1,200-call main plan.
4. Run the main plan with an explicit projected per-run cost and budget guard.
5. Analyze only after all terminal planned rows are present.

No provider command is run until API execution is explicitly authorized.

The surface-ranked ratios were selected after an offline gold-field retention
diagnostic on the benchmark instances. Model success outcomes were not viewed,
but the resulting sweep is post-hoc and must remain explicitly exploratory.

The plan checksum serialization now includes the method label. Older plan
checksums omitted this field, so checksum values produced by the earlier code
are not comparable even when the run entries otherwise match.
