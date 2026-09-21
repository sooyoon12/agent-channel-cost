# V2 T-H follow-up decisions (preregistered before held-out runs)

## Scope

- The v2 confirmatory scope is tool-observation compression (`T-H`) only.
- `I-H` is excluded. The v1 operation shortened a single short instruction, not a multi-turn history; its result may be described only as an exploratory destructive instruction-truncation result.
- Every v2 pair requires reading one tool result to construct the next tool call. There are no T-H no-op precise tasks.

## Leakage and difficulty controls

- V2 seeds are disjoint from v1.
- `notes` use eight repeated neutral templates; neither correctness labels nor unique target-only strings occur.
- Record and event identifiers use the same opaque hash format for every candidate.
- Sensitive long tasks place the correct amount and date strictly beyond the stated boundaries; exact boundary equality is not an answer shortcut.
- The offline audit executes the gold trajectory and verifies the final state with the production evaluator.

## Development gate and held-out freeze

- Every fourth semantic pair is development data: 20 long and 5 event pairs (50 bilingual runs).
- The untouched candidate held-out set contains 60 long and 15 event pairs (150 bilingual runs per method).
- Before any held-out run, development B0 must have complete coverage, no analysis-invalid rows, and success in `[0.90, 0.99)`. A rate at or above 0.99 is treated as unresolved ceiling effect; a rate below 0.90 is treated as excessive difficulty.
- The development gate is descriptive. Confirmatory uncertainty and resampling use `pair_id` as the cluster because Korean and English surfaces share hidden state.
- Only a passing development report can create `data/v2/manifest.json`. Held-out execution refuses to run without a matching frozen manifest.
- Development round 1 (64 records, four predicates on neutral tasks) produced 50/50 success and was blocked by the ceiling rule. Before any held-out run, candidate revision 2 increased long observations to 128 hard-negative records and added category/priority predicates (plus a risk bound for sensitive tasks). Round-1 logs remain archived and are not pooled with revision 2.
- Development round 2 produced 43/50 success (86%): event 10/10, long English 17/20, and long Korean 16/20. It was blocked by the 90% floor. Candidate revision 3 retains the same predicates and hard-negative construction but reduces long observations from 128 to 96 records. Earlier rounds remain archived and are not pooled.
- The round-3 stop decision was made at 27/50 planned runs and 21 successes, when even perfect remaining outcomes could reach only 44/50 (88%). One already in-flight run completed before interruption, so the archived file contains 28 rows and 22 successes. Candidate revision 4 uses 64 records; neutral tasks retain four equality predicates, while sensitive tasks use region, status, amount, date, and risk (five predicates). This changes one difficulty axis after the failed development diagnostic. Round 3 is permanently marked incomplete and is not pooled.
- Development round 4 passed at 48/50 (96%): event 10/10 and long 38/40, with one long failure in each language. Thirteen rows carried positive-residual warnings, but no row was analysis-invalid. Revision 4 is the version used to freeze the held-out set.

## Accounting decision

- `provider_input = local I + local T + protocol residual` remains an exact recorded identity.
- A negative residual remains fatal because local channel counts cannot exceed the provider total under the accounting definition.
- A positive residual above the diagnostic tolerance is retained as a warning in v2 rather than converted into a behavioral failure. It captures provider-side protocol/formatting items that the local content tokenizer does not classify. No row is excluded solely for a positive residual.
- V1 behavior is unchanged because the new setting defaults to the old fail-closed rule.

## Comparison conditions

| Label | Role | Information used |
|---|---|---|
| B0 | uncompressed reference | full observation |
| uniform | destructive baseline | evenly spaced entries/text units |
| head-tail | destructive baseline | beginning and end entries/text units |
| query-aware | semantic upper reference | perfect structured task constraints, never the gold ID |
| schema-oracle | oracle upper bound | predeclared non-contract key `notes` |

The schema-oracle condition is not a proposed compressor. Its saving is reported together with the observed token share of `notes`; the filler quantity therefore cannot be mistaken for general compression performance. The query-aware condition is also not a deployable end-to-end method in this study because parsing is supplied perfectly by the synthetic generator.

## Confirmatory execution rule

- No held-out method is run until the dataset is frozen.
- All methods use one run per language and pair at target ratio 0.8 where the algorithm has a ratio control.
- Algorithm name and version are stored at the run-record top level as well as in compression events.
- Paid runs are append-only and resumable. Output files are separate by method.
- The preferred `heldout-main` plan interleaves all five methods in one seeded random order and records `experiment_method`; this supersedes method-by-method execution for the confirmatory run.
- The first 21 held-out runs were launched sequentially. The remaining frozen plan is resumed with four independent workers solely to reduce wall-clock time; plan ordinals, prompts, model settings, and scoring are unchanged. Standard list-price cost is primary for method comparison because provider caching can depend on concurrent request timing; actual billed cost is retained as a secondary operational measure.

## Post-analysis corrections and interpretation

- Four strict bilingual-pair comparisons are adjusted with Holm's method. This multiplicity rule was specified after review of the held-out results, so adjusted p-values are descriptive rather than preregistered confirmatory evidence.
- Head-tail's point estimate for list-price cost per run is 5.2% above B0, but the paired 95% CI includes zero. The supported statement is that head-tail did not demonstrate a cost saving while success fell by 23.3 percentage points.
- Pair-cluster bootstrap intervals are reported for list-price cost per success. Uniform is +0.6% versus B0 with a 95% CI of [-10.7%, 15.5%]; head-tail is +38.4% with a 95% CI of [15.2%, 69.6%].
- Deterministic offline replay audits whether the exact gold item and all action-bearing fields survive. Uniform succeeds 122/122 when they survive and 0/28 when removed. Head-tail succeeds 111/118 when they survive and 0/32 when removed. Thus uniform's observed failures are completely explained by information removal; the retained head-tail subset does not establish an additional behavioral effect over B0.
- Query-aware filtering is separated from the main method table. It receives a perfect structured semantic form and reaches a realized ratio of 0.052, so it is a theoretical lower-bound reference rather than an evaluated deployable compressor.
- Positive residual warnings are summarized by method. The maximum warning request is 1,206 residual tokens, and the maximum residual fraction is 32.7%. The exact provider-side cause is not observable from response usage; cost claims use provider tokens and are unaffected, while channel-share claims retain this limitation.
