# Offline Phase 1 Coverage Matrix

This document records what the clean `retrainopt` Phase 1 harness covers and what is intentionally deferred.

## Covered In Phase 1

- Offline exact planning (`exact`)
- Offline approximate planning (`bayes`) with lightweight and BoTorch backends
- Synthetic sudden / gradual / recurring timelines used by the offline paper studies
- Real binary-timeline datasets consumed through `BinaryTimelineAdapter`
- Deterministic disk caching for initial-model and candidate evaluations
- Normalized run summaries for total cost, training cost, loss cost, candidate counts, cache fingerprint, and schedule output
- Puhti batch-script rendering for clean offline runs
- Comparison reporting against latest-paper reference metrics

## Explicitly Deferred To Phase 2

- Heuristic baselines such as Periodic, Never Retrain, Always Retrain, ADWIN, and DDM
- Timing-only cost-aware baselines such as Markov, Threshold, Cumulative Threshold, and UPF
- Online policy training and deployment (`base`, `learned`)
- Online figures and online-policy acceptance checks

## Source-Of-Truth Boundary

- `retrainopt` is the clean execution repo for Phase 1.
- `CARA-Extension` remains reference-only during the port and should not be imported by clean Phase 1 runtime code.
- Acceptance is against the latest paper conclusions rather than the originally submitted PDF.
