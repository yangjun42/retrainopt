# Changelog

## 0.2.0 — 2026-10-14

### Correctness

- Apply acquisition functions to complete weighted per-stage costs. Combine
  independent training/loss posterior variances with squared weights.
- Use zero uncertainty for measured costs, one observed stage-cost incumbent
  for EI/PI, and the paper's zero-standard-deviation EI/PI convention.
- Honor an explicit initial-loss baseline even with a nonempty first block.
- Validate configuration, timeline inputs and evaluated costs instead of
  silently substituting defaults for invalid values.

### Public API

- Add evaluator injection through `EvaluationProvider` and publicly expose
  `CandidateEvaluation`.
- Freeze planner configuration and isolate planner configuration snapshots.
- Add `PlanResult.weighted_loss_cost`; preserve the raw `loss_cost` field.
  Serialized results use schema version 2 and versioned metadata.

### Distribution

- Add an algorithm guide, public API reference, migration notes and runnable
  documentation, plus optional scikit-learn, PyTorch and BoTorch examples.
- Document source, installed-package and documentation validation, and a
  staged release procedure.

Approximate plans can change under the corrected acquisition semantics.
The frozen paper implementation and canonical experiment pin are unchanged.

## 0.1.0 (unpublished)

Initial standalone library with exact and approximate planning, timeline/model
adapters, and a binary-timeline CLI.

