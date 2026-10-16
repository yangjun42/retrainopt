# Changelog

## 0.4.1 — 2026-10-16

- Restore each Torch module's original training/evaluation mode after evaluation,
  including mixed module modes and exceptions.
- Reoptimize the final approximate schedule over all measured candidates after
  final measurement, without starting another round of candidate training.
  Report the returned schedule's cost and its estimate consistently.
- Document controller recovery, including the transition from a synthetic
  initial baseline to a trained model.
- Add an interactive demo with synthetic cost tables and schedules generated
  by the package's exact planner, alongside fixed retraining baselines.

## 0.4.0 — 2026-10-15

- Add `training_cost_fn` to offline planning, historical policy training and
  online execution. Custom costs replace weighted fit time and use the same
  objective units in all three paths.
- Preserve custom training-cost predictions in approximate search even when
  `time_cost_weight=0`, including costs supplied by an external evaluator.
- Allow historical policy training to start with a model's existing age and
  signed accumulated staleness. Deployment can also start with a model trained
  before the supplied timeline.
- Validate cost callbacks and initial history, and test cost accounting,
  failure retries, bounded data access and state continuity across windows.
- Make copyright attribution explicit in `LICENSE` and clarify academic
  citation guidance.
- Consolidate publishing instructions in the development guide and remove the
  redundant `setup.py` entry point.
- Simplify the README and remove unrelated design comparisons from the
  algorithm documentation.

## 0.3.1 — 2026-10-14

### Correctness and validation

- Reject fractional, non-finite, boolean and string class labels in the
  Prototype and sklearn adapters instead of silently converting them to integers.
  Integer-valued real labels remain supported.
- Reject non-finite, empty or inconsistent feature vectors in the Prototype
  adapter, including mismatches between training and evaluation dimensions.
- Validate the actual model IDs returned by custom evaluators and require a
  boolean `observed=True` flag.
- Exercise real GP fitting and predictions for unobserved candidates in the
  BoTorch integration test, and verify online continuation from nonzero history
  against uninterrupted execution.

### Documentation

- Distinguish Base's label-independent decision rule from the controller's
  current-loss measurement requirement.
- Clarify online cost and historical initialization limitations, and describe
  the approximate planner's implementation choices relative to the paper.
- Keep the simpler retraining-schedule introduction and citation guidance.

## 0.3.0 — 2026-10-14

### Online policies

- Add the paper's periodic Base and random-forest Learned online policies,
  using current staleness, accumulated staleness and time since retraining.
- Derive policy training examples from an explicitly bounded historical
  timeline. Online execution measures only the current step and retains the
  deployed model between calls.
- Keep exact and approximate offline planning available with their existing
  interfaces and objective semantics.

### Documentation

- Document online training and execution with runnable examples.
- Link the official ECML PKDD proceedings chapter and provide its verified
  publisher citation, alongside a software citation.

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

## 0.1.0 (unpublished)

Initial standalone library with exact and approximate planning, timeline/model
adapters, and a binary-timeline CLI.

