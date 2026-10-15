# Initial models, costs and evaluation

## Initial-model contract

The initial model represents the model already available before the first
decision. Its historical training cost is outside the planning horizon.
Choose one of these forms:

1. Pass `initial_model=artifact` to `plan()`. The model adapter evaluates it on
   every evaluation block.
2. Set `PlannerConfig(initial_loss=...)` for an explicit constant baseline
   loss at every timestep. This works even when training data exist at step 0.
3. Leave both unset to use the compatibility bootstrap: fit a model on step 0
   using `initial_model_samples` or the full first block. Its fit is outside
   the schedule's objective. Prefer an explicit artifact or baseline when
   comparing algorithms or accounting for all work.

`initial_loss` and an explicit `initial_model` are mutually exclusive. The
bootstrap needs training data at step 0. A timeline with no retrain candidates
can still produce an all-keep plan when an initial model or baseline is valid.
Result metadata records `initial_model_source`: `provided_initial_model`,
`configured_initial_loss`, `trained_on_step0`, or `provider_supplied` for a
custom provider without its own source label. Initial-model records carry zero
training cost; they do not account for bootstrap elapsed time.

## Units and accounting

The default evaluator measures `ModelAdapter.fit()` with a wall clock.
Sampling and later evaluations are outside that fit-time measurement. Timing
depends on hardware, process load, synchronization and the adapter's work;
fixed seeds alone do not make it deterministic.

| Value | Meaning |
| --- | --- |
| `CandidateEvaluation.training_seconds` | Raw seconds measured for a candidate's fit. |
| `CandidateEvaluation.training_cost` | Training cost in objective units; the default is `time_cost_weight * training_seconds`. |
| `CandidateEvaluation.future_losses[t]` | Raw modeling loss at evaluation time `t`. |
| `PlanResult.training_cost` | Sum of selected retraining actions' weighted training costs. |
| `PlanResult.loss_cost` | Sum of selected models' raw losses over the timeline. |
| `PlanResult.weighted_loss_cost` | `loss_cost_weight * loss_cost`. |
| `PlanResult.total_cost` | `training_cost + weighted_loss_cost`. |

The plan's training cost is not the expense of finding the plan. Exact search
trains all candidates; approximate search trains a subset and may perform
extra fits when measuring the final schedule or collecting diagnostic history.
`record_bayes_true_history=True` can add substantial diagnostic work. There is
no guarantee that a small candidate-selection setting caps total fit time.

## Custom evaluators

For precomputed costs, controlled timing, or external training code, implement
the public `EvaluationProvider` protocol and inject it:

```python
planner = RetrainingPlanner(
    timeline, model_adapter, config,
    evaluator=my_evaluator,
)
```

The provider exposes:

```python
evaluate_initial_model(initial_model=None) -> CandidateEvaluation
evaluate_candidate(step, n_samples) -> CandidateEvaluation
evaluate_candidate_uncached(step, n_samples) -> CandidateEvaluation
iter_cached_candidates() -> Iterable[CandidateEvaluation]
```

The initial record has ID `(-1, 0)`, zero training cost, and losses for the
entire timeline. A candidate `(s, n)` has losses for every timestep from `s`
through the end. Losses and measured costs must be finite and nonnegative.
The provider returns weighted training cost and raw losses;
RetrainOpt applies the loss weight. `iter_cached_candidates()` lists observed
candidate records, excluding the initial model. The uncached method supports
diagnostics without adding a new observation to the search cache.
If `initial_loss` is configured, the provider's initial losses must match that
baseline at every timestep.

Model IDs must be pairs of integers (excluding booleans), including on records
returned for a requested candidate. An equal-valued float is not an integer ID.
Every provider record must set `observed=True` as a boolean; strings such as
`"false"` and numeric flags are rejected.

See `examples/verify_exact_against_bruteforce.py` for a complete fixed-table
provider and independent exhaustive verification. No subclass of the default
evaluation engine is needed.

## Lifetime and errors

A planner uses a configuration snapshot and a cache scoped to that planner.
Treat the timeline, adapter, initial-model semantics, and custom provider as
unchanged during its lifetime. Create a new planner for a different experiment.
This in-memory cache is not a persistent cross-experiment result store.

Serialized results use `schema_version=2`; metadata uses
`metadata_schema_version=1` and `algorithm_version="0.2"`. The algorithm version
identifies cost and acquisition semantics independently of the package version.
Approximate runs record
`acquisition_semantics="complete_stage_independent_posteriors"`.

Invalid input or a failed fit/evaluation raises an error. Do not encode a
failed training run as a zero loss, an empty successful result, or an arbitrary
cost.

