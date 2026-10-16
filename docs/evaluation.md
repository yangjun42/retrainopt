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
| `PlanResult.training_cost` | Sum of selected retraining actions' costs in objective units. |
| `PlanResult.loss_cost` | Sum of selected models' raw losses over the timeline. |
| `PlanResult.weighted_loss_cost` | `loss_cost_weight * loss_cost`. |
| `PlanResult.total_cost` | `training_cost + weighted_loss_cost`. |
| `PlanResult.estimated_total_cost` | For Bayes, the returned schedule's cost using the table from before final measurement. |

The plan's training cost is not the expense of finding the plan. Exact search
trains all candidates; approximate search trains a subset and may perform
extra fits when measuring the final schedule or collecting diagnostic history.
`record_bayes_true_history=True` can add substantial diagnostic work. There is
no guarantee that a small candidate-selection setting caps total fit time.

After measuring the estimated proposal, Bayes reoptimizes over all measured
candidates. The returned plan can therefore differ from that proposal, and
every selected candidate has measured costs. This final optimization performs
no fits. `surrogate_evaluated_candidates` counts candidates observed before
final measurement; `evaluated_candidates` includes final measurement, while
`finalization_new_candidates` lists its additions. These counts exclude extra
diagnostic fits. Diagnostic history is collected after selection and cannot
change the returned plan or add candidates to the search cache.

(custom-training-costs)=
## Custom training costs

Pass `training_cost_fn` to `RetrainingPlanner`, `EvaluationEngine`,
`collect_online_training_data`, or `OnlineController` to change training-cost
accounting while retaining the built-in fitting and loss evaluation. The
`TrainingCostFunction` protocol accepts these keyword arguments:

```python
def training_cost_fn(*, step, n_samples, training_seconds, model) -> float:
    ...
```

Return a finite, nonnegative number in **final objective units**. This replaces
`time_cost_weight * training_seconds`; the time weight is not applied again.
`loss_cost_weight` still applies to raw losses. With no callback, the existing
wall-clock calculation remains the default. For example, a fixed cost per
sample can make a small experiment independent of timing noise:

```{doctest}
>>> from retrainopt import PlannerConfig, RetrainingPlanner
>>> from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter
>>> def sample_cost(*, step, n_samples, training_seconds, model):
...     return 0.125 * n_samples
>>> rows = [((0.0,), 0), ((1.0,), 1)]
>>> timeline = ArrayTimelineAdapter([rows], [rows])
>>> config = PlannerConfig(n_choices=(1.0,), initial_loss=1.0,
...                        time_cost_weight=999.0, loss_cost_weight=2.0)
>>> result = RetrainingPlanner(timeline, PrototypeModelAdapter(), config,
...                            training_cost_fn=sample_cost).plan("exact")
>>> (result.training_cost, result.weighted_loss_cost, result.total_cost)
(0.25, 0.0, 0.25)
```

The callback receives the artifact returned by `fit`, so it may read energy or
billing measurements stored by the adapter. It must not mutate that artifact.
It receives no timeline or future evaluation data. Historical tuning and
deployment should use the same cost definition; callbacks must not read future
data through external state or depend on the previously deployed model. The
offline recurrence assumes a candidate's training cost is independent of that
previous model.

The callback runs after a new candidate fit. Returning a cached evaluation does
not call it again; additional diagnostic fits can. KEEP decisions and initial
models, including compatibility bootstrap fits, have zero training cost and do
not invoke it. `training_seconds` continues to report fit wall-clock seconds,
whatever units the callback returns. A callback failure or invalid return value
raises an error; online controller state is committed only after the whole step
succeeds. Applications are responsible for any external callback side effects
when retrying.

## Custom evaluators

For precomputed costs, controlled timing, or external training code, implement
the public `EvaluationProvider` protocol and inject it:

```python
planner = RetrainingPlanner(
    timeline, model_adapter, config,
    evaluator=my_evaluator,
)
```

An explicit `evaluator` and `training_cost_fn` are mutually exclusive: a custom
provider already supplies the final training cost.

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
The provider returns training cost in final objective units and raw losses;
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
Treat the timeline, adapter, initial-model semantics, cost function, and custom
provider as unchanged during its lifetime. Create a new planner for a different
experiment.
This in-memory cache is not a persistent cross-experiment result store.

Serialized results use `schema_version=2`; metadata uses
`metadata_schema_version=1` and `algorithm_version="0.3"`. The algorithm version
identifies planning semantics independently of the package version; `0.3` adds
the final optimization over measured candidates.
Approximate runs record
`acquisition_semantics="complete_stage_independent_posteriors"`.

Invalid input or a failed fit/evaluation raises an error. Do not encode a
failed training run as a zero loss, an empty successful result, or an arbitrary
cost.

