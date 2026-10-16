# Migration guide

## From 0.4.1 to 0.4.2

For sklearn classification, pass the task's known class labels with
`SklearnModelAdapter(estimator, classes=(0, 1))` when a candidate sample may
omit a class. The adapter aligns probability columns to this declared class
space. Declare it from the task specification or available training history,
not future evaluation labels. Estimators that require multiple training
classes still need a suitable training sample. Without `classes`, evaluation
rejects a one-class probability model or labels outside the fitted class set
consistently across supported sklearn versions.

Nonempty `bayes_backend_options` now raise `ValueError`: neither built-in
backend implemented these options. Remove the dictionary and use the explicit
`PlannerConfig` fields. The default empty dictionary remains accepted.

For Torch, `device="cuda"` now resolves to the current CUDA device before
training. Random-state protection and seeding follow that same device;
explicit device strings such as `"cuda:1"` retain their meaning.

## From 0.4.0 to 0.4.1

The public APIs are unchanged. After measuring the approximate planner's final
proposal, `bayes` now reoptimizes over all measured candidates. It can return a
different, cheaper schedule without additional candidate fits. `total_cost`
is the returned schedule's measured cost; `estimated_total_cost` evaluates that
same schedule using the estimates from before final measurement. The two values
can differ. Algorithm metadata advances to `algorithm_version="0.3"`.

Torch evaluation now restores every module's original training/evaluation mode,
including when evaluation raises an exception. Code that relied on evaluation
leaving a model in evaluation mode should call `model.eval()` explicitly.

See [Online retraining](online.md) for restoring a controller after a completed
step, including a controller that started from a synthetic loss baseline.

## From 0.3.1 to 0.4.0

The default evaluator retains its wall-clock training costs and zero initial
history defaults. New optional arguments support custom costs and models that
predate a historical window:

- Pass `training_cost_fn` to the planner, evaluation engine, historical helper,
  or online controller. Its return value is already in final objective units
  and replaces `time_cost_weight * training_seconds`; it is not multiplied by
  the time weight again. A planner cannot accept both this callback and an
  explicit `evaluator`.
- Set `initial_model_age` and `initial_accumulated_staleness` on
  `collect_online_training_data` to carry pre-window history. Supply the
  model's original reference loss. Local step numbers stay unchanged; model
  age may exceed the step number. Implicit bootstrap rejects nonzero carry.
- `OnlineController.initial_trained_at` now accepts negative integers for a
  model trained before the supplied timeline starts. The accumulated value
  covers observations before `start_step`; the controller adds the current
  observation before deciding.

Approximate search now preserves costs from an injected evaluator when
`time_cost_weight=0`. Previously, it incorrectly zeroed the training-cost
predictions for unobserved candidates. Plans using that combination can change.

Keep callback units and feature initialization consistent between historical
tuning and deployment. See {ref}`costs <custom-training-costs>` and
[online retraining](online.md) for examples and retry semantics.

## From 0.3.0 to 0.3.1

The planning and online policy APIs are unchanged. This patch rejects malformed
inputs that could previously produce a plausible but incorrect loss or cost.

- Prototype and sklearn class labels must be integers or finite integer-valued
  real numbers such as `1.0`. Fractional, boolean, string and non-finite labels
  raise `ValueError`; encode classes explicitly instead of relying on coercion.
- Prototype features must be finite real scalars or nonempty flat lists/tuples
  with one consistent dimension across training and evaluation.
- Custom evaluator records must use integer model IDs, excluding booleans, and
  the actual boolean `observed=True`; equal-valued floats and truthy strings
  are no longer accepted.

Base still makes periodic decisions without loss features, but the shared
`OnlineController` requires current evaluation labels to measure loss before
executing either policy. See [Online retraining](online.md) for this distinction
and for carrying an existing model's accumulated staleness into a controller.

## From 0.2 to 0.3

Version 0.3 adds the Base and Learned online policies alongside the existing
offline planner. `RetrainingPlanner.plan("exact")` and `plan("bayes")` still
use the full supplied timeline; choosing one of these methods does not turn
it into an online policy.

Use `collect_online_training_data(..., history_steps=...)` to train a policy
from a bounded historical prefix, then `OnlineController.step()` to process
each deployment step. Construct `BaseOnlinePolicy` from the historical plan,
or fit `LearnedOnlinePolicy` to its states and decisions. Learned requires the
`sklearn` extra. See [Online retraining](online.md) for a complete example.

Historical policy training returns a plan and training examples, not a
deployed model. When starting a controller with an existing model, pass its
`initial_reference_loss` and set the deployment `start_step` and
`initial_trained_at` explicitly. Do not also supply `PlannerConfig.initial_loss`.

## From 0.1 to 0.2

Version 0.2 retains `plan("exact")` and `plan("bayes")`, with the following
behavior and contract changes.

- Acquisition transforms the complete weighted stage cost. Variance uses
  squared loss weights, observed entries have zero uncertainty, and EI/PI use
  one stage-cost incumbent and the paper's zero-variance convention. Approximate
  search results can change even with the same seed.
- Configuration is frozen and the planner owns a snapshot. Construct a new
  `PlannerConfig` and planner to change the experiment; do not mutate an active
  planner's settings, timeline, model adapter, or evaluator state.
- Invalid configuration, malformed input data, and non-finite costs fail
  explicitly. Replace misspelled acquisition names and formerly accepted
  invalid sample counts rather than relying on silent fallback.
- `initial_loss` defines the initial baseline whenever supplied, including
  when the first training block is nonempty. It is exclusive with an explicit
  `initial_model`. See the [initial-model contract](evaluation.md).
- `PlanResult.loss_cost` remains the **raw** summed loss.
  `weighted_loss_cost` makes its objective contribution explicit.
  `to_dict()` includes `schema_version=2`; metadata has its own version marker.
  Consumers that enforce an exact dictionary schema must be updated.
- Supply an `EvaluationProvider` through the `evaluator=` constructor argument
  instead of monkey-patching `planner.engine` or calling private methods.

