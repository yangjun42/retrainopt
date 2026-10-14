# Migration guide

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

