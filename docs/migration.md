# Migrating from 0.1

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

The published paper's frozen implementation and the canonical reproduction
pin are unchanged. Keep the pinned release when validating historical paper
artifacts. Version 0.2 is a maintained library release, not a replacement for
the original experiment environment.

