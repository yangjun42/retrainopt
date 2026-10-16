# Online retraining

The Base and Learned policies make sequential retraining decisions using
observed history. They implement the online strategies described in
[*Automated Retraining for Cost-Optimal ML*](https://link.springer.com/chapter/10.1007/978-3-032-37654-1_14).

(online-history)=
## Train on history, then deploy

`collect_online_training_data(..., history_steps=H)` runs the offline planner
on the first `H` timesteps and produces an `OnlineTrainingData` object:

- `plan`: the historical offline plan.
- `states`: the observed states immediately before the plan's decisions.
- `decisions`: the corresponding keep/retrain decisions.

The helper only reads that historical prefix. Keep deployment steps outside
the prefix, then create an `OnlineController` with `start_step=H`. Each call to
`step()` observes the current timestep and executes one decision. The helper
does not return candidate models or select a model for deployment;
the controller's initial model is an explicit choice.

Historical tuning follows the offline planner's initial-baseline rules. If
both `initial_model` and `config.initial_loss` are omitted, the helper fits a
bootstrap model on the first historical step and uses its loss there as the
reference. The deployment controller does not perform this implicit fit.

To tune on a later historical window, supply that window as the timeline and
set `initial_model_age` to the model's age at its first step. Pass
`initial_accumulated_staleness` for the signed sum observed since the last
retraining and **before** this window. Both default to zero. For example:

```python
history = collect_online_training_data(
    historical_window, model_adapter, config,
    history_steps=3,
    initial_model=existing_model,
    initial_reference_loss=0.2,
    initial_model_age=5,
    initial_accumulated_staleness=0.7,
)
```

The reference is this model's loss measured at its training time, not its loss
at the start of the window. If its first current loss is `0.1`, the first
pre-action features are approximately `(-0.1, 0.6, 5)`. Carry changes the
training features, not the offline objective. An implicit bootstrap creates a
new model, so it rejects nonzero age or accumulated staleness.

The helper still reads only `[0, history_steps)` of the supplied timeline, and
states and decisions retain those local step numbers. Age can exceed the local
step number. Window slicing is the caller's responsibility; renumbering steps
changes the sampling seeds, so a sliced run need not draw the same samples as
an unsliced run.

This example calibrates Base on three historical steps and processes the next
three steps sequentially. Training and evaluation use disjoint toy samples;
the initial model's reference loss is also measured on evaluation samples.
The same initial model is supplied to calibration and deployment. This small
example demonstrates the API, not an estimate of real-world performance:

```{literalinclude} ../examples/online_timeline.py
:language: python
```

An existing deployed model requires its measured `initial_reference_loss`.
`initial_trained_at` records the timestep of its last retraining;
`initial_accumulated_staleness` can carry previously observed staleness into
the controller. When `start_step > initial_trained_at`, supply the signed
staleness accumulated since that model's last retraining and before
`start_step`, or intentionally reset this feature by leaving its default of
zero. Negative `initial_trained_at` values represent training before the
supplied timeline starts: `start_step=0, initial_trained_at=-5` gives initial
age five. The controller does not reconstruct earlier observations automatically.

As an alternative to supplying a model, `PlannerConfig.initial_loss` defines
a synthetic constant-loss baseline until the controller first retrains. This
baseline is useful for controlled examples; it is not a fitted model. It also
accepts explicit initial age and accumulated staleness. An explicit model and
`initial_loss` are mutually exclusive.

## Resume a controller

After a successful step, save the current model, reference loss, last retraining
step, next step and accumulated staleness together. Also preserve the fitted
policy, configuration, training-cost function and the settings needed to
reconstruct the model adapter and timeline. Model and policy serialization,
and writing a consistent checkpoint, are the application's responsibility;
`OnlineStepResult.to_dict()` alone is not a checkpoint.

The following reconstructs a controller in the same process. After a restart,
use the corresponding loaded objects and saved values:

```python
from dataclasses import replace
from retrainopt import OnlineController

model = controller.current_model
resume_config = controller.config
if model is not None:
    resume_config = replace(resume_config, initial_loss=None)

saved_state = {
    "initial_model": model,
    "initial_reference_loss": controller.reference_loss if model is not None else None,
    "start_step": controller.next_step,
    "initial_trained_at": controller.last_retrain_step,
    "initial_accumulated_staleness": controller.accumulated_staleness,
}
restored = OnlineController(
    controller.timeline,
    controller.model_adapter,
    controller.policy,
    resume_config,
    training_cost_fn=controller.training_cost_fn,
    **saved_state,
)
```

Keep the original timeline coordinates: saved step `7` must still identify the
same data block after restoration. This preserves model age and sampling seeds.
Use the saved reference loss and signed accumulated staleness without measuring
a new reference or adding the last completed observation again.

If the controller started with a synthetic `initial_loss`, its configuration
still contains that value after the first retraining. Clear it in the restored
configuration when supplying the trained model, as above. Before the first
retraining, retain `initial_loss` and pass neither a model nor a reference loss.

## Base policy

`BaseOnlinePolicy.from_plan(plan)` computes the mean retraining sample count
and the interval $\tau=T/R$, where $T$ is the historical plan's length and $R$
is its number of retraining decisions. The policy requests retraining when
the elapsed time since the last retraining reaches $\tau$.

The implementation keeps $\tau$ as a floating-point value and compares integer
elapsed timesteps against it. The mean sample count is rounded to the nearest
integer, with halves rounded upward, then clipped to the number of currently
available training samples. A historical plan with no retraining decisions
produces a policy that always keeps the model. These are explicit engineering
conventions for applying the paper's strategy to discrete timesteps and sample
counts.

## Learned policy

Install the `sklearn` extra, then replace the Base policy construction with:

```python
from retrainopt import LearnedOnlinePolicy

policy = LearnedOnlinePolicy(random_seed=42).fit(history.states, history.decisions)
```

Learned uses current staleness, accumulated staleness and time since the last
retraining as features. Its 100-tree random forest classifier learns keep or
retrain from all historical decisions. A separate 100-tree random forest
regressor learns sample counts from retraining decisions only. The classifier
follows the paper's specification; the regressor family is a library choice.

If the training history has no retraining decisions, Learned keeps the model
and does not fit a sample-count regressor. Predicted sample counts use the
same rounding and clipping convention as Base; they are not snapped to the
offline planner's candidate grid. Staleness can be negative when the current
model improves relative to its reference loss; it is not clipped to zero.

## Observations, costs and assumptions

Current staleness is the raw loss difference $L_t(m)-L_{\mathrm{reference}}(m)$.
Accumulated staleness adds these signed differences since the last retraining.
The library uses raw loss for these features; loss weights affect the measured
objective only. The controller keeps the current model when fewer than `config.min_train_size`
training samples are available.

`OnlineStepResult.state` contains the features before the action, including
the current observation in accumulated staleness. The remaining fields record
the executed decision, raw loss, training time, training cost in objective units,
weighted loss cost and their sum `total_cost`. After a successful retraining,
the new model's current loss becomes its reference loss and accumulated
staleness resets for the next timestep.

By default, the online controller and historical training helper use fit
wall-clock time multiplied by `time_cost_weight` as training cost. Both accept
`training_cost_fn` to return a custom cost in final objective units, replacing
that calculation. Use the same callback for tuning and deployment; see
{ref}`Custom training costs <custom-training-costs>` for its signature,
units, caching and failure contract. KEEP has zero training cost. Fit timing
excludes sampling, evaluation and policy inference; those expenses are also
excluded from the default objective. A retraining request below
`config.min_train_size` is rejected when enough data are available; an
undersized prediction is not silently expanded.

Applications can implement `OnlinePolicy.decide(state, *, train_size)` to
return a `Decision`. This method must not mutate policy or deployment state.
Model fitting must return a new artifact, and evaluation and cost callbacks
must not mutate model artifacts. The controller advances only after a
successful step; an unavailable timestep raises `IndexError` and can be retried
when the timeline grows.

The Learned policy uses staleness derived from the current model's observed
predictive loss. The Base policy itself needs only elapsed time and the
available training size; its decision does not depend on current loss.
The shared `OnlineController` evaluates current loss before calling either
policy so that it can maintain features and report costs. When using a real
deployed model with this controller, labels must therefore be available for
the current evaluation observations. Applications that need Base decisions
without labels can call `BaseOnlinePolicy.decide` directly and manage execution
and measurement separately. Delayed-label handling and the cost of acquiring
labels require an application-level policy.

The policies use a historical prefix to calibrate deployment decisions. Their
usefulness depends on how well that history represents subsequent drift and
training costs. They do not provide an online optimality or competitive-ratio
guarantee.
