# Online retraining

The Base and Learned policies make sequential retraining decisions using
observed history. They implement the online strategies described in
[*Automated Retraining for Cost-Optimal ML*](https://link.springer.com/chapter/10.1007/978-3-032-37654-1_14).

## Train on history, then deploy

`collect_online_training_data(..., history_steps=H)` runs the offline planner
on the first `H` timesteps and produces an `OnlineTrainingData` object:

- `plan`: the historical offline plan.
- `states`: the observed states immediately before the plan's decisions.
- `decisions`: the corresponding keep/retrain decisions.

The helper only reads that historical prefix. Keep deployment steps outside
the prefix, then create an `OnlineController` with `start_step=H`. Each call to
`step()` observes the current timestep and executes one decision. The helper
does not return candidate models or silently select a model for deployment;
the controller's initial model is an explicit choice.

Historical tuning follows the offline planner's initial-baseline rules. If
both `initial_model` and `config.initial_loss` are omitted, the helper fits a
bootstrap model on the first historical step and uses its loss there as the
reference. The deployment controller does not perform this implicit fit.

This example calibrates Base on three historical steps and processes the next
three steps sequentially:

```{literalinclude} ../examples/online_timeline.py
:language: python
```

An existing deployed model requires its measured `initial_reference_loss`.
`initial_trained_at` records the timestep of its last retraining;
`initial_accumulated_staleness` can carry previously observed staleness into
the controller. It defaults to zero.

As an alternative to supplying a model, `PlannerConfig.initial_loss` defines
a synthetic constant-loss baseline until the controller first retrains. This
baseline is useful for controlled examples; it is not a fitted model. An
explicit model and `initial_loss` are mutually exclusive.

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
Loss weights affect the measured objective, not these policy features. The
controller keeps the current model when fewer than `config.min_train_size`
training samples are available.

`OnlineStepResult.state` contains the features before the action, including
the current observation in accumulated staleness. The remaining fields record
the executed decision, raw loss, training time, weighted training cost,
weighted loss cost and their sum `total_cost`. After a successful retraining,
the new model's current loss becomes its reference loss and accumulated
staleness resets for the next timestep.

Training time measures the model fit only. Sampling, evaluation and policy
inference time are excluded from this objective. A retraining request below
`config.min_train_size` is rejected when enough data are available; an
undersized prediction is not silently expanded.

Applications can implement `OnlinePolicy.decide(state, *, train_size)` to
return a `Decision`. This method must not mutate policy or deployment state.
Model fitting must return a new artifact, and evaluation must not mutate the
deployed model. The controller advances only after a successful step; an
unavailable timestep raises `IndexError` and can be retried when the timeline
grows.

Both policies use the current model's observed predictive loss. An application
therefore needs labels for the corresponding evaluation observations before
making that decision. Delayed-label handling and the cost of acquiring those
labels require an application-level policy.

The policies use a historical prefix to calibrate deployment decisions. Their
usefulness depends on how well that history represents subsequent drift and
training costs. They do not provide an online optimality or competitive-ratio
guarantee.
