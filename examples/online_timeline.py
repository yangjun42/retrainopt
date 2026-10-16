"""Calibrate on history, then deploy using separate train/evaluation samples."""

from retrainopt import (
    BaseOnlinePolicy,
    OnlineController,
    PlannerConfig,
    collect_online_training_data,
)
from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter


def make_timeline():
    # Each pair gives the left/right class labels at one timestep. Evaluation
    # points never appear in training, including the initial model's fit.
    regimes = [(0, 1), (1, 0), (1, 0), (0, 1), (0, 1), (1, 0)]
    train_blocks = [[((0.0,), left), ((1.0,), right)] for left, right in regimes]
    eval_blocks = [[((0.1,), left), ((0.9,), right)] for left, right in regimes]
    return ArrayTimelineAdapter(train_blocks=train_blocks, eval_blocks=eval_blocks)


def run(policy_kind="base"):
    timeline = make_timeline()
    adapter = PrototypeModelAdapter()
    config = PlannerConfig(n_choices=(1.0,), time_cost_weight=0.0)

    # Fit on training samples; measure the reference on held-out samples.
    initial_model = adapter.fit(
        timeline.get_train_block(0), step=0,
        n_samples=timeline.train_size(0), random_seed=42,
    )
    reference_loss = adapter.evaluate(initial_model, timeline.get_eval_block(0), step=0)
    history = collect_online_training_data(
        timeline, adapter, config, history_steps=3,
        initial_model=initial_model, initial_reference_loss=reference_loss,
    )
    if policy_kind == "learned":
        from retrainopt import LearnedOnlinePolicy

        policy = LearnedOnlinePolicy().fit(history.states, history.decisions)
    else:
        policy = BaseOnlinePolicy.from_plan(history.plan)

    # Deploy the initial model; policy training does not deploy historical candidates.
    accumulated = sum(
        adapter.evaluate(initial_model, timeline.get_eval_block(step), step=step) - reference_loss
        for step in range(3)
    )
    controller = OnlineController(
        timeline, adapter, policy, config, start_step=3,
        initial_model=initial_model, initial_reference_loss=reference_loss,
        initial_trained_at=0,
        initial_accumulated_staleness=accumulated,
    )
    results = []
    for _ in range(3):
        result = controller.step()
        results.append(result)
        print(result.state.step, result.decision.action.value, result.decision.n_samples)
        print(f"Measured step objective: {result.total_cost:.6f}")
    return results


if __name__ == "__main__":
    run()
