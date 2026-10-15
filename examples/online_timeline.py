"""Train the Base policy on history, then execute one timestep at a time."""

from retrainopt import (
    BaseOnlinePolicy,
    OnlineController,
    PlannerConfig,
    collect_online_training_data,
)
from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter


def run(policy_kind="base"):
    stable = [((0.0,), 0), ((1.0,), 1)]
    drifted = [((0.0,), 1), ((1.0,), 0)]
    blocks = [stable, drifted, drifted, stable, stable, drifted]
    timeline = ArrayTimelineAdapter(train_blocks=blocks, eval_blocks=blocks)
    adapter = PrototypeModelAdapter()
    config = PlannerConfig(n_choices=(1.0,), time_cost_weight=0.0)

    # Fit an initial model and measure its reference loss.
    initial_model = adapter.fit(stable, step=0, n_samples=2, random_seed=42)
    reference_loss = adapter.evaluate(initial_model, stable, step=0)
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
    for _ in range(3):
        result = controller.step()
        print(result.state.step, result.decision.action.value, result.decision.n_samples)
        print(f"Measured step objective: {result.total_cost:.6f}")


if __name__ == "__main__":
    run()
