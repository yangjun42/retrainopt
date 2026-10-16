"""Exercise the public online example with real models and policies."""

import importlib.util
from pathlib import Path

import pytest

from retrainopt import DecisionAction
from retrainopt.adapters import PrototypeModelAdapter


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "online_timeline.py"
SPEC = importlib.util.spec_from_file_location("online_example", EXAMPLE_PATH)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def test_example_training_and_evaluation_samples_are_disjoint():
    timeline = example.make_timeline()
    training = [row for step in range(timeline.num_steps) for row in timeline.get_train_block(step)]
    evaluation = [row for step in range(timeline.num_steps) for row in timeline.get_eval_block(step)]
    assert timeline.num_steps == 6
    assert {id(row) for row in training}.isdisjoint(id(row) for row in evaluation)
    assert {features for features, _ in training}.isdisjoint(features for features, _ in evaluation)
    for step in range(timeline.num_steps):
        assert timeline.get_train_block(step) is not timeline.get_eval_block(step)
        assert len(timeline.get_eval_block(step)) == 2


def test_example_reference_and_deployment_use_the_explicit_initial_model(monkeypatch, capsys):
    class RecordingAdapter(PrototypeModelAdapter):
        def __init__(self):
            self.fits = []
            self.evaluations = []

        def fit(self, train_data, **kwargs):
            model = super().fit(train_data, **kwargs)
            self.fits.append((model, tuple(train_data)))
            return model

        def evaluate(self, model, eval_data, **kwargs):
            self.evaluations.append((model, tuple(eval_data), kwargs["step"]))
            return super().evaluate(model, eval_data, **kwargs)

    adapter = RecordingAdapter()
    monkeypatch.setattr(example, "PrototypeModelAdapter", lambda: adapter)
    collect = example.collect_online_training_data
    calibrated_models = []

    def record_calibration(*args, **kwargs):
        calibrated_models.append(kwargs["initial_model"])
        first = len(adapter.evaluations)
        history = collect(*args, **kwargs)
        assert len(history.states) == len(history.decisions) == kwargs["history_steps"] == 3
        assert all(step < 3 for _, _, step in adapter.evaluations[first:])
        assert kwargs["initial_reference_loss"] == 0.0
        return history

    monkeypatch.setattr(example, "collect_online_training_data", record_calibration)
    example.run()
    initial_model, initial_train = adapter.fits[0]
    reference_model, reference_rows, reference_step = adapter.evaluations[0]
    assert reference_model is initial_model
    assert reference_step == 0
    assert {features for features, _ in initial_train}.isdisjoint(features for features, _ in reference_rows)
    assert len(calibrated_models) == 1
    assert calibrated_models[0] is initial_model
    first_deployment_model = next(model for model, _, step in adapter.evaluations if step == 3)
    assert first_deployment_model is initial_model
    # Every actual fit and evaluation, not only the initially built timeline,
    # must retain the split. All classifier calculations above are real.
    fit_features = {features for _, rows in adapter.fits for features, _ in rows}
    eval_features = {features for _, rows, _ in adapter.evaluations for features, _ in rows}
    assert fit_features.isdisjoint(eval_features)
    assert "Measured step objective:" in capsys.readouterr().out


def test_base_example_runs_a_periodic_policy_on_held_out_samples(capsys):
    results = example.run("base")
    assert [result.state.step for result in results] == [3, 4, 5]
    assert [result.decision.action for result in results] == [
        DecisionAction.RETRAIN, DecisionAction.KEEP, DecisionAction.KEEP,
    ]
    assert results[0].decision.n_samples == 2
    assert [result.loss for result in results] == [0.0, 0.0, 1.0]
    # Carry describes the same initial model kept during the historical period.
    assert results[0].state.accumulated_staleness == 2.0
    assert results[0].state.time_since_last_retrain == 3
    capsys.readouterr()


def test_learned_example_reacts_to_the_held_out_label_reversal(capsys):
    pytest.importorskip("sklearn")
    results = example.run("learned")
    assert [result.state.step for result in results] == [3, 4, 5]
    assert results[-1].state.current_staleness == 1.0
    assert results[-1].decision.action == DecisionAction.RETRAIN
    assert results[-1].decision.n_samples == 2
    assert results[-1].loss == 0.0
    assert any(result.decision.action == DecisionAction.KEEP for result in results)
    capsys.readouterr()
