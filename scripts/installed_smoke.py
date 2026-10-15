"""Exercise an installed distribution without relying on the source checkout."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.resources
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import retrainopt
from retrainopt import (
    BaseOnlinePolicy, OnlineController, PlannerConfig, RetrainingPlanner,
    collect_online_training_data,
)
from retrainopt.adapters import ArrayTimelineAdapter, PrototypeModelAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    version = importlib.metadata.version("retrainopt")
    assert retrainopt.__version__ == version
    if args.expected_version:
        assert version == args.expected_version, (version, args.expected_version)
    if args.source_root:
        assert not Path(retrainopt.__file__).resolve().is_relative_to(args.source_root.resolve())
    assert importlib.resources.files("retrainopt").joinpath("py.typed").is_file()
    distribution = importlib.metadata.distribution("retrainopt")
    assert any(str(path).endswith("licenses/LICENSE") for path in distribution.files)
    assert distribution.metadata["License-Expression"] == "Apache-2.0"
    blocks = [[((0.0,), 0), ((1.0,), 1)], [((0.2,), 0), ((0.8,), 1)]]
    timeline = ArrayTimelineAdapter(train_blocks=blocks * 2, eval_blocks=blocks * 2)
    costs = {}
    for method in ("exact", "bayes"):
        result = RetrainingPlanner(
            timeline, PrototypeModelAdapter(),
            config=PlannerConfig(
                n_choices=(1, 1.0), time_cost_weight=0.0, loss_cost_weight=2.0,
                initial_loss=1.0, bayes_anchor_steps=(0.0,),
                bayes_initial_samples=2, bayes_iterations=1,
            ),
        ).plan(method=method)
        assert len(result.decisions) == 4
        assert math.isfinite(result.total_cost)
        assert math.isclose(result.total_cost, result.training_cost + result.weighted_loss_cost)
        json.dumps(result.to_dict(), allow_nan=False)
        if method == "bayes":
            assert result.metadata["bayes_iterations_completed"] == 1
        costs[method] = result.total_cost
    assert costs["exact"] == costs["bayes"]
    def fixed_training_cost(*, step, n_samples, training_seconds, model):
        assert n_samples == 2 and training_seconds >= 0 and model is not None
        return 0.25

    online_config = PlannerConfig(n_choices=(1.0,), initial_loss=1.0, time_cost_weight=100.0)
    history = collect_online_training_data(
        timeline, PrototypeModelAdapter(), online_config, history_steps=2,
        initial_model_age=4, initial_accumulated_staleness=-0.5,
        training_cost_fn=fixed_training_cost,
    )
    assert history.states[0].features() == [0.0, -0.5, 4.0]
    assert history.plan.training_cost == 0.25
    online = OnlineController(
        timeline, PrototypeModelAdapter(), BaseOnlinePolicy.from_plan(history.plan),
        online_config, start_step=2, initial_trained_at=-4,
        initial_accumulated_staleness=-0.5, training_cost_fn=fixed_training_cost,
    )
    online_results = [online.step(), online.step()]
    assert [result.state.step for result in online_results] == [2, 3]
    assert online_results[0].state.features() == [0.0, -0.5, 6.0]
    assert online_results[0].training_cost == 0.25
    assert online_results[1].state.features() == [0.0, 0.0, 1.0]
    for result in online_results:
        assert math.isfinite(result.total_cost)
        assert math.isclose(result.total_cost, result.training_cost + result.weighted_loss_cost)
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run([sys.executable, "-m", "retrainopt", "--help"], cwd=directory, check=True, stdout=subprocess.DEVNULL)
    print(json.dumps({"version": version, "module": retrainopt.__file__, "costs": costs}))


if __name__ == "__main__":
    main()
