"""Public demo provenance and accounting, without optional dependencies."""

import importlib.util
import itertools
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_demo", ROOT / "examples" / "build_demo.py")
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


@pytest.fixture(scope="module")
def artifact():
    return json.loads(demo.OUTPUT.read_text(encoding="utf-8"))


def test_demo_covers_every_selectable_setting(artifact):
    actual = [(case["scenario"], case["quality_weight"], case["training_multiplier"])
              for case in artifact["cases"]]
    expected = set(itertools.product(
        (item["id"] for item in artifact["scenarios"]),
        artifact["quality_weights"], artifact["training_multipliers"],
    ))
    assert len(actual) == len(set(actual)) == 60
    assert set(actual) == expected
    assert artifact["package_version"] == demo.__version__


def test_demo_accounting_and_fixed_policy_contract(artifact):
    for case in artifact["cases"]:
        methods = case["policies"]
        assert methods["periodic"]["actions"] == [100 if step in (6, 12, 18) else 0 for step in range(24)]
        assert methods["always"]["actions"] == [0] + [100] * 23
        assert methods["never"]["actions"] == [0] * 24
        for plan in methods.values():
            assert len(plan["actions"]) == len(plan["step_losses"]) == len(plan["step_training_costs"]) == 24
            assert plan["actions"][0] == 0
            assert plan["step_training_costs"][0] == 0
            assert plan["step_losses"][0] == 0.04
            assert set(plan["actions"]) <= {0, 25, 50, 75, 100}
            assert plan["training_cost"] == pytest.approx(sum(plan["step_training_costs"]), abs=1e-6)
            assert plan["raw_loss"] == pytest.approx(sum(plan["step_losses"]), abs=1e-6)
            assert plan["weighted_loss_cost"] == pytest.approx(case["quality_weight"] * plan["raw_loss"], abs=1e-6)
            assert plan["total_cost"] == pytest.approx(plan["training_cost"] + plan["weighted_loss_cost"], abs=1e-6)
            assert plan["retrain_count"] == sum(n > 0 for n in plan["actions"])
            assert plan["total_samples"] == sum(plan["actions"])
            assert all(math.isfinite(value) and value >= 0 for value in plan["step_losses"] + plan["step_training_costs"])
            assert methods["optimized"]["total_cost"] <= plan["total_cost"] + 1e-6


@pytest.mark.parametrize("scenario,quality,price,expected_actions", [
    ("sudden", 2.0, 1.0, {8: 100, 17: 75}),
    ("sudden", 0.5, 4.0, {8: 25, 17: 25}),
    ("recurring", 0.5, 4.0, {}),
])
def test_demo_examples_are_real_package_plans(artifact, scenario, quality, price, expected_actions):
    # Recompute representative cases through the actual public planner. These
    # fixed examples cover full/partial fits and a genuinely keep-only optimum.
    result = demo.build_case(scenario, quality, price)
    optimized = result["policies"]["optimized"]
    assert {t: n for t, n in enumerate(optimized["actions"]) if n} == expected_actions
    stored = next(case for case in artifact["cases"] if (
        case["scenario"], case["quality_weight"], case["training_multiplier"]
    ) == (scenario, quality, price))
    for name, plan in result["policies"].items():
        assert plan["actions"] == stored["policies"][name]["actions"]
        assert plan["total_cost"] == pytest.approx(stored["policies"][name]["total_cost"], abs=1e-7)


def test_supplied_costs_include_a_known_drift_penalty():
    evaluator = demo.DemoEvaluator(demo.regime("sudden"), 1.0)
    initial = evaluator.evaluate_initial_model()
    small = evaluator.evaluate_candidate(8, 25)
    full = evaluator.evaluate_candidate(8, 100)
    assert initial.future_losses[8] == 0.68291304
    assert small.future_losses[8] == 0.13
    assert full.future_losses[8] == 0.04
    assert small.training_cost < full.training_cost
    assert evaluator.evaluate_candidate(8, 25) is small
