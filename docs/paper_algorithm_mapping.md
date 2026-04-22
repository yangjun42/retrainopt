# Paper Algorithm Mapping

`retrainopt` is the public algorithm package behind the paper's offline
planning claims. The companion repro repo is responsible for manifests, data
staging, Puhti automation, and paper-specific acceptance checks.

## Core Mapping

- Oracle planner: `RetrainingPlanner.plan(method="exact")`
  This is the exact dynamic-programming solver over the configured
  retraining candidates.
- Approximate planner: `RetrainingPlanner.plan(method="bayes")`
  This is the surrogate-guided search path that exactly re-evaluates the
  selected decision sequence before returning `total_cost`.
- Timeline abstraction: `TimelineDataset`
  This is the paper's time-sliced dataset interface.
- Candidate training and evaluation: `ModelAdapter`
  This is the reusable interface for fitting a candidate model at one
  timestep and scoring it on future evaluation slices.
- Search space and objective weights: `PlannerConfig`
  This carries the `n_choices`, train-size floor, random seed, and the
  time-vs-loss cost weights used by the planner.
- Decision output: `PlanResult`
  This records the chosen retraining schedule together with total,
  training, and loss cost.

## Correctness Evidence

- Exact-vs-brute-force: [examples/verify_exact_against_bruteforce.py](/Users/yangju/.config/superpowers/worktrees/retrainopt/offline-repro-phase1/examples/verify_exact_against_bruteforce.py)
- Core planner tests: [tests/test_retrainopt_planner.py](/Users/yangju/.config/superpowers/worktrees/retrainopt/offline-repro-phase1/tests/test_retrainopt_planner.py)
- Bayes backend validation: [tests/test_retrainopt_botorch.py](/Users/yangju/.config/superpowers/worktrees/retrainopt/offline-repro-phase1/tests/test_retrainopt_botorch.py)
- Public CLI and example smoke tests: [tests/test_retrainopt_smoke.py](/Users/yangju/.config/superpowers/worktrees/retrainopt/offline-repro-phase1/tests/test_retrainopt_smoke.py), [tests/test_retrainopt_example.py](/Users/yangju/.config/superpowers/worktrees/retrainopt/offline-repro-phase1/tests/test_retrainopt_example.py)

## Boundary

`retrainopt` deliberately does not ship the paper-only experiment harness.
Paper manifests, reference-number comparisons, frozen synthetic staging, and
Puhti submission scripts belong in the companion repository so the public
package remains small, reusable, and releaseable.
