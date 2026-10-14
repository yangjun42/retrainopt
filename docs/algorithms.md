# Algorithms and assumptions

## The objective

At time $t$, keep the current model or train a new model on a configured sample
count $n$. With loss weight $w_\ell$ and time weight $w_t$:

$$
c_t(\mathrm{keep},m)=w_\ell L_t(m),\qquad
c_t(\mathrm{retrain},n)=w_t\tau_{t,n}+w_\ell L_t(m_{t,n}).
$$

Here $\tau$ is measured fit time in seconds for the built-in evaluator. A
custom evaluator can supply a different measured training cost in the same
objective units. Retraining depends on the current data and selected sample
count, not on the previous model; warm-start-dependent training would require
a different state model.

For a finite timeline, dynamic programming uses

$$
V_t(m)=\min\left\{c_t(\mathrm{keep},m)+V_{t+1}(m),
\min_n[c_t(\mathrm{retrain},n)+V_{t+1}(m_{t,n})]\right\},
\quad V_T(m)=0.
$$

`exact` measures every configured candidate and solves this recurrence.
Negative acquisition scores are permitted in the approximate search; the
finite-horizon recurrence does not require positive scores.

## Approximate search

`bayes` measures an initial candidate set, fits training-cost and evaluation-loss
surrogates, and uses acquisition-guided candidate selection. It finally runs
DP on the measured/estimated cost table and measures the selected schedule's
remaining candidates. `total_cost` is that schedule's measured cost;
`estimated_total_cost` is the pre-measurement surrogate estimate. Neither
certifies that approximate search found the global optimum.

The default `lightweight` backend is a nearest-neighbor heuristic. Its
uncertainty scores are not calibrated GP posterior uncertainties. The optional
`botorch` backend fits Gaussian-process surrogates.

The default `sequence` selection follows an acquisition-minimizing DP schedule.
The current implementation also supports anchors, candidate ranking, and
supplementing iterations that propose too few new models. In particular,
`bayes_candidates_per_iteration` is a minimum target, not a hard evaluation
budget. Sequence selection can propose more, and measuring the final selected
schedule can require additional candidates.

To study the paper's random-design/GP/sequence formulation explicitly, use
`bayes_backend="botorch"`, `bayes_anchor_steps=()`, and
`bayes_selection_strategy="sequence"`. This still retains the library's
documented supplementation behavior and does not reproduce the published
experiments numerically by itself.

## Acquisition operates on a complete stage

Let $C=w_t\tau$ be training cost, already weighted, and $L$ be raw loss.
RetrainOpt assumes independent training-cost and loss posteriors when combining
their moments:

$$
\mu_K=w_\ell\mu_L,\quad v_K=w_\ell^2v_L,\qquad
\mu_R=\mu_C+w_\ell\mu_L,\quad v_R=v_C+w_\ell^2v_L.
$$

Independence is an implementation assumption for the separate surrogates, not
a claim proved by the paper. No joint covariance across stages is modeled:
the acquisition objective is a sum of per-stage scores, as in the paper.

The scores minimized by DP are:

| Acquisition | Stage score |
| --- | --- |
| `expected` | $\mu$ |
| `variance` | $-v$ |
| `lcb` | $\mu-\beta\sqrt v$ |
| `ucb` | $\mu+\beta\sqrt v$ |
| `ei` | negative expected improvement |
| `pi` | negative probability of improvement |

For EI/PI, $b$ is the global minimum measured **legal stage cost**, across
times and actions. It includes the initial model's keep costs, observed
retrain costs, and observed models' keep costs at later times. It is not the
minimum loss or training-cost component considered separately. The margin
$\xi$ has the same units as a stage cost.

Measured stage costs have zero uncertainty: Expected/LCB/UCB return the
measured cost, while Variance returns zero. EI and PI return zero at zero
standard deviation, following the paper's explicit convention. Changing from
0.1 to 0.2 can therefore change approximate candidate selection and plans.

## Research boundaries

The maintained package covers offline planning and adapters. Online policies,
large datasets, figures and experiment orchestration belong in the separate
`retrainopt-paper-repro` project. The canonical reproduction remains pinned to
`ecml-pkdd-2026-camera-ready-retrainopt`; upgrading this library does not update
that pin or replace its results.

Zephon's separation of curriculum and execution and Modyn's separation of
policy, selection and evaluation are useful engineering precedents. RetrainOpt
uses ordinary in-process interfaces; neither project is a dependency. See
[Zephon's architecture](https://github.com/datologyai/zephon#how-it-works) and
[Modyn's extension interfaces](https://github.com/eth-easl/modyn/blob/main/docs/TECHNICAL.md#coding).
Their distributed execution and orchestration features are outside this
library's scope.

