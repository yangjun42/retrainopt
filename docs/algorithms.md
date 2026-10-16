# Algorithms and assumptions

## The offline objective

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
surrogates, and uses acquisition-guided candidate selection. It then runs DP
on the measured/estimated cost table and measures that proposed schedule's
remaining candidates. A final DP over **all measured candidates** chooses the
returned schedule. This can replace an overly optimistic proposal with a
better measured alternative, including combinations of earlier candidates.
The result is optimal over the measured subset, which includes the fully
measured proposal; unmeasured candidates may still offer a better schedule.

`total_cost` is the returned schedule's measured cost. `estimated_total_cost`
evaluates that same returned schedule using the cost table from before final
measurement; it is not the estimate of a discarded proposal.

Search DP solves still consider the full configured candidate grid; only the
final measured-candidate solve uses a subset. Approximate search reduces model
fits but retains the full grid's DP state-space complexity during search.

The default `lightweight` backend is a nearest-neighbor heuristic. Its
uncertainty scores are not calibrated GP posterior uncertainties. The optional
`botorch` backend fits Gaussian-process surrogates. The library enriches the
paper's training-cost input $n$ with training step, available dataset size, and
sample fraction. Loss features also include evaluation step and model age.
With only one observed row, the BoTorch regressor uses that target as a constant
mean with zero standard deviation instead of fitting a GP. The planner clips
negative predicted cost and loss means to zero without changing their standard
deviations.

Use the explicit `PlannerConfig` fields to configure these backends.
`bayes_backend_options` currently accepts only an empty dictionary; unsupported
options raise an error.

The default `sequence` selection follows an acquisition-minimizing DP schedule.
Initialization measures every configured sample count at each anchor step,
then samples random candidates if needed to reach `bayes_initial_samples`.
Anchors can therefore exceed that initial target. Candidate ranking supplies
new candidates when the selected sequence has none, and supplements iterations
that propose too few. These are library extensions to the paper's sequence
search. `bayes_candidates_per_iteration` is a minimum target, not a hard
evaluation budget. Sequence selection can propose more. Final measurement adds
at most one previously unmeasured candidate per timestep, and the final DP
performs no fits or further surrogate-guided search. Metadata records these
additional candidates in `finalization_new_candidates`, with
`finalization_strategy="observed_candidates_dp"`.

For GP surrogates with random initialization and sequence selection, use
`bayes_backend="botorch"`, `bayes_anchor_steps=()`, and
`bayes_selection_strategy="sequence"`. Candidate supplementation still applies.

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
standard deviation, following the paper's explicit convention.

For the Base and Learned online policies, see [Online retraining](online.md).

