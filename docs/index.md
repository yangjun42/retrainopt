# RetrainOpt

RetrainOpt provides offline planning and online policies for **when to retrain
and how much data to use**, balancing training cost and predictive loss. It
implements algorithms from
[*Automated Retraining for Cost-Optimal ML*](https://link.springer.com/chapter/10.1007/978-3-032-37654-1_14)
by Jun Yang and Michael Mathioudakis (ECML PKDD 2026).

The offline planner uses exact dynamic programming or surrogate-guided
approximate search over a supplied timeline. The whole timeline, including
evaluation data, is available to this planner. A plan's evaluated cost is not
a prospective test-set estimate when the same evaluation data were used to
choose that plan.

The Base and Learned online policies support sequential retraining decisions
from observed history.

<p><a href="_static/demo/index.html"><strong>Try the interactive scheduling example →</strong></a></p>
Change the cost of training and prediction errors to explore when and how much
to retrain in a synthetic timeline.

```{toctree}
:maxdepth: 2

quickstart
demo
algorithms
online
evaluation
integrations
api
migration
development
```

