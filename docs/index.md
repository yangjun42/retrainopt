# RetrainOpt

RetrainOpt plans **when to retrain and how much data to use** over a supplied
timeline. It minimizes a sum of training and modeling costs. The package
implements the offline planning core of *Automated Retraining for Cost-Optimal
ML* by Jun Yang and Michael Mathioudakis (2026).

The whole timeline, including evaluation data, is available to the planner.
This is an offline optimization library, not a deployment service or an online
drift detector. A plan's evaluated cost is not a prospective test-set estimate
when the same evaluation data were used to choose that plan.

```{toctree}
:maxdepth: 2

quickstart
algorithms
evaluation
integrations
api
migration
development
```

