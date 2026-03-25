"""Paper-focused manifest definitions for the internal offline repro harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Sequence

from ..config import PlannerConfig


def _stable_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class DatasetSpec:
    """Definition for a real or synthetic timeline used in paper reproduction."""

    name: str
    source: str
    num_features: int
    binary_dir: str | None = None
    num_steps: int = 0
    samples_per_step: int = 0
    test_split_ratio: float = 0.2
    drift_pattern: str | None = None
    drift_timestep: int = 0
    pattern_params: Dict[str, Any] = field(default_factory=dict)
    model_defaults: Dict[str, Any] = field(default_factory=dict)
    data_state_strategy: str = "individual"
    sliding_window_k: int = 1

    def fingerprint(self) -> str:
        return _stable_hash(asdict(self))


@dataclass(frozen=True)
class RunSpec:
    """One executable offline repro run."""

    run_id: str
    study: str
    dataset: str
    method: str
    model: str
    seed: int = 42
    n_choices: Sequence[float | int] = (0.25, 0.5, 1.0)
    min_train_size: int = 1
    time_cost_weight: float = 1.0
    loss_cost_weight: float = 1.0
    bayes_backend: str = "lightweight"
    bayes_iterations: int = 5
    bayes_initial_samples: int = 6
    bayes_candidates_per_iteration: int = 1
    acquisition: str = "expected"
    expected_artifacts: Sequence[str] = ()
    tags: Sequence[str] = ()
    cache_group: str | None = None

    def cache_key(self, dataset_spec: DatasetSpec) -> str:
        if self.cache_group:
            return self.cache_group
        return self.run_id

    def planner_config(self) -> PlannerConfig:
        return PlannerConfig(
            n_choices=self.n_choices,
            min_train_size=self.min_train_size,
            time_cost_weight=self.time_cost_weight,
            loss_cost_weight=self.loss_cost_weight,
            random_seed=self.seed,
            bayes_iterations=self.bayes_iterations,
            bayes_initial_samples=self.bayes_initial_samples,
            bayes_candidates_per_iteration=self.bayes_candidates_per_iteration,
            acquisition=self.acquisition,
            bayes_backend=self.bayes_backend,
        )


@dataclass
class PaperManifest:
    """Container for paper datasets and executable runs."""

    version: str
    acceptance_baseline: str
    datasets: Dict[str, DatasetSpec]
    runs: List[RunSpec]

    def get_dataset(self, name: str) -> DatasetSpec:
        try:
            return self.datasets[name]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset in paper manifest: {name}") from exc

    def get_run(self, run_id: str) -> RunSpec:
        for run in self.runs:
            if run.run_id == run_id:
                return run
        raise KeyError(f"Unknown run_id in paper manifest: {run_id}")

    def find_reference_exact_run(self, run: RunSpec) -> RunSpec | None:
        for candidate in self.runs:
            if candidate.method != "exact":
                continue
            if candidate.dataset != run.dataset or candidate.seed != run.seed or candidate.model != run.model:
                continue
            if candidate.time_cost_weight != run.time_cost_weight:
                continue
            if candidate.loss_cost_weight != run.loss_cost_weight:
                continue
            return candidate
        return None


def default_phase1_manifest() -> PaperManifest:
    """Return the latest-paper offline scope for the clean Phase 1 harness."""
    common_real_n = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    common_synth_n = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    acquisitions = ("expected", "variance", "lcb_beta_1.0", "ucb_beta_2.0", "ei_xi_0.01", "pi_xi_0.1")
    kappa_ratios = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
    phi_tags = ("phi-0.2", "phi-0.1", "phi-0.05", "phi-0.01")

    datasets = {
        "extreme_sudden_M": DatasetSpec(
            name="extreme_sudden_M",
            source="synthetic",
            num_features=2,
            num_steps=50,
            samples_per_step=5000,
            drift_pattern="sudden",
            drift_timestep=25,
            pattern_params={"feature_shift": 3.0, "noise_std": 0.5},
            model_defaults={"hidden_dims": [128, 64], "epochs": 5, "learning_rate": 0.01},
            data_state_strategy="sliding_window",
            sliding_window_k=2,
        ),
        "gradual_drift_M": DatasetSpec(
            name="gradual_drift_M",
            source="synthetic",
            num_features=2,
            num_steps=50,
            samples_per_step=5000,
            drift_pattern="gradual",
            drift_timestep=15,
            pattern_params={"drift_duration": 15, "final_threshold": 1.5},
            model_defaults={"hidden_dims": [128, 64], "epochs": 3, "learning_rate": 0.01},
            data_state_strategy="sliding_window",
            sliding_window_k=5,
        ),
        "recurring_drift_M": DatasetSpec(
            name="recurring_drift_M",
            source="synthetic",
            num_features=2,
            num_steps=50,
            samples_per_step=5000,
            drift_pattern="recurring",
            drift_timestep=10,
            pattern_params={"cycle_length": 10, "amplitude": 3.0},
            model_defaults={"hidden_dims": [128, 64], "epochs": 3, "learning_rate": 0.01},
            data_state_strategy="sliding_window",
            sliding_window_k=3,
        ),
        "airlines": DatasetSpec(
            name="airlines",
            source="binary",
            num_features=7,
            binary_dir="./data/airlines",
            model_defaults={"hidden_dims": [128, 64], "epochs": 5, "learning_rate": 0.01},
            data_state_strategy="sliding_window",
            sliding_window_k=3,
        ),
        "electricity": DatasetSpec(
            name="electricity",
            source="binary",
            num_features=6,
            binary_dir="./data/electricity",
            model_defaults={"hidden_dims": [128, 64], "epochs": 5, "learning_rate": 0.01},
            data_state_strategy="sliding_window",
            sliding_window_k=3,
        ),
        "covertype": DatasetSpec(
            name="covertype",
            source="binary",
            num_features=54,
            binary_dir="./data/covertype",
            model_defaults={"hidden_dims": [256, 128], "epochs": 5, "learning_rate": 0.01},
            data_state_strategy="sliding_window",
            sliding_window_k=3,
        ),
    }

    runs: List[RunSpec] = []
    exact_defaults = {
        "extreme_sudden_M": (0.7, 1.6, common_synth_n),
        "gradual_drift_M": (1.0, 1.2, common_synth_n),
        "recurring_drift_M": (0.9, 1.6, common_synth_n),
        "airlines": (1.0, 1.2, common_real_n),
        "electricity": (1.0, 1.2, common_real_n),
        "covertype": (1.0, 1.2, common_real_n),
    }
    for dataset_name, (time_weight, loss_weight, n_choices) in exact_defaults.items():
        cache_group = f"{dataset_name}-seed42-core"
        runs.append(
            RunSpec(
                run_id=f"phase1_exact_{dataset_name}_seed42",
                study="objective-validity",
                dataset=dataset_name,
                method="exact",
                model="paper_mlp",
                seed=42,
                n_choices=n_choices,
                time_cost_weight=time_weight,
                loss_cost_weight=loss_weight,
                expected_artifacts=("summary", "cache", "comparison"),
                cache_group=cache_group,
            )
        )

    for acquisition in acquisitions:
        runs.append(
            RunSpec(
                run_id=f"phase1_bayes_recurring_drift_M_seed42_{acquisition}",
                study="convergence",
                dataset="recurring_drift_M",
                method="bayes",
                model="paper_mlp",
                seed=42,
                n_choices=common_synth_n,
                time_cost_weight=0.9,
                loss_cost_weight=1.6,
                bayes_backend="lightweight",
                bayes_iterations=50,
                bayes_initial_samples=6,
                bayes_candidates_per_iteration=1,
                acquisition=acquisition,
                expected_artifacts=("summary", "cache", "comparison"),
                cache_group="recurring-drift-seed42-core",
            )
        )

    for phi_tag in phi_tags:
        runs.append(
            RunSpec(
                run_id=f"phase1_scalability_extreme_sudden_M_seed42_{phi_tag}",
                study="scalability",
                dataset="extreme_sudden_M",
                method="bayes",
                model="paper_mlp",
                seed=42,
                n_choices=common_synth_n,
                time_cost_weight=0.7,
                loss_cost_weight=1.6,
                bayes_backend="lightweight",
                bayes_iterations=50,
                bayes_initial_samples=6,
                bayes_candidates_per_iteration=1,
                acquisition="lcb_beta_1.0",
                expected_artifacts=("summary", "cache", "comparison"),
                tags=(phi_tag,),
                cache_group="extreme-sudden-seed42-core",
            )
        )

    for ratio in kappa_ratios:
        runs.append(
            RunSpec(
                run_id=f"phase1_kappa_extreme_sudden_M_seed42_ratio{str(ratio).replace('.', '_')}",
                study="kappa-sensitivity",
                dataset="extreme_sudden_M",
                method="exact",
                model="paper_mlp",
                seed=42,
                n_choices=common_synth_n,
                time_cost_weight=0.7,
                loss_cost_weight=0.7 * ratio,
                expected_artifacts=("summary", "cache", "comparison"),
                tags=(f"ratio-{ratio}",),
                cache_group="extreme-sudden-seed42-core",
            )
        )

    for acquisition in ("lcb_beta_0.1", "lcb_beta_1.0", "lcb_beta_2.0", "ei_xi_0.0", "ei_xi_0.01", "ei_xi_0.1"):
        runs.append(
            RunSpec(
                run_id=f"phase1_param_recurring_drift_M_seed42_{acquisition}",
                study="parameter-sensitivity",
                dataset="recurring_drift_M",
                method="bayes",
                model="paper_mlp",
                seed=42,
                n_choices=common_synth_n,
                time_cost_weight=0.9,
                loss_cost_weight=1.6,
                bayes_backend="lightweight",
                bayes_iterations=50,
                bayes_initial_samples=6,
                bayes_candidates_per_iteration=1,
                acquisition=acquisition,
                expected_artifacts=("summary", "cache", "comparison"),
                cache_group="recurring-drift-seed42-core",
            )
        )

    return PaperManifest(
        version="phase1-latest-offline-v1",
        acceptance_baseline="latest-paper",
        datasets=datasets,
        runs=runs,
    )
