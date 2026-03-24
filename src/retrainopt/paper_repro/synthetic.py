"""Synthetic timeline generation for the internal offline repro harness."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from ..adapters import ArrayTimelineAdapter
from .manifest import DatasetSpec


Sample = Tuple[Tuple[float, ...], int]


def build_synthetic_timeline(spec: DatasetSpec, seed: int) -> ArrayTimelineAdapter:
    if spec.source != "synthetic":
        raise ValueError(f"Dataset {spec.name} is not synthetic.")
    if spec.num_steps <= 0:
        raise ValueError(f"Synthetic dataset {spec.name} must define num_steps > 0.")
    if spec.samples_per_step <= 0:
        raise ValueError(f"Synthetic dataset {spec.name} must define samples_per_step > 0.")

    train_blocks: List[List[Sample]] = []
    eval_blocks: List[List[Sample]] = []
    rng = random.Random(seed)

    for step in range(spec.num_steps):
        block_seed = rng.randint(0, 10_000_000)
        samples = _generate_block(spec, step, block_seed)
        local_rng = random.Random(block_seed + 991)
        local_rng.shuffle(samples)
        num_test = max(1, int(len(samples) * spec.test_split_ratio))
        eval_blocks.append(samples[:num_test])
        train_blocks.append(samples[num_test:])

    return ArrayTimelineAdapter(train_blocks=train_blocks, eval_blocks=eval_blocks)


def _generate_block(spec: DatasetSpec, step: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    pattern = (spec.drift_pattern or "sudden").strip().lower()
    samples: List[Sample] = []
    for _ in range(spec.samples_per_step):
        samples.append(_generate_sample(pattern, spec, step, rng))
    return samples


def _generate_sample(pattern: str, spec: DatasetSpec, step: int, rng: random.Random) -> Sample:
    num_features = max(1, spec.num_features)
    if pattern == "sudden":
        feature_shift = float(spec.pattern_params.get("feature_shift", 2.0))
        noise_std = float(spec.pattern_params.get("noise_std", 1.0))
        if step < spec.drift_timestep:
            primary = rng.gauss(0.0, noise_std)
            label = 1 if primary > 0.0 else 0
        else:
            primary = rng.gauss(feature_shift, noise_std)
            label = 1 if primary < 0.0 else 0
    elif pattern == "gradual":
        duration = max(1, int(spec.pattern_params.get("drift_duration", 3)))
        final_threshold = float(spec.pattern_params.get("final_threshold", 1.0))
        if step < spec.drift_timestep:
            threshold = 0.0
        elif step >= spec.drift_timestep + duration:
            threshold = final_threshold
        else:
            progress = float(step - spec.drift_timestep) / float(duration)
            threshold = final_threshold * progress
        primary = rng.gauss(0.0, 1.0)
        label = 1 if primary > threshold else 0
    elif pattern == "recurring":
        cycle_length = max(2, int(spec.pattern_params.get("cycle_length", 4)))
        amplitude = float(spec.pattern_params.get("amplitude", 1.5))
        primary = rng.gauss(0.0, 1.0)
        if step < spec.drift_timestep:
            label = 1 if primary > 0.0 else 0
        else:
            cycle_position = (step - spec.drift_timestep) % cycle_length
            if cycle_position < cycle_length // 2:
                label = 1 if primary > amplitude * 0.5 else 0
            else:
                label = 1 if primary < 0.0 else 0
    else:
        raise ValueError(f"Unsupported synthetic drift pattern: {pattern}")

    features = [primary]
    for _ in range(1, num_features):
        features.append(rng.gauss(0.0, 1.0))
    return tuple(float(value) for value in features), int(label)
