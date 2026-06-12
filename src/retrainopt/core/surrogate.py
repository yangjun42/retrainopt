"""Lightweight nearest-neighbor surrogate used by the bayes planner."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover - exercised only when numpy is installed.
    import numpy as _np
except ImportError:  # pragma: no cover - dependency-free fallback.
    _np = None


class NearestNeighborRegressor:
    """A tiny distance-weighted regressor with uncertainty estimates."""

    def __init__(self, k_neighbors: int = 5, uncertainty_floor: float = 0.05):
        self.k_neighbors = max(1, int(k_neighbors))
        self.uncertainty_floor = max(0.0, float(uncertainty_floor))
        self._rows: List[Tuple[Tuple[float, ...], float]] = []
        self._mins: Tuple[float, ...] = ()
        self._ranges: Tuple[float, ...] = ()
        self._matrix = None
        self._targets = None
        self._mins_array = None
        self._ranges_array = None

    def fit(self, rows: Iterable[Sequence[float]], targets: Iterable[float]) -> None:
        self._rows = [(tuple(map(float, row)), float(target)) for row, target in zip(rows, targets)]
        if not self._rows:
            self._mins = ()
            self._ranges = ()
            self._matrix = None
            self._targets = None
            self._mins_array = None
            self._ranges_array = None
            return

        dimension = len(self._rows[0][0])
        minima = [float("inf")] * dimension
        maxima = [float("-inf")] * dimension
        for row, _ in self._rows:
            for idx, value in enumerate(row):
                minima[idx] = min(minima[idx], value)
                maxima[idx] = max(maxima[idx], value)
        self._mins = tuple(minima)
        self._ranges = tuple(max(maxima[idx] - minima[idx], 1e-6) for idx in range(dimension))
        if _np is not None:
            self._matrix = _np.asarray([row for row, _ in self._rows], dtype=float)
            self._targets = _np.asarray([target for _, target in self._rows], dtype=float)
            self._mins_array = _np.asarray(self._mins, dtype=float)
            self._ranges_array = _np.asarray(self._ranges, dtype=float)

    def predict(self, row: Sequence[float]) -> Tuple[float, float]:
        if not self._rows:
            raise ValueError("NearestNeighborRegressor.predict() called before fit().")
        if self._matrix is not None:
            return self._predict_numpy(row)

        query = tuple(map(float, row))
        distances: List[Tuple[float, float]] = []
        exact_values: List[float] = []
        for stored_row, target in self._rows:
            distance = self._distance(query, stored_row)
            if distance <= 1e-12:
                exact_values.append(target)
            distances.append((distance, target))

        if exact_values:
            mean = sum(exact_values) / len(exact_values)
            if len(exact_values) == 1:
                return mean, 0.0
            variance = sum((value - mean) ** 2 for value in exact_values) / len(exact_values)
            return mean, math.sqrt(max(0.0, variance))

        nearest = sorted(distances, key=lambda item: item[0])[: self.k_neighbors]
        weights = [1.0 / (distance + 1e-6) for distance, _ in nearest]
        weight_sum = sum(weights)
        mean = sum(weight * target for weight, (_, target) in zip(weights, nearest)) / weight_sum
        variance = sum(
            weight * ((target - mean) ** 2) for weight, (_, target) in zip(weights, nearest)
        ) / weight_sum
        distance_scale = sum(distance for distance, _ in nearest) / len(nearest)
        std = math.sqrt(max(0.0, variance)) + self.uncertainty_floor * distance_scale
        return mean, std

    def _predict_numpy(self, row: Sequence[float]) -> Tuple[float, float]:
        query = _np.asarray(tuple(map(float, row)), dtype=float)
        scaled = (self._matrix - query) / self._ranges_array
        distances = _np.sqrt(_np.sum(scaled * scaled, axis=1))
        exact_mask = distances <= 1e-12
        if bool(_np.any(exact_mask)):
            exact_values = self._targets[exact_mask]
            mean = float(_np.mean(exact_values))
            if int(exact_values.size) == 1:
                return mean, 0.0
            variance = float(_np.mean((exact_values - mean) ** 2))
            return mean, math.sqrt(max(0.0, variance))

        k_value = min(self.k_neighbors, int(distances.size))
        if k_value <= 0:
            raise ValueError("NearestNeighborRegressor.predict() called before fit().")
        if k_value == int(distances.size):
            nearest_idx = _np.argsort(distances)
        else:
            nearest_idx = _np.argpartition(distances, k_value - 1)[:k_value]
            nearest_idx = nearest_idx[_np.argsort(distances[nearest_idx])]

        nearest_distances = distances[nearest_idx]
        nearest_targets = self._targets[nearest_idx]
        weights = 1.0 / (nearest_distances + 1e-6)
        weight_sum = float(_np.sum(weights))
        mean = float(_np.sum(weights * nearest_targets) / weight_sum)
        variance = float(_np.sum(weights * ((nearest_targets - mean) ** 2)) / weight_sum)
        distance_scale = float(_np.mean(nearest_distances))
        std = math.sqrt(max(0.0, variance)) + self.uncertainty_floor * distance_scale
        return mean, std

    def _distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        total = 0.0
        for idx, left_value in enumerate(left):
            scale = self._ranges[idx]
            diff = (left_value - right[idx]) / scale
            total += diff * diff
        return math.sqrt(total)
