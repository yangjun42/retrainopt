"""Lightweight nearest-neighbor surrogate used by the bayes planner."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


class NearestNeighborRegressor:
    """A tiny distance-weighted regressor with uncertainty estimates."""

    def __init__(self, k_neighbors: int = 5, uncertainty_floor: float = 0.05):
        self.k_neighbors = max(1, int(k_neighbors))
        self.uncertainty_floor = max(0.0, float(uncertainty_floor))
        self._rows: List[Tuple[Tuple[float, ...], float]] = []
        self._mins: Tuple[float, ...] = ()
        self._ranges: Tuple[float, ...] = ()

    def fit(self, rows: Iterable[Sequence[float]], targets: Iterable[float]) -> None:
        self._rows = [(tuple(map(float, row)), float(target)) for row, target in zip(rows, targets)]
        if not self._rows:
            self._mins = ()
            self._ranges = ()
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

    def predict(self, row: Sequence[float]) -> Tuple[float, float]:
        if not self._rows:
            raise ValueError("NearestNeighborRegressor.predict() called before fit().")

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

    def _distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        total = 0.0
        for idx, left_value in enumerate(left):
            scale = self._ranges[idx]
            diff = (left_value - right[idx]) / scale
            total += diff * diff
        return math.sqrt(total)

