"""Lightweight nearest-neighbor surrogate used by the bayes planner."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover - exercised only when numpy is installed.
    import numpy as _np
except ImportError:  # pragma: no cover - dependency-free fallback.
    _np = None

try:  # pragma: no cover - exercised only when the sklearn extra is installed.
    from sklearn.neighbors import NearestNeighbors as _SklearnNearestNeighbors
except ImportError:  # pragma: no cover - optional acceleration path.
    _SklearnNearestNeighbors = None


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
        self._scaled_matrix = None
        self._neighbor_index = None

    def fit(self, rows: Iterable[Sequence[float]], targets: Iterable[float]) -> None:
        self._rows = [(tuple(map(float, row)), float(target)) for row, target in zip(rows, targets)]
        if not self._rows:
            self._mins = ()
            self._ranges = ()
            self._matrix = None
            self._targets = None
            self._mins_array = None
            self._ranges_array = None
            self._scaled_matrix = None
            self._neighbor_index = None
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
            self._scaled_matrix = (self._matrix - self._mins_array) / self._ranges_array
            if _SklearnNearestNeighbors is not None:
                self._neighbor_index = _SklearnNearestNeighbors(
                    n_neighbors=min(self.k_neighbors, len(self._rows)),
                    algorithm="auto",
                    metric="euclidean",
                )
                self._neighbor_index.fit(self._scaled_matrix)
            else:
                self._neighbor_index = None

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

    def predict_many(
        self,
        rows: Iterable[Sequence[float]],
        *,
        batch_size: int = 128,
    ) -> List[Tuple[float, float]]:
        """Predict many rows while preserving ``predict`` semantics."""
        if not self._rows:
            raise ValueError("NearestNeighborRegressor.predict_many() called before fit().")
        query_rows = [tuple(map(float, row)) for row in rows]
        if not query_rows:
            return []
        if self._matrix is not None:
            if self._neighbor_index is not None:
                return self._predict_many_sklearn(query_rows, batch_size=max(1, int(batch_size)))
            return self._predict_many_numpy(query_rows, batch_size=max(1, int(batch_size)))
        return [self.predict(row) for row in query_rows]

    def _predict_numpy(self, row: Sequence[float]) -> Tuple[float, float]:
        query = _np.asarray(tuple(map(float, row)), dtype=float)
        scaled_query = (query - self._mins_array) / self._ranges_array
        scaled = self._scaled_matrix - scaled_query
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

    def _predict_many_numpy(
        self,
        rows: Sequence[Sequence[float]],
        *,
        batch_size: int,
    ) -> List[Tuple[float, float]]:
        predictions: List[Tuple[float, float]] = []
        target_count = int(self._targets.size)
        k_value = min(self.k_neighbors, target_count)
        if k_value <= 0:
            raise ValueError("NearestNeighborRegressor.predict_many() called before fit().")

        for start in range(0, len(rows), batch_size):
            query = _np.asarray(rows[start : start + batch_size], dtype=float)
            scaled = (self._matrix[None, :, :] - query[:, None, :]) / self._ranges_array
            distances = _np.sqrt(_np.sum(scaled * scaled, axis=2))
            exact_mask = distances <= 1e-12
            has_exact = _np.any(exact_mask, axis=1)
            batch_predictions: List[Tuple[float, float] | None] = [None] * int(query.shape[0])

            for row_index in _np.flatnonzero(has_exact):
                exact_values = self._targets[exact_mask[row_index]]
                mean = float(_np.mean(exact_values))
                if int(exact_values.size) == 1:
                    batch_predictions[int(row_index)] = (mean, 0.0)
                    continue
                variance = float(_np.mean((exact_values - mean) ** 2))
                batch_predictions[int(row_index)] = (mean, math.sqrt(max(0.0, variance)))

            non_exact_indices = _np.flatnonzero(~has_exact)
            if int(non_exact_indices.size) > 0:
                non_exact_distances = distances[non_exact_indices]
                if k_value == target_count:
                    nearest_idx = _np.argsort(non_exact_distances, axis=1)[:, :k_value]
                else:
                    nearest_idx = _np.argpartition(non_exact_distances, k_value - 1, axis=1)[:, :k_value]
                    order = _np.argsort(_np.take_along_axis(non_exact_distances, nearest_idx, axis=1), axis=1)
                    nearest_idx = _np.take_along_axis(nearest_idx, order, axis=1)

                nearest_distances = _np.take_along_axis(non_exact_distances, nearest_idx, axis=1)
                nearest_targets = self._targets[nearest_idx]
                weights = 1.0 / (nearest_distances + 1e-6)
                weight_sum = _np.sum(weights, axis=1)
                means = _np.sum(weights * nearest_targets, axis=1) / weight_sum
                variances = _np.sum(weights * ((nearest_targets - means[:, None]) ** 2), axis=1) / weight_sum
                distance_scales = _np.mean(nearest_distances, axis=1)
                stds = _np.sqrt(_np.maximum(0.0, variances)) + self.uncertainty_floor * distance_scales

                for offset, row_index in enumerate(non_exact_indices):
                    batch_predictions[int(row_index)] = (float(means[offset]), float(stds[offset]))

            predictions.extend(prediction for prediction in batch_predictions if prediction is not None)
        return predictions

    def _predict_many_sklearn(
        self,
        rows: Sequence[Sequence[float]],
        *,
        batch_size: int,
    ) -> List[Tuple[float, float]]:
        predictions: List[Tuple[float, float]] = []
        target_count = int(self._targets.size)
        k_value = min(self.k_neighbors, target_count)
        if k_value <= 0:
            raise ValueError("NearestNeighborRegressor.predict_many() called before fit().")

        for start in range(0, len(rows), batch_size):
            query = _np.asarray(rows[start : start + batch_size], dtype=float)
            scaled_query = (query - self._mins_array) / self._ranges_array
            nearest_distances, nearest_idx = self._neighbor_index.kneighbors(
                scaled_query,
                n_neighbors=k_value,
                return_distance=True,
            )
            exact_rows = _np.flatnonzero(nearest_distances[:, 0] <= 1e-12)
            batch_predictions: List[Tuple[float, float] | None] = [None] * int(query.shape[0])

            for row_index in exact_rows:
                exact_idx = self._neighbor_index.radius_neighbors(
                    scaled_query[int(row_index) : int(row_index) + 1],
                    radius=1e-12,
                    return_distance=False,
                )[0]
                exact_values = self._targets[exact_idx]
                mean = float(_np.mean(exact_values))
                if int(exact_values.size) == 1:
                    batch_predictions[int(row_index)] = (mean, 0.0)
                    continue
                variance = float(_np.mean((exact_values - mean) ** 2))
                batch_predictions[int(row_index)] = (mean, math.sqrt(max(0.0, variance)))

            non_exact_indices = _np.flatnonzero(nearest_distances[:, 0] > 1e-12)
            if int(non_exact_indices.size) > 0:
                non_exact_distances = nearest_distances[non_exact_indices]
                non_exact_idx = nearest_idx[non_exact_indices]
                nearest_targets = self._targets[non_exact_idx]
                weights = 1.0 / (non_exact_distances + 1e-6)
                weight_sum = _np.sum(weights, axis=1)
                means = _np.sum(weights * nearest_targets, axis=1) / weight_sum
                variances = _np.sum(weights * ((nearest_targets - means[:, None]) ** 2), axis=1) / weight_sum
                distance_scales = _np.mean(non_exact_distances, axis=1)
                stds = _np.sqrt(_np.maximum(0.0, variances)) + self.uncertainty_floor * distance_scales

                for offset, row_index in enumerate(non_exact_indices):
                    batch_predictions[int(row_index)] = (float(means[offset]), float(stds[offset]))

            predictions.extend(prediction for prediction in batch_predictions if prediction is not None)
        return predictions

    def _distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        total = 0.0
        for idx, left_value in enumerate(left):
            scale = self._ranges[idx]
            diff = (left_value - right[idx]) / scale
            total += diff * diff
        return math.sqrt(total)
