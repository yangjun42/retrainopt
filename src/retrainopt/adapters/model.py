"""Built-in model adapters."""

from __future__ import annotations

from collections.abc import Mapping, Set
import math
from numbers import Integral, Real
from typing import Any, Callable, Iterable, List, Sequence, Tuple

from ..interfaces import ModelAdapter


def _class_label(value: Any) -> int:
    """Normalize an integer-valued numeric label without changing its class."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Class labels must be finite integers, not booleans or strings.")
    if isinstance(value, Integral):
        return int(value)
    if not math.isfinite(value) or value != int(value):
        raise ValueError("Class labels must be finite integers.")
    return int(value)


def _split_labeled_block(block: Any) -> Tuple[List[Any], List[int]]:
    features: List[Any] = []
    labels: List[int] = []
    for item in block:
        if isinstance(item, (str, bytes, bytearray, Mapping, Set)):
            raise ValueError("Expected labeled items shaped like (features, label).")
        try:
            values, label = item
        except (TypeError, ValueError) as exc:
            raise ValueError("Expected labeled items shaped like (features, label).") from exc
        features.append(values)
        labels.append(_class_label(label))
    return features, labels


class SklearnModelAdapter(ModelAdapter):
    """ModelAdapter for sklearn classifiers with integer class labels.

    Finite integer-valued real labels, such as ``1.0``, are also accepted.
    Fractional labels, booleans and strings are rejected rather than coerced.
    """

    def __init__(self, model_factory: Callable[[], Any] | Any):
        self.model_factory = model_factory

    def fit(self, train_data: Any, *, step: int, n_samples: int, random_seed: int) -> Any:
        features, labels = _split_labeled_block(train_data)
        estimator = self._build_estimator()
        if hasattr(estimator, "get_params"):
            seed_params = {
                name: int(random_seed) % (2 ** 32)
                for name, value in estimator.get_params(deep=True).items()
                if (name == "random_state" or name.endswith("__random_state")) and value is None
            }
            if seed_params:
                estimator.set_params(**seed_params)
        estimator.fit(features, labels)
        return estimator

    def evaluate(self, model: Any, eval_data: Any, *, step: int) -> float:
        try:
            from sklearn.metrics import log_loss
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("SklearnModelAdapter requires the `sklearn` extra.") from exc

        features, labels = _split_labeled_block(eval_data)
        if not features:
            raise ValueError("Evaluation data must not be empty.")
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)
            return float(log_loss(labels, probabilities, labels=list(model.classes_)))
        predictions = model.predict(features)
        mistakes = sum(int(prediction != label) for prediction, label in zip(predictions, labels))
        return mistakes / max(1, len(labels))

    def _build_estimator(self) -> Any:
        try:
            from sklearn.base import clone
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("SklearnModelAdapter requires the `sklearn` extra.") from exc

        if callable(self.model_factory):
            return self.model_factory()
        return clone(self.model_factory)


class PrototypeModelAdapter(ModelAdapter):
    """Dependency-free prototype classifier with integer class labels.

    Labels follow the same numeric contract as ``SklearnModelAdapter``.
    Features are finite real scalars or nonempty flat lists/tuples of finite
    real values. Every train/evaluation sample must have the trained feature
    dimension. Boolean feature values remain valid numeric indicators.
    """

    def fit(self, train_data: Any, *, step: int, n_samples: int, random_seed: int) -> Any:
        features, labels = _split_labeled_block(train_data)
        if not features:
            return {"prototypes": {}, "majority": 0}

        vectors = self._feature_vectors(features)
        grouped = {}
        counts = {}
        for feature_vector, label in zip(vectors, labels):
            grouped.setdefault(label, []).append(feature_vector)
            counts[label] = counts.get(label, 0) + 1

        prototypes = {}
        for label, vectors in grouped.items():
            dimension = len(vectors[0])
            mean_vector = []
            for index in range(dimension):
                mean = sum(vector[index] for vector in vectors) / len(vectors)
                if not math.isfinite(mean):
                    raise ValueError("Prototype mean overflowed; rescale feature values.")
                mean_vector.append(mean)
            prototypes[label] = tuple(mean_vector)

        majority = max(counts.items(), key=lambda item: item[1])[0]
        return {"prototypes": prototypes, "majority": majority}

    def evaluate(self, model: Any, eval_data: Any, *, step: int) -> float:
        features, labels = _split_labeled_block(eval_data)
        if not features:
            raise ValueError("Evaluation data must not be empty.")

        mistakes = 0
        prototypes = model.get("prototypes", {})
        majority = model.get("majority", 0)
        dimension = len(next(iter(prototypes.values()))) if prototypes else None
        vectors = self._feature_vectors(features, dimension=dimension)
        for feature_vector, label in zip(vectors, labels):
            if prototypes:
                predicted = min(
                    prototypes.items(),
                    key=lambda item: self._squared_distance(item[1], feature_vector),
                )[0]
            else:
                predicted = majority
            if predicted != label:
                mistakes += 1
        return mistakes / len(labels)

    @classmethod
    def _feature_vectors(cls, features: Sequence[Any], *, dimension: int | None = None) -> List[Tuple[float, ...]]:
        vectors = []
        for values in features:
            vector = cls._flatten_features(values)
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError(f"Prototype features must have a consistent dimension of {dimension}.")
            vectors.append(vector)
        return vectors

    @staticmethod
    def _flatten_features(features: Any) -> Tuple[float, ...]:
        values = features if isinstance(features, (list, tuple)) else (features,)
        if not values:
            raise ValueError("Prototype features must not be empty.")
        vector = []
        for value in values:
            if not isinstance(value, Real):
                raise ValueError("Prototype features must be finite real scalars or a flat numeric vector.")
            try:
                number = float(value)
            except OverflowError as exc:
                raise ValueError("Prototype features must be finite real values.") from exc
            if not math.isfinite(number):
                raise ValueError("Prototype features must be finite real values.")
            vector.append(number)
        return tuple(vector)

    @staticmethod
    def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
        distance = 0.0
        for lv, rv in zip(left, right):
            difference = lv - rv
            distance += difference * difference
        if not math.isfinite(distance):
            raise ValueError("Prototype distance overflowed; rescale feature values.")
        return distance


class TorchModelAdapter(ModelAdapter):
    """ModelAdapter for PyTorch classifiers."""

    def __init__(
        self,
        model_factory: Callable[[], Any],
        *,
        optimizer_factory: Callable[[Iterable[Any]], Any] | None = None,
        epochs: int = 5,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        for name, value in (("epochs", epochs), ("batch_size", batch_size)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = device

    def fit(self, train_data: Any, *, step: int, n_samples: int, random_seed: int) -> Any:
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("TorchModelAdapter requires the `torch` extra.") from exc

        device = torch.device(self.device)
        device_type = device.type if device.type in {"cuda", "mps"} else "cuda"
        devices = []
        if device.type in {"cuda", "mps"}:
            devices = [device.index or 0]
        with torch.random.fork_rng(devices=devices, device_type=device_type):
            torch.random.default_generator.manual_seed(random_seed)
            if device.type == "cuda":
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(random_seed)
            elif device.type == "mps":
                torch.mps.manual_seed(random_seed)
            model = self.model_factory().to(self.device)
            optimizer = (
                self.optimizer_factory(model.parameters())
                if self.optimizer_factory is not None
                else torch.optim.Adam(model.parameters(), lr=1e-3)
            )
            criterion = torch.nn.CrossEntropyLoss()
            generator = torch.Generator().manual_seed(random_seed)
            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, generator=generator)

            model.train()
            for _ in range(self.epochs):
                for inputs, targets in train_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()

        return model

    def evaluate(self, model: Any, eval_data: Any, *, step: int) -> float:
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("TorchModelAdapter requires the `torch` extra.") from exc

        loader = DataLoader(eval_data, batch_size=self.batch_size, shuffle=False, generator=torch.Generator().manual_seed(0))
        criterion = torch.nn.CrossEntropyLoss()
        model.eval()
        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += float(loss.item()) * int(inputs.size(0))
                total_count += int(inputs.size(0))

        if total_count <= 0:
            raise ValueError("Evaluation data must not be empty.")
        return total_loss / total_count
