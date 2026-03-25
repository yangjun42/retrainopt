"""Built-in model adapters."""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable, List, Sequence, Tuple

from ..interfaces import ModelAdapter


def _split_labeled_block(block: Any) -> Tuple[List[Any], List[int]]:
    features: List[Any] = []
    labels: List[int] = []
    for item in block:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError("Expected labeled items shaped like (features, label).")
        features.append(item[0])
        labels.append(int(item[1]))
    return features, labels


class SklearnModelAdapter(ModelAdapter):
    """ModelAdapter for sklearn classifiers."""

    def __init__(self, model_factory: Callable[[], Any] | Any):
        self.model_factory = model_factory

    def fit(self, train_data: Any, *, step: int, n_samples: int, random_seed: int) -> Any:
        estimator = self._build_estimator()
        features, labels = _split_labeled_block(train_data)
        estimator.fit(features, labels)
        return estimator

    def evaluate(self, model: Any, eval_data: Any, *, step: int) -> float:
        try:
            from sklearn.metrics import log_loss
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("SklearnModelAdapter requires the `sklearn` extra.") from exc

        features, labels = _split_labeled_block(eval_data)
        if not features:
            return float("inf")
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
        try:
            return clone(self.model_factory)
        except Exception:
            return copy.deepcopy(self.model_factory)


class PrototypeModelAdapter(ModelAdapter):
    """Dependency-free prototype classifier for small structured datasets."""

    def fit(self, train_data: Any, *, step: int, n_samples: int, random_seed: int) -> Any:
        if not train_data:
            return {"prototypes": {}, "majority": 0}

        grouped = {}
        counts = {}
        for features, label in train_data:
            feature_vector = self._flatten_features(features)
            grouped.setdefault(int(label), []).append(feature_vector)
            counts[int(label)] = counts.get(int(label), 0) + 1

        prototypes = {}
        for label, vectors in grouped.items():
            dimension = len(vectors[0])
            mean_vector = []
            for index in range(dimension):
                mean_vector.append(sum(vector[index] for vector in vectors) / len(vectors))
            prototypes[label] = tuple(mean_vector)

        majority = max(counts.items(), key=lambda item: item[1])[0]
        return {"prototypes": prototypes, "majority": majority}

    def evaluate(self, model: Any, eval_data: Any, *, step: int) -> float:
        if not eval_data:
            return 0.0

        mistakes = 0
        prototypes = model.get("prototypes", {})
        majority = model.get("majority", 0)
        for features, label in eval_data:
            feature_vector = self._flatten_features(features)
            if prototypes:
                predicted = min(
                    prototypes.items(),
                    key=lambda item: self._squared_distance(item[1], feature_vector),
                )[0]
            else:
                predicted = majority
            if int(predicted) != int(label):
                mistakes += 1
        return mistakes / len(eval_data)

    @staticmethod
    def _flatten_features(features: Any) -> Tuple[float, ...]:
        if isinstance(features, (list, tuple)):
            return tuple(float(value) for value in features)
        return (float(features),)

    @staticmethod
    def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
        return sum((float(lv) - float(rv)) ** 2 for lv, rv in zip(left, right))


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
        self.epochs = max(1, int(epochs))
        self.batch_size = max(1, int(batch_size))
        self.device = device

    def fit(self, train_data: Any, *, step: int, n_samples: int, random_seed: int) -> Any:
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("TorchModelAdapter requires the `torch` extra.") from exc

        torch.manual_seed(random_seed)
        model = self.model_factory().to(self.device)
        optimizer = (
            self.optimizer_factory(model.parameters())
            if self.optimizer_factory is not None
            else torch.optim.Adam(model.parameters(), lr=1e-3)
        )
        criterion = torch.nn.CrossEntropyLoss()
        train_data = self._coerce_dataset(train_data, torch)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

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

        return model

    def evaluate(self, model: Any, eval_data: Any, *, step: int) -> float:
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("TorchModelAdapter requires the `torch` extra.") from exc

        eval_data = self._coerce_dataset(eval_data, torch)
        loader = DataLoader(eval_data, batch_size=self.batch_size, shuffle=False)
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
            return float("inf")
        return total_loss / total_count

    @staticmethod
    def _coerce_dataset(block: Any, torch: Any) -> Any:
        if not isinstance(block, Sequence) or not block:
            return block

        first_item = block[0]
        if not isinstance(first_item, Sequence) or len(first_item) != 2:
            return block

        from torch.utils.data import TensorDataset

        feature_tensors = []
        labels = []
        for features, label in block:
            if torch.is_tensor(features):
                feature_tensor = features.detach().to(dtype=torch.float32)
            elif isinstance(features, Sequence):
                feature_tensor = torch.tensor([float(value) for value in features], dtype=torch.float32)
            else:
                feature_tensor = torch.tensor([float(features)], dtype=torch.float32)
            feature_tensors.append(feature_tensor)
            labels.append(int(label))

        inputs = torch.stack(feature_tensors)
        targets = torch.tensor(labels, dtype=torch.long)
        return TensorDataset(inputs, targets)
