"""Built-in timeline adapters."""

from __future__ import annotations

import json
import math
import random
import struct
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from ..interfaces import TimelineDataset


class ArrayTimelineAdapter(TimelineDataset):
    """Timeline adapter backed by in-memory Python sequences."""

    def __init__(self, train_blocks: Sequence[Sequence[Any]], eval_blocks: Sequence[Sequence[Any]]):
        if len(train_blocks) != len(eval_blocks):
            raise ValueError("train_blocks and eval_blocks must have the same number of timesteps.")
        if len(train_blocks) == 0:
            raise ValueError("A timeline must contain at least one timestep.")
        self._train_blocks = [list(block) for block in train_blocks]
        self._eval_blocks = [list(block) for block in eval_blocks]
        if any(not block for block in self._eval_blocks):
            raise ValueError("Evaluation blocks must not be empty.")

    @property
    def num_steps(self) -> int:
        return len(self._train_blocks)

    def get_train_block(self, step: int) -> List[Any]:
        return self._train_blocks[step]

    def get_eval_block(self, step: int) -> List[Any]:
        return self._eval_blocks[step]

    def train_size(self, step: int) -> int:
        return len(self._train_blocks[step])

    def sample_train_block(self, step: int, n_samples: int, *, seed: int) -> List[Any]:
        block = self._train_blocks[step]
        actual_n = min(max(0, int(n_samples)), len(block))
        if actual_n >= len(block):
            return list(block)
        if actual_n <= 0:
            return []
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(block)), actual_n))
        return [block[idx] for idx in indices]


class TorchTimelineAdapter(TimelineDataset):
    """Timeline adapter for PyTorch datasets or lists of tensor/label pairs."""

    def __init__(self, train_blocks: Sequence[Any], eval_blocks: Sequence[Any]):
        if len(train_blocks) != len(eval_blocks):
            raise ValueError("train_blocks and eval_blocks must have the same number of timesteps.")
        if len(train_blocks) == 0:
            raise ValueError("A timeline must contain at least one timestep.")
        self._train_blocks = list(train_blocks)
        self._eval_blocks = list(eval_blocks)
        if any(len(block) == 0 for block in self._eval_blocks):
            raise ValueError("Evaluation blocks must not be empty.")

    @property
    def num_steps(self) -> int:
        return len(self._train_blocks)

    def get_train_block(self, step: int) -> Any:
        return self._train_blocks[step]

    def get_eval_block(self, step: int) -> Any:
        return self._eval_blocks[step]

    def train_size(self, step: int) -> int:
        return len(self._train_blocks[step])

    def sample_train_block(self, step: int, n_samples: int, *, seed: int) -> Any:
        block = self._train_blocks[step]
        actual_n = min(max(0, int(n_samples)), len(block))
        if actual_n >= len(block):
            return block
        if actual_n <= 0:
            return []

        try:
            from torch.utils.data import Subset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("TorchTimelineAdapter requires the `torch` extra to sample datasets.") from exc

        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(block)), actual_n))
        return Subset(block, indices)


class BinaryTimelineAdapter(ArrayTimelineAdapter):
    """Load a ``train/*.bin`` and ``test/*.bin`` timeline directory."""

    def __init__(self, root_dir: str | Path, *, num_features: int | None = None):
        root_path = Path(root_dir)
        metadata_path = root_path / "metadata.json"
        if num_features is None and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if not isinstance(metadata, dict):
                raise ValueError("metadata.json must contain a JSON object.")
            num_features = metadata.get("num_features")
        if isinstance(num_features, bool) or not isinstance(num_features, Integral) or num_features < 1:
            raise ValueError("num_features must be a positive integer, supplied explicitly or in metadata.json.")

        train_dir = root_path / "train"
        eval_dir = root_path / "test"
        train_ids = self._block_ids(train_dir)
        eval_ids = self._block_ids(eval_dir)
        if train_ids != eval_ids:
            raise ValueError("Train and evaluation block identifiers must match exactly.")
        train_blocks = self._load_blocks(train_dir, num_features)
        eval_blocks = self._load_blocks(eval_dir, num_features)
        super().__init__(train_blocks=train_blocks, eval_blocks=eval_blocks)

    @staticmethod
    def _block_ids(directory: Path) -> set[str]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Timeline directory not found: {directory}")
        stems = {path.stem for path in directory.glob("*.bin")}
        numeric_ids = [int(stem) for stem in stems if stem.lstrip("-").isdigit()]
        if len(set(numeric_ids)) != len(numeric_ids) or any(value < 0 for value in numeric_ids):
            raise ValueError(f"Ambiguous or negative timestep identifiers in {directory}.")
        return stems

    @classmethod
    def _load_blocks(cls, directory: Path, num_features: int) -> List[List[Any]]:
        if not directory.exists():
            raise FileNotFoundError(f"Timeline directory not found: {directory}")

        files = sorted(directory.glob("*.bin"), key=lambda path: cls._sort_key(path.stem))
        return [cls._read_bin_file(path, num_features) for path in files]

    @staticmethod
    def _sort_key(stem: str) -> Any:
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    @staticmethod
    def _read_bin_file(path: Path, num_features: int) -> List[Any]:
        payload = path.read_bytes()
        values_per_record = num_features + 1
        record_format = "<" + "f" * values_per_record
        record_size = struct.calcsize(record_format)
        if record_size <= 0 or len(payload) % record_size != 0:
            raise ValueError(f"Unexpected record layout in {path}")

        samples = []
        for record in struct.iter_unpack(record_format, payload):
            features = tuple(float(value) for value in record[:-1])
            label_value = record[-1]
            if not all(math.isfinite(value) for value in features):
                raise ValueError(f"Non-finite feature value in {path}.")
            if not math.isfinite(label_value) or label_value < 0 or not label_value.is_integer():
                raise ValueError(f"Labels in {path} must be finite nonnegative integers.")
            label = int(label_value)
            samples.append((features, label))
        return samples
