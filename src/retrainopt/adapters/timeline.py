"""Built-in timeline adapters."""

from __future__ import annotations

import json
import random
import struct
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from ..interfaces import TimelineDataset


class ArrayTimelineAdapter(TimelineDataset):
    """Timeline adapter backed by in-memory Python sequences."""

    def __init__(self, train_blocks: Sequence[Sequence[Any]], eval_blocks: Sequence[Sequence[Any]]):
        if len(train_blocks) != len(eval_blocks):
            raise ValueError("train_blocks and eval_blocks must have the same number of timesteps.")
        self._train_blocks = [list(block) for block in train_blocks]
        self._eval_blocks = [list(block) for block in eval_blocks]

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
        self._train_blocks = list(train_blocks)
        self._eval_blocks = list(eval_blocks)

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
            num_features = int(metadata.get("num_features", 0)) or None
        if num_features is None:
            raise ValueError("num_features is required when metadata.json is missing or incomplete.")

        train_dir = root_path / "train"
        eval_dir = root_path / "test"
        train_blocks = self._load_blocks(train_dir, num_features)
        eval_blocks = self._load_blocks(eval_dir, num_features)
        super().__init__(train_blocks=train_blocks, eval_blocks=eval_blocks)

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
            label = int(round(record[-1]))
            samples.append((features, label))
        return samples
