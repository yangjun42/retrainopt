"""Real-dataset download and conversion helpers for clean Phase 1 reruns."""

from __future__ import annotations

import bz2
import csv
import json
import random
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class RealDatasetSpec:
    name: str
    raw_relpath: str
    num_features: int
    label_candidates: Tuple[str, ...]
    urls: Tuple[str, ...]
    compressed: bool = False


REAL_DATASET_SPECS: Dict[str, RealDatasetSpec] = {
    "airlines": RealDatasetSpec(
        name="airlines",
        raw_relpath="airlines/raw.csv",
        num_features=7,
        label_candidates=("Delay", "delay", "dep_delayed_15min", "Class"),
        urls=(
            "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/airlines.csv",
            "https://www.openml.org/data/get_csv/66526/phpvcoG8S",
        ),
    ),
    "electricity": RealDatasetSpec(
        name="electricity",
        raw_relpath="elec/raw.csv",
        num_features=6,
        label_candidates=("class", "Class", "label", "Label", "UP"),
        urls=(
            "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/elec.csv",
            "https://www.openml.org/data/get_csv/2419/electricity-normalized.csv",
        ),
    ),
    "covertype": RealDatasetSpec(
        name="covertype",
        raw_relpath="covertype/raw/covtype.libsvm.binary.scale",
        num_features=54,
        label_candidates=(),
        urls=("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2",),
        compressed=True,
    ),
}


def prepare_real_datasets(
    *,
    datasets: Sequence[str],
    raw_root: Path | str,
    output_root: Path | str,
    num_steps: int = 50,
    test_split: float = 0.2,
    seed: int = 42,
    download_missing: bool = False,
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    names = list(REAL_DATASET_SPECS) if len(datasets) == 1 and datasets[0] == "all" else list(datasets)
    for name in names:
        results[name] = prepare_real_dataset(
            name,
            raw_root=raw_root,
            output_root=output_root,
            num_steps=num_steps,
            test_split=test_split,
            seed=seed,
            download_missing=download_missing,
        )
    return results


def prepare_real_dataset(
    dataset_name: str,
    *,
    raw_root: Path | str,
    output_root: Path | str,
    num_steps: int = 50,
    test_split: float = 0.2,
    seed: int = 42,
    download_missing: bool = False,
) -> dict:
    spec = REAL_DATASET_SPECS.get(dataset_name)
    if spec is None:
        raise ValueError(f"Unknown real dataset: {dataset_name}")

    raw_path = Path(raw_root) / spec.raw_relpath
    if not raw_path.exists():
        if not download_missing:
            raise FileNotFoundError(
                f"Raw data not found for {dataset_name}: {raw_path}. "
                "Pass download_missing=True or prepare the raw files first."
            )
        _download_raw_dataset(spec, raw_path)

    if dataset_name == "covertype":
        features, labels = _load_libsvm_dataset(raw_path, spec.num_features)
    else:
        features, labels = _load_csv_dataset(raw_path, spec)
        if dataset_name == "electricity":
            features = _minmax_scale(features)

    output_dir = Path(output_root) / dataset_name
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    paired = list(zip(features, labels))
    rng = random.Random(seed)
    rng.shuffle(paired)
    batches = _split_evenly(paired, num_steps)

    stats = []
    for step, batch in enumerate(batches):
        train_rows, test_rows = _split_batch(batch, test_split=test_split, rng=rng)
        _write_bin_file(train_dir / f"{step}.bin", train_rows, spec.num_features)
        _write_bin_file(test_dir / f"{step}.bin", test_rows, spec.num_features)
        stats.append(
            {
                "step": step,
                "train": len(train_rows),
                "test": len(test_rows),
                "total": len(batch),
            }
        )

    metadata = {
        "dataset": dataset_name,
        "num_features": spec.num_features,
        "num_steps": num_steps,
        "seed": seed,
        "test_split": test_split,
        "raw_path": str(raw_path),
        "stats": stats,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "dataset": dataset_name,
        "raw_path": str(raw_path),
        "output_dir": str(output_dir),
        "num_features": spec.num_features,
        "num_steps": num_steps,
        "total_samples": len(paired),
    }


def _download_raw_dataset(spec: RealDatasetSpec, raw_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in spec.urls:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if spec.compressed:
                payload = bz2.decompress(payload)
            raw_path.write_bytes(payload)
            if raw_path.stat().st_size > 0:
                return
        except Exception as exc:  # pragma: no cover - network and mirror availability vary
            last_error = exc
    raise RuntimeError(f"Failed to download raw dataset for {spec.name}") from last_error


def _load_csv_dataset(path: Path, spec: RealDatasetSpec) -> Tuple[List[Tuple[float, ...]], List[int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV dataset is empty: {path}")

    headers = list(rows[0].keys())
    label_column = _select_label_column(headers, spec.label_candidates)
    numeric_columns = [
        column
        for column in headers
        if column != label_column and _column_is_numeric(rows, column)
    ]
    numeric_columns = numeric_columns[: spec.num_features]
    features = _parse_numeric_matrix(rows, numeric_columns)
    labels = _normalize_labels([row.get(label_column, "") for row in rows])
    return [tuple(row) for row in features], labels


def _load_libsvm_dataset(path: Path, num_features: int) -> Tuple[List[Tuple[float, ...]], List[int]]:
    features: List[Tuple[float, ...]] = []
    raw_labels: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        raw_labels.append(tokens[0])
        vector = [0.0] * num_features
        for token in tokens[1:]:
            index_str, value_str = token.split(":", 1)
            index = int(index_str) - 1
            if 0 <= index < num_features:
                vector[index] = float(value_str)
        features.append(tuple(vector))
    if not features:
        raise ValueError(f"LIBSVM dataset is empty: {path}")
    return features, _normalize_labels(raw_labels)


def _select_label_column(headers: Sequence[str], candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in headers:
            return candidate
    return headers[-1]


def _column_is_numeric(rows: Sequence[dict], column: str) -> bool:
    for row in rows:
        value = (row.get(column) or "").strip()
        if not value:
            continue
        try:
            float(value)
        except ValueError:
            return False
    return True


def _parse_numeric_matrix(rows: Sequence[dict], columns: Sequence[str]) -> List[List[float]]:
    matrix: List[List[float | None]] = []
    for row in rows:
        matrix.append(
            [
                None if not (row.get(column) or "").strip() else float((row.get(column) or "").strip())
                for column in columns
            ]
        )

    means: List[float] = []
    for column_index in range(len(columns)):
        values = [row[column_index] for row in matrix if row[column_index] is not None]
        means.append(sum(values) / len(values) if values else 0.0)

    resolved: List[List[float]] = []
    for row in matrix:
        resolved.append(
            [
                float(means[index] if value is None else value)
                for index, value in enumerate(row)
            ]
        )
    return resolved


def _normalize_labels(raw_labels: Sequence[object]) -> List[int]:
    cleaned = [str(label).strip() for label in raw_labels]
    try:
        numeric = [float(label) for label in cleaned]
    except ValueError:
        unique = sorted(set(cleaned))
        mapping = {value: index for index, value in enumerate(unique)}
        if len(unique) == 2:
            mapping = {unique[0]: 0, unique[1]: 1}
        return [int(mapping[label]) for label in cleaned]

    unique = sorted(set(numeric))
    if len(unique) == 2 and set(unique) != {0.0, 1.0}:
        positive = unique[-1]
        return [1 if value == positive else 0 for value in numeric]
    if len(unique) > 2:
        positive = unique[0]
        return [1 if value == positive else 0 for value in numeric]
    return [int(value) for value in numeric]


def _minmax_scale(matrix: Sequence[Sequence[float]]) -> List[Tuple[float, ...]]:
    if not matrix:
        return []
    num_features = len(matrix[0])
    mins = [min(row[index] for row in matrix) for index in range(num_features)]
    maxs = [max(row[index] for row in matrix) for index in range(num_features)]
    scaled: List[Tuple[float, ...]] = []
    for row in matrix:
        scaled.append(
            tuple(
                0.0 if abs(maxs[index] - mins[index]) <= 1e-12 else (row[index] - mins[index]) / (maxs[index] - mins[index])
                for index in range(num_features)
            )
        )
    return scaled


def _split_evenly(rows: Sequence[Tuple[Tuple[float, ...], int]], num_steps: int) -> List[List[Tuple[Tuple[float, ...], int]]]:
    batches: List[List[Tuple[Tuple[float, ...], int]]] = [[] for _ in range(num_steps)]
    for index, row in enumerate(rows):
        batches[index % num_steps].append(row)
    return batches


def _split_batch(
    batch: Sequence[Tuple[Tuple[float, ...], int]],
    *,
    test_split: float,
    rng: random.Random,
) -> Tuple[List[Tuple[Tuple[float, ...], int]], List[Tuple[Tuple[float, ...], int]]]:
    rows = list(batch)
    if not rows:
        return [], []
    rng.shuffle(rows)
    test_count = max(1, int(len(rows) * test_split))
    if test_count >= len(rows) and len(rows) > 1:
        test_count = len(rows) - 1
    test_rows = rows[-test_count:]
    train_rows = rows[:-test_count]
    return train_rows, test_rows


def _write_bin_file(
    path: Path,
    rows: Iterable[Tuple[Tuple[float, ...], int]],
    num_features: int,
) -> None:
    values_per_record = num_features + 1
    record_format = "<" + "f" * values_per_record
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for features, label in rows:
            padded = list(features[:num_features])
            if len(padded) < num_features:
                padded.extend([0.0] * (num_features - len(padded)))
            handle.write(struct.pack(record_format, *(padded + [float(label)])))
