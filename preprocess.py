"""Prepare CSV/XES event logs into one portable, train-only-fitted artifact."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from settings import CAUSAL_EVENT_TIME_FIELDS, DATASETS, DatasetSpec

ARTIFACT_VERSION = "mfml-prepared-v1"
PAD_TOKEN, UNK_TOKEN, MISSING_TOKEN = "<pad>", "<unk>", "<missing>"
ACTIVITY_UNK_TOKEN = "<unk_activity>"
PAD_ID, UNK_ID = 0, 1
SECONDS_PER_DAY = 86400.0
_TEMPLATE_FIELD = re.compile(r"{{\s*([^{}]+?)\s*}}")
_NON_IDENTIFIER = re.compile(r"[^0-9A-Za-z_]+")


def normalize_category(value: Any) -> str:
    """Return one stable categorical representation for every pipeline stage."""

    if value is None:
        return MISSING_TOKEN
    try:
        if bool(pd.isna(value)):
            return MISSING_TOKEN
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        text = str(int(value))
    elif isinstance(value, (np.floating, float)):
        numeric = float(value)
        text = (
            str(int(numeric))
            if math.isfinite(numeric) and numeric.is_integer()
            else format(numeric, ".15g")
        )
    else:
        text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.strip().split()).casefold()
    return text or MISSING_TOKEN


def _case_identifier(value: Any) -> str:
    """Stringify case IDs without applying categorical case folding."""

    if value is None:
        raise ValueError("case_column contains missing values")
    try:
        if bool(pd.isna(value)):
            raise ValueError("case_column contains missing values")
    except (TypeError, ValueError) as exc:
        if (
            isinstance(exc, ValueError)
            and str(exc) == "case_column contains missing values"
        ):
            raise
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        text = str(int(value))
    elif isinstance(value, (np.floating, float)):
        numeric = float(value)
        text = (
            str(int(numeric))
            if math.isfinite(numeric) and numeric.is_integer()
            else format(numeric, ".15g")
        )
    else:
        text = str(value)
    text = unicodedata.normalize("NFKC", text).strip()
    if not text:
        raise ValueError("case_column contains an empty identifier")
    return text


def _display_value(value: Any) -> str:
    normalized = normalize_category(value)
    return "unknown" if normalized == MISSING_TOKEN else str(value).strip()


def _template_aliases(values: Mapping[str, Any]) -> Dict[str, Any]:
    aliases: Dict[str, Any] = {}
    for key, value in values.items():
        key_text = str(key)
        aliases[key_text] = value
        aliases[key_text.replace(":", "_").replace(" ", "_")] = value
        aliases[_NON_IDENTIFIER.sub("_", key_text).strip("_")] = value
    return aliases


def render_template(template: Optional[str], values: Mapping[str, Any]) -> str:
    """Render only ``{{dictionary keys}}`` without evaluating template code.

    Both exact CSV keys (for example ``case:Project``) and their safe aliases
    (``case_Project``) are supported.  This fixes the old Jinja colon-key issue
    while keeping untrusted CSV contents inert.
    """

    if not template:
        return ""
    aliases = _template_aliases(values)

    def replace(match: "re.Match[str]") -> str:
        key = match.group(1).strip()
        if key not in aliases:
            raise ValueError(
                "Template references an undeclared or prohibited input field: " + key
            )
        return _display_value(aliases[key])

    return " ".join(_TEMPLATE_FIELD.sub(replace, template).split())


@dataclass
class SampleRecord:
    """All modalities and both labels for exactly one case prefix."""

    case_id: str
    split: str
    prefix_len: int
    case_length: int
    progress: float
    prefix_end_time: str
    text: str
    event_segments: List[str]
    trace_text: str
    categorical_events: List[List[str]]
    numeric_events: List[List[float]]
    next_activity: str
    remaining_time_days: float
    next_time_days: float = 0.0
    numeric_raw_events: List[List[float]] = field(default_factory=list)

    @property
    def event_features(self) -> Dict[str, Any]:
        return {
            "categorical": self.categorical_events,
            "numeric": self.numeric_events,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SampleRecord":
        return cls(**dict(values))


def _encode_length(tokenizer: Any, text: str) -> int:
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=True))
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    ids = encoded["input_ids"]
    return len(ids[0] if ids and isinstance(ids[0], list) else ids)


def text_with_complete_event_budget(
    record: SampleRecord, tokenizer: Any, max_tokens: int
) -> str:
    """Keep the longest contiguous suffix of complete event snippets.

    No event snippet is cut in the middle.  Trace-level context is appended only
    when it fits after the event suffix.
    """

    selected: List[str] = []
    for segment in reversed(record.event_segments):
        candidate = " ".join([segment] + selected)
        if _encode_length(tokenizer, candidate) <= max_tokens:
            selected.insert(0, segment)
        else:
            break
    event_text = " ".join(selected)
    if record.trace_text:
        candidate = " ".join(item for item in (event_text, record.trace_text) if item)
        if _encode_length(tokenizer, candidate) <= max_tokens:
            return candidate
    if event_text:
        return event_text
    # An unusually long single event is omitted rather than partially exposed.
    fallback = "No complete event description fits the token budget."
    if _encode_length(tokenizer, fallback) <= max_tokens:
        return fallback
    return ""


class DynamicCollator:
    """Right-pad event and token sequences to each batch's actual maximum."""

    def __init__(
        self,
        tokenizer: Any,
        vocabs: Mapping[str, Mapping[str, int]],
        activity_vocab: Mapping[str, int],
        scalers: Mapping[str, Any],
        categorical_columns: Sequence[str],
        max_events: int,
        max_tokens: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.vocabs = {key: dict(value) for key, value in vocabs.items()}
        self.activity_vocab = dict(activity_vocab)
        self.scalers = dict(scalers)
        self.categorical_columns = list(categorical_columns)
        self.max_events = int(max_events)
        self.max_tokens = int(max_tokens)

    def __call__(
        self, records: Sequence[Union[SampleRecord, Mapping[str, Any]]]
    ) -> Dict[str, Any]:
        parsed = [
            item if isinstance(item, SampleRecord) else SampleRecord.from_dict(item)
            for item in records
        ]
        if not parsed:
            raise ValueError("Cannot collate an empty batch")
        selected_categories = [
            item.categorical_events[-self.max_events :] for item in parsed
        ]
        selected_numeric = [item.numeric_events[-self.max_events :] for item in parsed]
        lengths = torch.tensor(
            [len(item) for item in selected_categories], dtype=torch.long
        )
        if bool((lengths <= 0).any()):
            raise ValueError("Every prefix must contain at least one event")
        batch_size = len(parsed)
        time_steps = int(lengths.max().item())
        feature_count = len(self.categorical_columns)
        categorical_ids = torch.full(
            (batch_size, time_steps, feature_count), PAD_ID, dtype=torch.long
        )
        numeric_features = torch.zeros((batch_size, time_steps, 2), dtype=torch.float32)
        event_mask = torch.zeros((batch_size, time_steps), dtype=torch.bool)
        numeric_scaler = self.scalers["numeric"]
        numeric_mean = np.asarray(numeric_scaler["mean"], dtype=np.float64)
        numeric_scale = np.asarray(numeric_scaler["scale"], dtype=np.float64)

        for batch_index, (categories, numeric) in enumerate(
            zip(selected_categories, selected_numeric)
        ):
            length = len(categories)
            if length != len(numeric):
                raise ValueError("Categorical/numeric modality length mismatch")
            if any(len(event) != feature_count for event in categories):
                raise ValueError(
                    "Categorical event width does not match configured columns"
                )
            for time_index, event in enumerate(categories):
                for feature_index, (column, value) in enumerate(
                    zip(self.categorical_columns, event)
                ):
                    categorical_ids[batch_index, time_index, feature_index] = (
                        self.vocabs[column].get(normalize_category(value), UNK_ID)
                    )
            numeric_array = np.asarray(numeric, dtype=np.float64)
            numeric_array = np.log1p(np.maximum(numeric_array, 0.0))
            numeric_array = (numeric_array - numeric_mean) / numeric_scale
            numeric_features[batch_index, :length] = torch.as_tensor(
                numeric_array, dtype=torch.float32
            )
            event_mask[batch_index, :length] = True

        texts = [
            text_with_complete_event_budget(item, self.tokenizer, self.max_tokens)
            for item in parsed
        ]
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"].bool()
        if input_ids.shape[1] > self.max_tokens:
            raise RuntimeError("Whole-event token budgeting exceeded max_tokens")

        target_scaler = self.scalers["remaining_time"]
        remaining_days = torch.tensor(
            [float(item.remaining_time_days) for item in parsed], dtype=torch.float32
        )
        transformed = (
            torch.log1p(torch.clamp(remaining_days, min=0.0))
            if "log1p" in target_scaler["transform"]
            else remaining_days
        )
        remaining_z = (transformed - float(target_scaler["mean"])) / float(
            target_scaler["scale"]
        )
        activity = torch.tensor(
            [
                self.activity_vocab.get(
                    normalize_category(item.next_activity),
                    self.activity_vocab[ACTIVITY_UNK_TOKEN],
                )
                for item in parsed
            ],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "categorical_ids": categorical_ids,
            "numeric_features": numeric_features,
            "event_mask": event_mask,
            "lengths": lengths,
            "labels": {
                "activity": activity,
                "remaining_time_z": remaining_z,
                "remaining_time_days": remaining_days,
            },
            "meta": {
                "case_id": [item.case_id for item in parsed],
                "prefix_len": torch.tensor(
                    [item.prefix_len for item in parsed], dtype=torch.long
                ),
                "progress": torch.tensor(
                    [item.progress for item in parsed], dtype=torch.float32
                ),
                "prefix_end_time": [item.prefix_end_time for item in parsed],
            },
        }


def _deterministic_vector(token: str, dimension: int, seed: int) -> np.ndarray:
    digest = hashlib.sha256((str(seed) + "\0" + token).encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**32)
    generator = np.random.RandomState(local_seed)
    return generator.normal(0.0, 0.02, dimension).astype(np.float32)


@dataclass
class PreparedData:
    samples: dict[str, list[SampleRecord]]
    vocabs: dict[str, dict[str, int]]
    activity_vocab: dict[str, int]
    scalers: dict
    cbow_vectors: dict[str, torch.Tensor]
    metadata: dict
    path: Path
    identity: str
    config: dict = field(default_factory=dict)

    @property
    def categorical_columns(self) -> list[str]:
        return list(self.metadata["categorical_columns"])

    @property
    def num_classes(self) -> int:
        return len(self.activity_vocab)

    @property
    def max_events(self) -> int:
        return int(self.metadata["max_events"])

    def make_collator(self, tokenizer: Any) -> DynamicCollator:
        return DynamicCollator(
            tokenizer,
            self.vocabs,
            self.activity_vocab,
            self.scalers,
            self.categorical_columns,
            self.max_events,
            self.metadata["max_tokens"],
        )

    def inverse_remaining_time(self, values):
        scaler = self.scalers["remaining_time"]
        if isinstance(values, torch.Tensor):
            return torch.expm1(values * scaler["scale"] + scaler["mean"]).clamp_min(0)
        values = np.asarray(values, dtype=np.float64)
        return np.maximum(np.expm1(values * scaler["scale"] + scaler["mean"]), 0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xes_rows(path: Path):
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag.rsplit("}", 1)[-1] != "trace":
                continue
            trace = {}
            for child in element:
                if child.tag.rsplit("}", 1)[-1] != "event" and "key" in child.attrib:
                    key = "case:" + child.attrib["key"]
                    if key in trace:
                        raise ValueError("Duplicate XES trace attribute: " + key)
                    trace[key] = child.attrib.get("value", "")
            if not trace.get("case:concept:name", "").strip():
                raise ValueError("XES trace has no nonempty concept:name")
            for event in element:
                if event.tag.rsplit("}", 1)[-1] != "event":
                    continue
                values = dict(trace)
                for child in event:
                    if "key" in child.attrib:
                        key = child.attrib["key"]
                        if key in values:
                            raise ValueError(
                                "Duplicate or conflicting XES event attribute: " + key
                            )
                        values[key] = child.attrib.get("value", "")
                yield values
            element.clear()


_ALIASES = {
    "case": ("case", "case:concept:name", "case_id", "Case ID"),
    "activity": ("activity", "concept:name", "Activity"),
    "timestamp": ("timestamp", "time:timestamp", "Timestamp"),
    "resource": ("resource", "org:resource", "Resource"),
}


def _load_source_frame(path: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, dict]:
    name = path.name.lower()
    if name.endswith((".xes", ".xes.gz")):
        frame = pd.DataFrame(_xes_rows(path)).fillna("")
    elif name.endswith((".csv", ".csv.gz")):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding=spec.csv_encoding, newline="") as handle:
            headers = next(csv.reader(handle), [])
        if len(headers) != len(set(headers)):
            raise ValueError("CSV contains duplicate column names")
        frame = pd.read_csv(
            path, dtype=str, keep_default_na=False, encoding=spec.csv_encoding
        )
    else:
        raise ValueError("Expected CSV, CSV.gz, XES, or XES.gz input")
    required = {
        "case",
        "activity",
        "timestamp",
        *spec.event_categorical_columns,
        *spec.event_text_columns,
        *spec.static_columns,
    }
    mapping = {}
    for canonical in sorted(required):
        declared = [old for old, new in spec.column_mapping.items() if new == canonical]
        # Production has both start and completion times: the registered
        # completion column, or an already canonical timestamp, is required.
        if canonical == "timestamp" and "Complete Timestamp" in declared:
            candidates = ["timestamp", "Complete Timestamp"]
        else:
            candidates = [canonical, *declared, *_ALIASES.get(canonical, ())]
        present = sorted(set(candidates).intersection(frame.columns))
        if not present:
            raise ValueError(
                f"Missing required column {canonical!r}; accepted names: {sorted(set(candidates))}"
            )
        if len(present) > 1:
            raise ValueError(
                f"Conflicting aliases for {canonical!r}: {present}; keep exactly one"
            )
        mapping[present[0]] = canonical
    if len(mapping) != len(required):
        raise ValueError("One source column maps to more than one required field")
    frame = frame.rename(columns=mapping)
    for column in ("case", "activity", "timestamp"):
        if frame[column].str.strip().eq("").any():
            raise ValueError("Empty required value in " + column)
    frame["case"] = frame["case"].str.strip()
    timestamps = pd.to_datetime(
        frame["timestamp"], format="mixed", errors="coerce", utc=True
    )
    if timestamps.isna().any():
        raise ValueError(
            f"{int(timestamps.isna().sum())} timestamp value(s) could not be parsed"
        )
    frame["timestamp"] = timestamps.map(lambda value: value.isoformat())
    statistics = {
        "cases": int(frame["case"].nunique()),
        "events": int(len(frame)),
        "activities": int(frame["activity"].nunique()),
    }
    if spec.expected_statistics is not None and statistics != spec.expected_statistics:
        raise ValueError(
            f"Dataset statistics mismatch: observed={statistics}; expected={spec.expected_statistics}"
        )
    # Preserve the original pipeline's canonical-CSV type inference, without
    # retaining the intermediate CSV or any local path in the artifact.
    # Converters preserve identifier spelling and literal "NA"/"null" values;
    # dtype=str alone still applies pandas' missing-value conversion.
    frame = pd.read_csv(
        io.StringIO(frame.to_csv(index=False)),
        converters={"case": str, "activity": str},
    )
    return frame, statistics


def _enrich(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["_source_order"] = np.arange(len(frame), dtype=np.int64)
    frame["_case_id"] = frame["case"].map(_case_identifier)
    frame["_timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed", utc=True)
    frame = frame.sort_values(
        ["_case_id", "_timestamp", "_source_order"], kind="mergesort"
    ).reset_index(drop=True)
    grouped = frame.groupby("_case_id", sort=False)
    frame["seconds_since_last"] = (
        grouped["_timestamp"].diff().dt.total_seconds().fillna(0.0)
    )
    frame["seconds_since_start"] = (
        frame["_timestamp"] - grouped["_timestamp"].transform("min")
    ).dt.total_seconds()
    frame["_computed_remaining_days"] = (
        grouped["_timestamp"].transform("max") - frame["_timestamp"]
    ).dt.total_seconds() / SECONDS_PER_DAY
    frame["timesincelastevent"] = frame["seconds_since_last"] / SECONDS_PER_DAY
    frame["timesincecasestart"] = frame["seconds_since_start"] / SECONDS_PER_DAY
    return frame


def split_cases(frame: pd.DataFrame) -> dict[str, list[str]]:
    cases = frame.groupby("_case_id", sort=False)["_timestamp"].min().reset_index()
    cases = cases.sort_values(["_timestamp", "_case_id"], kind="mergesort")
    count = len(cases)
    if count < 3:
        raise ValueError("At least three cases are required for train/validation/test")
    development = int(
        (Decimal(count) * Decimal("0.70")).to_integral_value(rounding=ROUND_FLOOR)
    )
    development = min(max(development, 2), count - 1)
    train = int(
        (Decimal(development) * Decimal("0.80")).to_integral_value(rounding=ROUND_FLOOR)
    )
    train = min(max(train, 1), development - 1)
    ids = cases["_case_id"].tolist()
    return {
        "train": ids[:train],
        "validation": ids[train:development],
        "test": ids[development:],
    }


def _validate_static(train: pd.DataFrame, spec: DatasetSpec) -> dict:
    report = {}
    for column in spec.static_columns:
        grouped = train.groupby("_case_id", sort=False)[column]
        missing = int(grouped.nth(0).map(normalize_category).eq(MISSING_TOKEN).sum())
        changing = int(
            grouped.apply(
                lambda values: values.map(normalize_category).nunique(dropna=False) > 1
            ).sum()
        )
        report[column] = {
            "missing_at_case_start": missing,
            "changes_within_case": changing,
        }
        if missing or changing:
            raise ValueError(
                "Static field must be available at case start and constant in training cases: "
                + column
            )
    return report


def _samples(
    frame: pd.DataFrame, split_ids: dict, spec: DatasetSpec, max_events: int
) -> dict:
    samples = {split: [] for split in split_ids}
    case_split = {case: split for split, ids in split_ids.items() for case in ids}
    visible = (
        *spec.event_categorical_columns,
        *spec.event_text_columns,
        "timestamp",
        *CAUSAL_EVENT_TIME_FIELDS,
    )
    for case_id, case in frame.groupby("_case_id", sort=False):
        rows = case.to_dict(orient="records")
        if len(rows) <= 2:
            continue
        static = {column: rows[0][column] for column in spec.static_columns}
        trace_text = (
            render_template(spec.trace_template, static)
            if spec.trace_template
            else (
                "Case attributes: "
                + "; ".join(
                    f"{column}={_display_value(static[column])}"
                    for column in spec.static_columns
                )
                + "."
                if static
                else ""
            )
        )
        segments = [
            render_template(
                spec.event_template, {column: row[column] for column in visible}
            )
            if spec.event_template
            else "; ".join(
                f"{column}={_display_value(row[column])}"
                for column in spec.event_categorical_columns
            )
            + "."
            for row in rows
        ]
        categorical = [
            [
                normalize_category(row[column])
                for column in spec.event_categorical_columns
            ]
            for row in rows
        ]
        numeric = [
            [float(row["seconds_since_last"]), float(row["seconds_since_start"])]
            for row in rows
        ]
        for prefix_len in range(2, len(rows)):
            start = max(0, prefix_len - max_events)
            current, target = rows[prefix_len - 1], rows[prefix_len]
            split = case_split[case_id]
            samples[split].append(
                SampleRecord(
                    case_id=str(case_id),
                    split=split,
                    prefix_len=prefix_len,
                    case_length=len(rows),
                    progress=prefix_len / len(rows),
                    prefix_end_time=current["_timestamp"].isoformat(),
                    text="",
                    event_segments=segments[start:prefix_len],
                    trace_text=trace_text,
                    categorical_events=categorical[start:prefix_len],
                    numeric_events=numeric[start:prefix_len],
                    next_activity=str(target["activity"]),
                    remaining_time_days=float(current["_computed_remaining_days"]),
                    next_time_days=float(
                        (target["_timestamp"] - current["_timestamp"]).total_seconds()
                        / SECONDS_PER_DAY
                    ),
                )
            )
    if any(not records for records in samples.values()):
        raise ValueError(
            "Every partition must contain prefixes with 2 <= prefix_length < case_length"
        )
    return samples


def _fit(
    train: pd.DataFrame, train_samples: list[SampleRecord], spec: DatasetSpec, seed: int
):
    from gensim.models import Word2Vec

    vocabs, vectors = {}, {}
    for feature_index, column in enumerate(spec.event_categorical_columns):
        tokens = sorted(
            set(train[column].map(normalize_category)) - {PAD_TOKEN, UNK_TOKEN}
        )
        vocab = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
            **{token: i + 2 for i, token in enumerate(tokens)},
        }
        sequences = [
            [normalize_category(value) for value in case[column].tolist()]
            for _, case in train.groupby("_case_id", sort=False)
        ]
        model = Word2Vec(
            sentences=sequences,
            vector_size=32,
            window=5,
            min_count=1,
            workers=1,
            sg=0,
            seed=seed + feature_index,
            epochs=20,
        )
        matrix = torch.zeros(len(vocab), 32, dtype=torch.float32)
        # UNK is an explicit train-independent initialization, not a fallback
        # for missing gensim or failed CBOW. Every real token must be fitted.
        matrix[UNK_ID] = torch.from_numpy(
            _deterministic_vector(UNK_TOKEN + column, 32, seed)
        )
        for token, index in vocab.items():
            if index >= 2:
                matrix[index] = torch.from_numpy(model.wv[token].copy())
        vocabs[column], vectors[column] = vocab, matrix
    labels = sorted(
        {normalize_category(sample.next_activity) for sample in train_samples}
    )
    if ACTIVITY_UNK_TOKEN in labels:
        raise ValueError("Activity name collides with reserved <unk_activity> token")
    activity_vocab = {label: i for i, label in enumerate(labels)}
    activity_vocab[ACTIVITY_UNK_TOKEN] = len(activity_vocab)
    numeric = np.log1p(
        np.maximum(
            train[["seconds_since_last", "seconds_since_start"]].to_numpy(
                dtype=np.float64
            ),
            0,
        )
    )
    numeric_scale = numeric.std(axis=0)
    numeric_scale[numeric_scale < 1e-12] = 1.0
    target = np.log1p(
        np.asarray(
            [sample.remaining_time_days for sample in train_samples], dtype=np.float64
        )
    )
    scale = float(target.std())
    scalers = {
        "numeric": {
            "columns": ["seconds_since_last", "seconds_since_start"],
            "transform": "standardized_log1p_seconds",
            "mean": numeric.mean(axis=0).tolist(),
            "scale": numeric_scale.tolist(),
            "fit_split": "train",
        },
        "remaining_time": {
            "unit": "days",
            "transform": "standardized_log1p_days",
            "mean": float(target.mean()),
            "scale": 1.0 if scale < 1e-12 else scale,
            "fit_split": "train",
        },
    }
    return vocabs, activity_vocab, scalers, vectors


def _save_prepared(prepared: PreparedData) -> None:
    payload = {
        "version": ARTIFACT_VERSION,
        "samples": {
            split: [vars(sample).copy() for sample in records]
            for split, records in prepared.samples.items()
        },
        "vocabs": prepared.vocabs,
        "activity_vocab": prepared.activity_vocab,
        "scalers": prepared.scalers,
        "cbow_vectors": prepared.cbow_vectors,
        "metadata": prepared.metadata,
        "config": prepared.config,
    }
    prepared.path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a completed artifact.
    with prepared.path.open("xb") as handle:
        try:
            torch.save(payload, handle)
        except BaseException:
            handle.close()
            prepared.path.unlink(missing_ok=True)
            raise
    prepared.identity = sha256_file(prepared.path)


def load_prepared(path: str | Path) -> PreparedData:
    source = Path(path).resolve()
    if source.is_dir():
        source = source / "prepared.pt"
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported prepared artifact version")
    samples = {
        split: [SampleRecord.from_dict(record) for record in records]
        for split, records in payload["samples"].items()
    }
    prepared = PreparedData(
        samples=samples,
        vocabs=payload["vocabs"],
        activity_vocab=payload["activity_vocab"],
        scalers=payload["scalers"],
        cbow_vectors=payload["cbow_vectors"],
        metadata=payload["metadata"],
        path=source,
        identity=sha256_file(source),
        config=payload["config"],
    )
    partitions = prepared.metadata["split_case_ids"]
    if set(partitions) != {"train", "validation", "test"} or set(samples) != set(
        partitions
    ):
        raise ValueError("Prepared partitions are incomplete")
    all_cases = [case for ids in partitions.values() for case in ids]
    if len(all_cases) != len(set(all_cases)):
        raise ValueError("Prepared case partitions overlap")
    for split, records in samples.items():
        allowed = set(partitions[split])
        if not records or any(
            record.case_id not in allowed
            or record.split != split
            or not 2 <= record.prefix_len < record.case_length
            for record in records
        ):
            raise ValueError("Prepared sample identities do not match their partition")
    for column in prepared.categorical_columns:
        vocab, vector = prepared.vocabs[column], prepared.cbow_vectors[column]
        if (
            sorted(vocab.values()) != list(range(len(vocab)))
            or vocab.get(PAD_TOKEN) != 0
            or vocab.get(UNK_TOKEN) != 1
        ):
            raise ValueError("Invalid categorical vocabulary: " + column)
        if (
            not isinstance(vector, torch.Tensor)
            or vector.shape != (len(vocab), 32)
            or not torch.isfinite(vector).all()
        ):
            raise ValueError("Invalid CBOW tensor: " + column)
    if (
        sorted(prepared.activity_vocab.values()) != list(range(prepared.num_classes))
        or ACTIVITY_UNK_TOKEN not in prepared.activity_vocab
    ):
        raise ValueError("Invalid activity vocabulary")
    return prepared


def prepare_data(
    data_path: str | Path,
    dataset: str | DatasetSpec,
    output: str | Path | None = None,
    seed: int = 42,
) -> PreparedData:
    """Fit preprocessing on the chronological training cases and save prepared.pt.

    Registered datasets verify event/case/activity counts. Input bytes are
    fingerprinted for provenance; canonical CSV and XES exports are both accepted.
    No source path or external cache is needed to load the saved artifact.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if isinstance(dataset, str):
        if dataset not in DATASETS:
            raise ValueError(
                "Unknown dataset: " + dataset + "; choose " + ", ".join(DATASETS)
            )
        spec = DATASETS[dataset]
    elif isinstance(dataset, DatasetSpec):
        spec = dataset
    else:
        raise TypeError("dataset must be a registered name or DatasetSpec")
    spec.validate()
    destination = Path(output or "prepared.pt").resolve()
    if destination.suffix.lower() != ".pt":
        destination /= "prepared.pt"
    if destination.exists():
        raise FileExistsError(
            "Prepared artifact already exists; choose a new output: " + str(destination)
        )
    source = Path(data_path).resolve()
    frame, statistics = _load_source_frame(source, spec)
    frame = _enrich(frame)
    if (
        frame["activity"]
        .map(normalize_category)
        .isin({ACTIVITY_UNK_TOKEN, PAD_TOKEN, UNK_TOKEN})
        .any()
    ):
        raise ValueError("Activity names must not collide with reserved PAD/UNK tokens")
    partitions = split_cases(frame)
    train = frame[frame["_case_id"].isin(set(partitions["train"]))]
    static_report = _validate_static(train, spec)
    lengths = train.groupby("_case_id", sort=False).size()
    eligible = lengths[lengths > 2]
    if eligible.empty:
        raise ValueError("Training cases contain no eligible prefixes")
    max_events = min(int(eligible.max()) - 1, 256)
    samples = _samples(frame, partitions, spec, max_events)
    vocabs, activity_vocab, scalers, vectors = _fit(train, samples["train"], spec, seed)
    metadata = {
        "dataset": spec.name,
        "seed": seed,
        "max_tokens": 512,
        "max_events": max_events,
        "categorical_columns": list(spec.event_categorical_columns),
        "split_case_ids": partitions,
        "split_hash": hashlib.sha256(
            json.dumps(
                partitions, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "source_sha256": sha256_file(source),
        "source_statistics": statistics,
        "source_format": "xes" if ".xes" in source.name.lower() else "csv",
        "case_counts": {split: len(ids) for split, ids in partitions.items()},
        "sample_counts": {split: len(records) for split, records in samples.items()},
        "static_attribute_validation": static_report,
        "cbow": {
            "backend": "gensim-cbow",
            "algorithm": "CBOW (sg=0)",
            "dimension": 32,
            "window": 5,
            "epochs": 20,
            "workers": 1,
            "train_only": True,
        },
        "train_only_fitted": [
            "vocabs",
            "activity_vocab",
            "cbow_vectors",
            "numeric_scaler",
            "remaining_time_scaler",
        ],
        "unknown_activity_counts": {
            split: sum(
                normalize_category(record.next_activity) not in activity_vocab
                for record in records
            )
            for split, records in samples.items()
        },
    }
    prepared = PreparedData(
        samples,
        vocabs,
        activity_vocab,
        scalers,
        vectors,
        metadata,
        destination,
        "",
        spec.to_dict(),
    )
    _save_prepared(prepared)
    return prepared


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument(
        "--data", required=True, help="CSV, CSV.gz, XES, or XES.gz event log"
    )
    parser.add_argument(
        "--output", default="prepared.pt", help="New .pt path or output directory"
    )
    parser.add_argument("--seed", type=int, default=42)
    arguments = sys.argv[1:] if argv is None else list(argv)
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    try:
        prepared = prepare_data(args.data, args.dataset, args.output, args.seed)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "prepared": str(prepared.path),
                "identity": prepared.identity,
                "dataset": prepared.metadata["dataset"],
                "sample_counts": prepared.metadata["sample_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
