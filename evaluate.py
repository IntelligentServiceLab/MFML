"""Evaluate a saved MFML model without refitting preprocessing or downloading BERT."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from preprocess import (
    ACTIVITY_UNK_TOKEN,
    load_prepared,
    normalize_category,
    sha256_file,
)

CHECKPOINT_VERSION = "mfml-simple-checkpoint-v1"
INPUT_NAMES = (
    "input_ids",
    "attention_mask",
    "categorical_ids",
    "numeric_features",
    "event_mask",
)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Use CUDA when available; an explicit unavailable device is an error."""
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu, cuda, and auto")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return result


def model_inputs(batch: dict, device: str | torch.device) -> dict[str, torch.Tensor]:
    """Pass only observed-prefix inputs to the model, never labels or metadata."""
    return {name: batch[name].to(device) for name in INPUT_NAMES}


def prediction_metrics(
    rows: list[dict], train_mean_remaining_days: float, known_activities: set[str]
) -> dict:
    """Pool all prefixes; an UNK prediction cannot match an unseen true label."""
    if not rows:
        raise ValueError("Cannot score an empty prediction set")
    true = [("activity", normalize_category(row["true_activity_raw"])) for row in rows]
    predicted = [
        ("unknown_prediction", "")
        if row["predicted_is_unknown"]
        else ("activity", normalize_category(row["predicted_activity"]))
        for row in rows
    ]
    # Tag the unknown prediction separately even if a real label spells UNK.
    support, predicted_support = Counter(true), Counter(predicted)
    correct = Counter(t for t, p in zip(true, predicted) if t == p)
    labels = sorted(set(true) | set(predicted))
    precisions = {
        name: correct[name] / max(predicted_support[name], 1) for name in labels
    }
    recalls = {name: correct[name] / max(support[name], 1) for name in labels}
    f1s = {
        name: 2
        * precisions[name]
        * recalls[name]
        / max(precisions[name] + recalls[name], 1e-12)
        for name in labels
    }
    actual_days = np.asarray([row["remaining_days"] for row in rows], dtype=np.float64)
    predicted_days = np.asarray(
        [row["predicted_remaining_time_days"] for row in rows], dtype=np.float64
    )
    if not np.isfinite(actual_days).all() or not np.isfinite(predicted_days).all():
        raise ValueError("Time metrics require finite remaining days")
    residual = predicted_days - actual_days
    mae = float(np.abs(residual).mean())
    rmse = float(np.sqrt(np.square(residual).mean()))
    macro_f1 = sum(f1s.values()) / len(labels)
    total = len(rows)
    return {
        "accuracy": sum(correct.values()) / total,
        "weighted_precision": sum(precisions[name] * support[name] for name in labels)
        / total,
        "weighted_f1": sum(f1s[name] * support[name] for name in labels) / total,
        "macro_f1": macro_f1,
        "mae_days": mae,
        "rmse_days": rmse,
        "selection_score": 0.5 * macro_f1
        + 0.5 / (1.0 + mae / max(float(train_mean_remaining_days), 1e-12)),
        "sample_count": total,
        "unknown_true_activity_count": sum(
            label[1] not in known_activities for label in true
        ),
        "unit": "days",
        "aggregation": "all_valid_prefixes",
    }


def evaluate_model(
    model,
    prepared,
    tokenizer,
    split: str = "test",
    device: str | torch.device = "auto",
    batch_size: int = 16,
) -> tuple[dict, list[dict]]:
    """Shared validation/test implementation; validation never reads test samples."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split: " + split)
    samples = prepared.samples[split]
    if not samples or batch_size < 1:
        raise ValueError("Evaluation requires a nonempty split and positive batch size")
    resolved = resolve_device(device)
    model.to(resolved)
    previous_mode = model.training
    model.eval()
    collator = prepared.make_collator(tokenizer)
    names = [""] * prepared.num_classes
    for name, index in prepared.activity_vocab.items():
        names[index] = name
    unknown_id = prepared.activity_vocab[ACTIVITY_UNK_TOKEN]
    rows: list[dict] = []
    try:
        with torch.no_grad():
            for offset in range(0, len(samples), batch_size):
                records = samples[offset : offset + batch_size]
                output = model(**model_inputs(collator(records), resolved))
                logits = output["activity_logits"].detach().float().cpu().numpy()
                z = (
                    output["remaining_time_z"]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                gates = output["gates"].detach().float().cpu().numpy()
                if (
                    logits.shape != (len(records), prepared.num_classes)
                    or not np.isfinite(logits).all()
                ):
                    raise ValueError("Invalid activity logits or prediction count")
                if z.shape != (len(records),) or not np.isfinite(z).all():
                    raise ValueError("Invalid remaining-time predictions")
                days = np.asarray(prepared.inverse_remaining_time(z), dtype=np.float64)
                if not np.isfinite(days).all():
                    raise ValueError(
                        "Non-finite remaining days after inverse transformation"
                    )
                if gates.shape != (len(records), 3) or not np.isfinite(gates).all():
                    raise ValueError("Invalid view weights")
                if (gates < 0).any() or not np.allclose(gates.sum(1), 1.0, atol=1e-5):
                    raise ValueError("View weights must be nonnegative and sum to one")
                predicted = logits.argmax(axis=1)
                for index, record in enumerate(records):
                    activity_id = int(predicted[index])
                    rows.append(
                        {
                            "case_id": record.case_id,
                            "prefix_len": record.prefix_len,
                            "true_activity_raw": record.next_activity,
                            "predicted_activity": names[activity_id],
                            "predicted_is_unknown": activity_id == unknown_id,
                            "remaining_days": float(record.remaining_time_days),
                            "predicted_remaining_time_days": float(days[index]),
                            "gate_semantic": float(gates[index, 0]),
                            "gate_order": float(gates[index, 1]),
                            "gate_attribute": float(gates[index, 2]),
                        }
                    )
    finally:
        model.train(previous_mode)
    train_mean = float(
        np.mean([r.remaining_time_days for r in prepared.samples["train"]])
    )
    known = set(prepared.activity_vocab) - {ACTIVITY_UNK_TOKEN}
    metrics = prediction_metrics(rows, train_mean, known)
    metrics["split"] = split
    return metrics, rows


def save_results(directory: str | Path, metrics: dict, rows: list[dict]) -> None:
    """Write a new set of results while preserving earlier evaluations."""
    if not rows:
        raise ValueError("Cannot save an empty prediction set")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    metrics_path, predictions_path = target / "metrics.json", target / "predictions.csv"
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError("Results already exist: " + str(target))
    metrics_text = (
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    with predictions_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with metrics_path.open("x", encoding="utf-8") as handle:
        handle.write(metrics_text)


def load_checkpoint(
    checkpoint: str | Path, prepared, device: str | torch.device = "auto"
):
    """Rebuild BERT from its saved configuration and load the full MFML state."""
    from transformers import AutoTokenizer, BertConfig, BertModel
    from MFML import MFML

    path = Path(checkpoint).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != CHECKPOINT_VERSION
    ):
        raise ValueError("Unsupported checkpoint; use a checkpoint from this release")
    if payload.get("prepared_id") != prepared.identity:
        raise ValueError(
            "Checkpoint and prepared data differ; use the original prepared.pt"
        )
    kwargs = payload["model_kwargs"]
    cards = {name: len(prepared.vocabs[name]) for name in prepared.categorical_columns}
    if (
        list(kwargs["categorical_cardinalities"].items()) != list(cards.items())
        or kwargs["num_activities"] != prepared.num_classes
    ):
        raise ValueError(
            "Checkpoint vocabulary or attribute order differs from prepared data"
        )
    bert_config = payload["bert_config"]
    if bert_config.get("model_type", "bert") != "bert":
        raise ValueError("This MFML implementation requires a BERT encoder")
    tokenizer_dir = (path.parent / payload["tokenizer_dir"]).resolve()
    if not tokenizer_dir.is_relative_to(path.parent):
        raise ValueError("Tokenizer must be stored within the run directory")
    expected_files = payload.get("tokenizer_sha256")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("Checkpoint lacks the saved tokenizer identity")
    actual_files = {
        p.relative_to(tokenizer_dir).as_posix(): sha256_file(p)
        for p in sorted(tokenizer_dir.rglob("*"))
        if p.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(
            "Saved tokenizer files are missing or differ from the checkpoint"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, use_fast=True
    )
    encoder_config = BertConfig.from_dict(bert_config)
    # Transformers 4.46 omits the resolved backend from config.to_dict().
    # Restore it explicitly so checkpoint evaluation uses the training math.
    attention = payload.get("bert_attention_implementation")
    if attention not in {"eager", "sdpa"}:
        raise ValueError("Checkpoint lacks a supported BERT attention implementation")
    encoder_config._attn_implementation = attention
    model = MFML(**kwargs, bert_encoder=BertModel(encoder_config))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(resolve_device(device))
    model.eval()
    return model, tokenizer, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Run directory's best.pt")
    parser.add_argument(
        "--prepared", required=True, help="The original prepared.pt file"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    return parser


def main(argv: list[str] | None = None) -> int:
    import sys

    parser = build_parser()
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    if args.threads < 1 or args.batch_size < 1:
        parser.error("--threads and --batch-size must be positive")
    torch.set_num_threads(args.threads)
    prepared = load_prepared(args.prepared)
    model, tokenizer, payload = load_checkpoint(args.checkpoint, prepared, args.device)
    metrics, rows = evaluate_model(
        model, prepared, tokenizer, "test", args.device, args.batch_size
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = Path(args.checkpoint).resolve().parent / ("reevaluation-" + stamp)
    destination.mkdir(exist_ok=False)
    metrics.update(
        checkpoint_epoch=payload["best_epoch"], prepared_id=prepared.identity
    )
    save_results(destination, metrics, rows)
    print(
        json.dumps(
            {"output": str(destination), **metrics}, indent=2, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
