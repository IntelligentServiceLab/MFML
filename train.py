"""Train the full MFML model and evaluate its validation-selected checkpoint."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from attribute import MaskedGroupNorm1d
from evaluate import (
    CHECKPOINT_VERSION,
    evaluate_model,
    load_checkpoint,
    model_inputs,
    resolve_device,
    save_results,
)
from MFML import MFML
from preprocess import ACTIVITY_UNK_TOKEN, load_prepared, normalize_category
from settings import TrainingConfig

LOGGER = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """Call before model construction so initialization is reproducible too."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """Use the BERT rate for its trainable layers; do not decay norms or biases."""
    bert_ids = {
        id(parameter) for parameter in model.semantic_branch.encoder.parameters()
    }
    norm_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, MaskedGroupNorm1d))
        for parameter in module.parameters(recurse=False)
    }
    groups = {
        (True, True): [],
        (True, False): [],
        (False, True): [],
        (False, False): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        decay = id(parameter) not in norm_ids and not name.endswith(".bias")
        groups[(id(parameter) in bert_ids, decay)].append(parameter)
    parameters = [
        {
            "params": values,
            "lr": config.bert_learning_rate if is_bert else config.head_learning_rate,
            "weight_decay": config.weight_decay if decay else 0.0,
        }
        for (is_bert, decay), values in groups.items()
        if values
    ]
    if not parameters:
        raise ValueError("The model has no trainable parameters")
    return torch.optim.AdamW(parameters)


def class_weights_from_counts(counts, *, device=None) -> torch.Tensor:
    """Train-only inverse-square-root class frequencies, clipped to [0.5, 2]."""
    counts = np.asarray(counts, dtype=np.float64)
    if (
        counts.ndim != 1
        or not len(counts)
        or not np.isfinite(counts).all()
        or (counts < 0).any()
        or counts.sum() <= 0
    ):
        raise ValueError("Class counts must be finite, nonnegative and nonempty")
    present = counts > 0
    weights = np.ones_like(counts)
    inverse = 1.0 / np.sqrt(counts[present])
    weights[present] = np.clip(inverse / inverse.mean(), 0.5, 2.0)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _batch_indices(count: int, batch_size: int, seed: int):
    order = np.arange(count)
    np.random.default_rng(seed).shuffle(order)
    for start in range(0, count, batch_size):
        yield order[start : start + batch_size]


def _loss_for_group_part(
    output, activity, remaining, weights, ce_denominator, group_size, config
) -> torch.Tensor:
    """Normalize each microbatch by the entire accumulation group's totals."""
    classification = F.cross_entropy(
        output["activity_logits"],
        activity,
        weight=weights,
        label_smoothing=config.label_smoothing,
        reduction="sum",
    ) / max(ce_denominator, 1e-12)
    regression = (
        F.smooth_l1_loss(
            output["remaining_time_z"].reshape(-1),
            remaining,
            beta=config.smooth_l1_beta,
            reduction="sum",
        )
        / group_size
    )
    return (
        config.classification_loss_weight * classification
        + config.regression_loss_weight * regression
    ) / (config.classification_loss_weight + config.regression_loss_weight)


def _optimizer_step(optimizer, scaler, scheduler) -> bool:
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= previous_scale
    if updated:
        scheduler.step()
    return updated


def _scheduler(optimizer, updates: int, warmup_ratio: float):
    warmup = int(updates * warmup_ratio)

    def factor(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min(1.0, (step - warmup) / max(updates - warmup, 1))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _write_history(path: Path, history: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def _require_empty_output(directory: Path) -> None:
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {directory}")


def _checkpoint_metadata(model, prepared, tokenizer, directory, config) -> dict:
    tokenizer_directory = directory / "tokenizer"
    tokenizer.save_pretrained(tokenizer_directory)
    tokenizer_hashes = {
        path.relative_to(tokenizer_directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(tokenizer_directory.rglob("*"))
        if path.is_file()
    }
    if not tokenizer_hashes:
        raise ValueError("Tokenizer did not save any files")
    embeddings = model.categorical_embeddings
    return {
        "format_version": CHECKPOINT_VERSION,
        "model_kwargs": {
            "categorical_cardinalities": {
                name: embeddings[name].num_embeddings for name in model.attribute_names
            },
            "num_activities": prepared.num_classes,
            "embedding_dim": embeddings[model.attribute_names[0]].embedding_dim,
            "branch_dim": model.activity_head.layers[0].in_features,
            "dropout": model.activity_head.layers[2].p,
        },
        "bert_config": model.semantic_branch.encoder.config.to_dict(),
        "bert_attention_implementation": model.semantic_branch.encoder.config._attn_implementation,
        "training_config": config.to_dict(),
        "prepared_id": prepared.identity,
        "prepared_filename": str(prepared.path),
        "tokenizer_dir": "tokenizer",
        "tokenizer_sha256": tokenizer_hashes,
    }


def train_model(model, prepared, tokenizer, run_dir, config, device="auto") -> dict:
    """Fit train/validation only; save and restore the best validation model.

    The caller must seed before constructing the model. Test evaluation belongs
    to the caller after this function finishes and the checkpoint is fixed.
    """
    config.validate()
    samples = prepared.samples["train"]
    if not samples or not prepared.samples["validation"]:
        raise ValueError("Training and validation splits must contain prefixes")
    directory = Path(run_dir).resolve()
    _require_empty_output(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = _checkpoint_metadata(model, prepared, tokenizer, directory, config)
    resolved = resolve_device(device)
    model.to(resolved)
    optimizer = build_optimizer(model, config)
    updates = math.ceil(
        math.ceil(len(samples) / config.batch_size) / config.gradient_accumulation_steps
    )
    scheduler = _scheduler(optimizer, updates * config.max_epochs, config.warmup_ratio)
    amp = bool(config.use_amp and resolved.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    collator = prepared.make_collator(tokenizer)
    unknown = prepared.activity_vocab[ACTIVITY_UNK_TOKEN]
    labels = np.asarray(
        [
            prepared.activity_vocab.get(normalize_category(row.next_activity), unknown)
            for row in samples
        ]
    )
    weights = class_weights_from_counts(
        np.bincount(labels, minlength=prepared.num_classes), device=resolved
    )
    time_scaler = prepared.scalers["remaining_time"]
    remaining_days = np.asarray(
        [row.remaining_time_days for row in samples], dtype=np.float64
    )
    # Transform in float64 before rounding to match the original benchmark.
    targets = (
        (np.log1p(remaining_days) - float(time_scaler["mean"]))
        / float(time_scaler["scale"])
    ).astype(np.float32)
    train_mean = max(float(remaining_days.mean()), 1e-12)
    checkpoint = directory / "best.pt"
    best_score, best_epoch, stale = -float("inf"), -1, 0
    best_metrics, history = None, []
    LOGGER.info(
        "device=%s lambda1=%g lambda2=%g (normalized by their sum)",
        resolved,
        config.classification_loss_weight,
        config.regression_loss_weight,
    )
    for epoch in range(config.max_epochs):
        model.train()
        iterator = iter(
            _batch_indices(len(samples), config.batch_size, config.seed + epoch)
        )
        epoch_loss, seen = 0.0, 0
        while True:
            group = list(itertools.islice(iterator, config.gradient_accumulation_steps))
            if not group:
                break
            group_ids = np.concatenate(group)
            group_size = len(group_ids)
            group_labels = torch.as_tensor(
                labels[group_ids], dtype=torch.long, device=resolved
            )
            denominator = float(weights[group_labels].sum().item())
            optimizer.zero_grad(set_to_none=True)
            for ids in group:
                batch = collator([samples[int(index)] for index in ids])
                activity = torch.as_tensor(
                    labels[ids], dtype=torch.long, device=resolved
                )
                remaining = torch.as_tensor(targets[ids], device=resolved)
                with torch.autocast(device_type=resolved.type, enabled=amp):
                    output = model(**model_inputs(batch, resolved))
                    loss = _loss_for_group_part(
                        output,
                        activity,
                        remaining,
                        weights,
                        denominator,
                        group_size,
                        config,
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        "Non-finite training loss; test evaluation was not run"
                    )
                scaler.scale(loss).backward()
                epoch_loss += float(loss.detach().cpu()) * group_size
                seen += len(ids)
            scaler.unscale_(optimizer)
            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )
            _optimizer_step(optimizer, scaler, scheduler)
        metrics, _ = evaluate_model(
            model,
            prepared,
            tokenizer,
            split="validation",
            device=resolved,
            batch_size=config.batch_size,
        )
        score = 0.5 * float(metrics["macro_f1"]) + 0.5 / (
            1.0 + float(metrics["mae_days"]) / train_mean
        )
        if not math.isfinite(score):
            raise FloatingPointError(
                "Non-finite validation score; no test evaluation was run"
            )
        metrics = {**metrics, "selection_score": score}
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": epoch_loss / seen,
                **{
                    key: value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                },
            }
        )
        _write_history(directory / "history.csv", history)
        LOGGER.info(
            "epoch=%d train_loss=%.6f validation_score=%.6f",
            epoch + 1,
            epoch_loss / seen,
            score,
        )
        if score > best_score:
            best_score, best_epoch, stale, best_metrics = score, epoch + 1, 0, metrics
            payload = {
                **metadata,
                "state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "validation_metrics": best_metrics,
            }
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint)
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    if best_epoch < 0:
        raise RuntimeError("No validation-selected checkpoint was saved")
    payload = torch.load(checkpoint, map_location=resolved, weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return {
        "checkpoint": str(checkpoint),
        "best_epoch": best_epoch,
        "stopped_epoch": len(history),
        "validation_metrics": best_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainingConfig()
    parser.add_argument(
        "--prepared", required=True, help="Prepared .pt file from preprocess.py"
    )
    parser.add_argument(
        "--bert-model",
        default="prajjwal1/bert-medium",
        help="Hugging Face name or local directory",
    )
    parser.add_argument(
        "--output", required=True, help="New or empty training output directory"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--accumulation-steps", type=int, default=defaults.gradient_accumulation_steps
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=defaults.classification_loss_weight,
        help="Classification loss weight",
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=defaults.regression_loss_weight,
        help="Remaining-time loss weight",
    )
    parser.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        default=defaults.use_amp,
        help="Disable CUDA mixed precision",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use already downloaded BERT files",
    )
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    config = TrainingConfig(
        seed=args.seed,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        classification_loss_weight=args.lambda1,
        regression_loss_weight=args.lambda2,
        use_amp=args.use_amp,
    )
    config.validate()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    destination = Path(args.output).resolve()
    _require_empty_output(destination)
    prepared = load_prepared(args.prepared)
    set_random_seed(config.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.bert_model, use_fast=True, local_files_only=args.local_files_only
    )
    model = MFML(
        categorical_cardinalities={
            name: len(prepared.vocabs[name]) for name in prepared.categorical_columns
        },
        num_activities=prepared.num_classes,
        bert_model_name_or_path=args.bert_model,
        embedding_initial_weights=prepared.cbow_vectors,
        local_files_only=args.local_files_only,
    )
    result = train_model(model, prepared, tokenizer, destination, config, args.device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model, tokenizer, _ = load_checkpoint(result["checkpoint"], prepared, args.device)
    metrics, rows = evaluate_model(
        model, prepared, tokenizer, "test", args.device, config.batch_size
    )
    metrics.update(checkpoint_epoch=result["best_epoch"], prepared_id=prepared.identity)
    save_results(destination, metrics, rows)
    LOGGER.info("Best epoch: %d; outputs: %s", result["best_epoch"], destination)
    for key in (
        "accuracy",
        "weighted_precision",
        "weighted_f1",
        "mae_days",
        "rmse_days",
    ):
        LOGGER.info("%s: %.6f", key, metrics[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
