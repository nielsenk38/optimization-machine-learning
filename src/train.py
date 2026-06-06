"""Training and evaluation loops."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import build_model
from .utils import ensure_dir, set_global_seed


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for one experimental run."""

    order_mode: str
    seed: int
    experiment_name: str
    output_dir: str
    model: str
    optimizer: str
    epochs: int
    batch_size: int
    max_train_examples: int
    learning_rate: float
    momentum: float
    weight_decay: float
    device: str
    deterministic: bool
    save_embeddings: bool = False


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create an optimizer by name."""
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    """Concatenate all gradients into a single vector."""
    gradients = [parameter.grad.detach().reshape(-1) for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(gradients)


def gradient_norm(model: nn.Module) -> float:
    """Compute the global L2 norm of all available gradients."""
    gradient_vector = flatten_gradients(model)
    if gradient_vector.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(gradient_vector).detach().cpu())


def _metadata_from_config(config: TrainConfig) -> dict[str, Any]:
    return asdict(config)


def get_run_artifact_paths(config: TrainConfig, output_dir: str | Path) -> dict[str, Path]:
    """Return the output files expected for one run."""
    output_dir = ensure_dir(output_dir)
    order_mode = config.order_mode
    seed = config.seed
    paths = {
        "metrics": output_dir / f"metrics_{order_mode}_seed{seed}.csv",
        "prediction_distribution": output_dir / f"prediction_distribution_{order_mode}_seed{seed}.csv",
        "gradient_cosine": output_dir / f"gradient_cosine_{order_mode}_seed{seed}.csv",
        "class_accuracy": output_dir / f"class_accuracy_{order_mode}_seed{seed}.csv",
    }
    if config.save_embeddings:
        paths["embeddings"] = output_dir / f"embeddings_{order_mode}_seed{seed}.npz"
    return paths


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    show_progress: bool,
    previous_gradient: torch.Tensor | None,
    global_step_start: int,
) -> tuple[dict[str, float], list[dict[str, Any]], torch.Tensor | None, int]:
    """Train for one epoch and return aggregate metrics plus cosine traces."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    grad_norms: list[float] = []
    cosine_values: list[float] = []
    cosine_rows: list[dict[str, Any]] = []
    global_step = global_step_start

    # Gradient-variance accumulators: track E[||g||²] - ||E[g]||² = Tr(Cov(g))
    # Uses O(d) memory regardless of epoch length (no list of all gradient vectors).
    grad_sum: torch.Tensor | None = None
    grad_sq_norm_sum: float = 0.0
    grad_variance_count: int = 0

    iterator = tqdm(loader, desc=f"epoch {epoch:02d} train", leave=False) if show_progress else loader
    for batch_index, (inputs, targets) in enumerate(iterator, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()

        gradient_vector = flatten_gradients(model)
        grad_norms.append(float(torch.linalg.vector_norm(gradient_vector).detach().cpu()) if gradient_vector.numel() else 0.0)

        # Accumulate for Tr(Cov(g)) = E[||g||²] - ||E[g]||²
        if gradient_vector.numel() > 0:
            gv_cpu = gradient_vector.detach().cpu().float()
            if grad_sum is None:
                grad_sum = gv_cpu.clone()
            else:
                grad_sum.add_(gv_cpu)
            grad_sq_norm_sum += float(torch.dot(gv_cpu, gv_cpu))
            grad_variance_count += 1

        cosine_similarity = np.nan
        if previous_gradient is not None and previous_gradient.numel() and gradient_vector.numel():
            denominator = float(
                torch.linalg.vector_norm(previous_gradient).detach().cpu()
                * torch.linalg.vector_norm(gradient_vector).detach().cpu()
            )
            if denominator > 0.0:
                cosine_similarity = float(F.cosine_similarity(previous_gradient, gradient_vector, dim=0).detach().cpu())
                cosine_values.append(cosine_similarity)

        global_step += 1
        cosine_rows.append(
            {
                "epoch": epoch,
                "batch_index": batch_index,
                "global_step": global_step,
                "cosine_similarity": cosine_similarity,
            }
        )
        previous_gradient = gradient_vector.detach().clone()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().cpu())
        total_examples += batch_size

    grad_tensor = torch.tensor(grad_norms, dtype=torch.float32)
    cosine_tensor = torch.tensor(cosine_values, dtype=torch.float32) if cosine_values else torch.zeros(0, dtype=torch.float32)

    # Gradient variance: Tr(Cov(g)) = E[||g||²] - ||E[g]||²
    # This measures how spread the per-batch gradients are around their mean —
    # the σ² term that appears in SGD convergence bounds.
    grad_variance = np.nan
    if grad_variance_count > 0 and grad_sum is not None:
        mean_sq_norm = grad_sq_norm_sum / grad_variance_count
        mean_grad_sq_norm = float(torch.dot(grad_sum, grad_sum)) / (grad_variance_count ** 2)
        grad_variance = max(0.0, mean_sq_norm - mean_grad_sq_norm)

    metrics = {
        "train_loss": total_loss / total_examples,
        "train_accuracy": total_correct / total_examples,
        "grad_norm_mean": float(grad_tensor.mean()) if len(grad_norms) else 0.0,
        "grad_norm_std": float(grad_tensor.std(unbiased=False)) if len(grad_norms) else 0.0,
        "grad_cosine_mean": float(cosine_tensor.mean()) if len(cosine_values) else np.nan,
        "grad_cosine_std": float(cosine_tensor.std(unbiased=False)) if len(cosine_values) else np.nan,
        "grad_variance": grad_variance,
    }
    return metrics, cosine_rows, previous_gradient, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the model on the test set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().cpu())
        total_examples += batch_size

    return {
        "test_loss": total_loss / total_examples,
        "test_accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, int | float]]:
    """Compute class-wise accuracy on the evaluation set."""
    model.eval()
    num_classes = None
    class_correct = None
    class_total = None

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        predictions = logits.argmax(dim=1)

        if num_classes is None:
            num_classes = logits.shape[1]
            class_correct = torch.zeros(num_classes, dtype=torch.long)
            class_total = torch.zeros(num_classes, dtype=torch.long)

        batch_total = torch.bincount(targets.cpu(), minlength=num_classes)
        batch_correct = torch.bincount(targets[predictions == targets].cpu(), minlength=num_classes)
        class_total += batch_total
        class_correct += batch_correct

    if num_classes is None or class_correct is None or class_total is None:
        return []

    rows: list[dict[str, int | float]] = []
    for class_id in range(num_classes):
        total = int(class_total[class_id].item())
        correct = int(class_correct[class_id].item())
        rows.append(
            {
                "class_id": class_id,
                "num_correct": correct,
                "num_examples": total,
                "class_accuracy": correct / total if total > 0 else np.nan,
            }
        )
    return rows


@torch.no_grad()
def collect_prediction_distribution(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Count predicted classes over the full evaluation loader."""
    model.eval()
    counts: torch.Tensor | None = None

    for inputs, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        batch_counts = torch.bincount(logits.argmax(dim=1).cpu(), minlength=logits.shape[1])
        if counts is None:
            counts = torch.zeros(logits.shape[1], dtype=torch.long)
        counts += batch_counts

    if counts is None:
        return torch.zeros(0, dtype=torch.long)
    return counts


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Collect final embeddings, labels, and predictions on the evaluation set."""
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        if not hasattr(model, "extract_embedding"):
            raise AttributeError("Model does not expose an extract_embedding method.")

        embedding = model.extract_embedding(inputs)
        embeddings.append(embedding.detach().cpu().numpy())
        labels.append(targets.numpy())
        predictions.append(logits.argmax(dim=1).detach().cpu().numpy())

    if not embeddings:
        return {
            "embeddings": np.zeros((0, 0), dtype=np.float32),
            "labels": np.zeros(0, dtype=np.int64),
            "predictions": np.zeros(0, dtype=np.int64),
        }

    return {
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float32),
        "labels": np.concatenate(labels, axis=0).astype(np.int64),
        "predictions": np.concatenate(predictions, axis=0).astype(np.int64),
    }


def run_single_experiment(
    config: TrainConfig,
    train_loader: DataLoader,
    train_sampler,
    test_loader: DataLoader,
    output_dir: str | Path,
    show_progress: bool = True,
) -> Path:
    """Run one seed/order combination and save epoch metrics as CSV."""
    set_global_seed(config.seed, deterministic=config.deterministic)
    device = torch.device(config.device)
    model = build_model(config.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model=model,
        optimizer_name=config.optimizer,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    rows: list[dict[str, float | int | str | bool]] = []
    gradient_cosine_rows: list[dict[str, Any]] = []
    class_accuracy_rows: list[dict[str, Any]] = []
    output_dir = ensure_dir(output_dir)
    metadata = _metadata_from_config(config)
    previous_gradient: torch.Tensor | None = None
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        start = time.time()
        train_sampler.set_epoch(epoch)
        train_metrics, epoch_gradient_rows, previous_gradient, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            show_progress=show_progress,
            previous_gradient=previous_gradient,
            global_step_start=global_step,
        )
        test_metrics = evaluate(model, test_loader, criterion, device)
        per_class_rows = evaluate_per_class(model, test_loader, device)
        elapsed = time.time() - start

        row = {
            **metadata,
            "epoch": epoch,
            **train_metrics,
            **test_metrics,
            "epoch_time_sec": elapsed,
        }
        rows.append(row)

        gradient_cosine_rows.extend([{**metadata, **gradient_row} for gradient_row in epoch_gradient_rows])
        class_accuracy_rows.extend([{**metadata, "epoch": epoch, **per_class_row} for per_class_row in per_class_rows])

        if show_progress:
            print(
                f"[{config.order_mode} | seed={config.seed} | epoch={epoch:02d}] "
                f"train_loss={row['train_loss']:.4f} "
                f"test_acc={100 * row['test_accuracy']:.2f}% "
                f"grad_norm={row['grad_norm_mean']:.3f} "
                f"grad_cos={row['grad_cosine_mean']:.3f}"
            )

    artifact_paths = get_run_artifact_paths(config, output_dir)
    pd.DataFrame(rows).to_csv(artifact_paths["metrics"], index=False)
    pd.DataFrame(gradient_cosine_rows).to_csv(artifact_paths["gradient_cosine"], index=False)
    pd.DataFrame(class_accuracy_rows).to_csv(artifact_paths["class_accuracy"], index=False)

    prediction_counts = collect_prediction_distribution(model, test_loader, device)
    distribution_rows = [
        {
            **metadata,
            "class_id": class_id,
            "num_predictions": int(count),
        }
        for class_id, count in enumerate(prediction_counts.tolist())
    ]
    pd.DataFrame(distribution_rows).to_csv(artifact_paths["prediction_distribution"], index=False)

    if config.save_embeddings:
        embeddings = collect_embeddings(model, test_loader, device)
        np.savez_compressed(
            artifact_paths["embeddings"],
            embeddings=embeddings["embeddings"],
            labels=embeddings["labels"],
            predictions=embeddings["predictions"],
            order_mode=np.array(config.order_mode),
            seed=np.array(config.seed),
            learning_rate=np.array(config.learning_rate),
            epochs=np.array(config.epochs),
            max_train_examples=np.array(config.max_train_examples),
            optimizer=np.array(config.optimizer),
        )

    return artifact_paths["metrics"]
