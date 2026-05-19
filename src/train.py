"""Training and evaluation loops."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
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
    model: str
    optimizer: str
    epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    device: str
    deterministic: bool


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


def gradient_norm(model: nn.Module) -> float:
    """Compute the global L2 norm of all available gradients."""
    total_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total_sq += float(torch.sum(grad * grad).cpu())
    return total_sq ** 0.5


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    show_progress: bool,
) -> dict[str, float]:
    """Train for one epoch and return aggregate metrics."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    grad_norms: list[float] = []

    iterator = tqdm(loader, desc=f"epoch {epoch:02d} train", leave=False) if show_progress else loader
    for inputs, targets in iterator:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        grad_norms.append(gradient_norm(model))
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().cpu())
        total_examples += batch_size

    grad_tensor = torch.tensor(grad_norms, dtype=torch.float32)
    return {
        "train_loss": total_loss / total_examples,
        "train_accuracy": total_correct / total_examples,
        "grad_norm_mean": float(grad_tensor.mean()) if len(grad_norms) else 0.0,
        "grad_norm_std": float(grad_tensor.std(unbiased=False)) if len(grad_norms) else 0.0,
    }


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

    rows: list[dict[str, float | int | str]] = []
    output_dir = ensure_dir(output_dir)

    for epoch in range(1, config.epochs + 1):
        start = time.time()
        train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            show_progress=show_progress,
        )
        test_metrics = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start

        row = {
            **asdict(config),
            "epoch": epoch,
            **train_metrics,
            **test_metrics,
            "epoch_time_sec": elapsed,
        }
        rows.append(row)

        if show_progress:
            print(
                f"[{config.order_mode} | seed={config.seed} | epoch={epoch:02d}] "
                f"train_loss={row['train_loss']:.4f} "
                f"test_acc={100 * row['test_accuracy']:.2f}% "
                f"grad_norm={row['grad_norm_mean']:.3f}"
            )

    metrics_path = output_dir / f"metrics_{config.order_mode}_seed{config.seed}.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    prediction_counts = collect_prediction_distribution(model, test_loader, device)
    distribution_rows = [
        {
            "order_mode": config.order_mode,
            "seed": config.seed,
            "class_id": class_id,
            "num_predictions": int(count),
        }
        for class_id, count in enumerate(prediction_counts.tolist())
    ]
    distribution_path = output_dir / f"prediction_distribution_{config.order_mode}_seed{config.seed}.csv"
    pd.DataFrame(distribution_rows).to_csv(distribution_path, index=False)

    return metrics_path
