"""Dataset loading and custom data-order samplers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, transforms


ORDER_MODES = (
    "random",
    "fixed_random",
    "label_sorted",
    "label_block_random",
    "curriculum_easy",
    "curriculum_hard",
)


@dataclass(frozen=True)
class DatasetBundle:
    """Container for datasets and metadata aligned with the train subset."""

    train_dataset: Subset
    test_dataset: datasets.FashionMNIST
    train_labels: np.ndarray
    train_images: np.ndarray


class EpochOrderSampler(Sampler[int]):
    """Sampler whose order can depend on the epoch.

    The sampler returns integer positions for a Subset object. The only thing
    changing across experimental conditions is the sequence in which these
    positions are visited.
    """

    def __init__(
        self,
        labels: np.ndarray,
        mode: str,
        seed: int,
        images: np.ndarray | None = None,
    ) -> None:
        if mode not in ORDER_MODES:
            raise ValueError(f"Unknown order mode '{mode}'. Valid modes: {ORDER_MODES}")
        self.labels = np.asarray(labels)
        self.mode = mode
        self.seed = int(seed)
        self.epoch = 0
        self.n = len(self.labels)
        self.base_indices = np.arange(self.n)
        self.fixed_random_order = self._rng(0).permutation(self.n)
        self.curriculum_scores = self._compute_curriculum_scores(images) if images is not None else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, offset: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + 10_000 * offset)

    def _compute_curriculum_scores(self, images: np.ndarray | None) -> np.ndarray | None:
        """Compute a simple class-prototype difficulty score.

        Images close to their class mean are treated as easier examples. This is
        a cheap, deterministic proxy for difficulty, not an oracle.
        """
        if images is None:
            return None

        flat = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
        scores = np.zeros(len(self.labels), dtype=np.float32)
        for label in np.unique(self.labels):
            mask = self.labels == label
            class_flat = flat[mask]
            prototype = class_flat.mean(axis=0, keepdims=True)
            scores[mask] = np.linalg.norm(class_flat - prototype, axis=1)
        return scores

    def _random_order(self) -> np.ndarray:
        return self._rng(self.epoch + 1).permutation(self.n)

    def _fixed_random_order(self) -> np.ndarray:
        return self.fixed_random_order.copy()

    def _label_sorted_order(self) -> np.ndarray:
        # Stable sort keeps a deterministic order inside each class.
        return np.argsort(self.labels, kind="stable")

    def _label_block_random_order(self) -> np.ndarray:
        """Return shuffled examples but grouped into class blocks.

        Each epoch uses a new class order and new within-class permutations,
        while still exposing the optimizer to long non-iid class blocks.
        """
        rng = self._rng(self.epoch + 1)
        class_order = rng.permutation(np.unique(self.labels))
        blocks = []
        for label in class_order:
            positions = self.base_indices[self.labels == label]
            blocks.append(rng.permutation(positions))
        return np.concatenate(blocks)

    def _curriculum_order(self, reverse: bool) -> np.ndarray:
        if self.curriculum_scores is None:
            raise ValueError("Curriculum modes require train images.")
        order = np.argsort(self.curriculum_scores, kind="stable")
        if reverse:
            order = order[::-1]
        return order

    def __iter__(self) -> Iterator[int]:
        if self.mode == "random":
            order = self._random_order()
        elif self.mode == "fixed_random":
            order = self._fixed_random_order()
        elif self.mode == "label_sorted":
            order = self._label_sorted_order()
        elif self.mode == "label_block_random":
            order = self._label_block_random_order()
        elif self.mode == "curriculum_easy":
            order = self._curriculum_order(reverse=False)
        elif self.mode == "curriculum_hard":
            order = self._curriculum_order(reverse=True)
        else:
            raise RuntimeError(f"Unsupported mode: {self.mode}")
        return iter(order.tolist())

    def __len__(self) -> int:
        return self.n


def _stratified_subset_indices(labels: torch.Tensor, max_examples: int | None, seed: int) -> np.ndarray:
    """Select a balanced subset while preserving all classes."""
    n_total = len(labels)
    if max_examples is None or max_examples >= n_total:
        return np.arange(n_total)

    labels_np = labels.numpy()
    rng = np.random.default_rng(seed)
    classes = np.unique(labels_np)
    per_class = max_examples // len(classes)
    remainder = max_examples % len(classes)

    selected = []
    for i, label in enumerate(classes):
        candidates = np.where(labels_np == label)[0]
        quota = per_class + (1 if i < remainder else 0)
        selected.extend(rng.choice(candidates, size=quota, replace=False).tolist())
    return rng.permutation(np.array(selected, dtype=np.int64))


def load_fashion_mnist(
    data_dir: str | Path,
    max_train_examples: int | None,
    subset_seed: int,
) -> DatasetBundle:
    """Load Fashion-MNIST and optionally use a stratified train subset."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    full_train = datasets.FashionMNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    subset_indices = _stratified_subset_indices(full_train.targets, max_train_examples, subset_seed)
    train_dataset = Subset(full_train, subset_indices.tolist())
    train_labels = full_train.targets[subset_indices].numpy()
    train_images = full_train.data[subset_indices].numpy()

    return DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_labels=train_labels,
        train_images=train_images,
    )


def make_train_loader(
    bundle: DatasetBundle,
    order_mode: str,
    seed: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, EpochOrderSampler]:
    """Create a train DataLoader and its epoch-aware sampler."""
    sampler = EpochOrderSampler(
        labels=bundle.train_labels,
        images=bundle.train_images,
        mode=order_mode,
        seed=seed,
    )
    loader = DataLoader(
        bundle.train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


def make_test_loader(
    bundle: DatasetBundle,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Create a deterministic test DataLoader."""
    return DataLoader(
        bundle.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
