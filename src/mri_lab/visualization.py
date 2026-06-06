"""Plotting utilities for MRI simulation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_image(image: np.ndarray, path: str | Path, title: str | None = None, cmap: str = "gray") -> None:
    """Save a single image with axes removed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_comparison(images: dict[str, np.ndarray], path: str | Path, cmap: str = "gray") -> None:
    """Save a row of images for visual comparison."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    plt.figure(figsize=(4 * n, 4))
    for index, (title, image) in enumerate(images.items(), start=1):
        plt.subplot(1, n, index)
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_bar_chart(values: dict[str, float], path: str | Path, ylabel: str = "Relative signal") -> None:
    """Save a sequence contrast bar chart."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(list(values.keys()), list(values.values()))
    plt.ylabel(ylabel)
    plt.title("Sequence contrast comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
