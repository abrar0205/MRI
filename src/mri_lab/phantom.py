"""Synthetic MRI phantom generation."""

from __future__ import annotations

import numpy as np


def add_ellipse(image: np.ndarray, x0: float, y0: float, a: float, b: float, angle_deg: float, intensity: float) -> np.ndarray:
    """Add a rotated ellipse to an image using normalized coordinates."""
    size_y, size_x = image.shape
    y, x = np.mgrid[-1:1:complex(size_y), -1:1:complex(size_x)]
    theta = np.deg2rad(angle_deg)
    x_shift = x - x0
    y_shift = y - y0
    x_rot = x_shift * np.cos(theta) + y_shift * np.sin(theta)
    y_rot = -x_shift * np.sin(theta) + y_shift * np.cos(theta)
    mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1
    out = image.copy()
    out[mask] += intensity
    return out


def shepp_logan_like(size: int = 128, normalize: bool = True) -> np.ndarray:
    """Create a simple Shepp-Logan-like phantom for k-space experiments."""
    image = np.zeros((size, size), dtype=float)
    ellipses = [
        (0.00, 0.00, 0.70, 0.92, 0, 1.00),
        (0.00, -0.03, 0.62, 0.84, 0, -0.75),
        (0.22, 0.00, 0.16, 0.38, -18, -0.20),
        (-0.22, 0.00, 0.16, 0.38, 18, -0.20),
        (0.00, 0.28, 0.20, 0.12, 0, 0.30),
        (0.00, -0.35, 0.16, 0.10, 0, 0.25),
        (0.35, -0.25, 0.08, 0.12, 0, 0.20),
        (-0.35, -0.25, 0.08, 0.12, 0, 0.20),
    ]
    for ellipse in ellipses:
        image = add_ellipse(image, *ellipse)
    image = np.clip(image, 0, None)
    if normalize and image.max() > 0:
        image = image / image.max()
    return image
