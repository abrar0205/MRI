"""Bloch-inspired MRI signal simulation.

This module uses compact analytical signal expressions and a small magnetization
simulator. It is intended for academic/portfolio use, not scanner-grade sequence
engineering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tissue:
    """Simple tissue model using millisecond relaxation times."""

    name: str
    t1: float
    t2: float
    proton_density: float = 1.0


def rotation_x(angle_rad: float) -> np.ndarray:
    """Rotation matrix for an RF pulse around the x-axis."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def relax(m: np.ndarray, dt_ms: float, t1_ms: float, t2_ms: float, m0: float = 1.0) -> np.ndarray:
    """Apply T1/T2 relaxation over dt."""
    e1 = np.exp(-dt_ms / t1_ms)
    e2 = np.exp(-dt_ms / t2_ms)
    out = np.empty_like(m, dtype=float)
    out[0] = m[0] * e2
    out[1] = m[1] * e2
    out[2] = m0 - (m0 - m[2]) * e1
    return out


def simulate_single_echo(
    flip_angle_deg: float = 30,
    te_ms: float = 12,
    tr_ms: float = 600,
    t1_ms: float = 900,
    t2_ms: float = 80,
    repetitions: int = 60,
) -> complex:
    """Simulate repeated RF excitation and read one GRE-like echo."""
    m = np.array([0.0, 0.0, 1.0])
    rf = rotation_x(np.deg2rad(flip_angle_deg))
    signal = 0j
    for _ in range(repetitions):
        m = rf @ m
        echo_state = relax(m, te_ms, t1_ms, t2_ms)
        signal = echo_state[0] + 1j * echo_state[1]
        m = relax(m, max(tr_ms - te_ms, 1e-6), t1_ms, t2_ms)
    return signal


def gre_signal(
    flip_angle_deg: float = 20,
    te_ms: float = 8,
    tr_ms: float = 80,
    t1_ms: float = 900,
    t2star_ms: float = 50,
    proton_density: float = 1.0,
) -> float:
    """Approximate spoiled GRE / FLASH magnitude signal."""
    alpha = np.deg2rad(flip_angle_deg)
    e1 = np.exp(-tr_ms / t1_ms)
    e2s = np.exp(-te_ms / t2star_ms)
    numerator = (1 - e1) * np.sin(alpha)
    denominator = 1 - e1 * np.cos(alpha)
    return float(proton_density * numerator / denominator * e2s)


def spin_echo_signal(
    te_ms: float = 80,
    tr_ms: float = 1200,
    t1_ms: float = 900,
    t2_ms: float = 80,
    proton_density: float = 1.0,
) -> float:
    """Approximate spin echo magnitude signal."""
    return float(proton_density * (1 - np.exp(-tr_ms / t1_ms)) * np.exp(-te_ms / t2_ms))


def rare_echo_train(
    echo_times_ms: np.ndarray,
    tr_ms: float = 2500,
    t1_ms: float = 900,
    t2_ms: float = 80,
    proton_density: float = 1.0,
) -> np.ndarray:
    """Generate a simplified RARE/TSE echo train decay."""
    recovery = 1 - np.exp(-tr_ms / t1_ms)
    decay = np.exp(-echo_times_ms / t2_ms)
    return proton_density * recovery * decay


def bssfp_signal(
    flip_angle_deg: float = 45,
    tr_ms: float = 5,
    t1_ms: float = 900,
    t2_ms: float = 80,
    off_resonance_hz: float = 0,
    proton_density: float = 1.0,
) -> float:
    """Simplified balanced SSFP magnitude response with off-resonance dependence."""
    alpha = np.deg2rad(flip_angle_deg)
    e1 = np.exp(-tr_ms / t1_ms)
    e2 = np.exp(-tr_ms / t2_ms)
    phase = 2 * np.pi * off_resonance_hz * tr_ms / 1000
    numerator = proton_density * (1 - e1) * np.sin(alpha)
    denominator = 1 - e1 * np.cos(alpha) - (e2**2) * (e1 - np.cos(alpha)) * np.cos(phase)
    return float(abs(numerator / (denominator + 1e-12)))


def compare_sequence_contrast(t1: float = 900, t2: float = 80, pd: float = 1.0) -> dict[str, float]:
    """Return representative signal levels for common sequence families."""
    return {
        "GRE/FLASH": gre_signal(t1_ms=t1, t2star_ms=max(t2 * 0.7, 1), proton_density=pd),
        "Spin Echo": spin_echo_signal(t1_ms=t1, t2_ms=t2, proton_density=pd),
        "bSSFP": bssfp_signal(t1_ms=t1, t2_ms=t2, proton_density=pd),
    }
