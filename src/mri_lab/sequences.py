"""Sequence parameter objects and simplified event timelines."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class GREParams:
    flip_angle_deg: float = 20
    te_ms: float = 8
    tr_ms: float = 80
    readout_points: int = 128
    phase_encodes: int = 128


@dataclass(frozen=True)
class SpinEchoParams:
    excitation_deg: float = 90
    refocusing_deg: float = 180
    te_ms: float = 80
    tr_ms: float = 1200
    readout_points: int = 128
    phase_encodes: int = 128


@dataclass(frozen=True)
class RAREParams:
    excitation_deg: float = 90
    refocusing_deg: float = 180
    echo_spacing_ms: float = 12
    echo_train_length: int = 8
    tr_ms: float = 2500
    readout_points: int = 128


@dataclass(frozen=True)
class BSSFPParams:
    flip_angle_deg: float = 45
    tr_ms: float = 5
    readout_points: int = 128
    phase_encodes: int = 128


def sequence_summary(params: object) -> dict[str, float]:
    """Convert a sequence dataclass into a dictionary."""
    return asdict(params)


def rare_echo_times(params: RAREParams) -> np.ndarray:
    """Return echo times for a simplified RARE echo train."""
    return params.echo_spacing_ms * np.arange(1, params.echo_train_length + 1)


def gre_event_timeline(params: GREParams) -> list[dict[str, float | str]]:
    """Create a simplified GRE/FLASH event timeline."""
    return [
        {"event": "RF excitation", "time_ms": 0.0, "flip_deg": params.flip_angle_deg},
        {"event": "phase encoding gradient", "time_ms": params.te_ms * 0.35, "flip_deg": 0.0},
        {"event": "readout gradient and ADC", "time_ms": params.te_ms, "flip_deg": 0.0},
        {"event": "spoiling and recovery", "time_ms": params.tr_ms, "flip_deg": 0.0},
    ]


def spin_echo_event_timeline(params: SpinEchoParams) -> list[dict[str, float | str]]:
    """Create a simplified spin echo event timeline."""
    return [
        {"event": "90 degree RF excitation", "time_ms": 0.0, "flip_deg": params.excitation_deg},
        {"event": "180 degree refocusing pulse", "time_ms": params.te_ms / 2, "flip_deg": params.refocusing_deg},
        {"event": "spin echo readout", "time_ms": params.te_ms, "flip_deg": 0.0},
        {"event": "longitudinal recovery", "time_ms": params.tr_ms, "flip_deg": 0.0},
    ]
