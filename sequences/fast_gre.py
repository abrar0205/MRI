"""Fast Gradient Recalled Echo (Fast GRE / Spoiled GRE) 2-D sequence.

Implements a fast GRE sequence with RF spoiling to destroy residual
transverse coherences. The key differences from standard GRE are:

  - Very short TR (< T2) with low flip angles
  - Quadratic RF phase increment (spoiling) to suppress steady-state
    transverse magnetisation
  - RF-spoiling phase cycle: φ_n = φ_{n-1} + n·Δφ  (Δφ = 117°)

This produces predominantly T1-weighted contrast and is the basis for
clinical fast gradient echo imaging (FLASH / SPGR / T1-FFE).
"""

import numpy as np
from mri_sim.sequence import (
    Sequence,
    RFEvent,
    GradientEvent,
    ADCEvent,
    DelayEvent,
    readout_gradient,
    phase_encode_gradient,
)


def build_fast_gre_sequence(
    fov=0.22,
    n_read=64,
    n_phase=64,
    flip_angle_deg=10.0,
    TE=5e-3,
    TR=15e-3,
    readout_duration=None,
    grad_duration=0.5e-3,
    rf_spoil_increment_deg=117.0,
):
    """Build a 2-D Fast GRE (FLASH/SPGR) pulse sequence with RF spoiling.

    Parameters
    ----------
    fov : float
        Field of view in metres.
    n_read : int
        Number of readout samples.
    n_phase : int
        Number of phase-encode lines.
    flip_angle_deg : float
        Excitation flip angle in degrees.
    TE : float
        Echo time in seconds.
    TR : float
        Repetition time in seconds.
    readout_duration : float or None
        Total readout duration in seconds.
    grad_duration : float
        Gradient lobe duration in seconds.
    rf_spoil_increment_deg : float
        RF spoiling phase increment in degrees (commonly 117°).

    Returns
    -------
    Sequence
        Configured Fast GRE pulse sequence.
    """
    if readout_duration is None:
        readout_duration = n_read * 10e-6

    seq = Sequence(name="FastGRE_2D")
    flip_angle = np.radians(flip_angle_deg)

    G_read = readout_gradient(fov, n_read, readout_duration)

    # RF spoiling state
    rf_phase = 0.0
    rf_inc = np.radians(rf_spoil_increment_deg)

    for n, ii in enumerate(range(-n_phase // 2, n_phase // 2)):
        # Quadratic phase increment: φ_n = φ_{n-1} + n * Δφ
        rf_phase_rad = rf_phase

        # --- RF excitation with spoiled phase ---
        seq.add_event(RFEvent(flip_angle, phase=rf_phase_rad))

        # --- Pre-winder and phase encode ---
        G_pe = phase_encode_gradient(fov, n_phase, ii, grad_duration)
        G_pre = -G_read / 2 * (readout_duration / grad_duration)
        seq.add_event(GradientEvent(grad_duration, gx=G_pre, gy=G_pe))

        # --- TE delay ---
        elapsed = grad_duration
        delay_te = TE - elapsed - readout_duration / 2
        if delay_te > 0:
            seq.add_event(DelayEvent(delay_te))

        # --- Readout with ADC ---
        seq.add_event(ADCEvent(n_read, readout_duration, gx=G_read,
                               phase_offset=rf_phase_rad))

        # --- Rewinder ---
        seq.add_event(GradientEvent(grad_duration, gy=-G_pe))

        # --- TR delay ---
        time_used = grad_duration + max(delay_te, 0) + readout_duration + grad_duration
        delay_tr = TR - time_used
        if delay_tr > 0:
            seq.add_event(DelayEvent(delay_tr))

        # Update RF spoiling phase (quadratic increment)
        rf_phase += (n + 1) * rf_inc

    return seq


if __name__ == "__main__":
    seq = build_fast_gre_sequence()
    print(seq)
    print(f"Total ADC samples: {seq.n_adc_samples}")
    print(f"Approx duration: {seq.total_duration:.3f} s")
