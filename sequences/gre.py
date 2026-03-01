"""Gradient Recalled Echo (GRE) 2-D sequence.

Implements a standard Cartesian 2-D GRE pulse sequence with:
  - Low flip-angle RF excitation (typically ≤90°)
  - Frequency-encode (readout) gradient with pre-winder
  - Phase-encode gradient that varies each repetition
  - T1-weighted or PD-weighted contrast depending on TR/flip angle

The sequence loops over phase-encode lines from most-negative to
most-positive k_y, acquiring one line per TR.
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


def build_gre_sequence(
    fov=0.22,
    n_read=64,
    n_phase=64,
    flip_angle_deg=15.0,
    TE=10e-3,
    TR=50e-3,
    readout_duration=None,
    grad_duration=1e-3,
):
    """Build a 2-D Gradient Recalled Echo (GRE) pulse sequence.

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
        Total readout duration in seconds. Defaults to ``n_read * 10e-6``.
    grad_duration : float
        Duration of gradient lobes (pre-winder, phase encode) in seconds.

    Returns
    -------
    Sequence
        Configured GRE pulse sequence.
    """
    if readout_duration is None:
        readout_duration = n_read * 10e-6

    seq = Sequence(name="GRE_2D")
    flip_angle = np.radians(flip_angle_deg)

    # Readout gradient amplitude
    G_read = readout_gradient(fov, n_read, readout_duration)

    for ii in range(-n_phase // 2, n_phase // 2):
        # --- RF excitation ---
        seq.add_event(RFEvent(flip_angle))

        # --- Pre-winder and phase encode ---
        G_pe = phase_encode_gradient(fov, n_phase, ii, grad_duration)
        # Readout pre-winder: move to -kx_max
        G_pre = -G_read / 2 * (readout_duration / grad_duration)
        seq.add_event(GradientEvent(grad_duration, gx=G_pre, gy=G_pe))

        # --- Delay to reach TE ---
        elapsed = grad_duration  # time from RF centre to end of pre-winder
        delay_te = TE - elapsed - readout_duration / 2
        if delay_te > 0:
            seq.add_event(DelayEvent(delay_te))

        # --- Readout with ADC ---
        seq.add_event(ADCEvent(n_read, readout_duration, gx=G_read))

        # --- Spoiler / rewinder and TR delay ---
        # Phase-encode rewinder
        seq.add_event(GradientEvent(grad_duration, gy=-G_pe))

        # TR delay
        time_used = grad_duration + max(delay_te, 0) + readout_duration + grad_duration
        delay_tr = TR - time_used
        if delay_tr > 0:
            seq.add_event(DelayEvent(delay_tr))

    return seq


if __name__ == "__main__":
    seq = build_gre_sequence()
    print(seq)
    print(f"Total ADC samples: {seq.n_adc_samples}")
    print(f"Approx duration: {seq.total_duration:.3f} s")
