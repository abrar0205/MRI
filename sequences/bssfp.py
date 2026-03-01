"""Balanced Steady-State Free Precession (bSSFP) 2-D sequence.

Implements a balanced SSFP (also known as TrueFISP, FIESTA, or balanced
FFE) pulse sequence:

  - Alternating RF phase (0°/180°) or linear phase cycling
  - Fully balanced gradients: net gradient area is zero per TR
  - Readout gradient with symmetric pre-winder and rewinder
  - Phase-encode with matched rewinder
  - Half-alpha preparation pulse for smooth approach to steady state

bSSFP produces T2/T1-weighted contrast and is extremely efficient,
but sensitive to B0 inhomogeneity (banding artefacts at off-resonance
frequencies where ΔB0 = n/(2·TR)).
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


def build_bssfp_sequence(
    fov=0.22,
    n_read=64,
    n_phase=64,
    flip_angle_deg=30.0,
    TR=6e-3,
    readout_duration=None,
    grad_duration=0.5e-3,
):
    """Build a 2-D balanced SSFP (TrueFISP) pulse sequence.

    Parameters
    ----------
    fov : float
        Field of view in metres.
    n_read : int
        Number of readout samples.
    n_phase : int
        Number of phase-encode lines.
    flip_angle_deg : float
        Flip angle in degrees.
    TR : float
        Repetition time in seconds.
    readout_duration : float or None
        Total readout duration in seconds.
    grad_duration : float
        Gradient lobe duration in seconds.

    Returns
    -------
    Sequence
        Configured bSSFP pulse sequence.
    """
    if readout_duration is None:
        readout_duration = n_read * 10e-6

    seq = Sequence(name="bSSFP_2D")
    flip_angle = np.radians(flip_angle_deg)

    G_read = readout_gradient(fov, n_read, readout_duration)

    # === Half-alpha preparation pulse ===
    # Apply α/2 pulse to catalyse approach to steady state
    seq.add_event(RFEvent(flip_angle / 2, phase=0.0))
    # Wait half a TR to let magnetisation evolve
    prep_delay = TR / 2 - grad_duration
    if prep_delay > 0:
        seq.add_event(DelayEvent(prep_delay))

    # RF phase alternation: 0, π, 0, π, ...
    rf_phase = 0.0
    rf_increment = np.pi  # 180° alternation

    for ii in range(-n_phase // 2, n_phase // 2):
        rf_phase_current = rf_phase

        # === RF excitation ===
        seq.add_event(RFEvent(flip_angle, phase=rf_phase_current))

        # === Pre-winder (readout) + Phase encode ===
        G_pe = phase_encode_gradient(fov, n_phase, ii, grad_duration)
        G_pre = -G_read / 2 * (readout_duration / grad_duration)
        seq.add_event(GradientEvent(grad_duration, gx=G_pre, gy=G_pe))

        # === Readout with ADC ===
        seq.add_event(ADCEvent(n_read, readout_duration, gx=G_read,
                               phase_offset=rf_phase_current))

        # === Rewinder (balanced): undo readout and phase encode ===
        # Readout rewinder: same as pre-winder (negative half-area)
        # Phase-encode rewinder: negative of encode gradient
        seq.add_event(GradientEvent(grad_duration, gx=G_pre, gy=-G_pe))

        # === TR delay ===
        time_used = 2 * grad_duration + readout_duration
        delay_tr = TR - time_used
        if delay_tr > 0:
            seq.add_event(DelayEvent(delay_tr))

        # Alternate RF phase
        rf_phase += rf_increment

    return seq


if __name__ == "__main__":
    seq = build_bssfp_sequence()
    print(seq)
    print(f"Total ADC samples: {seq.n_adc_samples}")
    print(f"Approx duration: {seq.total_duration:.3f} s")
