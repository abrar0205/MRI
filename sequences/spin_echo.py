"""Spin Echo (SE) 2-D sequence.

Implements a standard Cartesian 2-D Spin Echo pulse sequence:
  - 90° excitation pulse
  - 180° refocusing pulse at TE/2
  - Frequency-encode gradient with pre-winder
  - Phase-encode gradient applied between excitation and refocusing
  - Signal sampled at the echo time TE

Spin echo sequences refocus static field inhomogeneities and produce
T2-weighted (rather than T2*-weighted) contrast. The 180° pulse
inverts the phase accumulated from off-resonance, causing a spin echo
at time TE.
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


def build_spin_echo_sequence(
    fov=0.22,
    n_read=64,
    n_phase=64,
    TE=20e-3,
    TR=500e-3,
    readout_duration=None,
    grad_duration=1e-3,
):
    """Build a 2-D Spin Echo pulse sequence.

    Parameters
    ----------
    fov : float
        Field of view in metres.
    n_read : int
        Number of readout samples.
    n_phase : int
        Number of phase-encode lines.
    TE : float
        Echo time in seconds.
    TR : float
        Repetition time in seconds.
    readout_duration : float or None
        Total readout duration in seconds.
    grad_duration : float
        Gradient lobe duration in seconds.

    Returns
    -------
    Sequence
        Configured Spin Echo pulse sequence.
    """
    if readout_duration is None:
        readout_duration = n_read * 10e-6

    seq = Sequence(name="SE_2D")

    G_read = readout_gradient(fov, n_read, readout_duration)

    for ii in range(-n_phase // 2, n_phase // 2):
        # === 90° excitation ===
        seq.add_event(RFEvent(np.pi / 2, phase=np.pi / 2))

        # === Pre-winder + phase encode (between 90° and 180°) ===
        G_pe = phase_encode_gradient(fov, n_phase, ii, grad_duration)
        G_pre = G_read / 2 * (readout_duration / grad_duration)
        seq.add_event(GradientEvent(grad_duration, gx=G_pre, gy=G_pe))

        # === Delay to TE/2 ===
        elapsed_half = grad_duration
        delay_1 = TE / 2 - elapsed_half
        if delay_1 > 0:
            seq.add_event(DelayEvent(delay_1))

        # === 180° refocusing pulse ===
        seq.add_event(RFEvent(np.pi, phase=0.0))

        # === Delay from 180° to readout centre ===
        delay_2 = TE / 2 - readout_duration / 2
        if delay_2 > 0:
            seq.add_event(DelayEvent(delay_2))

        # === Readout with ADC ===
        seq.add_event(ADCEvent(n_read, readout_duration, gx=G_read,
                               phase_offset=np.pi / 2))

        # === TR delay ===
        time_used = grad_duration + max(delay_1, 0) + max(delay_2, 0) + readout_duration
        delay_tr = TR - time_used
        if delay_tr > 0:
            seq.add_event(DelayEvent(delay_tr))

    return seq


if __name__ == "__main__":
    seq = build_spin_echo_sequence()
    print(seq)
    print(f"Total ADC samples: {seq.n_adc_samples}")
    print(f"Approx duration: {seq.total_duration:.3f} s")
