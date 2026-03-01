"""RARE (Rapid Acquisition with Relaxation Enhancement) / TSE sequence.

Also known as Turbo Spin Echo (TSE) or Fast Spin Echo (FSE). Acquires
multiple phase-encode lines per excitation by using a train of 180°
refocusing pulses (echo train). This dramatically reduces scan time
compared to conventional spin echo:

  - Single 90° excitation per echo train
  - Train of 180° refocusing pulses, each followed by a readout
  - Phase-encode and rewinder gradients bracket each readout
  - Echo Train Length (ETL) determines acceleration factor
  - Effective TE determined by which echo fills the centre of k-space
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


def build_rare_sequence(
    fov=0.22,
    n_read=64,
    n_phase=64,
    echo_train_length=8,
    esp=12e-3,
    TR=2000e-3,
    readout_duration=None,
    grad_duration=1e-3,
):
    """Build a 2-D RARE (Turbo Spin Echo) pulse sequence.

    Parameters
    ----------
    fov : float
        Field of view in metres.
    n_read : int
        Number of readout samples per echo.
    n_phase : int
        Total number of phase-encode lines.
    echo_train_length : int
        Number of echoes per excitation (ETL).
    esp : float
        Echo spacing in seconds (time between consecutive 180° pulses).
    TR : float
        Repetition time in seconds.
    readout_duration : float or None
        Total readout duration per echo in seconds.
    grad_duration : float
        Gradient lobe duration in seconds.

    Returns
    -------
    Sequence
        Configured RARE pulse sequence.
    """
    if readout_duration is None:
        readout_duration = n_read * 10e-6

    seq = Sequence(name="RARE_2D")

    G_read = readout_gradient(fov, n_read, readout_duration)

    # Number of excitations (shots)
    n_shots = int(np.ceil(n_phase / echo_train_length))

    # Phase-encode ordering: linear from -N/2 to N/2-1
    pe_indices = list(range(-n_phase // 2, n_phase // 2))

    acquired = 0
    for shot in range(n_shots):
        # === 90° excitation ===
        seq.add_event(RFEvent(np.pi / 2, phase=np.pi / 2))

        # === Pre-winder for readout ===
        G_pre = G_read / 2 * (readout_duration / grad_duration)
        seq.add_event(GradientEvent(grad_duration, gx=G_pre))

        # Initial delay before first 180°
        delay_to_first_180 = esp / 2 - grad_duration
        if delay_to_first_180 > 0:
            seq.add_event(DelayEvent(delay_to_first_180))

        for echo_idx in range(echo_train_length):
            if acquired >= n_phase:
                break

            ii = pe_indices[acquired]

            # === 180° refocusing pulse ===
            seq.add_event(RFEvent(np.pi, phase=0.0))

            # === Phase encode + readout prewinder ===
            G_pe = phase_encode_gradient(fov, n_phase, ii, grad_duration)
            G_rw = G_read / 2 * (readout_duration / grad_duration)
            seq.add_event(GradientEvent(grad_duration, gx=G_rw, gy=G_pe))

            # Delay to echo centre
            delay_to_echo = esp / 2 - grad_duration - readout_duration / 2
            if delay_to_echo > 0:
                seq.add_event(DelayEvent(delay_to_echo))

            # === Readout with ADC ===
            seq.add_event(ADCEvent(n_read, readout_duration, gx=G_read,
                                   phase_offset=np.pi / 2))

            # === Phase-encode rewinder + readout rewinder ===
            seq.add_event(GradientEvent(grad_duration, gx=G_rw, gy=-G_pe))

            # Delay to next 180° centre
            delay_post = esp / 2 - grad_duration - readout_duration / 2
            if delay_post > 0:
                seq.add_event(DelayEvent(delay_post))

            acquired += 1

        # === TR delay ===
        etl_used = min(echo_train_length, n_phase - shot * echo_train_length)
        time_used = grad_duration + max(delay_to_first_180, 0) + etl_used * esp
        delay_tr = TR - time_used
        if delay_tr > 0:
            seq.add_event(DelayEvent(delay_tr))

    return seq


if __name__ == "__main__":
    seq = build_rare_sequence()
    print(seq)
    print(f"Total ADC samples: {seq.n_adc_samples}")
    print(f"Approx duration: {seq.total_duration:.3f} s")
