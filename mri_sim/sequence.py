"""Pulse sequence definition framework.

Provides data classes for RF pulses, gradient events, ADC readouts, and
delays, plus a ``Sequence`` container that orders events into an executable
list for the Bloch simulator.
"""

import numpy as np


class RFEvent:
    """Radio-frequency excitation event.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    phase : float, optional
        RF phase in radians. Default 0.
    """

    def __init__(self, flip_angle, phase=0.0):
        self.flip_angle = float(flip_angle)
        self.phase = float(phase)

    def __repr__(self):
        return (
            f"RFEvent(flip_angle={np.degrees(self.flip_angle):.1f}°, "
            f"phase={np.degrees(self.phase):.1f}°)"
        )


class GradientEvent:
    """Gradient event with specified amplitude and duration.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    gx : float, optional
        X-gradient amplitude in T/m. Default 0.
    gy : float, optional
        Y-gradient amplitude in T/m. Default 0.
    gz : float, optional
        Z-gradient amplitude in T/m. Default 0.
    """

    def __init__(self, duration, gx=0.0, gy=0.0, gz=0.0):
        self.duration = float(duration)
        self.gx = float(gx)
        self.gy = float(gy)
        self.gz = float(gz)

    def __repr__(self):
        return (
            f"GradientEvent(dt={self.duration * 1e3:.2f}ms, "
            f"Gx={self.gx * 1e3:.2f}mT/m, Gy={self.gy * 1e3:.2f}mT/m)"
        )


class ADCEvent:
    """Analogue-to-digital converter readout event.

    Samples the transverse magnetisation during the readout window while
    a readout gradient is optionally active.

    Parameters
    ----------
    n_samples : int
        Number of readout samples.
    duration : float
        Total readout duration in seconds.
    gx : float, optional
        Readout gradient amplitude in T/m during sampling.
    gy : float, optional
        Phase-encode gradient amplitude during sampling.
    gz : float, optional
        Slice gradient amplitude during sampling.
    phase_offset : float, optional
        Phase offset for the ADC in radians. Default 0.
    """

    def __init__(self, n_samples, duration, gx=None, gy=None, gz=None,
                 phase_offset=0.0):
        self.n_samples = int(n_samples)
        self.duration = float(duration)
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.phase_offset = float(phase_offset)

    def __repr__(self):
        return (
            f"ADCEvent(n_samples={self.n_samples}, "
            f"duration={self.duration * 1e3:.2f}ms)"
        )


class DelayEvent:
    """Simple delay (dead time) event.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    """

    def __init__(self, duration):
        self.duration = float(duration)

    def __repr__(self):
        return f"DelayEvent({self.duration * 1e3:.2f}ms)"


class Sequence:
    """Pulse sequence container.

    Stores an ordered list of events (RF, gradient, ADC, delay) that
    define an MRI pulse sequence.

    Parameters
    ----------
    name : str, optional
        Human-readable name for the sequence.
    """

    def __init__(self, name="unnamed"):
        self.name = name
        self.events = []

    def add_event(self, event):
        """Append a single event to the sequence.

        Parameters
        ----------
        event : RFEvent, GradientEvent, ADCEvent, or DelayEvent
        """
        self.events.append(event)

    def add_block(self, *events):
        """Append multiple events executed sequentially.

        Parameters
        ----------
        *events : sequence event objects
        """
        for ev in events:
            self.events.append(ev)

    @property
    def n_events(self):
        """Total number of events."""
        return len(self.events)

    @property
    def n_adc_samples(self):
        """Total number of ADC samples in the sequence."""
        return sum(e.n_samples for e in self.events if isinstance(e, ADCEvent))

    @property
    def total_duration(self):
        """Approximate total duration in seconds."""
        dur = 0.0
        for e in self.events:
            if isinstance(e, (GradientEvent, ADCEvent, DelayEvent)):
                dur += e.duration
        return dur

    def __repr__(self):
        return (
            f"Sequence('{self.name}', events={self.n_events}, "
            f"ADC_samples={self.n_adc_samples})"
        )


# ---------------------------------------------------------------------------
# Helper functions for computing gradient amplitudes from k-space areas
# ---------------------------------------------------------------------------

GAMMA_HZ = 42.577e6
GAMMA_RAD = 2 * np.pi * GAMMA_HZ


def gradient_amplitude(area, duration):
    """Compute gradient amplitude for a desired k-space area.

    Parameters
    ----------
    area : float
        Target k-space area in 1/m (= gamma * G * dt / (2*pi)).
    duration : float
        Gradient duration in seconds.

    Returns
    -------
    float
        Gradient amplitude in T/m.
    """
    # area = gamma/(2*pi) * G * dt  =>  G = area * 2*pi / (gamma * dt)
    return area * 2 * np.pi / (GAMMA_RAD * duration)


def readout_gradient(fov, n_samples, duration):
    """Compute readout gradient amplitude for Cartesian sampling.

    Parameters
    ----------
    fov : float
        Field of view in metres.
    n_samples : int
        Number of readout samples.
    duration : float
        Readout duration in seconds.

    Returns
    -------
    float
        Gradient amplitude in T/m.
    """
    # dk = 1/FOV, total k-span = N * dk = N/FOV
    # G = total_k_span * 2*pi / (gamma * duration)
    total_k = n_samples / fov
    return total_k * 2 * np.pi / (GAMMA_RAD * duration)


def phase_encode_gradient(fov, n_phase, line_index, duration):
    """Compute phase-encode gradient amplitude for a specific line.

    Parameters
    ----------
    fov : float
        Field of view in metres.
    n_phase : int
        Total number of phase-encode lines.
    line_index : int
        Phase-encode index, ranging from -n_phase//2 to n_phase//2 - 1.
    duration : float
        Phase-encode gradient duration in seconds.

    Returns
    -------
    float
        Gradient amplitude in T/m.
    """
    area = line_index / fov  # k-space position in 1/m
    return gradient_amplitude(area, duration)
