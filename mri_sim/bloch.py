"""Bloch equation simulator for MRI signal formation.

Implements discrete-time Bloch equation simulation using rotation matrices
for RF excitation and gradient-induced precession, combined with exponential
relaxation for T1 recovery and T2/T2* decay. Supports arbitrary pulse
sequence event lists operating on multi-voxel phantoms.
"""

import numpy as np


# Gyromagnetic ratio for 1H in Hz/T
GAMMA_HZ = 42.577e6
GAMMA_RAD = 2 * np.pi * GAMMA_HZ


def rotation_x(angle):
    """Rotation matrix about the x-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=np.float64)


def rotation_y(angle):
    """Rotation matrix about the y-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def rotation_z(angle):
    """Rotation matrix about the z-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=np.float64)


def rf_rotation(flip_angle, phase=0.0):
    """Compute rotation matrix for an RF pulse.

    The RF pulse is applied in the transverse plane at the given phase angle.
    This is equivalent to rotating about an axis in the x-y plane defined by
    the phase.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    phase : float, optional
        Phase of the RF pulse in radians (0 = x-axis). Default 0.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    # Rotate to align RF axis with x, apply flip, rotate back
    return rotation_z(phase) @ rotation_x(flip_angle) @ rotation_z(-phase)


def relaxation(dt, T1, T2):
    """Compute relaxation decay factors for a time step.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    T1 : float or np.ndarray
        Longitudinal relaxation time in seconds.
    T2 : float or np.ndarray
        Transverse relaxation time in seconds.

    Returns
    -------
    E1 : float or np.ndarray
        T1 decay factor exp(-dt/T1).
    E2 : float or np.ndarray
        T2 decay factor exp(-dt/T2).
    """
    E1 = np.exp(-dt / T1)
    E2 = np.exp(-dt / T2)
    return E1, E2


class BlochSimulator:
    """Discrete-time Bloch equation simulator for multi-voxel phantoms.

    The simulator maintains the magnetization state of all voxels and steps
    through a pulse sequence event by event, applying RF rotations, gradient-
    induced phase accrual, and T1/T2 relaxation at each time step.

    Parameters
    ----------
    phantom : Phantom
        Phantom object containing tissue properties and spatial positions.

    Attributes
    ----------
    n_voxels : int
        Number of voxels in the phantom.
    M : np.ndarray
        Magnetization vectors, shape (n_voxels, 3) for [Mx, My, Mz].
    """

    def __init__(self, phantom):
        self.phantom = phantom
        self.n_voxels = phantom.n_voxels
        # Initialise magnetisation at thermal equilibrium: M = [0, 0, PD]
        self.M = np.zeros((self.n_voxels, 3), dtype=np.float64)
        self.M[:, 2] = phantom.PD.copy()

    def reset(self):
        """Reset magnetisation to thermal equilibrium."""
        self.M[:] = 0.0
        self.M[:, 2] = self.phantom.PD.copy()

    def apply_rf(self, flip_angle, phase=0.0):
        """Apply an RF pulse to all voxels.

        Parameters
        ----------
        flip_angle : float
            Flip angle in radians.
        phase : float, optional
            RF phase in radians. Default 0.
        """
        R = rf_rotation(flip_angle, phase)
        # Apply rotation to each voxel's magnetisation
        self.M = (R @ self.M.T).T

    def apply_relaxation_and_precession(self, dt, gradients=None):
        """Apply free precession and relaxation for duration *dt*.

        Precession is caused by gradients and B0 inhomogeneity. Relaxation
        follows the Bloch equations with T1 recovery and T2 decay.

        Parameters
        ----------
        dt : float
            Duration in seconds.
        gradients : tuple of float or None, optional
            Gradient amplitudes (Gx, Gy, Gz) in T/m. Default None (no gradient).
        """
        if dt <= 0:
            return

        T1 = self.phantom.T1
        T2 = self.phantom.T2
        PD = self.phantom.PD

        E1, E2 = relaxation(dt, T1, T2)

        # Off-resonance precession angle per voxel
        # dw = gamma * (Gx*x + Gy*y + Gz*z + dB0) * dt
        dw = np.zeros(self.n_voxels, dtype=np.float64)

        if gradients is not None:
            Gx, Gy, Gz = gradients
            dw += GAMMA_RAD * (
                Gx * self.phantom.x
                + Gy * self.phantom.y
                + Gz * self.phantom.z
            ) * dt

        # Add B0 inhomogeneity contribution
        dw += GAMMA_RAD * self.phantom.B0 * dt

        # Apply T2* decay (T2 + T2' dephasing via B0 is already in precession)
        cos_dw = np.cos(dw)
        sin_dw = np.sin(dw)

        Mx = self.M[:, 0]
        My = self.M[:, 1]
        Mz = self.M[:, 2]

        # Precession + T2 decay in transverse plane
        Mx_new = E2 * (Mx * cos_dw - My * sin_dw)
        My_new = E2 * (Mx * sin_dw + My * cos_dw)
        # T1 recovery
        Mz_new = E1 * Mz + PD * (1 - E1)

        self.M[:, 0] = Mx_new
        self.M[:, 1] = My_new
        self.M[:, 2] = Mz_new

    def readout(self):
        """Sample the transverse magnetisation of all voxels.

        Returns
        -------
        complex
            Sum of transverse magnetisation Mx + j*My over all voxels.
        """
        return np.sum(self.M[:, 0] + 1j * self.M[:, 1])

    def run_sequence(self, sequence):
        """Execute a pulse sequence and collect ADC samples.

        Parameters
        ----------
        sequence : Sequence
            Pulse sequence containing a list of events.

        Returns
        -------
        signal : np.ndarray
            Complex-valued acquired signal, shape (n_adc_samples,).
        kspace_loc : np.ndarray
            k-space sample locations, shape (n_adc_samples, 2) for (kx, ky).
        """
        from mri_sim.sequence import RFEvent, GradientEvent, ADCEvent, DelayEvent

        signal_list = []
        kspace_list = []
        kx, ky, kz = 0.0, 0.0, 0.0

        for event in sequence.events:
            if isinstance(event, RFEvent):
                self.apply_rf(event.flip_angle, event.phase)

            elif isinstance(event, GradientEvent):
                # Gradient event with given amplitude and duration
                self.apply_relaxation_and_precession(
                    event.duration,
                    gradients=(event.gx, event.gy, event.gz),
                )
                # Update k-space position
                kx += GAMMA_RAD * event.gx * event.duration / (2 * np.pi)
                ky += GAMMA_RAD * event.gy * event.duration / (2 * np.pi)
                kz += GAMMA_RAD * event.gz * event.duration / (2 * np.pi)

            elif isinstance(event, ADCEvent):
                # Sample n_samples during the ADC window
                dt_sample = event.duration / event.n_samples
                gx = event.gx if event.gx is not None else 0.0
                gy = event.gy if event.gy is not None else 0.0
                gz = event.gz if event.gz is not None else 0.0
                for _ in range(event.n_samples):
                    # Half-step precession, sample, half-step precession
                    self.apply_relaxation_and_precession(
                        dt_sample / 2, gradients=(gx, gy, gz)
                    )
                    kx += GAMMA_RAD * gx * (dt_sample / 2) / (2 * np.pi)
                    ky += GAMMA_RAD * gy * (dt_sample / 2) / (2 * np.pi)

                    # Apply phase offset for ADC
                    sample = self.readout()
                    if event.phase_offset != 0.0:
                        sample *= np.exp(-1j * event.phase_offset)
                    signal_list.append(sample)
                    kspace_list.append([kx, ky])

                    self.apply_relaxation_and_precession(
                        dt_sample / 2, gradients=(gx, gy, gz)
                    )
                    kx += GAMMA_RAD * gx * (dt_sample / 2) / (2 * np.pi)
                    ky += GAMMA_RAD * gy * (dt_sample / 2) / (2 * np.pi)

            elif isinstance(event, DelayEvent):
                self.apply_relaxation_and_precession(event.duration)

        signal = np.array(signal_list, dtype=np.complex128)
        kspace_loc = np.array(kspace_list, dtype=np.float64) if kspace_list else np.empty((0, 2))
        return signal, kspace_loc
