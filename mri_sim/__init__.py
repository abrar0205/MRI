"""MRI Pulse Sequence Simulation Framework.

A Python-based framework for MRI pulse sequence programming and Bloch
equation simulation. Implements structured sequence definition, signal
formation modelling, and image reconstruction for multiple encoding
strategies including GRE, Fast GRE, Spin Echo, RARE, and balanced SSFP.
"""

from mri_sim.bloch import BlochSimulator
from mri_sim.phantom import Phantom, numerical_brain_phantom
from mri_sim.sequence import (
    Sequence,
    RFEvent,
    GradientEvent,
    ADCEvent,
    DelayEvent,
)
from mri_sim.reconstruction import reconstruct_cartesian, reconstruct_nufft

__all__ = [
    "BlochSimulator",
    "Phantom",
    "numerical_brain_phantom",
    "Sequence",
    "RFEvent",
    "GradientEvent",
    "ADCEvent",
    "DelayEvent",
    "reconstruct_cartesian",
    "reconstruct_nufft",
]
