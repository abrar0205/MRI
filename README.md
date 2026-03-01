# MRI Pulse Sequence Programming & Simulation Framework

A Python-based framework for MRI pulse sequence design and Bloch equation
simulation, developed as part of MRT Forschung am Universitätsklinikum
Erlangen. The project implements structured sequence programming, signal
formation modelling, and image reconstruction for multiple encoding
strategies.

## Features

- **Bloch Equation Simulator** — Discrete-time simulation using rotation
  matrices for RF excitation and gradient-induced precession, with
  exponential T1/T2 relaxation. Supports multi-voxel phantoms with
  spatially varying tissue properties and B0 inhomogeneity.

- **Pulse Sequence Framework** — Modular event-based sequence definition
  (RF, gradient, ADC, delay) with helper functions for gradient amplitude
  calculation from k-space requirements.

- **Implemented Sequences**:
  - **GRE** (Gradient Recalled Echo) — Standard Cartesian 2-D GRE
  - **Fast GRE** (FLASH/SPGR) — Spoiled GRE with RF phase cycling
  - **Spin Echo** — 90°–180° refocused echo with T2 contrast
  - **RARE** (Turbo Spin Echo / FSE) — Multi-echo acquisition with
    configurable echo train length
  - **Balanced SSFP** (TrueFISP/FIESTA) — Fully balanced gradients with
    α/2 preparation and alternating RF phase

- **Non-Cartesian Trajectories** — Radial (golden-angle) and spiral
  (Archimedean) k-space sampling with NUFFT reconstruction.

- **Undersampling** — Regular and variable-density random undersampling
  masks for compressed sensing applications.

- **Digital Phantoms** — Numerical brain phantom with anatomically
  motivated tissue compartments (WM, GM, CSF) and realistic relaxation
  parameters at 1.5 T.

- **Reconstruction** — Cartesian FFT and non-Cartesian gridding (NUFFT)
  reconstruction pipelines.

## Project Structure

```
MRI/
├── mri_sim/                  # Core simulation library
│   ├── __init__.py
│   ├── bloch.py              # Bloch equation simulator
│   ├── phantom.py            # Digital phantom generation
│   ├── sequence.py           # Pulse sequence framework
│   ├── reconstruction.py     # Image reconstruction (FFT, NUFFT)
│   └── plotting.py           # Visualisation utilities
├── sequences/                # Pulse sequence implementations
│   ├── gre.py                # Gradient Recalled Echo
│   ├── fast_gre.py           # Fast GRE with RF spoiling
│   ├── spin_echo.py          # Spin Echo
│   ├── rare.py               # RARE / Turbo Spin Echo
│   ├── bssfp.py              # Balanced SSFP
│   └── non_cartesian.py      # Radial, spiral, undersampling
├── tests/                    # Test suite
│   ├── test_bloch.py         # Bloch simulator tests
│   ├── test_sequences.py     # Sequence builder tests
│   └── test_reconstruction.py # Reconstruction tests
├── examples/
│   └── run_all.py            # Run all sequence demonstrations
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Matplotlib ≥ 3.4
- pytest ≥ 7.0 (for testing)

## Quick Start

Run all sequence demonstrations:

```bash
python examples/run_all.py
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Usage Example

```python
from mri_sim.bloch import BlochSimulator
from mri_sim.phantom import numerical_brain_phantom
from mri_sim.reconstruction import reconstruct_cartesian
from sequences.gre import build_gre_sequence

# Create a digital brain phantom
phantom = numerical_brain_phantom(fov=0.22, n_pixels=64)

# Build a GRE pulse sequence
seq = build_gre_sequence(
    fov=0.22, n_read=64, n_phase=64,
    flip_angle_deg=15, TE=10e-3, TR=50e-3,
)

# Simulate signal acquisition
sim = BlochSimulator(phantom)
signal, kspace_loc = sim.run_sequence(seq)

# Reconstruct the image
image, kspace = reconstruct_cartesian(signal, n_read=64, n_phase=64)
```

## References

- Inspired by the [MRTwin_pulseq](https://github.com/mzaiss/MRTwin_pulseq)
  course framework by M. Zaiss et al.
- [Pulseq](https://pulseq.github.io/) — Open-source MRI sequence standard
- [PyPulseq](https://github.com/imr-framework/pypulseq) — Python Pulseq port
- Bloch, F. "Nuclear Induction." *Physical Review* 70 (1946): 460–474