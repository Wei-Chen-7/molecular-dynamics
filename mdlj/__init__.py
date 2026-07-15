"""mdlj — a Lennard-Jones molecular-dynamics engine in pure NumPy.

``mdlj`` simulates a classical gas or liquid of particles interacting through
the truncated-and-shifted Lennard-Jones 12-6 potential, in two (default) or
three dimensions, with periodic boundary conditions.  Trajectories are
generated with the velocity-Verlet integrator (microcanonical / NVE) and can be
thermostatted (Berendsen or Andersen) for canonical / NVT sampling.  Forces are
evaluated in ``O(N)`` time with a linked-cell list, fully vectorized with NumPy.

Reduced Lennard-Jones units
---------------------------
Everything is expressed with ``epsilon = sigma = m = k_B = 1``.  Consequently:

* length is measured in ``sigma``,
* energy in ``epsilon``,
* temperature in ``epsilon / k_B``,
* time in ``tau = sigma * sqrt(m / epsilon)``.

Typical building blocks
-----------------------
>>> import numpy as np
>>> from mdlj import (initialize_positions, maxwell_boltzmann_velocities,
...                   System, Simulation)
>>> rng = np.random.default_rng(0)
>>> pos, box = initialize_positions(64, density=0.7, dim=2, lattice="square")
>>> vel = maxwell_boltzmann_velocities(len(pos), 2, temperature=1.0, rng=rng)
>>> sim = Simulation(System(pos, vel, box), dt=0.005)
>>> history = sim.run(1000, sample_every=10)

What it is / is not
-------------------
It is a compact, physics-validated teaching and research scaffold: energy and
momentum conservation, the radial distribution function, the Maxwell-Boltzmann
speed distribution and the virial pressure are all checked against closed-form
references.  It is *not* a high-performance production code (no GPU, no MPI, no
neighbour-list rebuild heuristics) and it implements a single-component
monatomic fluid only.
"""

from __future__ import annotations

from . import analytic, plotting
from .initialize import (
    fcc_lattice,
    initialize_positions,
    maxwell_boltzmann_velocities,
    square_lattice,
    triangular_lattice,
)
from .integrator import velocity_verlet_step
from .neighbors import CellList, all_pairs
from .observables import (
    RadialDistribution,
    kinetic_energy,
    minimum_image_distances,
    speed_histogram,
    temperature,
    virial_pressure,
)
from .potential import LennardJones, lj_force_magnitude, lj_potential
from .simulation import Simulation, System, pair_forces
from .thermostat import AndersenThermostat, BerendsenThermostat

__version__ = "0.1.0"

__all__ = [
    # potential
    "LennardJones",
    "lj_potential",
    "lj_force_magnitude",
    # neighbours
    "CellList",
    "all_pairs",
    # dynamics
    "System",
    "Simulation",
    "pair_forces",
    "velocity_verlet_step",
    # initialization
    "initialize_positions",
    "maxwell_boltzmann_velocities",
    "square_lattice",
    "triangular_lattice",
    "fcc_lattice",
    # observables
    "kinetic_energy",
    "temperature",
    "speed_histogram",
    "virial_pressure",
    "minimum_image_distances",
    "RadialDistribution",
    # thermostats
    "BerendsenThermostat",
    "AndersenThermostat",
    # modules
    "analytic",
    "plotting",
    "__version__",
]
