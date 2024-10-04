from __future__ import annotations
from re import S

import xarray as xr
import numpy as np
import attrs

from orbitals import electron_functions
from orbitals.definitions import CartesianCoords, RadialCoords, QuantumNumbers
from orbitals import tools


@attrs.define
class WavefunctionVolume:

    wavefunction: xr.DataArray

    resolution: dict
    r_max: int

    def _normalize(self):
        # Normalise so sum of elements is 1
        self.wavefunction.data /= np.sum(np.abs(self.wavefunction.data))

    def meshgrid_coords(self):
        return np.meshgrid(*[coord for coord in self.get_coords().values()])

    def get_density(self):
        density = np.absolute(self.get_wavefunction().data) ** 2
        density /= np.sum(np.abs(density))
        return density

    def get_wavefunction(self):
        return self.wavefunction.data

    def get_coords(self):
        return self.wavefunction.coords

    def get_dims(self):
        return self.wavefunction.dims


@attrs.define
class OneEAtomicWavefunction(WavefunctionVolume):
    """
    Base class for 1-electron atomic wavefunctions.
    """
    
    def get_quantum_numbers(self):
        return (
            self.wavefunction.attrs[QuantumNumbers.N],
            self.wavefunction.attrs[QuantumNumbers.L],
            self.wavefunction.attrs[QuantumNumbers.M],
        )

    @classmethod
    def new_1e_atomic_wavefunction(cls, resolution: dict, r_max: int,  n: int, l: int, m: int) -> OneEAtomicWavefunction:
        raise NotImplementedError

@attrs.define
class RadialWavefunction(OneEAtomicWavefunction):
    """
    Radial electron wavefunction class.

    args:
    resolution: dict, resolution of the wavefunction
    r_max: int, maximum radius of the wavefunction

    attrs:
    wavefunction: xarray.DataArray, radial electron wavefunction
    """

    @classmethod
    def new_1e_atomic_wavefunction(cls, resolution: dict, r_max: int,  n: int, l: int, m: int) -> RadialWavefunction:

        # Check that we've been provided with physically meaningful inputs
        assert tools.validate_quantum_numbers(n, l, m)
        assert resolution.keys() == RadialCoords

        # Radial wavefunction with coords r, phi, psi
        wavefunction = xr.DataArray(
            data=np.ones(
                (
                    resolution[RadialCoords.R],
                    resolution[RadialCoords.THETA],
                    resolution[RadialCoords.PHI],
                )
            ),
            dims=[RadialCoords.R, RadialCoords.THETA, RadialCoords.PHI],
            coords={
                RadialCoords.R: np.linspace(
                    0, r_max, resolution[RadialCoords.R]
                ),
                RadialCoords.THETA: np.linspace(
                    0, 2 * np.pi, resolution[RadialCoords.THETA]
                ),
                RadialCoords.PHI: np.linspace(
                    0, np.pi, resolution[RadialCoords.PHI]
                ),
            },
            attrs={
                "resolution": resolution,
                QuantumNumbers.N: n,
                QuantumNumbers.L: l,
                QuantumNumbers.M: m,
            },
        )

        return cls(wavefunction=wavefunction, resolution=resolution, r_max=r_max)

    def eval_wavefunction(self):

        # Check that we've been provided with physically meaningful inputs
        assert tools.validate_quantum_numbers(
            self.wavefunction.attrs[QuantumNumbers.N],
            self.wavefunction.attrs[QuantumNumbers.L],
            self.wavefunction.attrs[QuantumNumbers.M],
        )

        rr, tt, pp = np.meshgrid(
            self.wavefunction.coords[RadialCoords.R],
            self.wavefunction.coords[RadialCoords.THETA],
            self.wavefunction.coords[RadialCoords.PHI],
        )

        self.wavefunction.data = electron_functions.wavefunction(
            self.wavefunction.attrs[QuantumNumbers.N],
            self.wavefunction.attrs[QuantumNumbers.L],
            self.wavefunction.attrs[QuantumNumbers.M],
            rr, tt, pp)

        self._normalize()



@attrs.define
class CartesianWavefunction(OneEAtomicWavefunction):
    """
    Cartesian electron wavefunction class.

    args:
    resolution: dict, resolution of the wavefunction
    r_max: int, maximum radius of the wavefunction

    attrs:
    wavefunction: xarray.DataArray, radial electron wavefunction
    """

    # Cartesian wavefunction with coords x, y, z
    resolution: dict

    # Cartesian wavefunction with coords x, y, z
    # wavefunction = attrs.field(init=False)

    @classmethod
    def new_1e_atomic_wavefunction(cls, resolution: dict, r_max: int,  n: int, l: int, m: int) -> CartesianWavefunction:
        wavefunction = xr.DataArray(
            data=np.ones(
                (
                    resolution[CartesianCoords.X],
                    resolution[CartesianCoords.Y],
                    resolution[CartesianCoords.Z],
                )
            ),
            dims=[CartesianCoords.X, CartesianCoords.Y, CartesianCoords.Z],
            coords={
                CartesianCoords.X: np.linspace(
                    -r_max, r_max, resolution[CartesianCoords.X]
                ),
                CartesianCoords.Y: np.linspace(
                    -r_max, r_max, resolution[CartesianCoords.Y]
                ),
                CartesianCoords.Z: np.linspace(
                    -r_max, r_max, resolution[CartesianCoords.Z]
                ),
            },
            attrs={
                "resolution": resolution,
                QuantumNumbers.N: n,
                QuantumNumbers.L: l,
                QuantumNumbers.M: m,
            },
        )

        return cls(wavefunction=wavefunction, resolution=resolution, r_max=r_max)

    def eval_wavefunction(self):

        # Check that we've been provided with physically meaningful inputs
        assert tools.validate_quantum_numbers(
            self.wavefunction.attrs[QuantumNumbers.N],
            self.wavefunction.attrs[QuantumNumbers.L],
            self.wavefunction.attrs[QuantumNumbers.M],
        )

        xx, yy, zz = np.meshgrid(
            self.wavefunction.coords[CartesianCoords.X],
            self.wavefunction.coords[CartesianCoords.Y],
            self.wavefunction.coords[CartesianCoords.Z],
        )

        rr, tt, pp = tools.convert_cartesian_to_radial(xx, yy, zz)

        self.wavefunction.data = electron_functions.wavefunction(
            self.wavefunction.attrs[QuantumNumbers.N],
            self.wavefunction.attrs[QuantumNumbers.L],
            self.wavefunction.attrs[QuantumNumbers.M],
            rr, tt, pp
        )

        self._normalize()