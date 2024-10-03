from __future__ import annotations

import xarray as xr
import numpy as np
import attrs

from orbitals import electron_functions
from orbitals.definitions import CartesianCoords, RadialCoords
from orbitals import tools


@attrs.define
class WavefunctionVolume:
    resolution: dict
    r_max: int = 1

    wavefunction: xr.DataArray = attrs.field(init=False)

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
class RadialWavefunction(WavefunctionVolume):
    """
    Radial electron wavefunction class.

    args:
    resolution: dict, resolution of the wavefunction
    r_max: int, maximum radius of the wavefunction

    attrs:
    wavefunction: xarray.DataArray, radial electron wavefunction
    """

    # Radial wavefunction with coords r, phi, psi
    resolution: dict

    wavefunction = attrs.field(init=False)

    def __attrs_post_init__(self):
        # Radial wavefunction with coords r, phi, psi
        self.wavefunction = xr.DataArray(
            data=np.ones(
                (
                    self.resolution[RadialCoords.R],
                    self.resolution[RadialCoords.THETA],
                    self.resolution[RadialCoords.PHI],
                )
            ),
            dims=[RadialCoords.R, RadialCoords.THETA, RadialCoords.PHI],
            coords={
                RadialCoords.R: np.linspace(
                    0, self.r_max, self.resolution[RadialCoords.R]
                ),
                RadialCoords.THETA: np.linspace(
                    0, 2 * np.pi, self.resolution[RadialCoords.THETA]
                ),
                RadialCoords.PHI: np.linspace(
                    0, np.pi, self.resolution[RadialCoords.PHI]
                ),
            },
            attrs={
                "resolution": self.resolution,
            },
        )

        self._normalize()

    def eval_wavefunction(self, n: int, l: int, m: int):

        # Check that we've been provided with physically meaningful inputs
        assert tools.validate_quantum_numbers(n, l, m)

        rr, tt, pp = np.meshgrid(
            self.wavefunction.coords[RadialCoords.R],
            self.wavefunction.coords[RadialCoords.THETA],
            self.wavefunction.coords[RadialCoords.PHI],
        )

        self.wavefunction.data = electron_functions.wavefunction(n, l, m, rr, tt, pp)

        self._normalize()


@attrs.define
class CartesianWavefunction(WavefunctionVolume):
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
    wavefunction = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.wavefunction = xr.DataArray(
            data=np.ones(
                (
                    self.resolution[CartesianCoords.X],
                    self.resolution[CartesianCoords.Y],
                    self.resolution[CartesianCoords.Z],
                )
            ),
            dims=[CartesianCoords.X, CartesianCoords.Y, CartesianCoords.Z],
            coords={
                CartesianCoords.X: np.linspace(
                    -self.r_max, self.r_max, self.resolution[CartesianCoords.X]
                ),
                CartesianCoords.Y: np.linspace(
                    -self.r_max, self.r_max, self.resolution[CartesianCoords.Y]
                ),
                CartesianCoords.Z: np.linspace(
                    -self.r_max, self.r_max, self.resolution[CartesianCoords.Z]
                ),
            },
            attrs={
                "resolution": self.resolution,
            },
        )

        self._normalize()

    def eval_wavefunction(self, n: int, l: int, m: int):

        # Check that we've been provided with physically meaningful inputs
        assert tools.validate_quantum_numbers(n, l, m)

        xx, yy, zz = np.meshgrid(
            self.wavefunction.coords[CartesianCoords.X],
            self.wavefunction.coords[CartesianCoords.Y],
            self.wavefunction.coords[CartesianCoords.Z],
        )

        rr, tt, pp = electron_functions.convert_cartesian_to_radial(xx, yy, zz)

        self.wavefunction.data = electron_functions.wavefunction(n, l, m, rr, tt, pp)

        self._normalize()
