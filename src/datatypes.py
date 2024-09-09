from __future__ import annotations

import xarray as xr
import numpy as np
import attrs
from src import electron_functions


@attrs.define
class WavefunctionVolume:
    resolution: dict
    r_max: int = 1

    def _normalize(self):
        self.density.data = self.density.data / np.nansum(self.density.data)

    def meshgrid_coords(self):
        return np.meshgrid(
            self.density.coords["x"],
            self.density.coords["y"],
            self.density.coords["z"],
        )
    
    def get_density(self):
        return np.absolute(self.density.data) ** 2


@attrs.define
class RadialWavefunction(WavefunctionVolume):
    """
    Radial electron density class.

    args:
    resolution: dict, resolution of the density
    r_max: int, maximum radius of the density

    attrs:
    density: xarray.DataArray, radial electron density
    """

    # Radial density with coords r, phi, psi
    resolution: dict

    density = attrs.field(init=False)

    def __attrs_post_init__(self):
        # Radial density with coords r, phi, psi
        self.density = xr.DataArray(
            data=np.ones(
                (self.resolution["r"], self.resolution["theta"], self.resolution["phi"])
            ),
            dims=["r", "theta", "phi"],
            coords={
                "r": np.linspace(0, self.r_max, self.resolution["r"]),
                "theta": np.linspace(0, 2 * np.pi, self.resolution["theta"]),
                "phi": np.linspace(0, np.pi, self.resolution["phi"]),
            },
            attrs={
                "resolution": self.resolution,
            },
        )

        self._normalize()

    def eval_density(self, n: int, l: int, m: int):
        rr, tt, pp = np.meshgrid(
            self.density.coords["r"],
            self.density.coords["theta"],
            self.density.coords["phi"],
        )

        self.density.data = electron_functions.wavefunction(n, l, m, rr, tt, pp)

        self._normalize()


@attrs.define
class CartesianWavefunction(WavefunctionVolume):
    """
    Cartesian electron density class.

    args:
    resolution: dict, resolution of the density
    r_max: int, maximum radius of the density

    attrs:
    density: xarray.DataArray, radial electron density
    """

    # Cartesian density with coords x, y, z
    resolution: dict

    # Cartesian density with coords x, y, z
    density = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.density = xr.DataArray(
            data=np.ones(
                (self.resolution["x"], self.resolution["y"], self.resolution["z"])
            ),
            dims=["x", "y", "z"],
            coords={
                "x": np.linspace(-self.r_max, self.r_max, self.resolution["x"]),
                "y": np.linspace(-self.r_max, self.r_max, self.resolution["y"]),
                "z": np.linspace(-self.r_max, self.r_max, self.resolution["z"]),
            },
            attrs={
                "resolution": self.resolution,
            },
        )

        self._normalize()

    def eval_density(self, n: int, l: int, m: int):
        xx, yy, zz = np.meshgrid(
            self.density.coords["x"],
            self.density.coords["y"],
            self.density.coords["z"],
        )

        rr, tt, pp = electron_functions.convert_cartesian_to_radial(xx, yy, zz)

        self.density.data = electron_functions.wavefunction(n, l, m, rr, tt, pp)

        self._normalize()
