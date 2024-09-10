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
        self.wavefunction.data = self.wavefunction.data / np.nansum(self.wavefunction.data)

    def meshgrid_coords(self):
        return np.meshgrid(
            self.wavefunction.coords["x"],
            self.wavefunction.coords["y"],
            self.wavefunction.coords["z"],
        )
    
    def get_density(self):
        return np.absolute(self.wavefunction.data) ** 2
    
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

    def eval_wavefunction(self, n: int, l: int, m: int):
        rr, tt, pp = np.meshgrid(
            self.wavefunction.coords["r"],
            self.wavefunction.coords["theta"],
            self.wavefunction.coords["phi"],
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

    def eval_wavefunction(self, n: int, l: int, m: int):
        xx, yy, zz = np.meshgrid(
            self.wavefunction.coords["x"],
            self.wavefunction.coords["y"],
            self.wavefunction.coords["z"],
        )

        rr, tt, pp = electron_functions.convert_cartesian_to_radial(xx, yy, zz)

        self.wavefunction.data = electron_functions.wavefunction(n, l, m, rr, tt, pp)

        self._normalize()
