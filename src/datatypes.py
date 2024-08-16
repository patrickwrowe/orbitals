from __future__ import annotations

import xarray as xr
import numpy as np
import attrs

@attrs.define
class ElectronDensityVolume:
    resolution: dict

    def _normalize(self):
        self.density.data = self.density.data / self.density.data.sum()

@attrs.define
class RadialElectronDensity(ElectronDensityVolume):

    # Radial density with coords r, phi, psi
    resolution: dict 

    density = attrs.field(init=False)

    def __attrs_post_init__(self):
        # Radial density with coords r, phi, psi
        self.density = xr.DataArray(
            data = np.ones(
                (self.resolution["r"], self.resolution["phi"], self.resolution["psi"])
            ),
            dims = ["r", "phi", "psi"],
            coords={
                "r": np.linspace(0, 1, self.resolution["r"]),
                "phi": np.linspace(0, 2*np.pi, self.resolution["phi"]),
                "psi": np.linspace(0, np.pi, self.resolution["psi"])
            },
            attrs = {
                "resolution": self.resolution,
            }
        )
        
        self._normalize()

        




@attrs.define
class CartesianElectronDensity(ElectronDensityVolume):

    # Cartesian density with coords x, y, z
    resolution: dict

    # Cartesian density with coords x, y, z
    density = attrs.field(init=False)

    def __attrs_post_init__(self):
        self._normalize()

        self.density = xr.DataArray(
        data = np.zeros(
            (self.resolution["x"], self.resolution["y"], self.resolution["z"])
        ),
        dims = ["x", "y", "z"],
        coords={
            "x": np.linspace(0, 1, self.resolution["x"]),
            "y": np.linspace(0, 1, self.resolution["y"]),
            "z": np.linspace(0, 1, self.resolution["z"])
        },
        attrs = {
            "resolution": self.resolution,
        }
    )