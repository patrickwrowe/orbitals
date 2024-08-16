import xarray as xr
import numpy as np
import attrs

@attrs.define
class ElectronDensityVolume:
    resolution: dict
    density: xr.DataArray

    def _normalize(self):
        self.density = self.density / self.density.sum()

@attrs.define
class RadialElectronDensity(ElectronDensityVolume):

    resolution: dict = {"r": 100, "phi": 200, "psi": 100}

    # Radial density with coords r, phi, psi
    density = xr.DataArray(
        data = np.zeroes(
            (resolution["r"], resolution["phi"], resolution["psi"])
        ),
        dims = ["r", "phi", "psi"],
        coords={
            "r": np.linspace(0, 1, resolution["r"]),
            "phi": np.linspace(0, 2*np.pi, resolution["phi"]),
            "psi": np.linspace(0, np.pi, resolution["psi"])
        },
        attrs = {
            "resolution": resolution,
        }
    )


@attrs.define
class CartesianElectronDensity (ElectronDensityVolume):

    resolution: dict = {"x": 100, "y": 100, "z": 100}

    # Cartesian density with coords x, y, z
    cartesian_density = xr.DataArray(
        data = np.zeroes(
            (resolution["x"], resolution["y"], resolution["z"])
        ),
        dims = ["x", "y", "z"],
        coords={
            "x": np.linspace(0, 1, resolution["x"]),
            "y": np.linspace(0, 1, resolution["y"]),
            "z": np.linspace(0, 1, resolution["z"])
        },
        attrs = {
            "resolution": resolution,
        }
    )

