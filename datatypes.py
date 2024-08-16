from __future__ import annotations

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

    def to_cartesian_coordinates(self) -> CartesianElectronDensity:
        
        # Compute cartesian coordinates
        x = self.density.r * np.sin(self.density.psi) * np.cos(self.density.phi)
        y = self.density.r * np.sin(self.density.psi) * np.sin(self.density.phi)
        z = self.density.r * np.cos(self.density.psi)

        # Create CartesianElectronDensity object
        return CartesianElectronDensity(
            resolution = self.resolution,
            density = xr.DataArray(
                data = np.zeroes(
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
        )


@attrs.define
class CartesianElectronDensity(ElectronDensityVolume):

    resolution: dict = {"x": 100, "y": 100, "z": 100}

    # Cartesian density with coords x, y, z
    density = xr.DataArray(
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

    def to_radial_coordinates(self) -> RadialElectronDensity:
            # Compute radial coordinates
            r = np.sqrt(self.cartesian_density.x**2 + self.cartesian_density.y**2 + self.cartesian_density.z**2)
            phi = np.arctan2(self.cartesian_density.y, self.cartesian_density.x)
            psi = np.arccos(self.cartesian_density.z / r)
    
            # Create RadialElectronDensity object
            return RadialElectronDensity(
                resolution = self.resolution,
                density = xr.DataArray(
                    data = np.zeroes(
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
            )

