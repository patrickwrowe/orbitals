from orbitals import datatypes
import pytest

def test_RadialElectronDensity():
    resolution = {"r": 100, "phi": 100, "psi": 100}
    density = datatypes.RadialElectronDensity(resolution=resolution)
    assert density.resolution == resolution
    assert density.density.shape == (100, 100, 100)
    assert density.density.dims == ("r", "phi", "psi")
    assert density.density.coords["r"].shape == 100
    assert density.density.coords["phi"].shape == 100
    assert density.density.coords["psi"].shape == 100
    assert density.density.attrs["resolution"] == resolution

def test_CartesianElectronDensity():
    resolution = {"x": 100, "y": 100, "z": 100}
    density = datatypes.CartesianElectronDensity(resolution=resolution)
    assert density.resolution == resolution
    assert density.density.shape == (100, 100, 100)
    assert density.density.dims == ("x", "y", "z")
    assert density.density.coords["x"].shape == 100
    assert density.density.coords["y"].shape == 100
    assert density.density.coords["z"].shape == 100
    assert density.density.attrs["resolution"] == resolution
