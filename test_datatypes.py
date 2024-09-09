from src import datatypes
import pytest

def test_RadialWavefunction():
    resolution = {"r": 100, "phi": 100, "psi": 100}
    density = datatypes.RadialWavefunction(resolution=resolution)
    assert density.resolution == resolution
    assert density.density.shape == (100, 100, 100)
    assert density.density.dims == ("r", "phi", "psi")
    assert density.density.coords["r"].shape == 100
    assert density.density.coords["phi"].shape == 100
    assert density.density.coords["psi"].shape == 100
    assert density.density.attrs["resolution"] == resolution

    assert sum(density.data) == 1

    clipped_density = density.get_clipped_density(0.5)
    assert clipped_density.shape == (100, 100, 100)

    meshgrid = density.meshgrid_coords()
    assert len(meshgrid) == 3
    
    # xx, yy, zz
    assert meshgrid[0].shape == (100, 100, 100)
    assert meshgrid[1].shape == (100, 100, 100)
    assert meshgrid[2].shape == (100, 100, 100)


def test_CartesianWavefunction():
    resolution = {"x": 100, "y": 100, "z": 100}
    density = datatypes.CartesianWavefunction(resolution=resolution)
    assert density.resolution == resolution
    assert density.density.shape == (100, 100, 100)
    assert density.density.dims == ("x", "y", "z")
    assert density.density.coords["x"].shape == 100
    assert density.density.coords["y"].shape == 100
    assert density.density.coords["z"].shape == 100
    assert density.density.attrs["resolution"] == resolution

    assert sum(density.data) == 1

    clipped_density = density.get_clipped_density(0.5)
    assert clipped_density.shape == (100, 100, 100)

    meshgrid = density.meshgrid_coords()
    assert len(meshgrid) == 3

    # xx, yy, zz
    assert meshgrid[0].shape == (100, 100, 100)
    assert meshgrid[1].shape == (100, 100, 100)
    assert meshgrid[2].shape == (100, 100, 100)

