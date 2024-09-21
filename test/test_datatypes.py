from scipy.sparse import data
import numpy as np
import pytest

from orbitals import datatypes, tools

def test_RadialWavefunction():
    resolution = {"r": 100, "theta": 100, "phi": 100}
    density = datatypes.RadialWavefunction(resolution=resolution)
    assert density.resolution == resolution
    assert density.wavefunction.shape == (100, 100, 100)
    assert density.wavefunction.dims == ("r", "theta", "phi")
    assert density.wavefunction.coords["r"].shape == (100,)
    assert density.wavefunction.coords["theta"].shape == (100,)
    assert density.wavefunction.coords["phi"].shape == (100,)
    assert density.wavefunction.attrs["resolution"] == resolution

    assert np.isclose(np.sum(density.get_density()), 1.0)

    clipped_density = tools.clip_density(density, 0.5)
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
    assert density.wavefunction.shape == (100, 100, 100)
    assert density.wavefunction.dims == ("x", "y", "z")
    assert density.wavefunction.coords["x"].shape == (100,)
    assert density.wavefunction.coords["y"].shape == (100,)
    assert density.wavefunction.coords["z"].shape == (100,)
    assert density.wavefunction.attrs["resolution"] == resolution

    assert np.isclose(np.sum(density.get_density()), 1.0)

    clipped_density = tools.clip_density(density, 0.5)
    assert clipped_density.shape == (100, 100, 100)

    meshgrid = density.meshgrid_coords()
    assert len(meshgrid) == 3

    # xx, yy, zz
    assert meshgrid[0].shape == (100, 100, 100)
    assert meshgrid[1].shape == (100, 100, 100)
    assert meshgrid[2].shape == (100, 100, 100)

