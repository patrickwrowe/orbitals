import pytest

import sys

sys.path.append("..")  # Adjust the path to import from the parent directory

from orbitals import datatypes
from orbitals.definitions import CartesianCoords, RadialCoords

import xarray as xa
import numpy as np

@pytest.fixture
def simple_test_volume():
    test_volume = datatypes.WavefunctionVolume(
        wavefunction=xa.DataArray(np.array(
            [
                [0.05, 0.2, 0.05],
                [0.2, 0.05, 0.05],
                [0.3, 0.05, 0.05]
            ]
        )
    ),
        resolution={
            CartesianCoords.X: 3, 
            CartesianCoords.Y: 3, 
            CartesianCoords.Z: 1
        },
        r_max=1
    )

    test_volume.wavefunction = xa.DataArray(np.array(
            [
                [0.05, 0.2, 0.05],
                [0.2, 0.05, 0.05],
                [0.3, 0.05, 0.05]
            ]
        )
    )

    return test_volume

@pytest.fixture()
def simple_radial_wavefunction():
    resolution = {
        RadialCoords.R: 10, 
        RadialCoords.THETA: 10, 
        RadialCoords.PHI: 10
    }
    
    density = datatypes.RadialWavefunction.new_1e_atomic_wavefunction(
        resolution=resolution,
        r_max=1,
        n=1,
        l=0,
        m=0
    )

    density.eval_wavefunction()
    return density