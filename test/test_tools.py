import numpy as np
import xarray as xa
import pytest
from orbitals import tools, datatypes

@pytest.fixture
def simple_test_volume():
    test_volume = datatypes.WavefunctionVolume(
        resolution={'x': 3, 'y': 3, 'z': 3},
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

def test_validate_quantum_numbers():
    assert tools.validate_quantum_numbers(1, 0, 0) == True # 1s
    assert tools.validate_quantum_numbers(2, 0, 0) == True # 2s

    # 2p
    assert tools.validate_quantum_numbers(2, 1, 0) == True
    assert tools.validate_quantum_numbers(2, 1, 1) == True
    assert tools.validate_quantum_numbers(2, 1, -1) == True


    # 3d    
    assert tools.validate_quantum_numbers(3, 2, 2) == True
    assert tools.validate_quantum_numbers(3, 2, -2) == True

    # failures
    with pytest.raises(AssertionError):
        tools.validate_quantum_numbers(0, 0, 0) # n must be a positive integer
    
    with pytest.raises(AssertionError):
        tools.validate_quantum_numbers(1, 2, 0) # l must be +ve int or 0, up to n-1
    
    with pytest.raises(AssertionError):
        tools.validate_quantum_numbers(1, 0, 1) # m can be +l to -l

    with pytest.raises(AssertionError):
        tools.validate_quantum_numbers(1, 0, 0, 0.4) # spin is +/- 0.5
    
    with pytest.raises(AssertionError):
        tools.validate_quantum_numbers(1, 0, 0, 1) # spin is +/- 0.5

def test_abs_threshold_from_relative(simple_test_volume):
    assert tools.abs_threshold_from_relative(simple_test_volume.get_wavefunction(), 0.5) == 0.125
    assert tools.abs_threshold_from_relative(simple_test_volume.get_wavefunction(), 0.1) == 0.025

    with pytest.raises(ValueError):
        tools.abs_threshold_from_relative(simple_test_volume, 1.1)

    with pytest.raises(ValueError):
        tools.abs_threshold_from_relative(simple_test_volume, -0.1)

def test_interpolate_grid_function():
    pass

def test_clip_density():
    pass

def test_convert_cartesian_to_radial():
    pass

def test_convert_radial_to_cartesian():
    pass