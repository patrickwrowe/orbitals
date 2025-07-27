import numpy as np
from scipy.sparse import data
import xarray as xa
import pytest
from orbitals import tools, datatypes
from orbitals.definitions import CartesianCoords, RadialCoords

def test_validate_quantum_numbers():
    # 1s/2s
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
        

def test_interpolate_grid_function(simple_radial_wavefunction):
    new_volume = tools.interpolate_grid_function(
                simple_radial_wavefunction, 
                new_resolution={RadialCoords.R: 15, 
                                RadialCoords.THETA: 15, 
                                RadialCoords.PHI: 15})

    # Check that the coordinates have the correct values
    coords = new_volume.get_coords()
    
    # Check r coordinate (0 to r_max=1)
    expected_r = np.linspace(0, 1, 15)
    assert np.allclose(coords[RadialCoords.R].values, expected_r)
    
    # Check theta coordinate (0 to 2π)
    expected_theta = np.linspace(0, 2 * np.pi, 15)
    assert np.allclose(coords[RadialCoords.THETA].values, expected_theta)
    
    # Check phi coordinate (0 to π)
    expected_phi = np.linspace(0, np.pi, 15)
    assert np.allclose(coords[RadialCoords.PHI].values, expected_phi)


def test_clip_density(simple_radial_wavefunction):
    # Test with a threshold of 0.5 (50% of density range)
    clipped_density = tools.clip_density(simple_radial_wavefunction, 0.5)
    
    # Check that the result is a numpy array
    assert isinstance(clipped_density, np.ndarray)
    
    # Check that values below threshold are NaN
    original_density = simple_radial_wavefunction.get_density()
    density_range = np.nanmax(original_density) - np.nanmin(original_density)
    abs_threshold = 0.5 * density_range
    
    # Values below threshold should be NaN
    low_values = original_density < abs_threshold
    assert np.all(np.isnan(clipped_density[low_values]))
    
    # Values above threshold should remain unchanged
    high_values = original_density >= abs_threshold
    assert np.allclose(clipped_density[high_values], original_density[high_values], equal_nan=True)    

def test_convert_cartesian_to_radial():
    # Test known conversions
    # Point (1, 0, 0) -> (r=1, theta=0, phi=π/2)
    r, theta, phi = tools.convert_cartesian_to_radial(1.0, 0.0, 0.0)
    assert np.isclose(r, 1.0)
    assert np.isclose(theta, 0.0)
    assert np.isclose(phi, np.pi/2)
    
    # Point (0, 1, 0) -> (r=1, theta=π/2, phi=π/2)
    r, theta, phi = tools.convert_cartesian_to_radial(0.0, 1.0, 0.0)
    assert np.isclose(r, 1.0)
    assert np.isclose(theta, np.pi/2)
    assert np.isclose(phi, np.pi/2)
    
    # Point (0, 0, 1) -> (r=1, theta=0, phi=0)
    r, theta, phi = tools.convert_cartesian_to_radial(0.0, 0.0, 1.0)
    assert np.isclose(r, 1.0)
    # theta can be any value when z is along the axis, but phi should be 0
    assert np.isclose(phi, 0.0)
    
    # Test origin
    r, theta, phi = tools.convert_cartesian_to_radial(0.0, 0.0, 0.0)
    assert np.isclose(r, 0.0)
    
    # Test with arrays
    x_arr = np.array([1.0, 0.0, 0.0])
    y_arr = np.array([0.0, 1.0, 0.0])
    z_arr = np.array([0.0, 0.0, 1.0])
    r_arr, theta_arr, phi_arr = tools.convert_cartesian_to_radial(x_arr, y_arr, z_arr)
    
    assert np.allclose(r_arr, [1.0, 1.0, 1.0])
    assert np.allclose(theta_arr, [0.0, np.pi/2, 0.0], atol=1e-10)
    assert np.allclose(phi_arr, [np.pi/2, np.pi/2, 0.0])

def test_convert_radial_to_cartesian():
    # Test known conversions
    # (r=1, theta=0, phi=π/2) -> (1, 0, 0)
    x, y, z = tools.convert_radial_to_cartesian(1.0, 0.0, np.pi/2)
    assert np.isclose(x, 1.0)
    assert np.isclose(y, 0.0)
    assert np.isclose(z, 0.0)
    
    # (r=1, theta=π/2, phi=π/2) -> (0, 1, 0)
    x, y, z = tools.convert_radial_to_cartesian(1.0, np.pi/2, np.pi/2)
    assert np.isclose(x, 0.0, atol=1e-10)
    assert np.isclose(y, 1.0)
    assert np.isclose(z, 0.0, atol=1e-10)
    
    # (r=1, theta=0, phi=0) -> (0, 0, 1)
    x, y, z = tools.convert_radial_to_cartesian(1.0, 0.0, 0.0)
    assert np.isclose(x, 0.0)
    assert np.isclose(y, 0.0)
    assert np.isclose(z, 1.0)
    
    # Test origin
    x, y, z = tools.convert_radial_to_cartesian(0.0, 0.0, 0.0)
    assert np.isclose(x, 0.0)
    assert np.isclose(y, 0.0)
    assert np.isclose(z, 0.0)
    
    # Test with arrays
    r_arr = np.array([1.0, 1.0, 1.0])
    theta_arr = np.array([0.0, np.pi/2, 0.0])
    phi_arr = np.array([np.pi/2, np.pi/2, 0.0])
    x_arr, y_arr, z_arr = tools.convert_radial_to_cartesian(r_arr, theta_arr, phi_arr)
    
    expected_x = np.array([1.0, 0.0, 0.0])
    expected_y = np.array([0.0, 1.0, 0.0])
    expected_z = np.array([0.0, 0.0, 1.0])
    
    assert np.allclose(x_arr, expected_x, atol=1e-10)
    assert np.allclose(y_arr, expected_y, atol=1e-10)
    assert np.allclose(z_arr, expected_z, atol=1e-10)
    
    # Test round-trip conversion
    original_x, original_y, original_z = 1.5, 2.3, 0.8
    r, theta, phi = tools.convert_cartesian_to_radial(original_x, original_y, original_z)
    x, y, z = tools.convert_radial_to_cartesian(r, theta, phi)
    
    assert np.isclose(x, original_x)
    assert np.isclose(y, original_y)
    assert np.isclose(z, original_z)