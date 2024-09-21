import numpy as np
import xarray as xr
from src import datatypes
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Optional


@np.vectorize
def convert_radial_to_cartesian(
    r: float, theta: float, phi: float
) -> Tuple[float, float, float]:
    """
    Converts radial coordinates to cartesian coordinates.

    args:
    r: float, radial coordinate
    theta: float, polar angle
    phi: float, azimuthal angle

    returns:
    tuple, (x, y, z) cartesian coordinates
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return x, y, z


@np.vectorize
def convert_cartesian_to_radial(
    x: float, y: float, z: float
) -> Tuple[float, float, float]:
    """
    Converts cartesian coordinates to radial coordinates.

    args:
    x: float, x coordinate
    y: float, y coordinate
    z: float, z coordinate

    returns:
    tuple, (r, theta, phi) radial coordinates
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)

    return r, theta, phi


def clip_density(
    wavefunction: datatypes.WavefunctionVolume, threshold: float
) -> np.ndarray:
    """
    Returns the electron density clipped to a threshold value.

    args:
    threshold: float, threshold value

    returns:
    np.ndarray, clipped electron density
    """

    electron_density = wavefunction.get_density()

    dens_range = np.nanmax(electron_density) - np.nanmin(electron_density)
    abs_threshold = threshold * dens_range

    return np.where(electron_density < abs_threshold, np.nan, electron_density)


def interpolate_grid_function(
    grid_function: datatypes.WavefunctionVolume, new_resolution: dict
) -> datatypes.WavefunctionVolume:
    """
    Interpolates a grid function to a new resolution. We do this because actually calculating the wavefunction
    at a high resolution is computationally expensive, so we start with a low resolution and interpolate.

    args:
    grid_function: datatypes.WavefunctionVolume, grid function to interpolate
    new_resolution: dict, new resolution

    returns:
    datatypes.WavefunctionVolume, interpolated grid function
    """

    assert new_resolution.keys() == grid_function.resolution.keys()

    interp = RegularGridInterpolator(
        [grid_function.get_coords()[dim].values for dim in grid_function.get_dims()],
        grid_function.get_wavefunction(),
    )

    # e.g. RadialWavefunction or CartesianWavefunction
    interp_grid = type(grid_function)(
        resolution=new_resolution, r_max=grid_function.r_max
    )

    # e.g. xx, yy, zz or rr, tt, pp
    c1, c2, c3 = interp_grid.meshgrid_coords()

    interp_grid.wavefunction.data = interp((c1, c2, c3))

    return interp_grid


def abs_threshold_from_relative(
    wavefunction: datatypes.WavefunctionVolume, relative_threshold: float
) -> float:
    """
    Returns the absolute threshold value from a relative threshold value.

    args:
    wavefunction: datatypes.WavefunctionVolume, wavefunction
    relative_threshold: float, relative threshold value

    returns:
    float, absolute threshold value
    """

    electron_density = wavefunction.get_density()

    dens_range = np.nanmax(electron_density) - np.nanmin(electron_density)
    abs_threshold = relative_threshold * dens_range


    return abs_threshold


def validate_quantum_numbers(n: int, l: int, m: int, s: Optional[float]) -> bool:
    """
    args:
        n: int, principle quantum number
        l: int, azimuthal quantum number
        m: int, magnetic quantum number
    """

    # n must be a positive integer
    assert n > 0

    # l must be +ve int or 0, up to n-1
    assert n - 1 >= l >= 0

    # m can be +l to -l
    assert l >= m >= -l

    # spin is +/- 0.5
    if s:
        assert np.isclose(np.absolute(s), 0.5)

    return True
