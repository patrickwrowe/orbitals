import numpy as np
import xarray as xr
from src import datatypes
from scipy.interpolate import RegularGridInterpolator

@np.vectorize
def convert_radial_to_cartesian(r, theta, phi):
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
def convert_cartesian_to_radial(x, y, z):
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


def clip_density(wavefunction: datatypes.WavefunctionVolume, threshold: float):
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

def interpolate_grid_function(grid_function: datatypes.WavefunctionVolume, new_resolution: dict):
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
            [grid_function.density.coords[dim].values for dim in grid_function.get_dims()],
            grid_function.density.data,
        )

    # e.g. RadialWavefunction or CartesianWavefunction
    interp_grid = type(grid_function)(resolution=new_resolution, r_max=grid_function.r_max)

    # e.g. xx, yy, zz or rr, tt, pp
    c1, c2, c3 = interp_grid.meshgrid_coords()

    interp_grid.density.data = interp((c1, c2, c3))

    return interp_grid