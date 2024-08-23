import numpy as np
import xarray as xr

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

def clip_density(density: xr.DataArray, threshold: float):
    """
    Returns the electron density clipped to a threshold value.
    
    args:
    threshold: float, threshold value
    
    returns:
    np.ndarray, clipped electron density
    """

    electron_density = np.absolute(density.data) ** 2

    dens_range = np.nanmax(electron_density) - np.nanmin(electron_density)
    abs_threshold = threshold * dens_range

    return np.where(electron_density < abs_threshold, np.nan, electron_density)