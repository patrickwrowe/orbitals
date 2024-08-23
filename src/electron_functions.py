import numpy as np
import scipy
from src import definitions as d


@np.vectorize(excluded=["n", "l", "m"])
def wavefunction(n: int, l: int, m: int, r: float, theta: float, phi: float) -> float:
    """
    Returns the wavefunction for a given electron in a hydrogen atom in radial coordinates.
    For quantum numbers n, l, and m, and spherical coordinates r, phi, and theta.

    args:
    n: int, principal quantum number
    l: int, azimuthal quantum number
    m: int, magnetic quantum number

    r: float, radial coordinate
    phi: float, azimuthal angle
    theta: float, polar angle

    returns:
    float, wavefunction value
    """

    def _prefactor(n, l):
        return (
            (2 / (3 * n * d.A_0_STAR)) ** 3
            * (np.math.factorial(n - l - 1) / np.math.factorial(2 * n * (n + 1)))
        ) ** (1 / 2)

    def _rho_terms(n, l):
        rho = 2 * r / (n * d.A_0_STAR)

        return (
            np.exp(-rho / 2)
            * pow(rho, l)
            * scipy.special.genlaguerre(n - l - 1, 2 * l + 1)(rho)
        )

    def _spherical_harmonic_term(l, m, phi, theta):
        return scipy.special.sph_harm(m, l, phi, theta)

    return (
        _prefactor(n, l) * _rho_terms(n, l) * _spherical_harmonic_term(l, m, phi, theta)
    )


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


@np.vectorize
def get_clipped_density(density: np.ndarray, threshold: float):
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
