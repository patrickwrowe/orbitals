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
        return ((2 / ( 3 * n * d.A_0_STAR))**3 * (np.math.factorial(n - l - 1)/np.math.factorial(2*n*(n + 1))))**(1/2)

    def _rho_terms(n, l):
        rho = 2 * r / (n * d.A_0_STAR)

        return np.exp(-rho/2) * pow(rho, l) * scipy.special.genlaguerre(n - l - 1, 2*l + 1)(rho)
    
    def _spherical_harmonic_term(l, m, phi, theta):
        return scipy.special.sph_harm(m, l, phi, theta)
    
    return _prefactor(n, l) * _rho_terms(n, l) * _spherical_harmonic_term(l, m, phi, theta)

