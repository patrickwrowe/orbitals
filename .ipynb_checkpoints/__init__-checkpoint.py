import os, sys
from src import datatypes
from src import electron_functions

root_dir = os.getcwd()
sys.path.append(root_dir)

__name__ = "orbitals"

__all__ = [
    "datatypes",
    "electron_functions"
]

def eval_wavefunction(n, l, m, resolution):
    radialdensity = datatypes.RadialElectronDensity(resolution=resolution)

    radialdensity.density = electron_functions.wavefunction(n, l, m)(radialdensity.density.coords["r"], radialdensity.density.coords["phi"], radialdensity.density.coords["psi"])