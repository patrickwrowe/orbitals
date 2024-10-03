import skimage as ski

from orbitals import datatypes, tools


def extract_isosurface(
    wavefunction: datatypes.WavefunctionVolume, relative_threshold: float
):
    """
    Extracts the isosurface of the electron density at a given threshold value.

    args:
    wavefunction: datatypes.WavefunctionVolume, wavefunction volume
    threshold: float, threshold value

    returns:
    tuple, (vertices, faces) of the isosurface
    """

    abs_threshold = tools.abs_threshold_from_relative(
        wavefunction.get_wavefunction(), relative_threshold
    )

    verts, faces, normals, values = ski.measure.marching_cubes(
        volume=wavefunction.get_density(),
        level=abs_threshold,
    )

    return verts, faces, normals, values
