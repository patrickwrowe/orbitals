import skimage as ski
import matplotlib.pyplot as plt

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
        wavefunction.get_density(), relative_threshold
    )

    verts, faces, normals, values = ski.measure.marching_cubes(
        volume=wavefunction.get_density(),
        level=abs_threshold,
    )

    return verts, faces, normals, values


def plot_clipped_points(wavefunction: datatypes.WavefunctionVolume, threshold: float):
    """
    Plots the points of the wavefunction volume clipped to a threshold value.

    args:
    wavefunction: datatypes.WavefunctionVolume, wavefunction volume
    threshold: float, threshold value

    returns:
    matplotlib figure and axis
    """
    
    clipped_density = tools.clip_density(wavefunction, threshold)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    xx, yy, zz = wavefunction.meshgrid_coords()
    ax.scatter3D(xs=xx, ys=yy, zs=zz, c=clipped_density)

    plt.show()

    return fig, ax
