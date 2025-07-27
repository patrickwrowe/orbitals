import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from orbitals import tools
import numpy as np

from orbitals import datatypes, analysis, definitions

def plot_clipped_points(wavefunction: datatypes.WavefunctionVolume, threshold: float, alpha: float = None):
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

    # Delete nan values from the clipped density for better visualisation
    # And performance... nan values are still points!?
    mask = ~np.isnan(clipped_density)
    clipped_density = clipped_density[mask]
    xx = xx[mask]
    yy = yy[mask]
    zz = zz[mask]

    ax.scatter3D(xs=xx, ys=yy, zs=zz, c=clipped_density, alpha=alpha)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_zlim(zz.min(), zz.max())

    plt.tight_layout()
    ax.set_title("Clipped Density Points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    return fig, ax

def plot_isosurface(
    wavefunction: datatypes.WavefunctionVolume, relative_threshold: float
):

    verts, faces, normals, values = analysis.extract_isosurface(
        wavefunction=wavefunction, relative_threshold=relative_threshold
    )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor("k")
    ax.add_collection3d(mesh)  # pyright: ignore

    ax.set_xlim(verts.min(), verts.max())
    ax.set_ylim(verts.min(), verts.max())
    ax.set_zlim(verts.min(), verts.max())  # pyright: ignore

    plt.tight_layout()

    # Turn off axes/background entirely
    ax.set_axis_off()

    return fig, ax
