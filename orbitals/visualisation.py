import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from orbitals import tools


from orbitals import datatypes, analysis, definitions

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

    return fig, ax
