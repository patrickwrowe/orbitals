from orbitals import analysis

def test_extract_isosurface(simple_radial_wavefunction):
    verts, faces, normals, values = analysis.extract_isosurface(simple_radial_wavefunction, relative_threshold=0.5)

    