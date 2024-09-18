import enum

# Reduced Bohr Radius
A_0_STAR = 5.29177210544e-1  # Angstrom

class CartesianCoords(enum.StrEnum):
    X = enum.auto()
    Y = enum.auto()
    Z = enum.auto()

class RadialCoords(enum.StrEnum):
    R = enum.auto()
    THETA = enum.auto()
    PHI = enum.auto()
