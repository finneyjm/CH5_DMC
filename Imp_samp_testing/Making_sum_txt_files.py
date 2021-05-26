import numpy as np
from Coordinerds.CoordinateSystems import *

monomer = [
           [0.00000012, 0.00000004, 0.04523643],
           [0.93273984, -0.03166913, -0.23931206],
           [-0.43894472, 0.82361013, -0.23931163],
           [-0.49379696, -0.79194164, -0.23931138],
           ]
print(np.array(monomer))
zmat1 = CoordinateSet(np.array(monomer), system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                                ordering=([[0, 0, 0, 0],
                                                                                [1, 0, 0, 0], [2, 0, 1, 0],
                                                                                           [3, 0, 1, 2]])).coords
print(zmat1)
print(np.rad2deg(zmat1[1:2, 3]))
print(np.rad2deg(zmat1[2, -1]))

zmat = [[1, 0.97569163, 1, 0, 1, 0], [1, 0.97569163, 2, np.deg2rad(120), 1, 0],
        [1, 0.97569163, 2, np.deg2rad(120), 3, np.deg2rad(180
                                                          )]]

coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
print(coords)