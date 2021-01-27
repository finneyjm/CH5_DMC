import numpy as np
# import h3o2dip_77
from ProtWaterPES import *
from Coordinerds.CoordinateSystems import *


har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

struct = [
    [2.06095307, 0.05378083, 0.],
    [0., 0., 0.],
    [-0.32643038, -1.70972841, 0.52193868],
    [4.70153912, 0., 0.],
    [5.20071798, 0.80543847, 1.55595785]
]

# struct = [
#     [0.3057893, 0.0000006, -0.0732555],
#     [2.3561450, 0.0000000, 0.0000000],
#     [-2.9386764, 0.0000000, -1.7263630],
#     [2.3561450, 0.0000000, 0.0000000],
#     [2.6741510, 0.0000000, 1.7887807]
# ]

a = np.array([struct]*1)
a = a.reshape((1, 5, 3))
# a = np.transpose(a, (1, 2, 0))

# h3o2dip_77.init_param()
# print(h3o2dip_77.h3o2dip(np.transpose(a, (1, 2, 0)), 200).T)

class PotHolder:
    pot = None
    @classmethod
    def get_pot(cls, coords):
        if cls.pot is None:
            cls.pot = Potential(coords.shape[1])
        return cls.pot.get_potential(coords)
#
class DipHolder:
    dip = None
    @classmethod
    def get_dip(cls, coords):
        if cls.dip is None:
            cls.dip = Dipole(coords.shape[1])
        return cls.dip.get_dipole(coords)
#
#
get_pot = PotHolder.get_pot
get_dip = DipHolder.get_dip
#
print(get_pot(a)*har2wave)
# print(get_dip(a))

struct = np.array([
    [2.06730145, 0.1, 0.],
    [0., 0., 0.],
    [-0.38632011, -1.6897269,   0.54637778],
    [4.69798083, 0., 0.],
    [5.16812905, 0.81066508, 1.56325854]
])

def min_func(a):
    x = np.array([struct]*1)
    # x[0, 0, 0] = a[0]/2
    # x[0, 0, 1] = a[1]
    # x[0, 0, 2] = a[2]
    x[0, 3, 0] = a[0]
    coords = np.array(x)
    coords = coords[:, (1, 3, 0, 2, 4)]

    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0], [3, 0, 2, 1],
                                                                                   [4, 1, 2, 0]])).coords
    zmat[:, 2, 1] = a[1]
    zmat[:, 3, 1] = a[1]
    zmat[:, 1, 1] = a[0]/2
    zmat[:, 1, 3] = a[2]
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    x = new_coords[:, (2, 0, 3, 1, 4)]
    print(x)

    return get_pot(x)


a = [4.61678487, 1.81894038, 0.01648]
from scipy import optimize
pot1 = min_func(a)
print(pot1*har2wave)

result = optimize.minimize(min_func, a, method='Nelder-Mead')
while result.fun < pot1:
    print(result.fun*har2wave)
    pot1 = result.fun
    result = optimize.minimize(min_func, result.x, method='Nelder-Mead')
    pot1 = result.fun
    result = optimize.minimize(min_func, result.x, method='CG')
print(result.x)