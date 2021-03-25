from Eckart_turny_turn import EckartsSpinz
from PAF_spinz import MomentOfSpinz

import numpy as np

class ConstantSpinz:

    def __init__(self, reference, coords, masses, planar=None):
        self.reference = reference
        self.coords = coords
        self.masses = masses
        self.planar = planar

    def put_that_reference_back_where_you_found_it(self):
        MOM = MomentOfSpinz(self.reference, self.masses)
        self.reference = MOM.coord_spinz()


ref = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                          [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                          [1.786540362044548, -1.386051328559878, 0.000000000000000],
                          [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                          [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                          [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])

coords_initial = np.array([ref]*5) + np.random.normal(0, 0.002, (5, 6, 3))
MOM = MomentOfSpinz(ref, np.ones(6))
coords = MOM.coord_spinz()
print(coords)
MOM2 = MomentOfSpinz(coords, np.ones(6))
coords2 = MOM2.coord_spinz()
print(coords2)
MOM3 = MomentOfSpinz(coords2, np.ones(6))
coords3 = MOM3.coord_spinz()
print(coords3)

eck = EckartsSpinz(coords3, coords_initial, np.ones(6))
coord_new = eck.get_rotated_coords()
eck2 = EckartsSpinz(coords3, coord_new, np.ones(6))
coord_newer = eck2.get_rotated_coords()

print(coord_new)
print(coord_newer)
print(np.average(coord_new[0, 0, 0]-coord_newer[0, 0, 0]))
print(np.average(coord_new[0, 2, 2]-coord_newer[0, 2, 2]))
