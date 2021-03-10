import numpy as np


class eckarts_spinz():

    def __init__(self, reference, coords, masses):
        self.reference = reference
        self.coords = coords
        self.masses = masses

    def com_calc(self):
        com = np.dot(self.masses, self.coords)/np.sum(self.masses)
        self.coords = self.coords - com[:, None, :]
        ref_com = np.dot(self.masses, self.reference)/np.sum(self.masses)
        self.reference = self.reference - ref_com
