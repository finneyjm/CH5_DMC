import numpy as np


class EckartsSpinz:

    def __init__(self, reference, coords, masses, planar=None):
        self.reference = reference
        self.coords = coords
        self.masses = masses
        self.planar = planar
        self.little_fs = np.zeros((len(coords), 3, 3))
        self.biggo_fs = np.zeros((len(coords), 3, 3))

    def com_calc(self):
        com = np.dot(self.masses, self.coords)/np.sum(self.masses)
        self.coords = self.coords - com[:, None, :]
        ref_com = np.dot(self.masses, self.reference)/np.sum(self.masses)
        self.reference = self.reference - ref_com

    def create_f_vector_bois(self):
        if self.planar is None:
            mass_weight_ref = self.masses[:, None]*self.reference
            self.little_fs = np.matmul(np.transpose(self.coords, (0, 2, 1)), mass_weight_ref)
            self.biggo_fs = np.matmul(self.little_fs.transpose((0, 2, 1)), self.little_fs)

    def get_eigs(self):
        if self.planar is None:
            self.create_f_vector_bois()
            self._eigs, self._eigvs = np.linalg.eigh(self.biggo_fs)

    def get_transformed_fs(self):
        self.com_calc()
        self.get_eigs()
        if self.planar is None:
            eig_1o2 = 1/np.sqrt(self._eigs)[:, None, :]
            big_F_m1o2 = self._eigvs@(eig_1o2*self._eigvs.T)
            self._f_vecs = np.matmul(self.coords, big_F_m1o2)

    def get_rotated_coords(self):
        self.get_transformed_fs()
        return self.coords@self._f_vecs

    def return_f_vecs(self):
        self.get_transformed_fs()
        return self._f_vecs