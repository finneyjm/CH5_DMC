import numpy as np


class MomentOfSpinz:

    def __init__(self, coords, masses):
        self.coords = coords
        self.masses = masses

    def io_matrix_big(self):
        ri_sq = np.linalg.norm(self.coords, axis=-1)**2
        ri_sq_diag = np.broadcast_to(ri_sq[..., None, None], (ri_sq.shape[0], ri_sq.shape[1], 3, 3))*np.identity(3)
        Io_o_m = ri_sq_diag - self.coords[..., None]*self.coords[..., None, :]
        self._Io = np.tensordot(Io_o_m, self.masses, axes=(1, 0))

    def io_matrix_little(self):
        ri_sq = np.linalg.norm(self.coords, axis=-1)**2
        ri_sq_diag = np.broadcast_to(ri_sq[:, None, None], (ri_sq.shape[0], 3, 3))*np.identity(3)
        Io_o_m = ri_sq_diag - self.coords[..., None]*self.coords[:, None, :]
        self._Io = np.tensordot(Io_o_m, self.masses, axes=(0, 0))

    def eval_evec(self):
        if len(self.coords.shape) == 2:
            self.io_matrix_little()
        else:
            self.io_matrix_big()
        self._eval, self._evec = np.linalg.eigh(self._Io)

    def gimme_dat_trans_vec(self):
        self.eval_evec()
        return self._evec

    def gimme_dat_eigval(self):
        self.eval_evec()
        return self._eval

    def coord_spinz(self):
        self.eval_evec()
        return self.coords@self._evec
