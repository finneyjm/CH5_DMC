import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


def dists(coords):
    bonds = [[4, 1], [4, 2], [4, 3]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds)-1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds)-1)]
    dis = np.linalg.norm(cd2-cd1, axis=2)
    return dis


def psi_t_extra(coords, interp_reg_oh, reg_oh=None):
    if reg_oh is None:
        reg_oh = dists(coords)

    psi = np.zeros((len(coords), int(3)))

    for i in range(3):
        psi[:, i] = interpolate.splev(reg_oh[:, i], interp_reg_oh, der=0)
    return psi


def get_da_psi(coords, interp_reg_oh, dx=1e-3):
    reg_oh = dists(coords)

    much_psi = np.zeros((len(coords), 3, 4, 3))
    psi = psi_t_extra(coords, interp_reg_oh, reg_oh)
    much_psi[:, 1] += np.broadcast_to(psi[:, -1][:, None, None], (len(coords), 4, 3))
    for atom_label in range(4):
        for xyz in range(3):
            coords[:, atom_label, xyz] -= dx
            much_psi[:, 0, atom_label, xyz] = psi_t_extra(coords, interp_reg_oh)[:, -1]
            coords[:, atom_label, xyz] += 2.*dx
            much_psi[:, 2, atom_label, xyz] = psi_t_extra(coords, interp_reg_oh)[:, -1]
            coords[:, atom_label, xyz] -= dx
    return much_psi



wvfn = np.load('../lets_go_girls/jobs/Prot_water_params/wvfns/hydronium_oh_wvfn.npy')
init_coords = np.load('../lets_go_girls/jobs/Prot_water_params/monomer_coords.npy')
free_oh_wvfn = interpolate.splrep(wvfn[:, 0], wvfn[:, 1], s=0)

# coords = np.array([init_coords]*1000)
from Coordinerds.CoordinateSystems import *
order_h = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2]]


def grid_dis(a, b, num, coords, order):
    spacing = np.linspace(a, b, num)
    zmat = CoordinateSet(np.flip(coords, axis=0), system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    g = np.array([zmat]*num)
    g[:, 0, 1] = spacing
    new_coords = CoordinateSet(g, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    return np.flip(new_coords, axis=1)

g = grid_dis(0.8, 2.5, 1000, init_coords, order_h)
psi = get_da_psi(g, free_oh_wvfn)
dx = 1e-3
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_O = 15.99491461957 / (Avo_num*me*1000)
dtau = 1
sigmaH = np.sqrt(dtau / m_H)
sigmaO = np.sqrt(dtau / m_O)
atoms = ['H', 'H', 'H', 'O']
sigmaOH = np.zeros((4, 3))
for i in range(len(atoms)):
    if atoms[i].upper() == 'H':
        sigmaOH[i] = np.array([[sigmaH] * 3])
    elif atoms[i].upper() == 'O':
        sigmaOH[i] = np.array([[sigmaO] * 3])
d2psidx2 = ((psi[:, 0] - 2. * psi[:, 1] + psi[:, 2]) / dx ** 2) / psi[:, 1]
kin = -1. / 2. * np.sum(np.sum(sigmaOH ** 2 / 1 * d2psidx2, axis=1), axis=1)

def potential_hydronium(grid):
    from ProtWaterPES import Potential
    pot = Potential(4)
    V = np.diag(pot.get_potential(grid))
    return V

har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
pot = np.diag(potential_hydronium(g))
plt.plot(np.linspace(1.2, 2, 1000)/ang2bohr, (pot+kin)*har2wave)
plt.xlabel(r'r$_{\rm{OH}}$')
plt.ylabel(r'Energy (cm$^{-1}$')
plt.show()
