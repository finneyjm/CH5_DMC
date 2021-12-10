import numpy as np
from ProtWaterPES import *
from Coordinerds.CoordinateSystems import *
import multiprocessing as mp

oxy_pos = 4.70153912
new_pos = oxy_pos

struct = np.array([
    [2.06095307, 0.05378083, 0.],
    [0., 0., 0.],
    [-0.32643038, -1.70972841, 0.52193868],
    [new_pos, 0., 0.],
    [5.20071798-oxy_pos+new_pos, 0.80543847, 1.55595785]
])

har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_red = (m_O*m_H)/(m_O+m_H)
m_red_D = (m_O*m_D)/(m_O+m_D)
m_red_sp = 1/(1/m_H + 1/(2*m_O + 2*m_H))
m_red_OO = (m_O**2)/(2*m_O)
omega = 3600./har2wave

new_struct = np.array([
    [2.30803545e+00, -3.02071334e-03, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [-4.65570340e-01, 1.67058080e+00, -5.46666468e-01],
    [4.61607485e+00, 0.00000000e+00, 0.00000000e+00],
    [5.12936209e+00, -8.18802009e-01, -1.54030505e+00]
])

new_struct = np.array([
    [0.000000000000000, 0.000000000000000, 0.000000000000000],
    [-2.304566686034061, 0.000000000000001, 0.000000000000000],
    [-2.740400260927908, 1.0814221449986587E-016, -1.766154718409233],
    [2.304566686034061, 0.000000000000001, 0.000000000000000],
    [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233]
])
new_struct[:, 0] = new_struct[:, 0] + 2.304566686034061


class PotHolder:
    pot = None
    @classmethod
    def get_pot(cls, coords):
        if cls.pot is None:
            cls.pot = Potential(coords.shape[1])
        return cls.pot.get_potential(coords)


get_pot = PotHolder.get_pot


def asym_grid(coords, r1, a):
    coords = np.array([coords]*1)
    coords = coords[:, (1, 3, 0, 2, 4)]
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0], [3, 0, 1, 2],
                                                                                   [4, 1, 0, 2]])).coords
    N = len(r1)
    zmat = np.array([zmat]*N).reshape((N, 4, 6))
    zmat[:, 2, 1] = r1
    zmat[:, 3, 1] = r1 - a
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    coords = new_coords[:, (2, 0, 3, 1, 4)]
    return coords


def all_dists(coords):
    bonds = [[1, 2],  [3, 4], [1, 3], [1, 0]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    a_oh = 1/np.sqrt(2)*(dis[:, 0]-dis[:, 1])
    s_oh = 1/np.sqrt(2)*(dis[:, 0]+dis[:, 1])
    mid = dis[:, 2]/2
    sp = mid - dis[:, -1]*np.cos(roh_roo_angle(coords, dis[:, -2], dis[:, -1]))
    return np.vstack((a_oh, dis[:, 0], dis[:, 1], s_oh, dis[:, -2], sp)).T


def roh_roo_angle(coords, roo_dist, roh_dist):
    v1 = (coords[:, 1]-coords[:, 3])/np.broadcast_to(roo_dist[:, None], (len(roo_dist), 3))
    v2 = (coords[:, 1]-coords[:, 0])/np.broadcast_to(roh_dist[:, None], (len(roh_dist), 3))
    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    return aang


def shared_prot_grid(coords, sp):
    mid = (coords[:, 3, 0] - coords[:, 1, 0])/2
    coords[:, 0, 0] = mid+sp
    return coords


def linear_combo_grid(coords, grid1, grid2):
    # re_sp = np.linalg.norm(coords[0, 0]-coords[0, 1])
    re_a = np.linalg.norm(coords[0, 2]-coords[0, 1])
    coords = coords[:, (1, 3, 0, 2, 4)] + 1e-14
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0], [3, 0, 1, 2],
                                                                                   [4, 1, 0, 2]])).coords
    zmat[:, 2, 1] = re_a + np.sqrt(2) / 2 * grid2
    zmat[:, 3, 1] = re_a - np.sqrt(2) / 2 * grid2
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    coords = new_coords[:, (2, 0, 3, 1, 4)]
    coords = shared_prot_grid(coords, grid1)
    coords[:, :, 1] = coords[:, :, 2]
    coords[:, :, 2] = np.zeros(coords[:, :, 2].shape)
    return coords


def oo_grid(coords, Roo):
    coords = np.array([coords] * len(Roo))
    equil_roo_roh_x = coords[0, 3, 0] - coords[0, 4, 0]
    coords[:, 3, 0] = Roo
    coords[:, 4, 0] = Roo - equil_roo_roh_x
    return coords


class DipHolder:
    dip = None
    @classmethod
    def get_dip(cls, coords):
        if cls.dip is None:
            cls.dip = Dipole(coords.shape[1])
        return cls.dip.get_dipole(coords)


get_dip = DipHolder.get_dip


def dip(coords):
    coords = np.array_split(coords, mp.cpu_count()-1)
    V = pool.map(get_dip, coords)
    dips = np.concatenate(V)
    return dips


pool = mp.Pool(mp.cpu_count()-1)


def pot(coords, grid1, grid2):
    print('started making our grid')
    mesh = np.array(np.meshgrid(grid1, grid2, indexing='ij'))
    gridz = np.reshape(mesh, (2, len(grid1)*len(grid2)))
    roo_coords = oo_grid(coords, gridz[1])
    full_coords = shared_prot_grid(roo_coords, gridz[0])
    print('finished making the grid, now to start the potential')
    mid = (full_coords[:, 3, 0] - full_coords[:, 1, 0])/2
    full_coords[:, :, 0] -= mid[:, None]
    pot = get_pot(full_coords)
    print('finished evaluating the potential')
    import scipy.sparse as sp
    return sp.diags([pot], [0]), pot.reshape((len(grid1), len(grid2)))


def pot2(coords, grid1, grid2):
    print('started making our grid')
    mesh = np.array(np.meshgrid(grid1, grid2, indexing='ij'))
    gridz = np.reshape(mesh, (2, len(grid1)*len(grid2)))
    roo_coords = oo_grid(coords, gridz[1])
    A = np.array([[42.200232187251913, -0.60594644269321474], [1.0206303697659393, 41.561937672470521]])
    fancy_grid = gridz[0]
    eh = np.matmul(np.linalg.inv(A), np.vstack((np.zeros(len(fancy_grid)), fancy_grid)))
    grid_sp = eh[1]
    grid_a = eh[0]
    full_coords = linear_combo_grid(roo_coords, grid_sp, grid_a)
    print('finished making the grid, now to start the potential')
    mid = (full_coords[:, 3, 0] - full_coords[:, 1, 0])/2
    full_coords[:, :, 0] -= mid[:, None]
    pot = get_pot(full_coords)
    np.save('coords_for_testing', full_coords)
    # pot[pot > 12000/har2wave] = 12000/har2wave
    print('finished evaluating the potential')
    import scipy.sparse as sp
    return sp.diags([pot], [0]), pot.reshape((len(grid1), len(grid2)))


def getting_coords(coords, grid1, grid2):
    mesh = np.array(np.meshgrid(grid1, grid2, indexing='ij'))
    gridz = np.reshape(mesh, (2, len(grid1) * len(grid2)))
    roo_coords = oo_grid(coords, gridz[1])
    full_coords = shared_prot_grid(roo_coords, gridz[0])
    mid = (full_coords[:, 3, 0] - full_coords[:, 1, 0]) / 2
    full_coords[:, :, 0] -= mid[:, None]
    return full_coords


def getting_coords2(coords, grid1, grid2):
    mesh = np.array(np.meshgrid(grid1, grid2, indexing='ij'))
    gridz = np.reshape(mesh, (2, len(grid1) * len(grid2)))
    roo_coords = oo_grid(coords, gridz[1])
    A = np.array([[42.200232187251913, -0.60594644269321474], [1.0206303697659393, 41.561937672470521]])
    fancy_grid = gridz[0]
    eh = np.matmul(np.linalg.inv(A), np.vstack((np.zeros(len(fancy_grid)), fancy_grid)))
    grid_sp = eh[1]
    grid_a = eh[0]
    full_coords = linear_combo_grid(roo_coords, grid_sp, grid_a)
    mid = (full_coords[:, 3, 0] - full_coords[:, 1, 0]) / 2
    full_coords[:, :, 0] -= mid[:, None]
    return full_coords


def HO_pots(mass, grid1, grid2):
    mesh = np.array(np.meshgrid(grid1, grid2))
    gridz = np.reshape(mesh, (2, len(grid1) * len(grid2)))
    pot = 1/2*mass*omega**2*gridz[0]**2 + 1/2*mass*omega**2*gridz[1]**2
    coupling = 1/2*mass*(550/har2wave)**2*(gridz[0]*gridz[1])
    pot = pot + coupling
    import scipy.sparse as sp
    return sp.diags([pot], [0]), pot.reshape((len(grid1), len(grid2)))


def Kinetic_Calc(grid1, grid2, red_m1, red_m2):
    import scipy.sparse as sp
    grids = [grid1, grid2]
    masses = [red_m1, red_m2]
    print('starting the kinetic energy calculation')

    def kin(grid, red_m):
        N = len(grid)
        a = grid[0]
        b = grid[-1]
        coeff = (1. / ((2. * red_m) / (((float(N) - 1.) / (b - a)) ** 2)))

        Tii = np.zeros(N)

        Tii += coeff * ((np.pi ** 2.) / 3.)
        T_initial = np.diag(Tii)
        for i in range(1, N):
            for j in range(i):
                T_initial[i, j] = coeff * ((-1.) ** (i - j)) * (2. / ((i - j) ** 2))
        T_final = T_initial + T_initial.T - np.diag(Tii)
        return T_final

    kinetic = [kin(g, m) for g, m in zip(grids, masses)]  # get a list of the two kinetic energy matrices
    print('done calculating kinetic energy')

    def kron_sum(der):
        '''Computes a Kronecker sum to build our Kronecker-Delta tensor product expression'''
        n_1 = len(der[1])  # len of grid 1
        ident_1 = sp.eye(n_1)  # the identity matrix of grid 1
        return sp.kron(sp.csr_matrix(der[0]), ident_1) + sp.kron(ident_1, sp.csr_matrix(der[1]))

    T = kron_sum(kinetic)
    print('threw those matrices into our sparse matrix')
    return T


def Energy(T, V, num_wvfns=20):
    H = (T + V)
    print('starting the diagonalization')
    import scipy.sparse.linalg as la
    En, Eigv = la.eigsh(H, num_wvfns, which='SM')
    ind = np.argsort(En)
    En = En[ind]
    Eigv = Eigv[:, ind]
    return En, Eigv


def run(grid1, grid2, mass1, mass2, structure, sp_description=None):
    print('starting DVR run')
    if sp_description is None:
        V, extraV = pot(structure, grid1, grid2)
    else:
        V, extraV = pot2(structure, grid1, grid2)
    # V, extraV = HO_pots(mass1, grid1, grid2)
    print('put the potential in a sparse matrix')
    T = Kinetic_Calc(grid1, grid2, mass1, mass2)
    En, Eig = Energy(T, V)
    print('whew! done with DVR!')
    print(f'ground state energy = {En[0] * har2wave}')
    if np.max(Eig[:, 0]) < 0.005:
        Eig[:, 0] *= -1.
    print(f'frequency of first transition = {(En[1] - En[0]) * har2wave}')
    print(f'frequency of second transition = {(En[2] - En[0]) * har2wave}')
    return En, Eig, extraV


grid_points = 100
Roo_grid = np.linspace(3.9, 5.8, grid_points)
sp_grid = np.linspace(-1.5, 1.5, grid_points)

en, eig, v = run(sp_grid, Roo_grid, m_red_sp, m_red_OO, new_struct)

np.savez('h3o2_2d_wvfn', gridz=[sp_grid, Roo_grid], wvfns=eig, energies=en, pot=v)

import DMC_Tools as dt

ground = dt.Derivatives(eig[:, 0].reshape((100, 100)), sp_grid, Roo_grid)
dx1 = ground.compute_derivative(dx=1)/eig[:, 0].reshape((100, 100))
dy1 = ground.compute_derivative(dy=1)/eig[:, 0].reshape((100, 100))
dx2 = ground.compute_derivative(dx=2)/eig[:, 0].reshape((100, 100))
dy2 = ground.compute_derivative(dy=2)/eig[:, 0].reshape((100, 100))
dx1_dy1 = ground.compute_derivative(dx=1, dy=1)/eig[:, 0].reshape((100, 100))

np.save('z_ground_dx1_2d_h3o2', dx1)
np.save('z_ground_dy1_2d_h3o2', dy1)
np.save('z_ground_dx2_2d_h3o2', dx2)
np.save('z_ground_dy2_2d_h3o2', dy2)
np.save('z_ground_dx1_dy1_2d_h3o2', dx1_dy1)

xh = dt.Derivatives(eig[:, 2].reshape((100, 100)), sp_grid, Roo_grid)
dx1 = xh.compute_derivative(dx=1)/eig[:, 2].reshape((100, 100))
dy1 = xh.compute_derivative(dy=1)/eig[:, 2].reshape((100, 100))
dx2 = xh.compute_derivative(dx=2)/eig[:, 2].reshape((100, 100))
dy2 = xh.compute_derivative(dy=2)/eig[:, 2].reshape((100, 100))
dx1_dy1 = xh.compute_derivative(dx=1, dy=1)/eig[:, 2].reshape((100, 100))

np.save('z_xh_dx1_2d_h3o2', dx1)
np.save('z_xh_dy1_2d_h3o2', dy1)
np.save('z_xh_dx2_2d_h3o2', dx2)
np.save('z_xh_dy2_2d_h3o2', dy2)
np.save('z_xh_dx1_dy1_2d_h3o2', dx1_dy1)


