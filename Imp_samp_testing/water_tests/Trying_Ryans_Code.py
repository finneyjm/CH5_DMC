import pyvibdmc as pv
import numpy as np
from pyvibdmc.simulation_utilities import potential_manager as pm
from scipy import interpolate
import matplotlib.pyplot as plt
import DMC_Tools as dt


pot_dir = '.'
py_file = 'water_1d_imp_samp_and_pot.py'
pot_func = 'get_pot'

test_structure = np.array([
        [0.1680942285,  0.2730106336, -0.2683675641],
        [0.7515744938, 1.1894828276,  -1.3309202486],
        [-0.4264345438, 0.9968892161, 0.9222274409],
])

test_structure2 = np.array([
            [0.,  0., 0.],
            [1.81005527, 0.,  0.],
            [-0.96710612, 1.89501907, 0.],
])

structures = np.array([test_structure, test_structure2])

me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)

anti = np.load('antisymmetric_stretch_water_wvfns.npz')
sym = np.load('symmetric_stretch_water_wvfns.npz')

interp_anti = interpolate.splrep(anti['grid'], np.abs(anti['ground']), s=0)
interp_sym = interpolate.splrep(sym['grid'], sym['excite'], s=0)
interp = [interp_anti, interp_sym]

mass = np.array([m_O, m_H, m_H])

kwargs = {
    "excite": "sym",
    "interp": interp,
    "shift": [0, 0.0, 0],
    "shift_rate": [0, 0.000005, 0],
    "mass": mass
}

kwargs2 = {
    "excite": "sym",
    "interp": interp,
    "shift": [0, 0.075, 0],
    "shift_rate": [0, -0.000005, 0],
    "mass": mass
}

kwargs3 = {
    "excite": False
}

kwargs_list = [kwargs, kwargs2, kwargs3]

# np.random.seed(76)

water_pot = pv.Potential(potential_function=pot_func,
                         python_file=py_file,
                         potential_directory=pot_dir,
                         num_cores=11)

for i in range(2):
    water_imp = pv.ImpSampManager(trial_function='psi_t',
                                  trial_directory=pot_dir,
                                  python_file=py_file,
                                  pot_manager=water_pot,
                                  deriv_function='full_dpsi_dx',
                                  trial_kwargs=kwargs_list[0],
                                  deriv_kwargs=kwargs_list[0],
                                  pass_timestep=True
                                  )

    test_water = pv.DMC_Sim(sim_name=f'1d_imp_samp_sym_testing_nodes_no_first_deriv_term_{i+1}',
                            output_folder='1d_imp_samp_test',
                            weighting='continuous',
                            cont_wt_thresh=[0.01, 20],
                            num_walkers=5000,
                            num_timesteps=20000,
                            equil_steps=500,
                            chkpt_every=500,
                            wfn_every=500,
                            desc_wt_steps=250,
                            atoms=['O', 'H', 'H'],
                            delta_t=1,
                            potential=water_pot,
                            start_structures=structures[i, None, :, :],
                            imp_samp=water_imp,
                            masses=[m_O, m_H, m_H])

    test_water.run()


from Coordinerds.CoordinateSystems import *


def linear_combo_stretch_grid(r1, r2, coords):
    coords = np.array([coords] * 1)
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0]])).coords
    N = len(r1)
    zmat = np.array([zmat]*N).squeeze()
    zmat[:, 0, 1] = r1
    zmat[:, 1, 1] = r2
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    return new_coords


def oh_dists(coords):
    bonds = [[1, 2], [1, 3]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis


num_points = 400
ang2bohr = 1.e-10 / 5.291772106712e-11
re = 0.95784 * ang2bohr
re2 = 0.95783997 * ang2bohr
anti = np.zeros(num_points)
sym = np.linspace(-0.55, 0.85, num_points) + (re + re2)/np.sqrt(2)
A = 1 / np.sqrt(2) * np.array([[-1, 1], [1, 1]])
X, Y = np.meshgrid(anti, sym, indexing='ij')
eh = np.matmul(np.linalg.inv(A), np.vstack((X.flatten(), Y.flatten())))
r1 = eh[0]
# r2 = np.zeros(num_points) + re2
r2 = eh[1]
# r1 = np.linspace(1.4, 2.8, num_points)
water = np.load('monomer_coords.npy')
grid = linear_combo_stretch_grid(r1, r2, water).reshape((num_points, num_points, 3, 3))[0]
mass = np.array([m_O, m_H, m_H])
mom = dt.MomentOfSpinz(water, mass)
water = mom.coord_spinz()
np.save('paf_monomer_coords', water)

eck = dt.EckartsSpinz(water, grid, mass, planar=True)
grid_rot = eck.get_rotated_coords()

har2wave = 219474.6

deriv, sderiv = water_imp.call_derivs(grid)
local_kin = test_water.impsamp.local_kin(test_water.inv_masses_trip, sderiv)
pot = test_water.potential(grid)

import matplotlib.pyplot as plt
plt.plot(sym, (local_kin + pot)*har2wave, label='local energy')
plt.plot(sym, pot*har2wave, label='potential')

deriv, sderiv = water_imp.call_derivs(grid_rot)
local_kin = test_water.impsamp.local_kin(test_water.inv_masses_trip, sderiv)
pot = test_water.potential(grid_rot)

plt.plot(sym, (local_kin + pot)*har2wave, label='local energy after Eckart')
plt.plot(sym, pot*har2wave, label='potential after Eckart')
plt.ylim(-100, 10000)
# plt.xlabel(r'r$\rm{_{OH}}$ (Bohr)')
plt.xlabel('s (Bohr)')
plt.ylabel(r'Energy cm$^{-1}$')
plt.tight_layout()
plt.legend()
plt.show()


