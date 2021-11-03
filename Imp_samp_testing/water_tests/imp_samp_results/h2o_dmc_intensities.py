import DMC_Tools as dt
import numpy as np
import sys
from Potential.Water_monomer_pot_fns import dipole_h2o

ground_wvfn = dt.waveFunction()
ground_wvfn.load_dvr_wvfn('wvfn_derivs_ground_state')
excite_wvfn = dt.waveFunction()
excite_wvfn.load_dvr_wvfn('wvfn_derivs_anti_excite_state')
ang2bohr = 1.e-10/5.291772106712e-11
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)


def water_symmetry(coords, weights):
    coords = np.hstack((coords, coords[:, :, [0, 2, 1]]))
    weights = np.hstack((weights, weights))
    return coords, weights


def asym(coords):
    dis = ground_wvfn.oh_dists(coords)
    return 1/np.sqrt(2)*(dis[:, 1] - dis[:, 0])


ground = dt.DMCIntensities(filename='ground_state_h2o_chain_rule_2d',
                           dipole_func=dipole_h2o,
                           ground_wvfn=ground_wvfn.psi_t_2d,
                           ref_struct=np.load('monomer_coords.npy'),
                           mass=np.array([m_O, m_H, m_H]),
                           walkers=10000,
                           num_wvfns=27,
                           num_sims=10,
                           num_time_steps=20000,
                           symmetry=water_symmetry,
                           planar=True,
                           excited_state_wvfn=False,
                           append='anti',
                           excited_wvfn=excite_wvfn.psi_t_2d
                           )

intens_ground, std_intens_ground = ground.lets_do_some_calcs('anti', rel_dis=asym)

anti_excite = dt.DMCIntensities(filename='anti_excite_state_h2o_chain_rule_2d',
                                dipole_func=dipole_h2o,
                                ground_wvfn=ground_wvfn.psi_t_2d,
                                ref_struct=np.load('monomer_coords.npy'),
                                mass=np.array([m_O, m_H, m_H]),
                                walkers=10000,
                                num_wvfns=27,
                                num_sims=5,
                                num_time_steps=20000,
                                symmetry=water_symmetry,
                                planar=True,
                                excited_state_wvfn=True,
                                append='anti',
                                excited_wvfn=excite_wvfn.psi_t_2d,
                                filename1='asym_left_state_h2o_chain_rule_2d',
                                filename2='asym_right_state_h2o_chain_rule_2d'
                                )

intens_excite, std_intens_excite = anti_excite.lets_do_some_calcs('anti', rel_dis=asym)

ground.intensities_w_freq(ground.a_eref, ground.s_eref, anti_excite.a_eref, anti_excite.s_eref)
anti_excite.intensities_w_freq(ground.a_eref, ground.s_eref, anti_excite.a_eref, anti_excite.s_eref)

print(ground.freqs_from_erefs(ground.a_eref, ground.s_eref, anti_excite.a_eref, anti_excite.s_eref))

print(f'{ground.a_dis} +/- {ground.std_dis}')
print(f'{anti_excite.a_dis} +/- {anti_excite.std_dis}')

