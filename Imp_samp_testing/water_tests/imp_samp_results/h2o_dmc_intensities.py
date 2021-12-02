import DMC_Tools as dt
import numpy as np
import sys
from Potential.Water_monomer_pot_fns import dipole_h2o

sym_excite_wvfn = dt.waveFunction()
sym_excite_wvfn.load_dvr_wvfn('wvfn_derivs_sym_excite_state')
ground_wvfn = dt.waveFunction()
ground_wvfn.load_dvr_wvfn('wvfn_derivs_ground_state')
excite_wvfn = dt.waveFunction()
excite_wvfn.load_dvr_wvfn('wvfn_derivs_anti_excite_state')
ang2bohr = 1.e-10 / 5.291772106712e-11
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num * me * 1000)
m_H = 1.007825 / (Avo_num * me * 1000)
m_D = 2.01410177812 / (Avo_num * me * 1000)


def water_symmetry(coords, weights):
    coords = np.hstack((coords, coords[:, :, [0, 2, 1]]))
    weights = np.hstack((weights, weights))
    return coords, weights


def asym(coords):
    dis = ground_wvfn.oh_dists(coords)
    return 1 / np.sqrt(2) * (dis[:, 1] - dis[:, 0])


def sym(coords):
    dis = ground_wvfn.oh_dists(coords)
    return 1 / np.sqrt(2) * (dis[:, 1] + dis[:, 0])


ground2 = dt.DMCIntensities(filename='ground_state_h2o_chain_rule_2d',
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
                            append='sym',
                            excited_wvfn=sym_excite_wvfn.psi_t_2d
                            )


wvfn_args = {
    "sym_shift": 0.015741
}

intens_ground_sym, std_intens_ground_sym = ground2.lets_do_some_calcs('sym', rel_dis=sym, **wvfn_args)


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

sym_excite = dt.DMCIntensities(filename='sym_excite_state_h2o_chain_rule_2d',
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
                               append='sym',
                               excited_wvfn=sym_excite_wvfn.psi_t_2d,
                               filename1='sym_left_state_h2o_chain_rule_2d',
                               filename2='sym_right_state_h2o_chain_rule_2d'
                               )

intens_excite_sym, std_intens_excite_sym = sym_excite.lets_do_some_calcs('sym', rel_dis=sym, **wvfn_args)

term1_intens, term1_intens_std = ground.intensities_w_freq(ground.a_eref, ground.s_eref,
                                                           anti_excite.a_eref, anti_excite.s_eref)
term2_intens, term2_intens_std = anti_excite.intensities_w_freq(ground.a_eref, ground.s_eref,
                                                                anti_excite.a_eref, anti_excite.s_eref)

print(ground.freqs_from_erefs(ground.a_eref, ground.s_eref, anti_excite.a_eref, anti_excite.s_eref))

print(f'{ground.a_dis} +/- {ground.std_dis}')
print(f'{anti_excite.a_dis} +/- {anti_excite.std_dis}')


term3_intens = 54.79168892327036
term3_intens_std = 0.015854463882979435

intens_full = term1_intens + term2_intens - term3_intens
intens_full_std = np.sqrt(term1_intens_std ** 2 + term2_intens_std ** 2 + term3_intens_std ** 2)

print(f'full intensity = {intens_full} +/- {intens_full_std}')

sym_excite.a_eref = 0.01673356385379158 + ground2.a_eref
sym_excite.s_eref = 0

term1_intens_sym, term1_intens_std_sym = ground2.intensities_w_freq(ground2.a_eref, ground2.s_eref,
                                                           sym_excite.a_eref, sym_excite.s_eref)
term2_intens_sym, term2_intens_std_sym = sym_excite.intensities_w_freq(ground2.a_eref, ground2.s_eref,
                                                                sym_excite.a_eref, sym_excite.s_eref)

freq2, freq2_std = ground2.freqs_from_erefs(ground2.a_eref, ground2.s_eref, sym_excite.a_eref, sym_excite.s_eref)

term3_intens_sym = 0.0009829591958287651
term3_intens_std_sym = freq2_std*term3_intens_sym

print(f'{[freq2, freq2_std]}')

print(f'{[term3_intens_sym*freq2, term3_intens_std_sym]}')

print(f'{ground2.a_dis} +/- {ground2.std_dis}')
print(f'{sym_excite.a_dis} +/- {sym_excite.std_dis}')

intens_full_sym = term1_intens_sym + term2_intens_sym - term3_intens_sym
intens_full_std_sym = np.sqrt(term1_intens_std_sym**2 + term2_intens_std_sym**2 + term3_intens_std_sym**2)

print(f'full intensity = {intens_full_sym} +/- {intens_full_std_sym}')

