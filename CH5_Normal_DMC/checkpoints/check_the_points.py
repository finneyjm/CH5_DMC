import numpy as np
from CH5_funcs.Dev_dep_fd_imp_samp import *
# from CH5_funcs.CH5_trial_wvfn import *

hh_relate = np.load('../../lets_go_girls/jobs/params/sigma_hh_to_rch_exp_relationship_params.npy')
anharm_ch_wvfn = np.load('../../lets_go_girls/jobs/params/min_wvfns/new_GSW_min_CH_2.npy')

anharm_interp_filler = interpolate.splrep(anharm_ch_wvfn[0], anharm_ch_wvfn[1], s=0)
anharm_interp = []
for i in range(5):
    anharm_interp.append(anharm_interp_filler)


a = np.load('walkers_1.npz')
coords1 = a['coords']
starting_coords = a['parent_coords']
weights1 = a['weights']
starting_weights = a['parent_weights']

psi1_from_Mark = psi_t(ch_dist(coords1), anharm_interp, 'dev_dep', coords1, interp_exp=hh_relate, multicore=True, analytic_rch=None)

rch_start = ch_dist(starting_coords)

starting_psi = psi_t(rch_start, anharm_interp, 'dev_dep', starting_coords, interp_exp=hh_relate, multicore=True, analytic_rch=None)


coords2 = np.load('walkers_2.npz')['coords']
psi2_from_Mark = psi_t(ch_dist(coords2), anharm_interp, 'dev_dep', coords2, interp_exp=hh_relate, multicore=True, analytic_rch=None)
weights2 = np.load('walkers_2.npz')['weights']

coords3 = np.load('walkers_3.npz')['coords']
psi3_from_Mark = psi_t(ch_dist(coords3), anharm_interp, 'dev_dep', coords3, interp_exp=hh_relate, multicore=True, analytic_rch=None)
weights3 = np.load('walkers_3.npz')['weights']

coords4 = np.load('walkers_4.npz')['coords']
psi4_from_Mark = psi_t(ch_dist(coords4), anharm_interp, 'dev_dep', coords4, interp_exp=hh_relate, multicore=True)
weights4 = np.load('walkers_4.npz')['weights']

coords5 = np.load('walkers_5.npz')['coords']
psi5_from_Mark = psi_t(ch_dist(coords5), anharm_interp, 'dev_dep', coords5, interp_exp=hh_relate, multicore=True)
weights5 = np.load('walkers_5.npz')['weights']

coords6 = np.load('walkers_6.npz')['coords']
psi6_from_Mark = psi_t(ch_dist(coords6), anharm_interp, 'dev_dep', coords6, interp_exp=hh_relate)
weights6 = np.load('walkers_6.npz')['weights']


coords7 = np.load('walkers_7.npz')['coords']
psi7_from_Mark = psi_t(ch_dist(coords7), anharm_interp, 'dev_dep', coords7, interp_exp=hh_relate)
weights7 = np.load('walkers_7.npz')['weights']

coords8 = np.load('walkers_8.npz')['coords']
psi8_from_Mark = psi_t(ch_dist(coords8), anharm_interp, 'dev_dep', coords8, interp_exp=hh_relate)
weights8 = np.load('walkers_8.npz')['weights']

coords9 = np.load('walkers_9.npz')['coords']
psi9_from_Mark = psi_t(ch_dist(coords9), anharm_interp, 'dev_dep', coords9, interp_exp=hh_relate)
weights9 = np.load('walkers_9.npz')['weights']

coords10 = np.load('walkers_10.npz')['coords']
psi10_from_Mark = psi_t(ch_dist(coords10), anharm_interp, 'dev_dep', coords10, interp_exp=hh_relate)
weights10 = np.load('walkers_10.npz')['weights']


dx = 1e-3


def drift_test(psi):
    return (psi[:, 2] - psi[:, 0]) / dx / psi[:, 1]


starting_drift = drift_test(starting_psi)
dr1 = drift_test(psi1_from_Mark)
dr2 = drift_test(psi2_from_Mark)
dr3 = drift_test(psi3_from_Mark)
dr4 = drift_test(psi4_from_Mark)
dr5 = drift_test(psi5_from_Mark)
dr6 = drift_test(psi6_from_Mark)
dr7 = drift_test(psi7_from_Mark)
dr8 = drift_test(psi8_from_Mark)
dr9 = drift_test(psi9_from_Mark)
dr10 = drift_test(psi10_from_Mark)


def easy_Parr_Potential(coords):
    crds = np.array_split(coords, mp.cpu_count()-1)
    V = pool.map(get_pot, crds)
    new_V = np.concatenate(V)
    return new_V


starting_v = easy_Parr_Potential(starting_coords)
v1 = easy_Parr_Potential(coords1)
v2 = easy_Parr_Potential(coords2)
v3 = easy_Parr_Potential(coords3)
v4 = easy_Parr_Potential(coords4)
v5 = easy_Parr_Potential(coords5)
v6 = easy_Parr_Potential(coords6)
v7 = easy_Parr_Potential(coords7)
v8 = easy_Parr_Potential(coords8)
v9 = easy_Parr_Potential(coords9)
v10 = easy_Parr_Potential(coords10)


def easy_kin(psi, sigma, dtau):
    d2psidx2 = ((psi[:, 0] - 2. * psi[:, 1] + psi[:, 2]) / dx ** 2) / psi[:, 1]
    kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * d2psidx2, axis=1), axis=1)
    return kin


dtau = 1
sigmaH = np.sqrt(dtau/m_H)
sigmaC = np.sqrt(dtau/m_C)
sigmaCH = np.zeros((6, 3))
sigmaCH[0] = np.array([[sigmaC] * 3])
for i in np.arange(1, 6):
    sigmaCH[i] = np.array([[sigmaH] * 3])

starting_kin = easy_kin(starting_psi, sigmaCH, dtau)
kin1 = easy_kin(psi1_from_Mark, sigmaCH, dtau)
kin2 = easy_kin(psi2_from_Mark, sigmaCH, dtau)
kin3 = easy_kin(psi3_from_Mark, sigmaCH, dtau)
kin4 = easy_kin(psi4_from_Mark, sigmaCH, dtau)
kin5 = easy_kin(psi5_from_Mark, sigmaCH, dtau)
kin6 = easy_kin(psi6_from_Mark, sigmaCH, dtau)
kin7 = easy_kin(psi7_from_Mark, sigmaCH, dtau)
kin8 = easy_kin(psi8_from_Mark, sigmaCH, dtau)
kin9 = easy_kin(psi9_from_Mark, sigmaCH, dtau)
kin10 = easy_kin(psi10_from_Mark, sigmaCH, dtau)


def easy_eref(weights, local_en, dtau):
    alpha = 0.5*dtau
    P = np.sum(weights)
    Eref = np.average(local_en, weights=weights) - alpha*np.log(P/len(weights))
    return Eref

starting_eref = easy_eref(starting_weights, starting_kin+starting_v, dtau)
eref1 = easy_eref(weights1, kin1-v1, dtau)
eref2 = easy_eref(weights2, kin2+v2, dtau)
eref3 = easy_eref(weights3, kin3+v3, dtau)
eref4 = easy_eref(weights4, kin4+v4, dtau)
eref5 = easy_eref(weights5, kin5+v5, dtau)
eref6 = easy_eref(weights6, kin6+v6, dtau)
eref7 = easy_eref(weights7, kin7+v7, dtau)
eref8 = easy_eref(weights8, kin8+v8, dtau)
eref9 = easy_eref(weights9, kin9+v9, dtau)
eref10 = easy_eref(weights10, kin10+v10, dtau)


stop = True