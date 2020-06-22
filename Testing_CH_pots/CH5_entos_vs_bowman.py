import numpy as np
import matplotlib.pyplot as plt

ang2bohr = (1.e-10)/(5.291772106712e-11)

coords_initial_min = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                  [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                  [1.786540362044548, -1.386051328559878, 0.000000000000000],
                  [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                  [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                  [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]]).reshape((1, 6, 3))

coords_initial_cs = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                       [1.931652478009080, -4.5126502395556294E-008, -0.6830921182334913],
                       [5.4640011799588715E-017, 0.8923685824271653, 2.083855680290835],
                       [-5.4640011799588715E-017, -0.8923685824271653, 2.083855680290835],
                       [-1.145620108130841, -1.659539840225091, -0.4971351597887673],
                       [-1.145620108130841, 1.659539840225091, -0.4971351597887673]]).reshape((1, 6, 3))

coords_initial_c2v = np.array([[0.000000000000000, 0.000000000000000, 0.386992362158741],
                       [0.000000000000000, 0.000000000000000, -1.810066283748844],
                       [1.797239666982623, 0.000000000000000, 1.381637275550612],
                       [-1.797239666982623, 0.000000000000000, 1.381637275550612],
                       [0.000000000000000, -1.895858229423645, -0.6415748897955779],
                       [0.000000000000000, 1.895858229423645, -0.6415748897955779]]).reshape((1, 6, 3))

coords_initial_min_entos = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                                     [0.0697906130287307, 1.105419471355168, 0.000000000000000],
                                     [0.945396445536106, -0.733466775892764, 0.000000000000000],
                                     [1.182079747453685, 0.1887626416923842, 0.000000000000000],
                                     [-0.4364188710110507, -0.3331332529394641, -0.939465377540584],
                                     [-0.4364188710110507, -0.3331332529394641, 0.939465377540584]]).reshape((1, 6, 3))*ang2bohr

coords_initial_c2v_entos = np.array([[0.000000000000000, 0.000000000000000, 0.2047875387577572],
                                     [0.000000000000000, 0.000000000000000, -0.957845827162026],
                                     [0.951058273879344, 0.000000000000000, 0.731130959633571],
                                     [-0.951058273879344, 0.000000000000000, 0.731130959633571],
                                     [0.000000000000000, -1.003244969672169, -0.3395068106179365],
                                     [0.000000000000000, 1.003244969672169, -0.3395068106179365]]).reshape((1, 6, 3))*ang2bohr

coords_initial_cs_entos = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                                    [1.022186470296639, -0.000000023879916664973551, -0.3614767817574607],
                                    [0.000000000000000028914249035082243, 0.4722211173383493, 1.102728936335139],
                                    [-0.000000000000000028914249035082243, -0.4722211173383493, 1.102728936335139],
                                    [-0.606236053308142, -0.878190663646051, -0.2630725971830046],
                                    [-0.606236053308142, 0.878190663646051, -0.2630725971830046]]).reshape((1, 6, 3))*ang2bohr

bonds = 5


def hh_dist(coords):
    N = len(coords)
    hh = np.zeros((N, 5, 4))
    for i in range(4):
        for j in np.arange(i, 4):
            hh[:, i, j] = np.sqrt((coords[:, j + 2, 0] - coords[:, i + 1, 0]) ** 2 +
                                  (coords[:, j + 2, 1] - coords[:, i + 1, 1]) ** 2 +
                                  (coords[:, j + 2, 2] - coords[:, i + 1, 2]) ** 2)
            hh[:, j+1, i] = hh[:, i, j]
    hh_std = np.std(hh, axis=2)
    return hh_std


def ch_dist(coords):
    N = len(coords)
    rch = np.zeros((N, 5))
    for i in range(5):
        rch[:, i] = np.sqrt((coords[:, i + 1, 0] - coords[:, 0, 0]) ** 2 +
                            (coords[:, i + 1, 1] - coords[:, 0, 1]) ** 2 +
                            (coords[:, i + 1, 2] - coords[:, 0, 2]) ** 2)
    return rch


def hh_dist_for_correlation(carts, rch):
    N = len(carts)
    coords = np.array(carts)
    coords -= np.broadcast_to(coords[:, None, 0], (N, bonds + 1, 3))
    coords[:, 1:] /= np.broadcast_to(rch[:, :, None], (N, bonds, 3))
    hh = np.zeros((N, 5, 4))
    for i in range(4):
        for j in np.arange(i, 4):
            hh[:, i, j] = np.sqrt((coords[:, j + 2, 0] - coords[:, i + 1, 0]) ** 2 +
                                  (coords[:, j + 2, 1] - coords[:, i + 1, 1]) ** 2 +
                                  (coords[:, j + 2, 2] - coords[:, i + 1, 2]) ** 2)
            hh[:, j+1, i] = hh[:, i, j]
    hh_std = np.std(hh, axis=2)
    return hh_std


rch_bowman_min = ch_dist(coords_initial_min)
hh_bowman_min = hh_dist(coords_initial_min)
hh_corr_bowman_min = hh_dist_for_correlation(coords_initial_min, rch_bowman_min)

rch_entos_min = ch_dist(coords_initial_min_entos)
hh_entos_min = hh_dist(coords_initial_min_entos)
hh_corr_entos_min = hh_dist_for_correlation(coords_initial_min_entos, rch_entos_min)

rch_bowman_cs = ch_dist(coords_initial_cs)
hh_bowman_cs = hh_dist(coords_initial_cs)
hh_corr_bowman_cs = hh_dist_for_correlation(coords_initial_cs, rch_bowman_cs)

rch_entos_cs = ch_dist(coords_initial_cs_entos)
hh_entos_cs = hh_dist(coords_initial_cs_entos)
hh_corr_entos_cs = hh_dist_for_correlation(coords_initial_cs_entos, rch_entos_cs)

rch_bowman_c2v = ch_dist(coords_initial_c2v)
hh_bowman_c2v = hh_dist(coords_initial_c2v)
hh_corr_bowman_c2v = hh_dist_for_correlation(coords_initial_c2v, rch_bowman_c2v)

rch_entos_c2v = ch_dist(coords_initial_c2v_entos)
hh_entos_c2v = hh_dist(coords_initial_c2v_entos)
hh_corr_entos_c2v = hh_dist_for_correlation(coords_initial_c2v_entos, rch_entos_c2v)

print(rch_bowman_min-rch_entos_min)
print(rch_bowman_cs-rch_entos_cs)
print(rch_bowman_c2v-rch_entos_c2v)

print(hh_bowman_min-hh_entos_min)
print(hh_bowman_cs-hh_entos_cs)
print(hh_bowman_c2v-hh_entos_c2v)








