

def short_intensity_calcs(filename):
    import matplotlib.pyplot as plt
    import numpy as np

    au_to_Debye = 1 / 0.3934303
    conv_fac = 4.702e-7
    km_mol = 5.33e6

    ground_stuff = np.load(f'ground_{filename}_eck_dips.npz')
    ground_coords = ground_stuff['coords']
    ground_weights = ground_stuff['weights']
    ground_dips = ground_stuff['dips']
    average_zpe = ground_stuff['a_zpe']
    std_zpe = ground_stuff['std_zpe']

    ground_fracs = np.load(f'ground_{filename}_wvfn_fractions.npz')
    ground_dists = ground_fracs['dist']

    def a_prime(a, z):
        return -0.60594644269321474*z + 42.200232187251913*a

    def z_prime(a, z):
        return 41.561937672470521*z + 1.0206303697659393*a

    def angle1(coords):
        bonds = [[1, 2], [1, 0]]
        cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
        cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
        dis = np.linalg.norm(cd2 - cd1, axis=2)
        v1 = (coords[:, 1] - coords[:, 2]) / np.broadcast_to(dis[:, 0, None], (len(dis[:, 0]), 3))
        v2 = (coords[:, 1] - coords[:, 0]) / np.broadcast_to(dis[:, 1, None], (len(dis[:, 1]), 3))
        v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
        v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
        aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
        return aang

    def angle2(coords):
        bonds = [[3, 4], [3, 0]]
        cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
        cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
        dis = np.linalg.norm(cd2 - cd1, axis=2)
        v1 = (coords[:, 3] - coords[:, 4]) / np.broadcast_to(dis[:, 0, None], (len(dis[:, 0]), 3))
        v2 = (coords[:, 3] - coords[:, 0]) / np.broadcast_to(dis[:, 1, None], (len(dis[:, 1]), 3))
        v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
        v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
        aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
        return aang

    binzz = 100
    grca = len(ground_coords)
    term1 = np.zeros((grca, 3))
    term1_vec = np.zeros((grca, 3))
    term1_ov = np.zeros(grca)
    term1_dis = np.zeros(grca)
    term1_dis_a_w_xh = np.zeros(grca)
    term1_dis_xh_w_a = np.zeros(grca)
    xh_term1 = np.zeros((grca, 3))
    xh_term1_vec = np.zeros((grca, 3))
    xh_term1_ov = np.zeros(grca)
    xh_term1_dis = np.zeros(grca)
    frac = ground_fracs['frac1']
    frac2 = ground_fracs['frac2']
    for i in range(grca):
        blah = np.where(np.isfinite(frac[i]))
        term1_ov[i] = np.dot(ground_weights[i, blah].squeeze(), frac[i, blah].squeeze()) \
                      / np.sum(ground_weights[i, blah].squeeze())
        xh_term1_ov[i] = np.dot(ground_weights[i, blah].squeeze(), frac2[i, blah].squeeze()) \
                         / np.sum(ground_weights[i, blah])
        dists = ground_dists[i, blah].squeeze()
        term1_dis[i] = np.dot(ground_weights[i, blah].squeeze(), frac[i, blah].squeeze() * dists[:, 0])\
                       / np.sum(ground_weights[i, blah].squeeze())
        xh_term1_dis[i] = np.dot(ground_weights[i, blah].squeeze(), frac2[i].squeeze() * dists[:, -1]) \
                          / np.sum(ground_weights[i, blah].squeeze())
        term1_dis_a_w_xh[i] = np.dot(ground_weights[i, blah].squeeze(), frac2[i].squeeze() * dists[:, 0]) \
                              / np.sum(ground_weights[i, blah].squeeze())
        term1_dis_xh_w_a[i] = np.dot(ground_weights[i, blah].squeeze(), frac[i].squeeze() * dists[:, -1])\
                              / np.sum(ground_weights[i, blah].squeeze())
        aang1 = np.rad2deg(angle1(ground_coords[i, blah].squeeze()))
        aang2 = np.rad2deg(angle2(ground_coords[i, blah].squeeze()))
        ap = a_prime(dists[:, 0], dists[:, -1])
        plt.hist2d(dists[:, 0], ground_coords[i, blah, 0, 0].squeeze(), bins=binzz, weights=ground_weights[i, blah].squeeze())
        plt.xlabel(r"$\rm{a}$")
        plt.ylabel(r'$\rm{r_{Hx}}$')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'f0_{filename}_a_vs_rhx_{i + 1}')
        plt.close()
        plt.hist2d(dists[:, 0], ground_coords[i, blah, 0, 1].squeeze(), bins=binzz, weights=ground_weights[i, blah].squeeze())
        plt.xlabel(r"$\rm{a}$")
        plt.ylabel(r'$\rm{r_{Hy}}$')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'f0_{filename}_a_vs_rhy_{i + 1}')
        plt.close()
        plt.hist2d(dists[:, 0], ground_coords[i, blah, 0, 2].squeeze(), bins=binzz, weights=ground_weights[i, blah].squeeze())
        plt.xlabel(r"$\rm{a}$")
        plt.ylabel(r'$\rm{r_{Hz}}$')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'f0_{filename}_a_vs_rhz_{i + 1}')
        plt.close()

        # plt.hist2d(ap, aang1 + aang2, bins=binzz, weights=ground_weights[i, blah].squeeze())
        # plt.xlabel(r"$\rm{a'}$")
        # plt.ylabel(r'$\rm{\Theta_1 + \Theta_2}$')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f'f0_{filename}_theta12_sum_vs_ap_{i+1}')
        # plt.close()
        # plt.hist2d(dists[:, 0], aang1 - aang2, bins=binzz, weights=ground_weights[i, blah].squeeze())
        # plt.xlabel(r"$\rm{a'}$")
        # plt.ylabel(r'$\rm{\Theta_1 - \Theta_2}$')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f'f0_{filename}_theta12_diff_vs_ap_{i + 1}')
        # plt.close()
        for j in range(3):
            term1[i, j] = np.dot(ground_weights[i, blah].squeeze(), frac[i, blah].squeeze() *
                                 ground_dips[i, blah, j].squeeze() * au_to_Debye) / np.sum(
                ground_weights[i, blah].squeeze())
            term1_vec[i, j] = np.dot(ground_weights[i, blah].squeeze(),
                                     frac[i, blah].squeeze() * ((ground_coords[i, blah, 2, j].squeeze()
                                                                 - ground_coords[i, blah, 1, j].squeeze()) +
                                                (ground_coords[i, blah, 4, j].squeeze()
                                                 - ground_coords[i, blah, 3, j].squeeze()))) \
                              / np.sum(ground_weights[i, blah].squeeze())
            xh_term1[i, j] = np.dot(ground_weights[i, blah].squeeze(), frac2[i, blah].squeeze()
                                    * ground_dips[i, blah, j].squeeze() * au_to_Debye) / np.sum(
                ground_weights[i, blah].squeeze())
            mid = (ground_coords[i, blah, 3, j].squeeze() - ground_coords[i, blah, 1, j].squeeze()) / 2
            xh_term1_vec[i, j] = np.dot(ground_weights[i, blah].squeeze(), frac2[i, blah].squeeze()
                                        * (mid - ground_coords[i, blah, 0, j].squeeze())) \
                                 / np.sum(ground_weights[i, blah].squeeze())

    avg_term1_vec = np.average(term1_vec, axis=0)
    std_term1_vec = np.std(term1_vec, axis=0)
    avg_term1_o = np.average(term1_ov)
    std_term1_o = np.std(term1_ov)
    avg_term1_d = np.average(term1_dis)
    std_term1_d = np.std(term1_dis)
    avg_term1_d_a_w_xh = np.average(term1_dis_a_w_xh)
    std_term1_d_a_w_xh = np.std(term1_dis_a_w_xh)
    avg_term1_d_xh_w_a = np.average(term1_dis_xh_w_a)
    std_term1_d_xh_w_a = np.std(term1_dis_xh_w_a)
    avg_xh_term1_v = np.average(xh_term1_vec, axis=0)
    std_xh_term1_v = np.std(xh_term1_vec, axis=0)
    avg_xh_term1_o = np.average(xh_term1_ov)
    std_xh_term1_o = np.std(xh_term1_ov)
    avg_xh_term1_d = np.average(xh_term1_dis)
    std_xh_term1_d = np.std(xh_term1_dis)
    print(f'average term1 asym components = {np.average(term1, axis=0)}')
    print(f'std term1 asym components = {np.std(term1, axis=0)}')
    print(f'average term1 XH components = {np.average(xh_term1, axis=0)}')
    print(f'std term1 XH components = {np.std(xh_term1, axis=0)}')
    xh_term1 = np.linalg.norm(xh_term1, axis=-1)
    term1 = np.linalg.norm(term1, axis=-1)
    std_xh_term1 = np.std(xh_term1)
    std_term1 = np.std(term1)
    term1 = np.average(term1)
    xh_term1 = np.average(xh_term1)
    print(f'term1 asym = {term1} +/- {std_term1}')
    print(f'term1 xh = {xh_term1} +/- {std_xh_term1}')

    asym_left = np.load(f'asym_left_{filename}_eck_dips.npz')
    excite_neg_coords = asym_left['coords']
    excite_neg_weights = asym_left['weights']
    excite_neg_dips = asym_left['dips']
    average_excite_neg_energy = asym_left['a_zpe']
    std_excite_neg_energy = asym_left['std_zpe']

    asym_right = np.load(f'asym_right_{filename}_eck_dips.npz')
    excite_pos_coords = asym_right['coords']
    excite_pos_weights = asym_right['weights']
    excite_pos_dips = asym_right['dips']
    average_excite_pos_energy = asym_right['a_zpe']
    std_excite_pos_energy = asym_right['std_zpe']

    XH_left = np.load(f'XH_left_{filename}_eck_dips.npz')
    xh_excite_neg_coords = XH_left['coords']
    xh_excite_neg_weights = XH_left['weights']
    xh_excite_neg_dips = XH_left['dips']
    average_xh_excite_neg_energy = XH_left['a_zpe']
    std_xh_excite_neg_energy = XH_left['std_zpe']

    XH_right = np.load(f'XH_right_{filename}_eck_dips.npz')
    xh_excite_pos_coords = XH_right['coords']
    xh_excite_pos_weights = XH_right['weights']
    xh_excite_pos_dips = XH_right['dips']
    average_xh_excite_pos_energy = XH_right['a_zpe']
    std_xh_excite_pos_energy = XH_right['std_zpe']

    excited_fracs = np.load(f'excited_{filename}_wvfn_fractions.npz')
    dists1 = excited_fracs['dist1']
    dists2 = excited_fracs['dist2']

    average_excite_energy = np.average(np.array([average_excite_pos_energy, average_excite_neg_energy]))
    std_excite_energy = np.sqrt(std_excite_pos_energy ** 2 + std_excite_neg_energy ** 2)
    average_xh_excite_energy = np.average(np.array([average_xh_excite_pos_energy, average_xh_excite_neg_energy]))
    std_xh_excite_energy = np.sqrt(std_xh_excite_pos_energy ** 2 + std_xh_excite_neg_energy ** 2)

    freq = average_excite_energy - average_zpe
    freq2 = average_xh_excite_energy - average_zpe
    std_freq2 = np.sqrt(std_zpe ** 2 + std_xh_excite_energy ** 2)
    std_freq = np.sqrt(std_zpe ** 2 + std_excite_energy ** 2)
    std_term1_sq = term1 ** 2 * np.sqrt((std_term1 / term1) ** 2 + (std_term1 / term1) ** 2)
    std_term1_sq_freq = term1 ** 2 * freq * np.sqrt((std_term1_sq / term1 ** 2) ** 2 + (std_freq / freq) ** 2)

    std_term1_xh_sq = xh_term1 ** 2 * np.sqrt((std_xh_term1 / xh_term1) ** 2 + (std_xh_term1 / xh_term1) ** 2)
    std_term1_xh_freq = xh_term1 ** 2 * freq2 * np.sqrt(
        (std_term1_xh_sq / xh_term1 ** 2) ** 2 + (std_freq2 / freq2) ** 2)
    conversion = conv_fac * km_mol
    print(f'asym term1 intensity = {conversion * term1 ** 2 * freq} +/- {std_term1_sq_freq * conversion}')

    excite = len(excite_pos_coords)
    term2 = np.zeros((excite, 3))
    term2_vec = np.zeros((excite, 3))
    term2_ov = np.zeros(excite)
    xh_term2 = np.zeros((excite, 3))
    xh_term2_vec = np.zeros((excite, 3))
    xh_term2_ov = np.zeros(excite)
    term2_dis = np.zeros(excite)
    term2_dis_a_w_xh = np.zeros(excite)
    term2_dis_xh_w_a = np.zeros(excite)
    xh_term2_dis = np.zeros(excite)
    frac = excited_fracs['frac1']
    frac2 = excited_fracs['frac2']
    for i in range(excite):
        blah1 = np.where(np.isfinite(frac[i]))
        blah2 = np.where(np.isfinite(frac2[i]))
        combine_coords = np.vstack((excite_neg_coords[i], excite_pos_coords[i]))[blah1].squeeze()
        xh_combine_coords = np.vstack((xh_excite_neg_coords[i], xh_excite_pos_coords[i]))[blah2].squeeze()
        combine_weights = np.hstack((excite_neg_weights[i], excite_pos_weights[i]))[blah1].squeeze()
        combine_dips = np.vstack((excite_neg_dips[i], excite_pos_dips[i]))[blah1].squeeze()
        xh_combine_weights = np.hstack((xh_excite_neg_weights[i], xh_excite_pos_weights[i]))[blah2].squeeze()
        xh_combine_dips = np.vstack((xh_excite_neg_dips[i], xh_excite_pos_dips[i]))[blah2].squeeze()
        term2_ov[i] = np.dot(combine_weights, frac[i, blah1].squeeze()) / np.sum(combine_weights)
        xh_term2_ov[i] = np.dot(xh_combine_weights, frac2[i, blah2].squeeze()) / np.sum(xh_combine_weights)
        dists = dists1[i, blah1].squeeze()
        term2_dis[i] = np.average(frac[i, blah1].squeeze() * dists[:, 0], weights=combine_weights)
        term2_dis_xh_w_a[i] = np.average(frac[i, blah1].squeeze() * dists[:, -1], weights=combine_weights)
        dists = dists2[i, blah2].squeeze()
        term2_dis_a_w_xh[i] = np.average(frac2[i, blah2].squeeze() * dists[:, 0], weights=xh_combine_weights)
        xh_term2_dis[i] = np.dot(xh_combine_weights, frac2[i, blah2].squeeze() * dists[:, -1]) \
                          / np.sum(xh_combine_weights)
        aang1 = np.rad2deg(angle1(combine_coords))
        aang2 = np.rad2deg(angle2(combine_coords))
        ap = a_prime(dists1[i, blah1, 0].squeeze(), dists1[i, blah1, 0].squeeze())
        plt.hist2d(dists1[i, blah1, 0].squeeze(), combine_coords[:, 0, 0], bins=binzz, weights=combine_weights)
        plt.xlabel(r"$\rm{a}$")
        plt.ylabel(r'$\rm{r_{Hx}}$')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'f1_{filename}_a_vs_rhx_{i + 1}')
        plt.close()
        plt.hist2d(dists1[i, blah1, 0].squeeze(), combine_coords[:, 0, 1], bins=binzz, weights=combine_weights)
        plt.xlabel(r"$\rm{a}$")
        plt.ylabel(r'$\rm{r_{Hy}}$')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'f1_{filename}_a_vs_rhy_{i + 1}')
        plt.close()
        plt.hist2d(dists1[i, blah1, 0].squeeze(), combine_coords[:, 0, 2], bins=binzz, weights=combine_weights)
        plt.xlabel(r"$\rm{a}$")
        plt.ylabel(r'$\rm{r_{Hz}}$')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'f1_{filename}_a_vs_rhz_{i + 1}')
        plt.close()

        # plt.hist2d(ap, aang1 + aang2, bins=binzz, weights=combine_weights)
        # plt.xlabel(r"$\rm{a'}$")
        # plt.ylabel(r'$\rm{\Theta_1 + \Theta_2}$')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f'f1_{filename}_theta12_sum_vs_ap_{i + 1}')
        # plt.close()
        # plt.hist2d(ap, aang1 - aang2, bins=binzz, weights=combine_weights)
        # plt.xlabel(r"$\rm{a'}$")
        # plt.ylabel(r'$\rm{\Theta_1 - \Theta_2}$')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f'f1_{filename}_theta12_diff_vs_ap_{i + 1}')
        # plt.close()
        for j in range(3):
            term2[i, j] = np.dot(combine_weights, frac[i, blah1].squeeze() * combine_dips[:, j] * au_to_Debye) \
                          / np.sum(combine_weights)
            term2_vec[i, j] = np.dot(combine_weights,
                                     frac[i, blah1].squeeze() * ((combine_coords[:, 2, j] - combine_coords[:, 1, j]) +
                                                (combine_coords[:, 4, j] - combine_coords[:, 3, j]))) \
                              / np.sum(combine_weights)
            xh_term2[i, j] = np.dot(xh_combine_weights, frac2[i, blah2].squeeze() * xh_combine_dips[:, j] * au_to_Debye) \
                             / np.sum(xh_combine_weights)
            mid = (xh_combine_coords[:, 3, j] - xh_combine_coords[:, 1, j]) / 2
            xh_term2_vec[i, j] = np.dot(xh_combine_weights, frac2[i, blah2].squeeze() * (mid - xh_combine_coords[:, 0, j])) \
                                 / np.sum(xh_combine_weights)

    avg_term2_vec = np.average(term2_vec, axis=0)
    std_term2_vec = np.std(term2_vec, axis=0)
    avg_term2_o = np.average(term2_ov)
    std_term2_o = np.std(term2_ov)
    avg_term2_d = np.average(term2_dis)
    std_term2_d = np.std(term2_dis)
    avg_term2_d_a_w_xh = np.average(term2_dis_a_w_xh)
    std_term2_d_a_w_xh = np.std(term2_dis_a_w_xh)
    avg_term2_d_xh_w_a = np.average(term2_dis_xh_w_a)
    std_term2_d_xh_w_a = np.std(term2_dis_xh_w_a)
    avg_xh_term2_v = np.average(xh_term2_vec, axis=0)
    std_xh_term2_v = np.std(xh_term2_vec, axis=0)
    avg_xh_term2_o = np.average(xh_term2_ov)
    std_xh_term2_o = np.std(xh_term2_ov)
    avg_xh_term2_d = np.average(xh_term2_dis)
    std_xh_term2_d = np.std(xh_term2_dis)
    print(f'average term2 asym components = {np.average(term2, axis=0)}')
    print(f'std term2 asym components = {np.std(term2, axis=0)}')
    print(f'average term2 XH components = {np.average(xh_term2, axis=0)}')
    print(f'std term2 XH components = {np.std(xh_term2, axis=0)}')

    term2 = np.linalg.norm(term2, axis=-1)
    xh_term2 = np.linalg.norm(xh_term2, axis=-1)

    std_term2 = np.std(term2)
    xh_std_term2 = np.std(xh_term2)
    term2 = np.average(term2)
    xh_term2 = np.average(xh_term2)
    print(f'term2 asym = {term2} +/- {std_term2}')
    print(f'term2 xh = {xh_term2} +/- {xh_std_term2}')

    std_term2_xh_sq = xh_term2 ** 2 * np.sqrt((xh_std_term2 / xh_term2) ** 2 + (xh_std_term2 / xh_term2) ** 2)
    std_term2_sq = term2 ** 2 * np.sqrt((std_term2 / term2) ** 2 + (std_term2 / term2) ** 2)
    std_term2_sq_freq = term2 ** 2 * freq * np.sqrt((std_term2_sq / term2 ** 2) ** 2 + (std_freq / freq) ** 2)
    xh_std_term2_sq_freq = xh_term2 ** 2 * freq2 * np.sqrt(
        (std_term2_xh_sq / xh_term2 ** 2) ** 2 + (std_freq2 / freq2) ** 2)
    print(f'term2 asym intensity = {conversion * term2 ** 2 * freq} +/- {std_term2_sq_freq * conversion}')

    # term3 = 0.14379963852224506
    term3 = 0.0424328790886425
    term3_mod = 0.057139489481379306
    xh_term3 = 1.1154257246577732
    xh_term3_mod = 1.0723059556104413
    std_term3 = 0.0
    std_term3_sq = 0.0
    std_term3_sq_freq = term3 ** 2 * freq * np.sqrt((std_term3_sq / term3 ** 2) ** 2 + (std_freq / freq) ** 2)
    xh_std_term3_sq_freq = xh_term3 ** 2 * freq2 * np.sqrt((std_freq2 / freq2) ** 2)
    xh_std_term3_mod_sq_freq = xh_term3_mod ** 2 * freq2 * np.sqrt((std_freq2 / freq2) ** 2)
    std_term3_mod_sq_freq = term3 ** 2 * freq * np.sqrt((std_freq / freq) ** 2)
    print(f'term3 asym intensity = {term3 ** 2 * freq * conversion} +/- '
          f'{std_term3_sq_freq * conversion}')
    print(f'term3 asym prime intensity = {term3_mod ** 2 * freq * conversion} +/- '
          f'{std_term3_mod_sq_freq * conversion}')

    full_error = np.sqrt(std_term1 ** 2 + std_term2 ** 2 + std_term3 ** 2)
    full_error_xh = np.sqrt(std_xh_term1 ** 2 + xh_std_term2 ** 2)
    dipole_xh = xh_term1 + xh_term2 - xh_term3_mod
    dipole = term1 + term2 - term3_mod
    full_error_sq = dipole ** 2 * np.sqrt((full_error / dipole) ** 2 + (full_error / dipole) ** 2)
    full_error_sq_xh = dipole_xh ** 2 * np.sqrt((full_error_xh / dipole_xh) ** 2 + (full_error_xh / dipole_xh) ** 2)
    full_error = dipole ** 2 * freq * np.sqrt((full_error_sq / dipole ** 2) ** 2 + (std_freq / freq) ** 2)
    full_error_xh = dipole_xh ** 2 * freq2 * np.sqrt(
        (full_error_sq_xh / dipole_xh ** 2) ** 2 + (std_freq2 / freq2) ** 2)
    print(f'full asym prime intensity = {dipole ** 2 * freq * conversion} +/- '
          f'{full_error * conversion}')

    print(f'term1 XH intensity = {conversion * xh_term1 ** 2 * freq2} +/- '
          f'{std_term1_xh_freq * conversion}')
    print(f'term2 XH intensity = {conversion * xh_term2 ** 2 * freq2} +/- '
          f'{xh_std_term2_sq_freq * conversion}')
    print(f'term3 XH intensity = {xh_term3 ** 2 * freq2 * conversion} +/- '
          f'{xh_std_term3_sq_freq * conversion}')
    print(f'term3 z prime intensity = {xh_term3_mod ** 2 * freq2 * conversion} +/- '
          f'{xh_std_term3_mod_sq_freq * conversion}')
    print(f'full z prime intensity = {dipole_xh ** 2 * freq2 * conversion} +/- '
          f'{full_error_xh * conversion}')

    print(f'term1 a overlap = {avg_term1_o} {std_term1_o}')
    print(f'term2 a overlap = {avg_term2_o} {std_term2_o}')

    print(f'term1 xh overlap = {avg_xh_term1_o} {std_xh_term1_o}')
    print(f'term2 xh overlap = {avg_xh_term2_o} {std_xh_term2_o}')

    print(f'term1 a dis = {avg_term1_d} {std_term1_d}')
    print(f'term2 a dis = {avg_term2_d} {std_term2_d}')

    print(f'term1 xh dis = {avg_xh_term1_d} {std_xh_term1_d}')
    print(f'term2 xh dis = {avg_xh_term2_d} {std_xh_term2_d}')

    print(f'term1 a vec = {avg_term1_vec} {std_term1_vec}')
    print(f'term2 a vec = {avg_term2_vec} {std_term2_vec}')

    print(f'term1 xh vec = {avg_xh_term1_v} {std_xh_term1_v}')
    print(f'term2 xh vec = {avg_xh_term2_v} {std_xh_term2_v}')

    print(f'term1 a dis with xh excite = {avg_term1_d_a_w_xh} {std_term1_d_a_w_xh}')
    print(f'term2 a dis with xh excite = {avg_term2_d_a_w_xh} {std_term2_d_a_w_xh}')

    print(f'term1 xh dis with a excite = {avg_term1_d_xh_w_a} {std_term1_d_xh_w_a}')
    print(f'term2 xh dis with a excite = {avg_term2_d_xh_w_a} {std_term2_d_xh_w_a}')


def intensity_calcs(filename, wvfn_type):
    import numpy as np
    try:
        np.load(f'excited_{filename}_wvfn_fractions.npz')
        print('loading in previous calculations')
    except:
        print('starting from scratch')
        from H3O2_intensities import full_intensity_calcs
        full_intensity_calcs(filename, wvfn_type)
    short_intensity_calcs(filename)


intensity_calcs('excite_state_chain_rule2_biggest_full_h3o2', 'linear_combo')

