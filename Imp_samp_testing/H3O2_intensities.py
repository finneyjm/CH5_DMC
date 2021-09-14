import matplotlib.pyplot as plt
from ProtWaterPES import Dipole
import multiprocessing as mp
from Eckart_turny_turn import EckartsSpinz
from PAF_spinz import MomentOfSpinz
from Coordinerds.CoordinateSystems import *
from scipy import interpolate


def full_intensity_calcs(filename, wvfn_type):

    har2wave = 219474.6

    ref = np.array([
      [0.000000000000000, 0.000000000000000, 0.000000000000000],
      [-2.304566686034061, 0.000000000000000, 0.000000000000000],
      [-2.740400260927908, 1.0814221449986587E-016, -1.766154718409233],
      [2.304566686034061, 0.000000000000000, 0.000000000000000],
      [2.740400260927908, 1.0814221449986587E-016, 1.766154718409233],
    ])

    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num*me*1000)
    m_H = 1.007825 / (Avo_num*me*1000)

    mass = np.array([m_H, m_O, m_H, m_O, m_H])

    MOM = MomentOfSpinz(ref, mass)
    ref = MOM.coord_spinz()

    small_grid_points = 100
    Roo_grid = np.linspace(3.9, 5.8, small_grid_points)
    if wvfn_type == 'linear_combo':
        sp_grid = np.linspace(-65, 65, small_grid_points)
        wvfns = np.load(f'2d_h3o2_new_def_{small_grid_points}_points_no_cutoff.npz')['wvfns']
    else:
        sp_grid = np.linspace(-1.5, 1.5, small_grid_points)
        wvfns = np.load('small_grid_2d_h3o2_no_cutoff.npz')['wvfns']

    z_ground_no_der = wvfns[:, 0].reshape((small_grid_points, small_grid_points))

    ground_no_der = interpolate.interp2d(sp_grid, Roo_grid, z_ground_no_der.T, kind='cubic')

    z_excite_no_der = wvfns[:, 2].reshape((small_grid_points, small_grid_points))

    excite_no_der = interpolate.interp2d(sp_grid, Roo_grid, z_excite_no_der.T, kind='cubic')

    def a_prime(a, z):
        return -0.60594644269321474*z + 42.200232187251913*a

    def z_prime(a, z):
        return 41.561937672470521*z + 1.0206303697659393*a

    def interp(x, y, poiuy):
        out = np.zeros(len(x))
        for i in range(len(x)):
            out[i] = poiuy(x[i], y[i])
        return out

    def get_da_psi(coords, excite):
        psi = np.ones((len(coords), 2))
        dists = all_dists(coords)
        if wvfn_type == 'linear_combo':
            mw_h = omega_asym
            a = a_prime(dists[:, 0], dists[:, -1])
            z = z_prime(dists[:, 0], dists[:, -1])
        else:
            mw_h = m_OH * omega_asym
            a = dists[:, 0]
            z = dists[:, -1]
        if excite == 'sp':
            psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * a ** 2))
            psi[:, 1] = interp(z, dists[:, -2], excite_no_der)
        elif excite == 'a':
            psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * a ** 2)) * \
                        (2 * mw_h) ** (1 / 2) * a
            psi[:, 1] = interp(z, dists[:, -2], ground_no_der)
        else:
            psi[:, 0] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * a ** 2))
            psi[:, 1] = interp(z, dists[:, -2], ground_no_der)
        return np.prod(psi, axis=1)

    def all_dists(coords):
        bonds = [[1, 2],  [3, 4], [1, 3], [1, 0]]
        cd1 = coords[:, tuple(x[0] for x in np.array(bonds))]
        cd2 = coords[:, tuple(x[1] for x in np.array(bonds))]
        dis = np.linalg.norm(cd2 - cd1, axis=2)
        a_oh = 1/np.sqrt(2)*(dis[:, 0]-dis[:, 1])
        mid = (coords[:, 1] + coords[:, 3])/2
        rxh = coords[:, 0] - mid
        rxh_dist = np.linalg.norm(rxh, axis=-1)
        sp = rxh_dist*np.cos(roh_roo_angle(coords, rxh, dis[:, -2], rxh_dist))
        return np.vstack((a_oh, dis[:, 0], dis[:, 1], dis[:, -2], sp)).T

    def roh_roo_angle(coords, rxh, roo_dist, rxh_dist):
        v1 = (coords[:, 1]-coords[:, 3])/np.broadcast_to(roo_dist[:, None], (len(roo_dist), 3))
        v2 = (rxh)/np.broadcast_to(rxh_dist[:, None], (len(rxh_dist), 3))
        v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
        v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
        aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
        return aang

    walkers = 20000

    ground_calcs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ground_coords = np.zeros((len(ground_calcs), 27, walkers, 5, 3))
    ground_erefs = np.zeros((len(ground_calcs), 20000))
    ground_weights = np.zeros((len(ground_calcs), 27, walkers))
    for i in range(len(ground_calcs)):
        blah = np.load(f'ground_{filename}_{ground_calcs[i]+1}.npz')
        coords = blah['coords']
        eref = blah['Eref']
        weights = blah['weights']
        ground_coords[i] = coords
        ground_erefs[i] = eref
        ground_weights[i] = weights

    print(f'average zpe = {np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave}')
    average_zpe = np.mean(np.mean(ground_erefs[:, 5000:], axis=1), axis=0)*har2wave
    std_zpe = np.std(np.mean(ground_erefs[:, 5000:]*har2wave, axis=1))
    print(f'std zpe = {std_zpe}')

    # walkers = 5000
    asm_left = [0, 1, 2, 3, 4]

    excite_neg_coords = np.zeros((len(asm_left), 27, walkers, 5, 3))
    excite_neg_erefs = np.zeros((len(asm_left), 20000))
    excite_neg_weights = np.zeros((len(asm_left), 27, walkers))
    for i in range(len(asm_left)):
        blah = np.load(f'asym_left_{filename}_{asm_left[i]+1}.npz')
        coords = blah['coords']
        eref = blah['Eref']
        weights = blah['weights']
        excite_neg_coords[i] = coords
        excite_neg_erefs[i] = eref
        excite_neg_weights[i] = weights

    print(f'average asym neg excite energy = {np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave}')
    average_excite_neg_energy = np.mean(np.mean(excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave
    std_excite_neg_energy = np.std(np.mean(excite_neg_erefs[:, 5000:]*har2wave, axis=1))

    asm_right = [0, 1, 2, 3, 4]
    excite_pos_coords = np.zeros((len(asm_right), 27, walkers, 5, 3))
    excite_pos_erefs = np.zeros((len(asm_right), 20000))
    excite_pos_weights = np.zeros((len(asm_right), 27, walkers))
    for i in range(len(asm_right)):
        blah = np.load(f'asym_right_{filename}_{asm_right[i]+1}.npz')
        coords = blah['coords']
        eref = blah['Eref']
        weights = blah['weights']
        excite_pos_coords[i] = coords
        excite_pos_erefs[i] = eref
        excite_pos_weights[i] = weights

    print(f'average asym pos excite energy = {np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave}')
    average_excite_pos_energy = np.mean(np.mean(excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave
    std_excite_pos_energy = np.std(np.mean(excite_pos_erefs[:, 5000:]*har2wave, axis=1))

    print(f'average asym neg frequency = {average_excite_neg_energy-average_zpe} +/-'
          f' {np.sqrt(std_zpe**2 + std_excite_neg_energy**2)}')
    print(f'average asym pos frequency = {average_excite_pos_energy-average_zpe} +/- '
          f'{np.sqrt(std_zpe**2 + std_excite_pos_energy**2)}')

    average_excite_energy = np.average(np.array([average_excite_pos_energy, average_excite_neg_energy]))
    std_excite_energy = np.sqrt(std_excite_pos_energy**2 + std_excite_neg_energy**2)
    print(f'average asym frequency = {average_excite_energy-average_zpe} +/- '
          f'{np.sqrt(std_zpe**2 + std_excite_energy**2)}')

    xh_left = [0, 1, 2, 3, 4]

    xh_excite_neg_coords = np.zeros((len(xh_left), 27, walkers, 5, 3))
    xh_excite_neg_erefs = np.zeros((len(xh_left), 20000))
    xh_excite_neg_weights = np.zeros((len(xh_left), 27, walkers))
    for i in range(len(xh_left)):
        blah = np.load(f'XH_left_{filename}_{xh_left[i]+1}.npz')
        coords = blah['coords']
        eref = blah['Eref']
        weights = blah['weights']
        xh_excite_neg_coords[i] = coords
        xh_excite_neg_erefs[i] = eref
        xh_excite_neg_weights[i] = weights

    print(f'average XH neg excite energy = {np.mean(np.mean(xh_excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave}')
    average_xh_excite_neg_energy = np.mean(np.mean(xh_excite_neg_erefs[:, 5000:], axis=1), axis=0)*har2wave
    std_xh_excite_neg_energy = np.std(np.mean(xh_excite_neg_erefs[:, 5000:]*har2wave, axis=1))

    xh_right = [0, 1, 2, 3, 4]
    xh_excite_pos_coords = np.zeros((len(xh_right), 27, walkers, 5, 3))
    xh_excite_pos_erefs = np.zeros((len(xh_right), 20000))
    xh_excite_pos_weights = np.zeros((len(xh_right), 27, walkers))
    for i in range(len(xh_right)):
        blah = np.load(f'XH_right_{filename}_{xh_right[i]+1}.npz')
        coords = blah['coords']
        eref = blah['Eref']
        weights = blah['weights']
        xh_excite_pos_coords[i] = coords
        xh_excite_pos_erefs[i] = eref
        xh_excite_pos_weights[i] = weights

    print(f'average XH pos excite energy = {np.mean(np.mean(xh_excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave}')
    average_xh_excite_pos_energy = np.mean(np.mean(xh_excite_pos_erefs[:, 5000:], axis=1), axis=0)*har2wave
    std_xh_excite_pos_energy = np.std(np.mean(xh_excite_pos_erefs[:, 5000:]*har2wave, axis=1))

    print(f'average XH neg frequency = {average_xh_excite_neg_energy-average_zpe} +/-'
          f' {np.sqrt(std_zpe**2 + std_xh_excite_neg_energy**2)}')
    print(f'average XH pos frequency = {average_xh_excite_pos_energy-average_zpe} +/- '
          f'{np.sqrt(std_zpe**2 + std_xh_excite_pos_energy**2)}')

    average_xh_excite_energy = np.average(np.array([average_xh_excite_pos_energy, average_xh_excite_neg_energy]))
    std_xh_excite_energy = np.sqrt(std_xh_excite_pos_energy**2 + std_xh_excite_neg_energy**2)
    print(f'average XH frequency = {average_xh_excite_energy-average_zpe} +/- '
          f'{np.sqrt(std_zpe**2 + std_xh_excite_energy**2)}')
    # au_to_Debye = 1/0.3934303
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num*me*1000)
    m_H = 1.007825 / (Avo_num*me*1000)
    m_OH = (m_H*m_O)/(m_H+m_O)
    omega_asym = 3815.044564/har2wave
    # mw = m_OH*omega_asym
    # conv_fac = 4.702e-7
    # km_mol = 5.33e6

    ground_coords = np.reshape(ground_coords, (len(ground_calcs), 27*walkers, 5, 3))
    ground_coords = np.hstack((ground_coords, ground_coords[:, :, [0, 3, 4, 1, 2]]))
    ground_weights = np.reshape(ground_weights, (len(ground_calcs), 27*walkers))
    ground_weights = np.hstack((ground_weights, ground_weights))
    ground_dips = np.zeros((len(ground_calcs), 27*walkers*2, 3))
    for i in range(len(ground_calcs)):
        eck = EckartsSpinz(ref, ground_coords[i], mass, planar=True)
        ground_coords[i] = np.ma.masked_invalid(eck.get_rotated_coords())
        ground_dips[i] = dip(ground_coords[i])

    print('ground wvfn dipoles calculated')

    np.savez(f'ground_{filename}_eck_dips', coords=ground_coords,
             weights=ground_weights, dips=ground_dips, a_zpe=average_zpe, std_zpe=std_zpe)

    excite_neg_coords = np.reshape(excite_neg_coords, (len(asm_left), 27*walkers, 5, 3))
    excite_neg_coords = np.hstack((excite_neg_coords, excite_neg_coords[:, :, [0, 3, 4, 1, 2]]))
    excite_neg_weights = np.reshape(excite_neg_weights, (len(asm_left), 27*walkers))
    excite_neg_weights = np.hstack((excite_neg_weights, excite_neg_weights))
    excite_neg_dips = np.zeros((len(asm_left), 27*walkers*2, 3))
    for i in range(len(asm_left)):
        eck = EckartsSpinz(ref, excite_neg_coords[i], mass, planar=True)
        excite_neg_coords[i] = np.ma.masked_invalid(eck.get_rotated_coords())
        excite_neg_dips[i] = dip(excite_neg_coords[i])

    print('asym neg dipoles calculated')

    np.savez(f'asym_left_{filename}_eck_dips', coords=excite_neg_coords,
             weights=excite_neg_weights, dips=excite_neg_dips, a_zpe=average_excite_neg_energy,
             std_zpe=std_excite_neg_energy)

    excite_pos_coords = np.reshape(excite_pos_coords, (len(asm_right), 27*walkers, 5, 3))
    excite_pos_coords = np.hstack((excite_pos_coords, excite_pos_coords[:, :, [0, 3, 4, 1, 2]]))
    excite_pos_weights = np.reshape(excite_pos_weights, (len(asm_right), 27*walkers))
    excite_pos_weights = np.hstack((excite_pos_weights, excite_pos_weights))
    excite_pos_dips = np.zeros((len(asm_right), 27*walkers*2, 3))
    for i in range(len(asm_right)):
        eck = EckartsSpinz(ref, excite_pos_coords[i], mass, planar=True)
        excite_pos_coords[i] = np.ma.masked_invalid(eck.get_rotated_coords())
        excite_pos_dips[i] = dip(excite_pos_coords[i])

    print('asym pos dipoles calculated')
    np.savez(f'asym_right_{filename}_eck_dips', coords=excite_pos_coords,
             weights=excite_pos_weights, dips=excite_pos_dips, a_zpe=average_excite_pos_energy,
             std_zpe=std_excite_pos_energy)

    xh_excite_neg_coords = np.reshape(xh_excite_neg_coords, (len(xh_left), 27*walkers, 5, 3))
    xh_excite_neg_coords = np.hstack((xh_excite_neg_coords, xh_excite_neg_coords[:, :, [0, 3, 4, 1, 2]]))
    xh_excite_neg_weights = np.reshape(xh_excite_neg_weights, (len(xh_left), 27*walkers))
    xh_excite_neg_weights = np.hstack((xh_excite_neg_weights, xh_excite_neg_weights))
    xh_excite_neg_dips = np.zeros((len(xh_left), 27*walkers*2, 3))
    for i in range(len(xh_left)):
        eck = EckartsSpinz(ref, xh_excite_neg_coords[i], mass, planar=True)
        xh_excite_neg_coords[i] = np.ma.masked_invalid(eck.get_rotated_coords())
        xh_excite_neg_dips[i] = dip(xh_excite_neg_coords[i])

    print('XH neg dipoles calculated')
    np.savez(f'XH_left_{filename}_eck_dips', coords=xh_excite_neg_coords,
             weights=xh_excite_neg_weights, dips=xh_excite_neg_dips, a_zpe=average_xh_excite_neg_energy,
             std_zpe=std_xh_excite_neg_energy)

    xh_excite_pos_coords = np.reshape(xh_excite_pos_coords, (len(xh_right), 27*walkers, 5, 3))
    xh_excite_pos_coords = np.hstack((xh_excite_pos_coords, xh_excite_pos_coords[:, :, [0, 3, 4, 1, 2]]))
    xh_excite_pos_weights = np.reshape(xh_excite_pos_weights, (len(xh_right), 27*walkers))
    xh_excite_pos_weights = np.hstack((xh_excite_pos_weights, xh_excite_pos_weights))
    xh_excite_pos_dips = np.zeros((len(xh_right), 27*walkers*2, 3))
    for i in range(len(xh_right)):
        eck = EckartsSpinz(ref, xh_excite_pos_coords[i], mass, planar=True)
        xh_excite_pos_coords[i] = np.ma.masked_invalid(eck.get_rotated_coords())
        xh_excite_pos_dips[i] = dip(xh_excite_pos_coords[i])

    print('XH pos dipoles calculated')
    np.savez(f'XH_right_{filename}_eck_dips', coords=xh_excite_pos_coords,
             weights=xh_excite_pos_weights, dips=xh_excite_pos_dips, a_zpe=average_xh_excite_pos_energy,
             std_zpe=std_xh_excite_pos_energy)

    # binzz = 100
    grca = len(ground_calcs)
    # term1 = np.zeros((grca, 3))
    # term1_vec = np.zeros((grca, 3))
    # term1_ov = np.zeros(grca)
    # term1_dis = np.zeros(grca)
    # term1_dis_a_w_xh = np.zeros(grca)
    # term1_dis_xh_w_a = np.zeros(grca)
    # xh_term1 = np.zeros((grca, 3))
    # xh_term1_vec = np.zeros((grca, 3))
    # xh_term1_ov = np.zeros(grca)
    # xh_term1_dis = np.zeros(grca)
    dists1 = np.zeros((grca, len(ground_coords[0]), 5))
    frac = np.zeros((grca, len(ground_coords[0])))
    frac2 = np.zeros((grca, len(ground_coords[0])))
    for i in range(grca):
        frac[i] = get_da_psi(ground_coords[i], 'a')/get_da_psi(ground_coords[i], None)
        frac2[i] = get_da_psi(ground_coords[i], 'sp')/get_da_psi(ground_coords[i], None)
        # term1_ov[i] = np.dot(ground_weights[i], frac[i])/np.sum(ground_weights[i])
        # xh_term1_ov[i] = np.dot(ground_weights[i], frac2[i])/np.sum(ground_weights[i])
        dists = all_dists(ground_coords[i])
        dists1[i] = dists
        # term1_dis[i] = np.dot(ground_weights[i], frac[i]*dists[:, 0])/np.sum(ground_weights[i])
        # xh_term1_dis[i] = np.dot(ground_weights[i], frac2[i]*dists[:, -1])/np.sum(ground_weights[i])
        # term1_dis_a_w_xh[i] = np.dot(ground_weights[i], frac2[i]*dists[:, 0])/np.sum(ground_weights[i])
        # term1_dis_xh_w_a[i] = np.dot(ground_weights[i], frac[i]*dists[:, -1])/np.sum(ground_weights[i])
        # aang1 = np.rad2deg(angle1(ground_coords[i]))
        # aang2 = np.rad2deg(angle2(ground_coords[i]))
        # plt.hist2d(dists[:, 0], aang1 + aang2, bins=binzz, weights=ground_weights[i])
        # plt.xlabel(r'$\rm{a}$')
        # plt.ylabel(r'$\rm{\Theta_1 + \Theta_2}$')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f'f0_theta12_sum_vs_a_{i+1}')
        # plt.close()
        # plt.hist2d(dists[:, 0], aang1 - aang2, bins=binzz, weights=ground_weights[i])
        # plt.xlabel(r'$\rm{a}$')
        # plt.ylabel(r'$\rm{\Theta_1 - \Theta_2}$')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f'f0_theta12_diff_vs_a_{i + 1}')
        # plt.close()
        # for j in range(3):
        #     term1[i, j] = np.dot(ground_weights[i], frac[i]*ground_dips[i, :, j]*au_to_Debye)/np.sum(ground_weights[i])
        #     term1_vec[i, j] = np.dot(ground_weights[i], frac[i] * ((ground_coords[i, :, 2, j] - ground_coords[i, :, 1, j]) +
        #                                                          (ground_coords[i, :, 4, j] - ground_coords[i, :, 3, j]))) \
        #                       / np.sum(ground_weights[i])
        #     xh_term1[i, j] = np.dot(ground_weights[i], frac2[i] * ground_dips[i, :, j] * au_to_Debye) / np.sum(ground_weights[i])
        #     mid = (ground_coords[i, :, 3, j] - ground_coords[i, :, 1, j]) / 2
        #     xh_term1_vec[i, j] = np.dot(ground_weights[i], frac2[i] * (mid - ground_coords[i, :, 0, j])) \
        #                          / np.sum(ground_weights[i])

    np.savez(f'ground_{filename}_wvfn_fractions', frac1=frac, frac2=frac2, dist=dists1)
    # avg_term1_vec = np.average(term1_vec, axis=0)
    # std_term1_vec = np.std(term1_vec, axis=0)
    # avg_term1_o = np.average(term1_ov)
    # std_term1_o = np.std(term1_ov)
    # avg_term1_d = np.average(term1_dis)
    # std_term1_d = np.std(term1_dis)
    # avg_term1_d_a_w_xh = np.average(term1_dis_a_w_xh)
    # std_term1_d_a_w_xh = np.std(term1_dis_a_w_xh)
    # avg_term1_d_xh_w_a = np.average(term1_dis_xh_w_a)
    # std_term1_d_xh_w_a = np.std(term1_dis_xh_w_a)
    # avg_xh_term1_v = np.average(xh_term1_vec, axis=0)
    # std_xh_term1_v = np.std(xh_term1_vec, axis=0)
    # avg_xh_term1_o = np.average(xh_term1_ov)
    # std_xh_term1_o = np.std(xh_term1_ov)
    # avg_xh_term1_d = np.average(xh_term1_dis)
    # std_xh_term1_d = np.std(xh_term1_dis)
    # print(f'average term1 asym components = {np.average(term1, axis=0)}')
    # print(f'std term1 asym components = {np.std(term1, axis=0)}')
    # print(f'average term1 XH components = {np.average(xh_term1, axis=0)}')
    # print(f'std term1 XH components = {np.std(xh_term1, axis=0)}')
    # xh_term1 = np.linalg.norm(xh_term1, axis=-1)
    # term1 = np.linalg.norm(term1, axis=-1)
    # std_xh_term1 = np.std(xh_term1)
    # std_term1 = np.std(term1)
    # term1 = np.average(term1)
    # xh_term1 = np.average(xh_term1)
    # print(f'term1 asym = {term1} +/- {std_term1}')
    # print(f'term1 xh = {xh_term1} +/- {std_xh_term1}')
    #
    # freq = average_excite_energy-average_zpe
    # freq2 = average_xh_excite_energy - average_zpe
    # std_freq2 = np.sqrt(std_zpe**2 + std_xh_excite_energy**2)
    # std_freq = np.sqrt(std_zpe**2 + std_excite_energy**2)
    # std_term1_sq = term1**2*np.sqrt((std_term1/term1)**2 + (std_term1/term1)**2)
    # std_term1_sq_freq = term1**2*freq*np.sqrt((std_term1_sq/term1**2)**2 + (std_freq/freq)**2)
    #
    # std_term1_xh_sq = xh_term1**2*np.sqrt((std_xh_term1/xh_term1)**2 + (std_xh_term1/xh_term1)**2)
    # std_term1_xh_freq = xh_term1**2*freq2*np.sqrt((std_term1_xh_sq/xh_term1**2)**2 + (std_freq2/freq2)**2)
    # conversion = conv_fac*km_mol
    # print(f'asym term1 intensity = {conversion*term1**2*freq} +/- {std_term1_sq_freq*conversion}')

    excite = len(asm_right)
    # term2 = np.zeros((excite, 3))
    # term2_vec = np.zeros((excite, 3))
    # term2_ov = np.zeros(excite)
    # xh_term2 = np.zeros((excite, 3))
    # xh_term2_vec = np.zeros((excite, 3))
    # xh_term2_ov = np.zeros(excite)
    # term2_dis = np.zeros(excite)
    # term2_dis_a_w_xh = np.zeros(excite)
    # term2_dis_xh_w_a = np.zeros(excite)
    # xh_term2_dis = np.zeros(excite)
    # combine_dips = np.zeros((excite, 27*walkers*4, 3))
    # combine_weights = np.zeros((excite, 27*walkers*4))
    # xh_combine_dips = np.zeros((excite, 27*walkers*4, 3))
    # xh_combine_weights = np.zeros((excite, 27*walkers*4))
    dists1 = np.zeros((grca, len(ground_coords[0])*2, 5))
    dists2 = np.zeros((grca, len(ground_coords[0])*2, 5))
    frac = np.zeros((excite, 27*walkers*4))
    frac2 = np.zeros((excite, 27*walkers*4))
    for i in range(excite):
        combine_coords = np.vstack((excite_neg_coords[i], excite_pos_coords[i]))
        xh_combine_coords = np.vstack((xh_excite_neg_coords[i], xh_excite_pos_coords[i]))
        # combine_weights[i] = np.hstack((excite_neg_weights[i], excite_pos_weights[i]))
        # combine_dips[i] = np.vstack((excite_neg_dips[i], excite_pos_dips[i]))
        # xh_combine_weights[i] = np.hstack((xh_excite_neg_weights[i], xh_excite_pos_weights[i]))
        # xh_combine_dips[i] = np.vstack((xh_excite_neg_dips[i], xh_excite_pos_dips[i]))
        H0 = get_da_psi(combine_coords, None)
        H0x = get_da_psi(xh_combine_coords, None)
        H1 = get_da_psi(combine_coords, 'a')
        H2 = get_da_psi(xh_combine_coords, 'sp')
        frac[i] = H0/H1
        frac2[i] = H0x/H2
        # term2_ov[i] = np.dot(combine_weights[i], frac[i])/np.sum(combine_weights[i])
        # xh_term2_ov[i] = np.dot(xh_combine_weights[i], frac2[i])/np.sum(xh_combine_weights[i])
        dists = all_dists(combine_coords)
        dists1[i] = dists
        # term2_dis[i] = np.average(frac[i]*dists[:, 0], weights=combine_weights[i])
        # term2_dis_xh_w_a[i] = np.average(frac[i]*dists[:, -1], weights=combine_weights[i])
        dists = all_dists(xh_combine_coords)
        dists2[i] = dists
        # term2_dis_a_w_xh[i] = np.average(frac2[i]*dists[:, 0], weights=combine_weights[i])
        # xh_term2_dis[i] = np.dot(xh_combine_weights[i], frac2[i]*dists[:, -1])/np.sum(xh_combine_weights[i])
        # aang1 = np.rad2deg(angle1(combine_coords))
        # aang2 = np.rad2deg(angle2(combine_coords))
        # plt.hist2d(dists1[i, :, 0], aang1 + aang2, bins=binzz, weights=combine_weights[i])
        # plt.xlabel(r'$\rm{a}$')
        # plt.ylabel(r'$\rm{\Theta_1 + \Theta_2}$')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f'f1_theta12_sum_vs_a_{i + 1}')
        # plt.close()
        # plt.hist2d(dists1[i, :, 0], aang1 - aang2, bins=binzz, weights=combine_weights[i])
        # plt.xlabel(r'$\rm{a}$')
        # plt.ylabel(r'$\rm{\Theta_1 - \Theta_2}$')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(f'f1_theta12_diff_vs_a_{i + 1}')
        # plt.close()
        # for j in range(3):
        #     term2[i, j] = np.dot(combine_weights[i], frac[i] * combine_dips[i, :, j] * au_to_Debye) \
        #                   / np.sum(combine_weights[i])
        #     term2_vec[i, j] = np.dot(combine_weights[i], frac[i] * ((combine_coords[:, 2, j] - combine_coords[:, 1, j]) +
        #                                                          (combine_coords[:, 4, j] - combine_coords[:, 3, j]))) \
        #                       / np.sum(combine_weights[i])
        #     xh_term2[i, j] = np.dot(xh_combine_weights[i], frac2[i]*xh_combine_dips[i, :, j] * au_to_Debye) \
        #                      /np.sum(xh_combine_weights[i])
        #     mid = (xh_combine_coords[:, 3, j] - xh_combine_coords[:, 1, j])/2
        #     xh_term2_vec[i, j] = np.dot(xh_combine_weights[i], frac2[i] * (mid - xh_combine_coords[:, 0, j])) \
        #                          / np.sum(xh_combine_weights[i])

    np.savez(f'excited_{filename}_wvfn_fractions', frac1=frac, frac2=frac2, dist1=dists1, dist2=dists2)
    # avg_term2_vec = np.average(term2_vec, axis=0)
    # std_term2_vec = np.std(term2_vec, axis=0)
    # avg_term2_o = np.average(term2_ov)
    # std_term2_o = np.std(term2_ov)
    # avg_term2_d = np.average(term2_dis)
    # std_term2_d = np.std(term2_dis)
    # avg_term2_d_a_w_xh = np.average(term2_dis_a_w_xh)
    # std_term2_d_a_w_xh = np.std(term2_dis_a_w_xh)
    # avg_term2_d_xh_w_a = np.average(term2_dis_xh_w_a)
    # std_term2_d_xh_w_a = np.std(term2_dis_xh_w_a)
    # avg_xh_term2_v = np.average(xh_term2_vec, axis=0)
    # std_xh_term2_v = np.std(xh_term2_vec, axis=0)
    # avg_xh_term2_o = np.average(xh_term2_ov)
    # std_xh_term2_o = np.std(xh_term2_ov)
    # avg_xh_term2_d = np.average(xh_term2_dis)
    # std_xh_term2_d = np.std(xh_term2_dis)
    # print(f'average term2 asym components = {np.average(term2, axis=0)}')
    # print(f'std term2 asym components = {np.std(term2, axis=0)}')
    # print(f'average term2 XH components = {np.average(xh_term2, axis=0)}')
    # print(f'std term2 XH components = {np.std(xh_term2, axis=0)}')
    #
    # term2 = np.linalg.norm(term2, axis=-1)
    # xh_term2 = np.linalg.norm(xh_term2, axis=-1)
    #
    # std_term2 = np.std(term2)
    # xh_std_term2 = np.std(xh_term2)
    # term2 = np.average(term2)
    # xh_term2 = np.average(xh_term2)
    # print(f'term2 asym = {term2} +/- {std_term2}')
    # print(f'term2 xh = {xh_term2} +/- {xh_std_term2}')
    #
    # std_term2_xh_sq = xh_term2**2*np.sqrt((xh_std_term2/xh_term2)**2 + (xh_std_term2/xh_term2)**2)
    # std_term2_sq = term2**2*np.sqrt((std_term2/term2)**2 + (std_term2/term2)**2)
    # std_term2_sq_freq = term2**2*freq*np.sqrt((std_term2_sq/term2**2)**2 + (std_freq/freq)**2)
    # xh_std_term2_sq_freq = xh_term2**2*freq2*np.sqrt((std_term2_xh_sq/xh_term2**2)**2 + (std_freq2/freq2)**2)
    # print(f'term2 asym intensity = {conversion*term2**2*freq} +/- {std_term2_sq_freq*conversion}')
    #
    # # term3 = 0.14379963852224506
    # term3 = 0.0424328790886425
    # term3_mod = 0.057139489481379306
    # xh_term3 = 1.1154257246577732
    # xh_term3_mod = 1.0723059556104413
    # std_term3 = 0.0
    # std_term3_sq = 0.0
    # std_term3_sq_freq = term3**2*freq*np.sqrt((std_term3_sq/term3**2)**2 + (std_freq/freq)**2)
    # xh_std_term3_sq_freq = xh_term3**2*freq2*np.sqrt((std_freq2/freq2)**2)
    # xh_std_term3_mod_sq_freq = xh_term3_mod**2*freq2*np.sqrt((std_freq2/freq2)**2)
    # std_term3_mod_sq_freq = term3**2*freq*np.sqrt((std_freq/freq)**2)
    # print(f'term3 asym intensity = {term3**2*freq*conversion} +/- '
    #       f'{std_term3_sq_freq*conversion}')
    # print(f'term3 asym prime intensity = {term3_mod**2*freq*conversion} +/- '
    #       f'{std_term3_mod_sq_freq*conversion}')
    #
    # full_error = np.sqrt(std_term1**2 + std_term2**2 + std_term3**2)
    # full_error_xh = np.sqrt(std_xh_term1**2 + xh_std_term2**2)
    # dipole_xh = xh_term1 + xh_term2 - xh_term3_mod
    # dipole = term1 + term2 - term3_mod
    # full_error_sq = dipole**2*np.sqrt((full_error/dipole)**2 + (full_error/dipole)**2)
    # full_error_sq_xh = dipole_xh**2*np.sqrt((full_error_xh/dipole_xh)**2 + (full_error_xh/dipole_xh)**2)
    # full_error = dipole**2*freq*np.sqrt((full_error_sq/dipole**2)**2 + (std_freq/freq)**2)
    # full_error_xh = dipole_xh**2*freq2*np.sqrt((full_error_sq_xh/dipole_xh**2)**2 + (std_freq2/freq2)**2)
    # print(f'full asym prime intensity = {dipole**2*freq*conversion} +/- '
    #       f'{full_error*conversion}')
    #
    # print(f'term1 XH intensity = {conversion*xh_term1**2*freq2} +/- '
    #       f'{std_term1_xh_freq*conversion}')
    # print(f'term2 XH intensity = {conversion*xh_term2**2*freq2} +/- '
    #       f'{xh_std_term2_sq_freq*conversion}')
    # print(f'term3 XH intensity = {xh_term3**2*freq2*conversion} +/- '
    #       f'{xh_std_term3_sq_freq*conversion}')
    # print(f'term3 z prime intensity = {xh_term3_mod**2*freq2*conversion} +/- '
    #       f'{xh_std_term3_mod_sq_freq*conversion}')
    # print(f'full z prime intensity = {dipole_xh**2*freq2*conversion} +/- '
    #       f'{full_error_xh*conversion}')
    #
    # print(f'term1 a overlap = {avg_term1_o} {std_term1_o}')
    # print(f'term2 a overlap = {avg_term2_o} {std_term2_o}')
    #
    # print(f'term1 xh overlap = {avg_xh_term1_o} {std_xh_term1_o}')
    # print(f'term2 xh overlap = {avg_xh_term2_o} {std_xh_term2_o}')
    #
    # print(f'term1 a dis = {avg_term1_d} {std_term1_d}')
    # print(f'term2 a dis = {avg_term2_d} {std_term2_d}')
    #
    # print(f'term1 xh dis = {avg_xh_term1_d} {std_xh_term1_d}')
    # print(f'term2 xh dis = {avg_xh_term2_d} {std_xh_term2_d}')
    #
    # print(f'term1 a vec = {avg_term1_vec} {std_term1_vec}')
    # print(f'term2 a vec = {avg_term2_vec} {std_term2_vec}')
    #
    # print(f'term1 xh vec = {avg_xh_term1_v} {std_xh_term1_v}')
    # print(f'term2 xh vec = {avg_xh_term2_v} {std_xh_term2_v}')
    #
    # print(f'term1 a dis with xh excite = {avg_term1_d_a_w_xh} {std_term1_d_a_w_xh}')
    # print(f'term2 a dis with xh excite = {avg_term2_d_a_w_xh} {std_term2_d_a_w_xh}')
    #
    # print(f'term1 xh dis with a excite = {avg_term1_d_xh_w_a} {std_term1_d_xh_w_a}')
    # print(f'term2 xh dis with a excite = {avg_term2_d_xh_w_a} {std_term2_d_xh_w_a}')


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
