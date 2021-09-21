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

    walkers = 60000

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

    ground_coords = np.reshape(ground_coords, (len(ground_calcs), 27*walkers, 5, 3))
    ground_coords = np.hstack((ground_coords, ground_coords[:, :, [0, 3, 4, 1, 2]]))
    ground_coords = np.hstack((ground_coords, ground_coords*-1))
    ground_weights = np.reshape(ground_weights, (len(ground_calcs), 27*walkers))
    ground_weights = np.hstack((ground_weights, ground_weights))
    ground_weights = np.hstack((ground_weights, ground_weights))
    ground_dips = np.zeros((len(ground_calcs), 27*walkers*4, 3))
    for i in range(len(ground_calcs)):
        eck = EckartsSpinz(ref, ground_coords[i], mass, planar=True)
        ground_coords[i] = np.ma.masked_invalid(eck.get_rotated_coords())
        ground_dips[i] = dip(ground_coords[i])

    print('ground wvfn dipoles calculated')

    np.savez(f'ground_{filename}_eck_dips', coords=ground_coords,
             weights=ground_weights, dips=ground_dips, a_zpe=average_zpe, std_zpe=std_zpe)

    excite_neg_coords = np.reshape(excite_neg_coords, (len(asm_left), 27*walkers, 5, 3))
    excite_neg_coords = np.hstack((excite_neg_coords, excite_neg_coords[:, :, [0, 3, 4, 1, 2]]))
    excite_neg_coords = np.hstack((excite_neg_coords, excite_neg_coords*-1))
    excite_neg_weights = np.reshape(excite_neg_weights, (len(asm_left), 27*walkers))
    excite_neg_weights = np.hstack((excite_neg_weights, excite_neg_weights))
    excite_neg_weights = np.hstack((excite_neg_weights, excite_neg_weights))
    excite_neg_dips = np.zeros((len(asm_left), 27*walkers*4, 3))
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
    excite_pos_coords = np.hstack((excite_pos_coords, excite_pos_coords*-1))
    excite_pos_weights = np.reshape(excite_pos_weights, (len(asm_right), 27*walkers))
    excite_pos_weights = np.hstack((excite_pos_weights, excite_pos_weights))
    excite_pos_weights = np.hstack((excite_pos_weights, excite_pos_weights))
    excite_pos_dips = np.zeros((len(asm_right), 27*walkers*4, 3))
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
    xh_excite_neg_coords = np.hstack((xh_excite_neg_coords, xh_excite_neg_coords*-1))
    xh_excite_neg_weights = np.reshape(xh_excite_neg_weights, (len(xh_left), 27*walkers))
    xh_excite_neg_weights = np.hstack((xh_excite_neg_weights, xh_excite_neg_weights))
    xh_excite_neg_weights = np.hstack((xh_excite_neg_weights, xh_excite_neg_weights))
    xh_excite_neg_dips = np.zeros((len(xh_left), 27*walkers*4, 3))
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
    xh_excite_pos_coords = np.hstack((xh_excite_pos_coords, xh_excite_pos_coords*-1))
    xh_excite_pos_weights = np.reshape(xh_excite_pos_weights, (len(xh_right), 27*walkers))
    xh_excite_pos_weights = np.hstack((xh_excite_pos_weights, xh_excite_pos_weights))
    xh_excite_pos_weights = np.hstack((xh_excite_pos_weights, xh_excite_pos_weights))
    xh_excite_pos_dips = np.zeros((len(xh_right), 27*walkers*4, 3))
    for i in range(len(xh_right)):
        eck = EckartsSpinz(ref, xh_excite_pos_coords[i], mass, planar=True)
        xh_excite_pos_coords[i] = np.ma.masked_invalid(eck.get_rotated_coords())
        xh_excite_pos_dips[i] = dip(xh_excite_pos_coords[i])

    print('XH pos dipoles calculated')
    np.savez(f'XH_right_{filename}_eck_dips', coords=xh_excite_pos_coords,
             weights=xh_excite_pos_weights, dips=xh_excite_pos_dips, a_zpe=average_xh_excite_pos_energy,
             std_zpe=std_xh_excite_pos_energy)

    grca = len(ground_calcs)
    dists1 = np.zeros((grca, len(ground_coords[0]), 5))
    frac = np.zeros((grca, len(ground_coords[0])))
    frac2 = np.zeros((grca, len(ground_coords[0])))
    for i in range(grca):
        frac[i] = get_da_psi(ground_coords[i], 'a', wvfn_type)/get_da_psi(ground_coords[i], None, wvfn_type)
        frac2[i] = get_da_psi(ground_coords[i], 'sp', wvfn_type)/get_da_psi(ground_coords[i], None, wvfn_type)
        dists = all_dists(ground_coords[i])
        dists1[i] = dists

    print('ground fracs calculated')
    np.savez(f'ground_{filename}_wvfn_fractions', frac1=frac, frac2=frac2, dist=dists1)

    excite = len(asm_right)
    dists1 = np.zeros((excite, len(ground_coords[0])*2, 5))
    dists2 = np.zeros((excite, len(ground_coords[0])*2, 5))
    frac = np.zeros((excite, len(ground_coords[0])*2))
    frac2 = np.zeros((excite, len(ground_coords[0])*2))
    for i in range(excite):
        combine_coords = np.vstack((excite_neg_coords[i], excite_pos_coords[i]))
        xh_combine_coords = np.vstack((xh_excite_neg_coords[i], xh_excite_pos_coords[i]))
        H0 = get_da_psi(combine_coords, None, wvfn_type)
        H0x = get_da_psi(xh_combine_coords, None, wvfn_type)
        H1 = get_da_psi(combine_coords, 'a', wvfn_type)
        H2 = get_da_psi(xh_combine_coords, 'sp', wvfn_type)
        frac[i] = H0/H1
        frac2[i] = H0x/H2
        dists = all_dists(combine_coords)
        dists1[i] = dists
        dists = all_dists(xh_combine_coords)
        dists2[i] = dists

    print('excited state fracs calculated')
    np.savez(f'excited_{filename}_wvfn_fractions', frac1=frac, frac2=frac2, dist1=dists1, dist2=dists2)


def a_prime(a, z):
    return -0.60594644269321474*z + 42.200232187251913*a


def z_prime(a, z):
    return 41.561937672470521*z + 1.0206303697659393*a


def interp(x, y, poiuy):
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = poiuy(x[i], y[i])
    return out


def get_da_psi(coords, excite, wvfn_type):
    from itertools import repeat
    coordz = np.array_split(coords, mp.cpu_count()-1)
    psi = pool.starmap(psi_t, zip(coordz, repeat(excite), repeat(wvfn_type)))
    psi = np.concatenate(psi)
    return psi


def psi_t(coords, excite, wvfn_type):
    har2wave = 219474.6
    me = 9.10938356e-31
    Avo_num = 6.0221367e23
    m_O = 15.994915 / (Avo_num*me*1000)
    m_H = 1.007825 / (Avo_num*me*1000)
    m_OH = (m_H*m_O)/(m_H+m_O)
    omega_asym = 3815.044564/har2wave

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
    v2 = rxh/np.broadcast_to(rxh_dist[:, None], (len(rxh_dist), 3))
    v1_new = np.reshape(v1, (v1.shape[0], 1, v1.shape[1]))
    v2_new = np.reshape(v2, (v2.shape[0], v2.shape[1], 1))
    aang = np.arccos(np.matmul(v1_new, v2_new).squeeze())
    return aang


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
