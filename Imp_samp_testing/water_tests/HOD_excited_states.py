import copy
from scipy import interpolate
from Coordinerds.CoordinateSystems import *
import Water_monomer_pot_fns as wm
import multiprocessing as mp

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_OD = (m_D*m_O)/(m_D+m_O)
m_OH = (m_H*m_O)/(m_H+m_O)
dtau = 1
alpha = 1./(2.*dtau)
sigmaH = np.sqrt(dtau/m_H)
sigmaO = np.sqrt(dtau/m_O)
sigmaD = np.sqrt(dtau/m_D)
sigma = np.broadcast_to(np.array([sigmaO, sigmaH, sigmaH])[:, None], (3, 3))
omega_OD = 2832.531899782715
omega_OH = 3890.7865072878913

har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

mw_d = m_OD * omega_OD/har2wave
mw_h = m_OH * omega_OH/har2wave
# mw_d = mw_h

coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                           [0.957840000000000, 0.000000000000000, 0.000000000000000],
                           [-0.23995350000000, 0.927297000000000, 0.000000000000000]])*ang2bohr


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers, initial_struct, excite, initial_shifts):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([initial_struct]*walkers)
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.excite = excite
        self.shift = initial_shifts


def psi_t(coords, excite, shift):
    psi = np.zeros((len(coords), 3))
    dists = oh_dists(coords)
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    req = [r1, r2]
    dists = dists - req
    if excite == 'both stretches' or excite == 'all':
        dists = dists - shift[:2]
        psi[:, 0] = (mw_d / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_d * dists[:, 0] ** 2)) * (2 * mw_d) ** (1 / 2) * dists[:, 0]
        psi[:, 1] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 1] ** 2)) * (2 * mw_h) ** (1 / 2) * dists[:, 1]
    elif excite == 'od' or excite == 'od and ang':
        dists[:, 0] = dists[:, 0] - shift[0]
        psi[:, 0] = (mw_d / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_d * dists[:, 0] ** 2)) * (2 * mw_d) ** (
                    1 / 2) * dists[:, 0]
        psi[:, 1] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 1] ** 2))
    elif excite == 'oh' or excite == 'oh and ang':
        dists[:, 1] = dists[:, 1] - shift[1]
        psi[:, 0] = (mw_d / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_d * dists[:, 0] ** 2))
        psi[:, 1] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 1] ** 2)) * (2 * mw_h) ** (
                    1 / 2) * dists[:, 1]
    else:
        psi[:, 0] = (mw_d / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_d * dists[:, 0] ** 2))
        psi[:, 1] = (mw_h / np.pi) ** (1. / 4.) * np.exp(-(1. / 2. * mw_h * dists[:, 1] ** 2))
    psi[:, 2] = angle_function(coords, excite, shift)
    return psi


def dpsidx(coords, excite, shift):
    dists = oh_dists(coords)
    drx = drdx(coords, dists, shift)
    dthet = dthetadx(coords, shift)
    dr = np.concatenate((drx, dthet[..., None]), axis=-1)
    collect = dpsidrtheta(coords, excite, dists, shift)
    return np.matmul(dr, collect[:, None, :, None]).squeeze()


def d2psidx2(coords, excite, shift):
    import pyvibdmc as pv
    check = pv.ChainRuleHelper.dth_dx(coords, [[1, 0, 2]])

    dists = oh_dists(coords)
    drx = drdx(coords, dists, shift)
    dthet = dthetadx(coords, shift)
    dr1 = np.concatenate((drx, dthet[..., None]), axis=-1)
    drx2 = drdx2(coords, dists, shift)
    dthet2 = dthetadx2(coords, angle(coords), shift)
    dr2 = np.concatenate((drx2, dthet2[..., None]), axis=-1)
    first_dir = dpsidrtheta(coords, excite, dists, shift)
    second_dir = d2psidrtheta(coords, excite, dists, shift)
    part1 = np.matmul(dr2, first_dir[:, None, :, None]).squeeze()
    part2 = np.matmul(dr1**2, second_dir[:, None, :, None]).squeeze()
    part3 = np.matmul(dr1*dr1[..., [1, 2, 0]], first_dir[:, None, :, None]
                      *first_dir[:, None, [1, 2, 0], None]).squeeze()
    return part1+part2+2*part3


def dpsidrtheta(coords, excite, dists, shift):
    collect = np.zeros((len(coords), 3))
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    req = [r1, r2]
    dists = dists - req
    if excite == 'both stretches' or excite == 'all':
        dists = dists - shift[:2]
        collect[:, 0] = (1 - mw_d*dists[:, 0]**2)/dists[:, 0]
        collect[:, 1] = (1 - mw_h*dists[:, 1]**2)/dists[:, 1]
    elif excite == 'od' or excite == 'od and ang':
        dists[:, 0] = dists[:, 0] - shift[0]
        collect[:, 0] = (1 - mw_d*dists[:, 0]**2)/dists[:, 0]
        collect[:, 1] = -mw_h*dists[:, 1]
    elif excite == 'oh' or excite == 'oh and ang':
        dists[:, 1] = dists[:, 1] - shift[1]
        collect[:, 0] = -mw_d*dists[:, 0]
        collect[:, 1] = (1 - mw_h*dists[:, 1]**2)/dists[:, 1]
    else:
        collect[:, 0] = -mw_d*dists[:, 0]
        collect[:, 1] = -mw_h*dists[:, 1]
    collect[:, 2] = dangle(coords, excite, shift)
    return collect


def d2psidrtheta(coords, excite, dists, shift):
    collect = np.zeros((len(coords), 3))
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    req = [r1, r2]
    dists = dists - req
    if excite == 'both stretches' or excite == 'all':
        dists = dists - shift[:2]
        collect[:, 0] = mw_d*(mw_d*dists[:, 0]**2 - 3)
        collect[:, 1] = mw_h*(mw_h*dists[:, 1]**2 - 3)
    elif excite == 'od' or excite == 'od and ang':
        dists[:, 0] = dists[:, 0] - shift[0]
        collect[:, 0] = mw_d*(mw_d*dists[:, 0]**2 - 3)
        collect[:, 1] = mw_h**2*dists[:, 1]**2 - mw_h
    elif excite == 'oh' or excite == 'oh and ang':
        dists[:, 1] = dists[:, 1] - shift[1]
        collect[:, 0] = mw_d**2*dists[:, 0]**2 - mw_d
        collect[:, 1] = mw_h*(mw_h*dists[:, 1]**2 - 3)
    else:
        collect[:, 0] = mw_d**2*dists[:, 0]**2 - mw_d
        collect[:, 1] = mw_h**2*dists[:, 1]**2 - mw_h
    collect[:, 2] = d2angle(coords, excite, shift)
    return collect


def oh_dists(coords):
    bonds = [[1, 2], [1, 3]]
    cd1 = coords[:, tuple(x[0] for x in np.array(bonds) - 1)]
    cd2 = coords[:, tuple(x[1] for x in np.array(bonds) - 1)]
    dis = np.linalg.norm(cd2 - cd1, axis=2)
    return dis


def angle(coords):
    dists = oh_dists(coords)
    v1 = (coords[:, 1] - coords[:, 0]) / np.broadcast_to(dists[:, 0, None], (len(dists), 3))
    v2 = (coords[:, 2] - coords[:, 0]) / np.broadcast_to(dists[:, 1, None], (len(dists), 3))

    ang1 = np.arccos(np.matmul(v1[:, None, :], v2[..., None]).squeeze())

    return ang1.T


def angle_function(coords, excite, shift):
    angs = angle(coords)
    r1 = 0.9616036495623883*ang2bohr
    r2 = 0.9616119936423067*ang2bohr
    theta = np.deg2rad(104.1747712)
    muH = 1 / m_H
    muO = 1 / m_O
    muD = 1 / m_D
    G = gmat(muD, muH, muO, r1, r2, theta)
    freq = 1462.5810039828614
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
        angs = angs - shift[2]
        return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2) * (2*alpha) ** (1/2) * (angs-theta)
    else:
        return (alpha / np.pi) ** (1 / 4) * np.exp(-alpha * (angs - theta) ** 2 / 2)


def dangle(coords, excite, shift):
    angs = angle(coords)
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    theta = np.deg2rad(104.1747712)
    muH = 1 / m_H
    muO = 1 / m_O
    muD = 1 / m_D
    G = gmat(muD, muH, muO, r1, r2, theta)
    freq = 1462.5810039828614
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
        angs = angs - shift[2]
        return (1 - alpha * (angs-theta) ** 2) / (angs-theta)
    else:
        return -alpha*(angs-theta)


def d2angle(coords, excite, shift):
    angs = angle(coords)
    r1 = 0.9616036495623883 * ang2bohr
    r2 = 0.9616119936423067 * ang2bohr
    theta = np.deg2rad(104.1747712)
    muH = 1 / m_H
    muO = 1 / m_O
    muD = 1 / m_D
    G = gmat(muD, muH, muO, r1, r2, theta)
    freq = 1462.5810039828614
    freq /= har2wave
    alpha = freq / G
    if excite == 'ang' or excite == 'all' or excite == 'oh and ang' or excite == 'od and ang':
        angs = angs - shift[2]
        return alpha * (alpha * (angs-theta) ** 2 - 3)
    else:
        return alpha**2*(angs-theta)**2 - alpha


def gmat(mu1, mu2, mu3, r1, r2, ang):
    return mu1/r1**2 + mu2/r2**2 + mu3*(1/r1**2 + 1/r2**2 - 2*np.cos(ang)/(r1*r2))


def drdx(coords, dists, shift):
    # dists = oh_dists(coords)
    chain = np.zeros((len(coords), 3, 3, 2))
    dists = dists - shift[:2]
    for bond in range(2):
        chain[:, 0, :, bond] += ((coords[:, 0]-coords[:, bond+1])/dists[:, bond, None])
        chain[:, bond+1, :, bond] += ((coords[:, bond+1]-coords[:, 0])/dists[:, bond, None])
    return chain


def dthetadx(coords, shift):
    chain = np.zeros((len(coords), 3, 3, 4))
    dx = 1e-3  #Bohr
    coeffs = np.array([1/12, -2/3, 2/3, -1/12])/dx
    for atom in range(3):
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = angle(coords) - shift[2]
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = angle(coords) - shift[2]
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 2] = angle(coords) - shift[2]
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 3] = angle(coords) - shift[2]
            coords[:, atom, xyz] -= 2*dx
    return np.dot(chain, coeffs)


def drdx2(coords, dists, shift):
    # dists = oh_dists(coords)
    chain = np.zeros((len(coords), 3, 3, 2))
    dists = dists - shift[:2]
    for bond in range(2):
        chain[:, 0, :, bond] = (1./dists[:, bond, None] - (coords[:, 0]-coords[:, bond+1])**2/dists[:, bond, None]**3)
        chain[:, bond + 1, :, bond] = (1./dists[:, bond, None] - (coords[:, bond + 1] - coords[:, 0])**2 / dists[:, bond, None]**3)
    return chain


def dthetadx2(coords, angs, shift):
    chain = np.zeros((len(coords), 3, 3, 5))
    chain[:, :, :, 2] = np.broadcast_to(angs[..., None, None], (len(coords), 3, 3))
    dx = 1e-3
    coeffs = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])/(dx**2)
    for atom in range(3):
        for xyz in range(3):
            coords[:, atom, xyz] -= 2*dx
            chain[:, atom, xyz, 0] = angle(coords) - shift[2]
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 1] = angle(coords) - shift[2]
            coords[:, atom, xyz] += 2*dx
            chain[:, atom, xyz, 3] = angle(coords) - shift[2]
            coords[:, atom, xyz] += dx
            chain[:, atom, xyz, 4] = angle(coords) - shift[2]
            coords[:, atom, xyz] -= 2*dx
    return np.dot(chain, coeffs)



from Coordinerds.CoordinateSystems import *


def linear_combo_stretch_grid(r1, r2, coords):
    re = np.linalg.norm(coords[0]-coords[1])
    re2 = np.linalg.norm(coords[0]-coords[2])
    re = 0.9616036495623883 * ang2bohr
    re2 = 0.9616119936423067 * ang2bohr

    coords = np.array([coords] * 1)
    zmat = CoordinateSet(coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates,
                                                                        ordering=([[0, 0, 0, 0], [1, 0, 0, 0],
                                                                                   [2, 0, 1, 0]])).coords
    N = len(r1)
    zmat = np.array([zmat]*N).squeeze()
    zmat[:, 0, 1] = re + r1
    zmat[:, 1, 1] = re2 + r2
    new_coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
    return new_coords


molecule = np.load('monomer_coords.npy')


def local_kinetic_finite(Psi):
    dx = 1e-3
    d2psidx2 = ((Psi[:, 0] - 2. * Psi[:, 1] + Psi[:, 2]) / dx ** 2) / Psi[:, 1]
    kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * d2psidx2, axis=1), axis=1)
    return d2psidx2


def all_da_psi(coords, excite, shift):
    dx = 1e-3
    psi = np.zeros((len(coords), 3, 3, 3))
    psi[:, 1] = np.broadcast_to(np.prod(psi_t(coords, excite, shift), axis=-1)[:, None, None],
                                (len(coords), 3, 3))
    for atom in range(3):
        for xyz in range(3):
            coords[:, atom, xyz] -= dx
            psi[:, 0, atom, xyz] = np.prod(psi_t(coords, excite, shift), axis=-1)
            coords[:, atom, xyz] += 2*dx
            psi[:, 2, atom, xyz] = np.prod(psi_t(coords, excite, shift), axis=-1)
            coords[:, atom, xyz] -= dx
    return psi


anti = np.linspace(-0.75, 0.75, 200)
sym = np.zeros(200) + 0.02
A = 1/np.sqrt(2)*np.array([[-1, 1], [1, 1]])
eh = np.matmul(np.linalg.inv(A), np.vstack((anti, sym)))
r1 = eh[0]
r2 = eh[1]

grid = linear_combo_stretch_grid(r1, r2, molecule)

psi1 = psi_t(grid, 'oh', [0, 0, 0])

blah = d2psidx2(grid, 'oh', [0, 0, 0])*psi1

psi2 = psi_t(grid, 'od', [0, 0, 0])

blah2 = d2psidx2(grid, 'od', [0, 0, 0])*psi2

full_psi = np.prod(1/np.sqrt(2)*(psi2 - psi1), axis=-1)

test = (blah2 - blah)/np.sqrt(2)/full_psi[:, None, None]

psi = all_da_psi(grid, 'oh', [0, 0, 0])

blah_fd = local_kinetic_finite(psi)

psi = all_da_psi(grid, 'od', [0, 0, 0])

blah_fd2 = local_kinetic_finite(psi)

test_fd = (blah_fd2 - blah_fd)/np.sqrt(2)

water_coord = np.array([[0., 0., 0.],
                        [1.81005599, 0., 0.],
                        [-0.45344658, 1.75233806, 0.]
                            ]) * 1.01

blah = d2psidx2(np.array([water_coord]*1), 'od', [0, 0, 0])
hahahahahahaha = 7


def drift(coords, excite, shift):
    return 2*dpsidx(coords, excite, shift)


def metropolis(Fqx, Fqy, x, y, excite, shift):
    psi_1 = psi_t(x, excite, shift)
    psi_2 = psi_t(y, excite, shift)
    psi_ratio = np.prod((psi_2/psi_1)**2, axis=1)
    a = np.exp(1. / 2. * (Fqx + Fqy) * (sigma ** 2 / 4. * (Fqx - Fqy) - (y - x)))
    a = np.prod(np.prod(a, axis=1), axis=1) * psi_ratio
    remove = np.argwhere(psi_2 * psi_1 < 0)
    a[remove] = 0.
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx):
    Drift = sigma**2/2.*Fqx
    randomwalk = np.random.normal(0.0, sigma, size=(len(Psi.coords), sigma.shape[0], sigma.shape[1]))
    y = randomwalk + Drift + np.array(Psi.coords)
    Fqy = drift(y, Psi.excite, Psi.shift)
    a = metropolis(Fqx, Fqy, Psi.coords, y, Psi.excite, Psi.shift)
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    Fqx[accept] = Fqy[accept]
    acceptance = float(len(accept)/len(Psi.coords))*100.
    return Psi, Fqx, acceptance


def get_pot(coords):
    V = wm.PatrickShinglePotential(coords)
    return V


def Potential(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    return Psi


def local_kinetic(Psi):
    kin = -1. / 2. * np.sum(np.sum(sigma ** 2 / dtau * d2psidx2(Psi.coords, Psi.excite, Psi.shift), axis=1), axis=1)
    return kin


def E_loc(Psi):
    Psi.El = local_kinetic(Psi) + Psi.V
    return Psi


def E_ref_calc(Psi):
    P0 = sum(Psi.weights_i)
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/P0)
    return E_ref


def Weighting(Eref, Psi, DW, Fqx):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Eref) * dtau)
    threshold = 0.01
    max_thresh = 20
    death = np.argwhere(Psi.weights < threshold)
    for i in death:
        ind = np.argmax(Psi.weights)
        if DW is True:
            Biggo_num = int(Psi.walkers[ind])
            Psi.walkers[i[0]] = Biggo_num
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_pot = float(Psi.V[ind])
        Biggo_el = float(Psi.El[ind])
        Biggo_force = np.array(Fqx[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Fqx[i[0]] = Biggo_force

    death = np.argwhere(Psi.weights > max_thresh)
    for i in death:
        ind = np.argmin(Psi.weights)
        if DW is True:
            Biggo_num = float(Psi.walkers[i[0]])
            Psi.walkers[ind] = Biggo_num
        Biggo_weight = float(Psi.weights[i[0]])
        Biggo_pos = np.array(Psi.coords[i[0]])
        Biggo_pot = float(Psi.V[i[0]])
        Biggo_el = float(Psi.El[i[0]])
        Biggo_force = np.array(Fqx[i[0]])
        Psi.weights[i[0]] = Biggo_weight / 2.
        Psi.weights[ind] = Biggo_weight / 2.
        Psi.coords[ind] = Biggo_pos
        Psi.V[ind] = Biggo_pot
        Psi.El[ind] = Biggo_el
        Fqx[ind] = Biggo_force
    return Psi


def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < len(Psi.coords):
        d = np.append(d, 0.)
    return d


def run(N_0, time_steps, propagation, excite, initial_struct, initial_shifts, shift_rate):
    DW = False
    psi = Walkers(N_0, initial_struct, excite, initial_shifts)
    Fqx = drift(psi.coords, psi.excite, psi.shift)
    Psi, Fqx, acceptance = Kinetic(psi, Fqx)
    Psi = Potential(Psi)
    Psi = E_loc(Psi)
    time = np.array([])
    weights = np.array([])
    accept = np.array([])
    Eref_array = np.array([])
    Eref = E_ref_calc(Psi)
    Eref_array = np.append(Eref_array, Eref)
    new_psi = Weighting(Eref, Psi, DW, Fqx)
    time = np.append(time, 1)
    weights = np.append(weights, np.sum(new_psi.weights))
    accept =np.append(accept, acceptance)

    Psi_tau = 0
    shift = np.zeros((time_steps + 1, 3))
    shift[0] = Psi.shift
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)

        Psi, Fqx, acceptance = Kinetic(new_psi, Fqx)
        Psi = Potential(Psi)
        Psi = E_loc(Psi)
        shift[i + 1] = Psi.shift

        if i >= 5000:
            Psi.shift = Psi.shift + shift_rate

        if DW is False:
            prop = float(propagation)
        elif DW is True:
            prop -= 1.
            if Psi_tau == 0:
                Psi_tau = copy.deepcopy(Psi)
        new_psi = Weighting(Eref, Psi, DW, Fqx)

        Eref = E_ref_calc(new_psi)



        Eref_array = np.append(Eref_array, Eref)
        time = np.append(time, 2 + i)
        weights = np.append(weights, np.sum(new_psi.weights))
        accept = np.append(accept, acceptance)

        if i >= (time_steps - 1. - float(propagation)) and prop > 0.:
            DW = True
        elif i >= (time_steps - 1. - float(propagation)) and prop == 0.:
            d_values = descendants(new_psi)
    return Eref_array, weights, shift, d_values, Psi_tau.coords


pool = mp.Pool(mp.cpu_count()-1)

# coords = np.load('h2o_test_ground_coords.npy')
# #
# psi = Walkers(1000, coords_initial, None, [0, 0, 0])
# psi.coords = coords[0]
# psi = Potential(psi)
# psi = E_loc(psi)
# psit = psi_t(psi.coords, psi.excite, psi.shift)
# psit = psit[:, 0]*psit[:, 1]
#
# eref, weights, shift, d, coords = run(1000, 1000, 250, None, coords_initial, [0, 0, 0], [0, 0, 0])
# import matplotlib.pyplot as plt
# plt.plot(eref*har2wave)
# print(np.mean(eref[500:]))
# plt.show()
#
# blah = 4
#
#
# def cub_fit(x, *params):
#     a, b, c, d = params
#     return a*x**3 + b*x**2 + c*x + d
#
# import scipy.optimize
#
# # eref, weights, shift, d, coords = run(5000, 10000, 250, None, coords_initial*1.05, np.array([0.0259616, 0, 0]), np.array([-0.000000, 0, 0]))
# #
# # np.save('ground_state_coords_od', coords)
# # np.save('ground_state_weights_od', d)
#
# coords = np.load('ground_state_coords_od.npy')
# d = np.load('ground_state_weights_od.npy')
#
# psi1 = np.prod(psi_t(coords, 'od', np.array([0.0259616, 0, 0])), axis=1)
# psi0 = np.prod(psi_t(coords, None, np.array([0.0259616, 0, 0])), axis=1)
#
# dists = oh_dists(coords)[:, 0]
#
# import matplotlib.pyplot as plt
# amp, bins = np.histogram(dists, weights=d, bins=75)
# x = (bins[1:] + bins[:-1]) / 2.
#
# plt.plot(x, amp/np.linalg.norm(amp))
# plt.show()
#
# c = np.sum(d*dists*psi1/psi0)/np.sum(d)
#
# print(c)
#
# # a = np.load('od_stretch_long.npy')
# # b = np.load('od_stretch_short.npy')
# # params = [-0.28662516,  0.31322083, -0.13109515,  0.02856335]
# # fitted_params1, _ = scipy.optimize.curve_fit(cub_fit, a[0, 5000:], a[1, 5000:], p0=params)
# #
# # params = [-0.3359762,   0.15442415,  0.06611351,  0.02092535]
# # fitted_params2, _ = scipy.optimize.curve_fit(cub_fit, b[0, 5000:], b[1, 5000:], p0=params)
# #
# #
# # import matplotlib.pyplot as plt
# # plt.plot(a[0], a[1], label='right side of OD stretch')
# # plt.plot(a[0, 5000:], cub_fit(a[0, 5000:], *fitted_params1))
# # plt.plot(b[0], b[1], label='left side of OD stretch')
# # plt.plot(b[0, 5000:], cub_fit(b[0, 5000:], *fitted_params2))
# # plt.legend()
# # plt.show()
# # eref, weights, shift, d, coords = run(500, 10000, 250, 'od', coords_initial*1.05, np.array([0.0259616, 0, 0]), np.array([-0.000000, 0, 0]))
# # import matplotlib.pyplot as plt
# # plt.plot(eref*har2wave)
# # print(np.mean(eref[500:]*har2wave))
# # plt.show()
# # plt.plot(weights)
# # plt.show()
# # plt.plot(shift[:, 0], eref*har2wave)
# # plt.show()
# #
# # np.save('od_stretch_long_coords', coords)
# # np.save('od_stretch_long_weights', d)
# #
# # # np.save('od_stretch_long', np.vstack((shift[:, 0], eref*har2wave)))
# #
# # eref, weights, shift, d, coords = run(5000, 10000, 250, 'od', coords_initial*0.95, np.array([0.0259616, 0, 0]), np.array([0.0000000, 0, 0]))
# #
# # import matplotlib.pyplot as plt
# # plt.plot(eref*har2wave)
# # print(np.mean(eref[500:]*har2wave))
# # plt.show()
# # plt.plot(weights)
# # plt.show()
# # plt.plot(shift[:, 0], eref*har2wave)
# # plt.show()
#
# # np.save('od_stretch_short', np.vstack((shift[:, 0], eref*har2wave)))
#
# # psi = Walkers(5, coords_initial, None, [0, 0, 0])
# # psi = Potential(psi)
# # psi = E_loc(psi)
# # print(psi.V*har2wave)
# # print(psi.El*har2wave)
# print((psi.El-psi.V)*har2wave)