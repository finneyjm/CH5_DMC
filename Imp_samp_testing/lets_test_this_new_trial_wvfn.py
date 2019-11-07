import copy
import CH5pot
from scipy import interpolate
from Coordinerds.CoordinateSystems import *
import multiprocessing as mp
# import Timing_p3 as tm

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.0107 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_CH = (m_C*m_H)/(m_H+m_C)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

# Starting orientation of walkers
coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                          [0.1318851447521099, 2.088940054609643, 0.000000000000000],
                          [1.786540362044548, -1.386051328559878, 0.000000000000000],
                          [2.233806981137821, 0.3567096955165336, 0.000000000000000],
                          [-0.8247121421923925, -0.6295306113384560, -1.775332267901544],
                          [-0.8247121421923925, -0.6295306113384560, 1.775332267901544]])

bonds = 5
order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]

Psi_t = np.load('params/min_wvfns/GSW_min_CH_2.npy')
x = np.linspace(0.4, 6., 5000)
if np.max(Psi_t) < 0.02:
    shift = x[np.argmin(Psi_t)]
else:
    shift = x[np.argmax(Psi_t)]
# shift = np.dot(Psi_t**2, x)
# shift = 2.2611644678388316
# shift = 2.0930991957283878
shift = 0.
x = x - shift
exp = np.load('params/sigma_hh_to_rch_cub_relationship.npy')
interp_exp = interpolate.splrep(exp[0, :], exp[1, :], s=0)
interp = interpolate.splrep(x, Psi_t, s=0)
dx = 1.e-4


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.arange(0, walkers)
        self.coords = np.array([coords_initial]*walkers)
        # rand_idx = np.random.rand(walkers, 5).argsort(axis=1) + 1
        # b = self.coords[np.arange(walkers)[:, None], rand_idx]
        # self.coords[:, 1:6, :] = b
        self.weights = np.ones(walkers)
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)
        self.psit = np.zeros((walkers, 3, 6, 3))


def hh_dist(carts, rch):
    N = len(carts)
    coords = np.array(carts)
    coords -= np.broadcast_to(coords[:, None, 0], (N, bonds+1, 3))
    coords[:, 1:] /= np.broadcast_to(rch[:, :, None], (N, bonds, 3))
    hh = np.zeros((N, 5, 5))
    a = np.full((5, 5), True)
    np.fill_diagonal(a, False)
    mask = np.broadcast_to(a, (N, 5, 5))
    for i in range(4):
        for j in np.arange(i+1, 5):
            hh[:, i, j] = np.sqrt((coords[:, j+1, 0] - coords[:, i+1, 0])**2 +
                                  (coords[:, j+1, 1] - coords[:, i+1, 1])**2 +
                                  (coords[:, j+1, 2] - coords[:, i+1, 2])**2)
    hh += np.transpose(hh, (0, 2, 1))
    blah = hh[mask].reshape(N, 5, 4)
    hh_std = np.std(blah, axis=2)
    return hh_std


def ch_dist(coords):
    N = len(coords)
    rch = np.zeros((N, bonds))
    for i in range(bonds):
        rch[:, i] = np.sqrt((coords[:, i+1, 0] - coords[:, 0, 0])**2 +
                            (coords[:, i+1, 1] - coords[:, 0, 1])**2 +
                            (coords[:, i+1, 2] - coords[:, 0, 2])**2)
    return rch


def one_ch_dist(coords, CH):
    rch = np.sqrt((coords[:, CH, 0] - coords[:, 0, 0])**2 +
                  (coords[:, CH, 1] - coords[:, 0, 1])**2 +
                  (coords[:, CH, 2] - coords[:, 0, 2])**2)
    return rch


def full_psi_t(coords):
    rch = ch_dist(coords)
    hh = hh_dist(coords, rch)
    psi = np.zeros((len(coords), bonds))
    shift = np.zeros((len(coords), bonds))
    for i in range(bonds):
        # shift[:, i] = interpolate.splev(hh[:, i], interp_exp, der=0)
        # shift[:, i] += 2.2842168433686734
        psi[:, i] = interpolate.splev(rch[:, i]-shift[:, i], interp, der=0)
    return psi, shift


def changin_psi_t(coords, atom, shift, psi):
    psit = np.zeros(psi.shape)
    if atom is 0:
        rch = ch_dist(coords)
        for i in range(bonds):
            psit[:, i] = interpolate.splev(rch[:, i]-shift[:, i], interp, der=0)
    else:
        rch = one_ch_dist(coords, atom)
        psit = np.array(psi)
        psit[:, atom-1] = interpolate.splev(rch - shift[:, atom-1], interp, der=0)
    return np.prod(psit, axis=1)


def get_da_psi(coords):
    much_psi = np.zeros((len(coords), 3, 6, 3))
    psi, shift = full_psi_t(coords)
    asdf = np.broadcast_to(np.prod(psi, axis=1)[:, None, None], (len(coords), 6, 3))
    much_psi[:, 1] += asdf
    for atoms in range(6):
        for xyz in range(3):
            coords[:, atoms, xyz] -= dx
            much_psi[:, 0, atoms, xyz] = changin_psi_t(coords, atoms, shift, psi)
            coords[:, atoms, xyz] += 2.*dx
            much_psi[:, 2, atoms, xyz] = changin_psi_t(coords, atoms, shift, psi)
            coords[:, atoms, xyz] -= dx
    return much_psi


def all_da_psi(coords):
    coords = np.array_split(coords, mp.cpu_count()-1)
    psi = pool.map(get_da_psi, coords)
    psi = np.concatenate(psi)
    return psi


def drift(psi):
    dpsidx = (psi[:, 2] - psi[:, 0])/(2.*dx)
    return 2.*dpsidx/psi[:, 1]


# Function for the potential for the mp to use
def get_pot(coords):
    V = CH5pot.mycalcpot(coords, len(coords))
    return V


# Split up those coords to speed up dat potential
def Potential(Psi):
    coords = np.array_split(Psi.coords, mp.cpu_count()-1)
    V = pool.map(get_pot, coords)
    Psi.V = np.concatenate(V)
    return Psi


def E_loc(Psi, sigmaCH, dtau):
    d2psidx2 = ((Psi.psit[:, 0] - 2.*Psi.psit[:, 1] + Psi.psit[:, 2])/dx**2)/Psi.psit[:, 1]
    kin = -1./2.*np.sum(np.sum(sigmaCH**2/dtau*d2psidx2, axis=1), axis=1)
    Psi.El = kin + Psi.V
    blah = Psi.El*har2wave
    return Psi


pool = mp.Pool(mp.cpu_count()-1)
import Timing_p3 as tm
psi = Walkers(10000)
psi.coords = np.load('Non_imp_sampled/DMC_CH5_coords_10000_walkers_5.npy')
rch, rch_time = tm.time_me(ch_dist, psi.coords)
tm.print_time_list(ch_dist, rch_time)
hh, hh_time = tm.time_me(hh_dist, psi.coords, rch)
tm.print_time_list(hh_dist, hh_time)
# psi_trial, trial_time = tm.time_me(full_psi_t, psi.coords)
# tm.print_time_list(full_psi_t, trial_time)
psit, psit_time = tm.time_me(get_da_psi, psi.coords)
tm.print_time_list(get_da_psi, psit_time)
psi.psit = get_da_psi(psi.coords)
# print(drift(psi.psit))
psi.V, v_time = tm.time_me(get_pot, psi.coords)
tm.print_time_list(get_pot, v_time)
dtau = 1
sigmaH = np.sqrt(dtau/m_H)
sigmaC = np.sqrt(dtau/m_C)
sigmaCH = np.array([[sigmaC]*3, [sigmaH]*3, [sigmaH]*3, [sigmaH]*3, [sigmaH]*3, [sigmaH]*3])
psi = E_loc(psi, sigmaCH, dtau)
print(psi.El*har2wave)
#
#
# import matplotlib.pyplot as plt
# from Coordinerds.CoordinateSystems import *
# N_0 = 10000
#
# fig, axes = plt.subplots(1, 5, figsize=(20, 8))
# for i in range(5):
#     Psi = Walkers(N_0)
#     zmat = CoordinateSet(Psi.coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
#     zmat[:, i, 1] = np.linspace(0.6, 1.8, N_0)*ang2bohr
#     Psi.coords = CoordinateSet(zmat, system=ZMatrixCoordinates).convert(CartesianCoordinates3D).coords
#     Psi.psit = get_da_psi(Psi.coords)
#     Psi = Potential(Psi)
#     Psi = E_loc(Psi, sigmaCH, dtau)
#     axes[i].plot(zmat[:, i, 1] / ang2bohr, Psi.V * har2wave, label='Potential')
#     axes[i].plot(zmat[:, i, 1] / ang2bohr, Psi.El * har2wave, label=f'Local Energy')
#     axes[i].plot(zmat[:, i, 1] / ang2bohr, Psi.El * har2wave - Psi.V * har2wave, label=f'Kinetic')
#     axes[i].set_xlabel('rCH (Angstrom)')
#     axes[i].set_ylabel('Energy (cm^-1)')
#     axes[i].set_ylim(-20000, 20000)
#     axes[i].legend(loc='lower left')
# plt.tight_layout()
# plt.show()
# plt.close()

# N, run_time = tm.time_me(run, 100, 20000, 1, 5000, 500, 'testing_dis_function')
# tm.print_time_list(run, run_time)


# The metropolis step based on those crazy Green's functions
def metropolis(psi_1, psi_2, Fqx, Fqy, x, y, sigmaC, sigmaH):
    a = (psi_2[:, 1, 0, 0]/psi_1[:, 1, 0, 0])**2
    for atom in range(6):
        for xyz in range(3):
            if atom == 0:
                sigma = sigmaC
            else:
                sigma = sigmaH
            # Use dat Green's function
            a *= np.exp(1./2.*(Fqx[:, atom, xyz] + Fqy[:, atom, xyz])*(sigma**2/4.*(Fqx[:, atom, xyz]-Fqy[:, atom, xyz])
                                                                       - (y[:, atom, xyz]-x[:, atom, xyz])))
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx, sigmaCH, sigmaH, sigmaC):
    Drift = sigmaCH**2/2.*Fqx   # evaluate the drift term from the F that was calculated in the previous step
    randomwalk = np.zeros((len(Psi.coords), 6, 3))  # normal randomwalk from DMC
    randomwalk[:, 1:6, :] = np.random.normal(0.0, sigmaH, size=(len(Psi.coords), 5, 3))
    randomwalk[:, 0, :] = np.random.normal(0.0, sigmaC, size=(len(Psi.coords), 3))
    y = randomwalk + Drift + np.array(Psi.coords)  # the proposed move for the walkers
    psi_y = all_da_psi(y)
    Fqy = drift(psi_y)  # evaluate new F
    a = metropolis(Psi.psit, psi_y, Fqx, Fqy, Psi.coords, y, sigmaC, sigmaH)  # Is it a good move?
    check = np.random.random(size=len(Psi.coords))
    accept = np.argwhere(a > check)
    # Update everything that is good
    Psi.coords[accept] = y[accept]
    nah = np.argwhere(a <= check)
    Fqy[nah] = Fqx[nah]
    Psi.psit[accept] = psi_y[accept]
    acceptance = float(len(accept)/len(Psi.coords))*100.
    return Psi, Fqy, acceptance


# Calculate the Eref for use in the weighting
def E_ref_calc(Psi, alpha):
    P = sum(Psi.weights)
    E_ref = sum(Psi.weights*Psi.El)/P - alpha*np.log(P/len(Psi.coords))
    return E_ref


# Calculate the weights of the walkers and figure out the birth/death if needed
def Weighting(Eref, Psi, Fqx, dtau, DW):
    Psi.weights = Psi.weights * np.exp(-(Psi.El - Eref) * dtau)
    threshold = 1./float(len(Psi.coords))
    death = np.argwhere(Psi.weights < threshold)  # should I kill a walker?
    for i in death:
        ind = np.argmax(Psi.weights)
        # copy things over
        if DW is True:
            Biggo_num = int(Psi.walkers[ind])
            Psi.walkers[i[0]] = Biggo_num
        Biggo_weight = float(Psi.weights[ind])
        Biggo_pos = np.array(Psi.coords[ind])
        Biggo_pot = float(Psi.V[ind])
        Biggo_el = float(Psi.El[ind])
        Biggo_psit = np.array(Psi.psit[ind])
        Biggo_force = np.array(Fqx[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Psi.psit[i[0]] = Biggo_psit
        Fqx[i[0]] = Biggo_force
    return Psi


# Adding up all the descendant weights
def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < len(Psi.coords):
        d = np.append(d, 0.)
    return d


class JacobHasNoFile(FileNotFoundError):
    pass


class JacobIsDumb(ValueError):
    pass


# Function to go through the DMC algorithm
def run(N_0, time_steps, dtau, equilibration, wait_time, output,
        DW=False, dw_num=None, dwfunc=None):
    alpha = 1. / (2. * dtau)
    sigmaH = np.sqrt(dtau / m_H)
    sigmaC = np.sqrt(dtau / m_C)
    sigmaCH = np.array([[sigmaC] * 3, [sigmaH] * 3, [sigmaH] * 3, [sigmaH] * 3, [sigmaH] * 3, [sigmaH] * 3])
    psi = Walkers(N_0)
    if DW is True:
        if dw_num is None:
            raise JacobIsDumb('Indicate the walkers that you want to use with an integer value')
        if dwfunc is None:
            raise JacobHasNoFile('Indicate the walkers to use for des weighting')
        wvfn = np.load(dwfunc)
        psi.coords = wvfn['coords'][dw_num-1]
        psi.weights = wvfn['weights'][dw_num-1]
    psi.psit = all_da_psi(psi.coords)
    Fqx = drift(psi.psit)

    num_o_collections = int((time_steps-equilibration)/wait_time) + 1
    time = np.zeros(time_steps)
    accept = np.zeros(time_steps)
    Eref_array = np.zeros(time_steps)
    sum_weights = np.zeros(time_steps)
    coords = np.zeros(np.append(num_o_collections, psi.coords.shape))
    weights = np.zeros(np.append(num_o_collections, psi.weights.shape))
    des = 0
    num = 0
    wait = float(wait_time)
    for i in range(int(time_steps)):
        wait -= 1.
        if i % 1000 == 0:
            print(i)

        psi, Fqx, acceptance = Kinetic(psi, Fqx, sigmaCH, sigmaH, sigmaC)
        psi = Potential(psi)
        psi = E_loc(psi, sigmaCH, dtau)

        if i == 0:
            Eref = E_ref_calc(psi, alpha)

        psi = Weighting(Eref, psi, Fqx, dtau, DW)

        Eref = E_ref_calc(psi, alpha)
        # print(Eref*har2wave)
        Eref_array[i] += Eref
        time[i] += i + 1
        sum_weights[i] += np.sum(psi.weights)
        accept[i] += acceptance

        if i >= int(equilibration)-1 and wait <= 0.:
            wait = float(wait_time)
            Psi_tau = copy.deepcopy(psi)
            coords[num] += Psi_tau.coords
            weights[num] += Psi_tau.weights
            num += 1
    if DW is True:
        des = descendants(psi)
        coords = wvfn['coords'][dw_num-1]
        weights = wvfn['weights'][dw_num-1]
    np.savez(output, coords=coords, weights=weights, time=time, Eref=Eref_array,
             sum_weights=sum_weights, accept=accept, des=des)
    print(np.mean(Eref_array[5000:])*har2wave)


# pool = mp.Pool(mp.cpu_count()-1)
# run(5000, 10000, 1, 5000, 500, 'testytest')
# print(0.04936915844038702*har2wave)




















