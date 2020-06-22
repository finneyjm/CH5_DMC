import copy
from scipy import interpolate
from Coordinerds.CoordinateSystems import *
import Water_monomer_pot_fns as wm
# import Timing_p3 as tm

# DMC parameters
dtau = 1.
time_steps = 20000.
alpha = 1./(2.*dtau)

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_O = 15.994915 / (Avo_num*me*1000)
m_H = 1.007825 / (Avo_num*me*1000)
m_OH = (m_H*m_O)/(m_H+m_O)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11

# Values for Simulation
sigmaH = np.sqrt(dtau/m_H)
sigmaO = np.sqrt(dtau/m_O)
sigmaOH = np.array([[sigmaO]*3, [sigmaH]*3, [sigmaH]*3])

bonds = 2
atoms = 3
coords_initial = np.array([[0.000000000000000, 0.000000000000000, 0.000000000000000],
                           [0.957840000000000, 0.000000000000000, 0.000000000000000],
                           [-0.23995350000000, 0.927297000000000, 0.000000000000000]])*ang2bohr

order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 1]]

Psi_t = np.load('Water_oh_stretch_GSW.npy')
interp = interpolate.splrep(Psi_t[0, :], Psi_t[1, :], s=0)


# Creates the walkers with all of their attributes
class Walkers(object):
    walkers = 0

    def __init__(self, walkers):
        self.walkers = np.arange(0, N_0)
        self.coords = np.array([coords_initial]*walkers)
        self.zmat = CoordinateSet(self.coords, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
        self.weights = np.zeros(walkers) + 1.
        self.d = np.zeros(walkers)
        self.weights_i = np.zeros(walkers) + 1.
        self.V = np.zeros(walkers)
        self.El = np.zeros(walkers)


def psi_t(zmatrix):
    psi = np.zeros((N_0, bonds))
    for i in range(bonds):
        psi[:, i] += interpolate.splev(zmatrix[:, i, 1], interp, der=0)
    return psi


def drdx(zmatrix, coords):
    chain = np.zeros((N_0, bonds, atoms, 3))
    for xyz in range(3):
        for CH in range(bonds):
            chain[:, CH, 0, xyz] += ((coords[:, 0, xyz]-coords[:, CH+1, xyz])/zmatrix[:, CH, 1])
            chain[:, CH, CH+1, xyz] += ((coords[:, CH+1, xyz]-coords[:, 0, xyz])/zmatrix[:, CH, 1])
    return chain


def drift(zmatrix, coords):
    psi = psi_t(zmatrix)
    dr1 = drdx(zmatrix, coords)
    der = np.zeros((N_0, bonds))
    for i in range(bonds):
        der[:, i] += (interpolate.splev(zmatrix[:, i, 1], interp, der=1)/psi[:, i])
    a = dr1.reshape((N_0, bonds, 9))
    b = der.reshape((N_0, 1, bonds))
    drift = np.matmul(b, a)
    return 2.*drift.reshape((N_0, atoms, 3))


def metropolis(r1, r2, Fqx, Fqy, x, y):
    psi_1 = psi_t(r1)
    psi_2 = psi_t(r2)
    psi_ratio = 1.
    for i in range(bonds):
        psi_ratio *= (psi_2[:, i]/psi_1[:, i])**2
    a = psi_ratio
    for atom in range(atoms):
        for xyz in range(3):
            if atom == 0:
                sigma = sigmaO
            else:
                sigma = sigmaH
            a *= np.exp(1./2.*(Fqx[:, atom, xyz] + Fqy[:, atom, xyz])*(sigma**2/4.*(Fqx[:, atom, xyz]-Fqy[:, atom, xyz])
                                                                       - (y[:, atom, xyz]-x[:, atom, xyz])))
    return a


# Random walk of all the walkers
def Kinetic(Psi, Fqx):
    Drift = sigmaOH**2/2.*Fqx
    randomwalk = np.zeros((N_0, atoms, 3))
    randomwalk[:, 1:3, :] = np.random.normal(0.0, sigmaH, size=(N_0, bonds, 3))
    randomwalk[:, 0, :] = np.random.normal(0.0, sigmaO, size=(N_0, 3))
    y = randomwalk + Drift + np.array(Psi.coords)
    zmatriy = CoordinateSet(y, system=CartesianCoordinates3D).convert(ZMatrixCoordinates, ordering=order).coords
    Fqy = drift(zmatriy, y)
    a = metropolis(Psi.zmat, zmatriy, Fqx, Fqy, Psi.coords, y)
    check = np.random.random(size=N_0)
    accept = np.argwhere(a > check)
    Psi.coords[accept] = y[accept]
    nah = np.argwhere(a <= check)
    Fqy[nah] = Fqx[nah]
    Psi.zmat[accept] = zmatriy[accept]
    acceptance = float(len(accept)/N_0)*100.
    return Psi, Fqy, acceptance


def Potential(Psi):
    V = wm.PatrickShinglePotential(Psi.coords, 4)
    Psi.V = np.array(V)
    return Psi


def local_kinetic(Psi):
    psi = psi_t(Psi.zmat)
    der1 = np.zeros((N_0, bonds))
    der2 = np.zeros((N_0, bonds))
    for i in range(bonds):
        der1[:, i] += (interpolate.splev(Psi.zmat[:, i, 1], interp, der=1)/psi[:, i]*(2./Psi.zmat[:, i, 1]))
        der2[:, i] += (interpolate.splev(Psi.zmat[:, i, 1], interp, der=2)/psi[:, i])
    kin = -1./(2.*m_OH)*np.sum(der2+der1, axis=1)
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
    threshold = 1./float(N_0)
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
        Biggo_zmat = np.array(Psi.zmat[ind])
        Biggo_force = np.array(Fqx[ind])
        Psi.weights[i[0]] = Biggo_weight/2.
        Psi.weights[ind] = Biggo_weight/2.
        Psi.coords[i[0]] = Biggo_pos
        Psi.V[i[0]] = Biggo_pot
        Psi.El[i[0]] = Biggo_el
        Psi.zmat[i[0]] = Biggo_zmat
        Fqx[i[0]] = Biggo_force
    return Psi


def descendants(Psi):
    d = np.bincount(Psi.walkers, weights=Psi.weights)
    while len(d) < N_0:
        d = np.append(d, 0.)
    return d


def run(propagation, test_number):
    DW = False
    psi = Walkers(N_0)
    Fqx = drift(psi.zmat, psi.coords)
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
    for i in range(int(time_steps)):
        if i % 1000 == 0:
            print(i)

        Psi, Fqx, acceptance = Kinetic(new_psi, Fqx)
        Psi = Potential(Psi)
        Psi = E_loc(Psi)

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
    # np.save(f'Imp_samp_water_coords_{N_0}_walkers_{test_number}', Psi_tau.coords)
    # np.save(f'Imp_samp_water_weights_{N_0}_walkers_{test_number}', np.vstack((Psi_tau.weights, d_values)))
    # np.save(f'Imp_samp_water_energy_{N_0}_walkers_{test_number}', np.vstack((time, Eref_array, weights, accept)))
    return Eref_array


# tests = [100, 200, 500, 1000, 2000, 5000, 10000]
# for j in range(5):
#     for i in range(7):
#         N_0 = tests[i]
#         run(250, j+6)
#         print(f'{tests[i]} Walker Test {j+1} is done!')
for i in range(10):
    N_0 = 20000
    run(250, i+1)
    print(f'{N_0} Walker Test {i+1} is done!')
# N_0 = 10000
# eref, time_list = tm.time_me(run, 0, 'testtesttest')
# tm.print_time_list(run, time_list)
