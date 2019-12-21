from scipy import interpolate
from Coordinerds.CoordinateSystems import *
import multiprocessing as mp

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_O = 15.99491461957 / (Avo_num*me*1000)
m_OD = (m_O*m_D)/(m_D+m_O)
m_OH = (m_O*m_H)/(m_H+m_O)
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11


class JacobHasNoFile(FileNotFoundError):
    pass


class JacobIsDumb(ValueError):
    pass


def run(N_0, time_steps, dtau, equilibration, wait_time, output, atoms, coords_initial, propagation=250,
        imp_samp=False, trial_wvfn=None, multicore=True, rand_samp=True):
    alpha = 1./(2.*dtau)
    sigmaH = np.sqrt(dtau/m_H)
    sigmaD = np.sqrt(dtau/m_D)
    sigmaO = np.sqrt(dtau/m_O)
    sigma = np.zeros((len(atoms), 3))

    for i in range(len(atoms)):
        if atoms[i].upper() == 'H':
            sigma[i] = np.array([sigmaH] * 3)
        elif atoms[i].upper() == 'D':
            sigma[i] = np.array([sigmaD] * 3)
        elif atoms[i].upper() == 'O':
            sigma[i] = np.array([sigmaO] * 3)
        else:
            raise JacobIsDumb("We don't serve that kind of atom in these here parts")

    if imp_samp is True:
        if trial_wvfn is None:
            raise JacobHasNoFile('Please supply a trial wavefunction if you wanna do importance sampling')
































