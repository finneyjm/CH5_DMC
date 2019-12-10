from scipy import interpolate
from Coordinerds.CoordinateSystems import *
import multiprocessing as mp
from CH5_funcs import *

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.000000000 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_CD = (m_C*m_D)/(m_D+m_C)
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


class JacobHasNoFile(FileNotFoundError):
    pass


class JacobIsDumb(ValueError):
    pass


# Function to go through the DMC algorithm
def run(N_0, time_steps, dtau, equilibration, wait_time, output, atoms=None,
        imp_samp=False, imp_samp_type='dev_indep', hh_relate=None, multicore=True,
        trial_wvfn=None, DW=False, dw_num=None, dwfunc=None, rand_samp=True):
    interp_exp = None
    if imp_samp is True:
        if trial_wvfn is None:
            raise JacobHasNoFile('Please supply a trial wavefunction if you wanna do importance sampling')
        if imp_samp_type == 'dev_indep':
            psi = di.Walkers(N_0, atoms, rand_samp)
            if len(trial_wvfn) == 2:
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(trial_wvfn[0, :], trial_wvfn[1, :], s=0))
            elif len(trial_wvfn) == 5000:
                x = np.linspace(0.4, 6., 5000)
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(x, trial_wvfn, s=0))
            else:
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(trial_wvfn[CH, 0], trial_wvfn[CH, 1], s=0))
        elif imp_samp_type == 'dev_dep':
            psi = dd.Walkers(N_0, atoms, rand_samp)
            if hh_relate is None:
                raise JacobIsDumb('Give me dat hh-rch function')
            interp_exp = interpolate.splrep(hh_relate[0, :], hh_relate[1, :], s=0)
            if len(trial_wvfn) == 5:
                for CH in range(bonds):
                    if np.max(trial_wvfn[CH, 1, :]) < 0.02:
                        shift = trial_wvfn[CH, 0, np.argmin(trial_wvfn[CH, 1, :])]
                    else:
                        shift = trial_wvfn[CH, 0, np.argmax(trial_wvfn[CH, 1, :])]
                    trial_wvfn[CH, 0, :] -= shift
                    for CH in range(bonds):
                        psi.interp.append(interpolate.splrep(trial_wvfn[CH, 0, :], trial_wvfn[CH, 1, :], s=0))
            if len(trial_wvfn) == 2:
                if np.max(trial_wvfn[1, :]) < 0.02:
                    shift = trial_wvfn[0, np.argmin(trial_wvfn[1, :])]
                else:
                    shift = trial_wvfn[0, np.argmax(trial_wvfn[1, :])]
                trial_wvfn[0, :] -= shift
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(trial_wvfn[0, :], trial_wvfn[1, :], s=0))
            elif len(trial_wvfn) == 5000:
                x = np.linspace(0.4, 6., 5000)
                if np.max(trial_wvfn) < 0.02:
                    shift = x[np.argmin(trial_wvfn)]
                else:
                    shift = x[np.argmax(trial_wvfn)]
                x -= shift
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(x, trial_wvfn, s=0))
        elif imp_samp_type == 'fd':
            psi = dd.Walkers(N_0, atoms, rand_samp)
            if len(trial_wvfn) == 2:
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(trial_wvfn[0, :], trial_wvfn[1, :], s=0))
            elif len(trial_wvfn) == 5000:
                x = np.linspace(0.4, 6., 5000)
                for CH in range(bonds):
                    psi.interp.append(interpolate.splrep(x, trial_wvfn, s=0))
        else:
            raise JacobIsDumb('Not a valid type of importance sampling yet')
    else:
        psi = ni.Walkers(N_0, atoms, rand_samp)

    alpha = 1./(2.*dtau)
    sigmaH = np.sqrt(dtau/m_H)
    sigmaC = np.sqrt(dtau/m_C)
    sigmaD = np.sqrt(dtau/m_D)
    sigmaCH = np.zeros((6, 3))
    sigmaCH[0] = np.array([[sigmaC] * 3])
    for i in np.arange(1, 6):
        if psi.atoms[i].upper() == 'H':
            sigmaCH[i] = np.array([[sigmaH] * 3])
        elif psi.atoms[i].upper() == 'D':
            sigmaCH[i] = np.array([[sigmaD] * 3])
        else:
            raise JacobIsDumb("We don't serve that kind of atom in these here parts")

    if DW is True:
        if dw_num is None:
            raise JacobIsDumb('Indicate the walkers that you want to use with an integer value')
        if dwfunc is None:
            raise JacobHasNoFile('Indicate the walkers to use for des weighting')
        wvfn = np.load(dwfunc)
        psi.coords = wvfn['coords'][dw_num-1]
        psi.weights = wvfn['weights'][dw_num-1]

    if imp_samp is True:
        if imp_samp_type == 'dev_indep':
            Fqx, psi.drdx = di.drift(psi.zmat, psi.coords, psi.interp)
        elif imp_samp_type == 'dev_dep':
            Fqx, psi.psit = dd.drift(psi.zmat, psi.coords, psi.interp, imp_samp_type, multicore=multicore, interp_exp=interp_exp)
        elif imp_samp_type == 'fd':
            Fqx, psi.psit = dd.drift(psi.zmat, psi.coords, psi.interp, imp_samp_type, multicore=multicore)

    if imp_samp is True:
        if imp_samp_type == 'dev_indep':
            coords, weights, time, Eref_array, sum_weights, accept, des = di.simulation_time(psi, alpha, sigmaCH,
                                                                                             Fqx, time_steps, dtau,
                                                                                             equilibration, wait_time,
                                                                                             multicore, DW)
        else:
            coords, weights, time, Eref_array, sum_weights, accept, des = dd.simulation_time(psi, alpha, sigmaCH, Fqx,
                                                                                             imp_samp_type, time_steps,
                                                                                             dtau, equilibration, wait_time,
                                                                                             multicore, DW, interp_exp)
        np.savez(output, coords=coords, weights=weights, time=time, Eref=Eref_array,
                 sum_weights=sum_weights, accept=accept, des=des)
    else:
        coords, weights, time, Eref_array, sum_weights, des = ni.simulation_time(psi, sigmaCH, time_steps, dtau,
                                                                                 equilibration, wait_time, multicore, DW)
        np.savez(output, coords=coords, weights=weights, time=time, Eref=Eref_array,
                 sum_weights=sum_weights, des=des)
    return time


pool = mp.Pool(mp.cpu_count()-1)


run(100, 200, 1, 200, 123, 'file')
