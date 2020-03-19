from scipy import interpolate
from Coordinerds.CoordinateSystems import *
import multiprocessing as mp
from CH5_funcs import *
from Prot_water_funcs import *

# constants and conversion factors
me = 9.10938356e-31
Avo_num = 6.0221367e23
m_C = 12.000000000 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
m_D = 2.01410177812 / (Avo_num*me*1000)
m_O = 15.99491461957 / (Avo_num*me*1000)
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
def run(N_0, time_steps, dtau, equilibration, wait_time, output, propagation=150, atoms=None,
        imp_samp=False, imp_samp_type='dev_indep', hh_relate=None, multicore=True,
        trial_wvfn=None, rand_samp=True, system='CH5', imp_samp_equilibration=True,
        imp_samp_equilibration_time=5000, equilibrations_dtau=10, threshold=None, weighting='continuous'):
    interp_exp = None
    if threshold is None:
        threshold = 1/float(N_0)
    if system == 'CH5':
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
                if len(hh_relate) == 4 or len(hh_relate) == 3:
                    interp_exp = hh_relate
                else:
                    interp_exp = interpolate.splrep(hh_relate[0, :], hh_relate[1, :], s=0)
                if len(trial_wvfn) == 5:
                    for CH in range(bonds):
                        if len(trial_wvfn[0]) == 2:
                            if np.max(trial_wvfn[CH, 1, :]) < 0.02:
                                shift = trial_wvfn[CH, 0, np.argmin(trial_wvfn[CH, 1, :])]
                            else:
                                shift = trial_wvfn[CH, 0, np.argmax(trial_wvfn[CH, 1, :])]
                            trial_wvfn[CH, 0, :] -= shift

                            psi.interp.append(interpolate.splrep(trial_wvfn[CH, 0, :], trial_wvfn[CH, 1, :], s=0))
                        else:
                            x = np.linspace(0.4, 6., 5000)
                            if np.max(trial_wvfn[CH, :]) < 0.02:
                                shift = x[np.argmin(trial_wvfn[CH, :])]
                            else:
                                shift = x[np.argmax(trial_wvfn[CH, :])]
                            x -= shift
                            psi.interp.append(interpolate.splrep(x, trial_wvfn[CH, :], s=0))
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
                                                                                                 propagation, multicore)
            else:
                coords, weights, time, Eref_array, sum_weights, accept, des = dd.simulation_time(psi, alpha, sigmaCH, Fqx,
                                                                                                 imp_samp_type, time_steps,
                                                                                                 dtau, equilibration, wait_time,
                                                                                                 propagation, multicore, interp_exp)
            np.savez(output, coords=coords, weights=weights, time=time, Eref=Eref_array,
                     sum_weights=sum_weights, accept=accept, des=des)
        else:
            coords, weights, time, Eref_array, sum_weights, des = ni.simulation_time(psi, sigmaCH, time_steps, dtau,
                                                                                     equilibration, wait_time, propagation,
                                                                                     multicore)
            np.savez(output, coords=coords, weights=weights, time=time, Eref=Eref_array,
                     sum_weights=sum_weights, des=des)
    elif system == 'ptrimer' or system == 'ptetramer':
        if system == 'ptrimer':
            coords = np.load('../../jobs/Prot_water_params/trimer_coords.npy')
        else:
            coords = np.load('../../jobs/Prot_water_params/tetramer_coords.npy')
        if imp_samp is True:
            if trial_wvfn is None:
                raise JacobHasNoFile('Please supply a trial wavefunction if you wanna do importance sampling')
            psi = pi.Walkers(N_0, atoms, coords)
            psi.interp_reg_oh = trial_wvfn['reg_oh']
            psi.interp_hbond = trial_wvfn['hbond']
            psi.interp_OO_shift = trial_wvfn['OO_shift']
            psi.interp_OO_scale = trial_wvfn['OO_scale']
            psi.interp_ang = trial_wvfn['ang']
        else:
            psi = pni.Walkers(N_0, atoms, coords)

        alpha = 1./(2.*dtau)
        sigmaH = np.sqrt(dtau / m_H)
        sigmaO = np.sqrt(dtau / m_O)
        sigmaD = np.sqrt(dtau / m_D)
        sigmaOH = np.zeros((len(atoms), 3))
        for i in range(len(atoms)):
            if psi.atoms[i].upper() == 'H':
                sigmaOH[i] = np.array([[sigmaH] * 3])
            elif psi.atoms[i].upper() == 'D':
                sigmaOH[i] = np.array([[sigmaD] * 3])
            elif psi.atoms[i].upper() == 'O':
                sigmaOH[i] = np.array([[sigmaO] * 3])
            else:
                raise JacobIsDumb("We don't serve that kind of atom in these here parts")

        if imp_samp is True:
            if imp_samp_equilibration:
                _, _, _, Eref, _, _ = pni.simulation_time(psi, sigmaOH, imp_samp_equilibration_time,
                                                                     equilibrations_dtau,
                                                                     imp_samp_equilibration_time,
                                                                     wait_time, propagation, multicore, threshold)

                Eref = Eref[-1]
            else:
                Eref = None

            Fqx, psi.psit = pi.drift(psi.coords, psi.atoms, (len(atoms)-1)/3, psi.interp_reg_oh, psi.interp_hbond,
                                     psi.interp_OO_shift, psi.interp_OO_scale, psi.interp_ang, multicore)

            coords, weights, time, Eref_array, sum_weights, accept, des = pi.simulation_time(psi, alpha, sigmaOH, Fqx,
                                                                                             time_steps, dtau, equilibration,
                                                                                             wait_time, propagation,
                                                                                             threshold, multicore, Eref)
            np.savez(output, coords=coords, weights=weights, time=time, Eref=Eref_array,
                     sum_weights=sum_weights, accept=accept, des=des)
        else:
            coords, weights, time, Eref_array, sum_weights, des = pni.simulation_time(psi, sigmaOH, time_steps, dtau,
                                                                                      equilibration, wait_time, propagation,
                                                                                      multicore, threshold, weighting)
            np.savez(output, coords=coords, weights=weights, time=time, Eref=Eref_array,
                     sum_weights=sum_weights, des=des)

    elif system == 'pmonomer' or system == 'pdimer':
        if system == 'pdimer':
            coords = np.load('../../jobs/Prot_water_params/dimer_coords.npy')

            if imp_samp is True:
                if trial_wvfn is None:
                    raise JacobHasNoFile('Please supply a trial wavefunction if you wanna do importance sampling')
                psi = pi.Walkers(N_0, atoms, coords)
                psi.interp_reg_oh = trial_wvfn['reg_oh']
                psi.interp_hbond = trial_wvfn['hbond']
                psi.interp_ang = trial_wvfn['ang']
            else:
                psi = pni.Walkers(N_0, atoms, coords)

        elif system == 'pmonomer':
            coords = np.load('../../jobs/Prot_water_params/monomer_coords.npy')

            if imp_samp is True:
                if trial_wvfn is None:
                    raise JacobHasNoFile('Please supply a trial wavefunction if you wanna do importance sampling')
                psi = pi.Walkers(N_0, atoms, coords)
                psi.interp_reg_oh = trial_wvfn['reg_oh']
            else:
                psi = pni.Walkers(N_0, atoms, coords)

        alpha = 1. / (2. * dtau)
        sigmaH = np.sqrt(dtau / m_H)
        sigmaO = np.sqrt(dtau / m_O)
        sigmaD = np.sqrt(dtau / m_D)
        sigmaOH = np.zeros((len(atoms), 3))
        for i in range(len(atoms)):
            if psi.atoms[i].upper() == 'H':
                sigmaOH[i] = np.array([[sigmaH] * 3])
            elif psi.atoms[i].upper() == 'D':
                sigmaOH[i] = np.array([[sigmaD] * 3])
            elif psi.atoms[i].upper() == 'O':
                sigmaOH[i] = np.array([[sigmaO] * 3])
            else:
                raise JacobIsDumb("We don't serve that kind of atom in these here parts")

        if imp_samp is True:
            if imp_samp_equilibration:
                _, _, _, Eref, _, _ = pni.simulation_time(psi, sigmaOH, imp_samp_equilibration_time,
                                                          equilibrations_dtau,
                                                          imp_samp_equilibration_time,
                                                          wait_time, propagation, multicore, threshold)

                Eref = Eref[-1]
            else:
                Eref = None

            Fqx, psi.psit = pi.drift(psi.coords, psi.atoms, (len(atoms)-1)/3, psi.interp_reg_oh, psi.interp_hbond,
                                     psi.interp_OO_shift, psi.interp_OO_scale, psi.interp_ang, multicore)

            coords, weights, time, Eref_array, sum_weights, accept, des = pi.simulation_time(psi, alpha, sigmaOH, Fqx,
                                                                                             time_steps, dtau, equilibration,
                                                                                             wait_time, propagation,
                                                                                             threshold, multicore, Eref)

            np.savez(output, coords=coords, weights=weights, time=time, Eref=Eref_array,
                     sum_weights=sum_weights, accept=accept, des=des)
        else:
            coords, weights, time, Eref_array, sum_weights, des = pni.simulation_time(psi, sigmaOH, time_steps, dtau,
                                                                                      equilibration, wait_time,
                                                                                      propagation,
                                                                                      multicore, threshold, weighting)
            np.savez(output, coords=coords, weights=weights, time=time, Eref=Eref_array,
                     sum_weights=sum_weights, des=des)

    return


pool = mp.Pool(mp.cpu_count()-1)

