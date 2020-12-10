import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from Coordinerds.CoordinateSystems import *

order = [[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 1, 0], [3, 0, 1, 2], [4, 0, 1, 2], [5, 0, 1, 2]]
har2wave = 219474.6
ang2bohr = 1.e-10/5.291772106712e-11
ni = ['_CD', '_1H', '_2H', '_3H', '_4H', '']
ens = [8044, 8563, 9097, 9694, 10300, 10917]
ens_std = [2, 3, 6, 3, 6, 3]
# ens = [8044, 8563, 9097, 9698.52384684, 10304.93858414, 23370.6]
lowlim = [8020, 8540, 9080, 9680, 10280, 10895]
highlim = [8070, 8600, 9130, 9730, 10330, 10945]
avg_beg = 5000
avg_end = 20000


def lets_get_some_energies(non_imp_samp_walkers, imp_samp_walkers, trials_ni, trials_i, isotop, system, extra=False):
    N_i = len(imp_samp_walkers)
    N_n = len(non_imp_samp_walkers)
    if system == 'tetramer':
        energies_imp = np.zeros((N_i+2, trials_i))
        energies_imp2 = np.zeros((N_i+1, trials_i))
    elif extra is True:
        energies_imp = np.zeros((N_i, trials_i))

    else:
        energies_imp = np.zeros((N_i, trials_i))
        energies_imp2 = np.zeros((N_i, trials_i))
    if extra:
        walk = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 10000]
        energies_imp3 = np.zeros((len(walk), trials_i))
        energies_imp2 = np.zeros((len(walk), trials_i))
    else:
        energies_imp3 = np.zeros((N_i, trials_i))
    energies_non = np.zeros((N_n, trials_ni))

    for j in range(trials_ni):
        for i in range(N_n):
            if system == 'CH5':
                Energy = np.load(f'Trial_wvfn_testing/results/Non_imp_sampled{ni[isotop]}/' +
                                 f'Non_imp_sampled{ni[isotop]}_{non_imp_samp_walkers[i]}_' +
                                 f'Walkers_Test_{j+1}.npz')['Eref']*har2wave
                energies_non[i, j] += np.mean(Energy[5000:])
            elif system == 'dimer':
                Energy = np.load(f'Trial_wvfn_testing/results/Non_imp_sampled_pdimer/' +
                                  f'Non_imp_sampled_pdimer_{non_imp_samp_walkers[i]}_' +
                                  f'Walkers_Test_{j+1}.npz')['Eref']*har2wave
                energies_non[i, j] += np.mean(Energy[5000:]) + 11804.25054877
            elif system == 'trimer':
                Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_non_imp_samp/' +
                                 f'ptrimer_non_imp_samp_{non_imp_samp_walkers[i]}_' +
                                 f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
                a = -9.129961343400107E-002 * har2wave
                energies_non[i, j] = np.mean(Energy[5000:]) - a
            elif system == 'tetramer':
                Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp/' +
                                f'ptetramer_non_imp_samp_{non_imp_samp_walkers[i]}_' +
                                f'Walkers_Test_{j+1}.npz')
                Energy = Energy['Eref']*har2wave
                a = -0.122146858971399 * har2wave
                energies_non[i, j] = np.mean(Energy[avg_beg:avg_end]) - a
            elif system == 'monomer':
                Energy = np.load(f'Trial_wvfn_testing/results/pmonomer_non_imp_samp/' +
                                 f'pmonomer_non_imp_samp_{non_imp_samp_walkers[i]}_' +
                                 f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave

                a = 0*har2wave
                energies_non[i, j] = np.mean(Energy[5000:]) - a

    for j in range(trials_i):
        for i in range(N_i):

            if system == 'CH5':
                Energy = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_{isotop}H_GSW2/' +
                                 f'HH_to_rCHrCD_{isotop}H_GSW2_{imp_samp_walkers[i]}_' +
                                 f'Walkers_Test_{j+1}.npz')['Eref']*har2wave
                energies_imp[i, j] += np.mean(Energy[5000:])

                # Energy = np.load(f'Trial_wvfn_testing/results/Average_fd/' +
                #                  f'Average_fd_{imp_samp_walkers[i]}_Walkers' +
                #                  f'_Test_{j+1}.npz')['Eref']*har2wave
                # energies_imp[i, j] += np.mean(Energy[5000:])
            elif system == 'dimer':
                Energy = np.load(f'Trial_wvfn_testing/results/pdimer_full_imp_samp/' +
                                  f'pdimer_full_imp_samp_{imp_samp_walkers[i]}_' +
                                  f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
                energies_imp[i, j] = np.mean(Energy[5000:]) + 11804.25054877
            elif system == 'trimer':

                Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_full_imp_samp/' +
                                 f'ptrimer_full_imp_samp_{imp_samp_walkers[i]}_' +
                                 f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
                a = -9.129961343400107E-002 * har2wave
                energies_imp[i, j] = np.mean(Energy[5000:]) - a

                Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_imp_samp_waters/' +
                                 f'ptrimer_imp_samp_waters_{imp_samp_walkers[i]}_' +
                                 f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
                a = -9.129961343400107E-002 * har2wave
                energies_imp2[i, j] = np.mean(Energy[5000:]) - a

                Energy = np.load(f'Trial_wvfn_testing/results/ptrimer_full_imp_samp_w_tetramer_params_patched/' +
                                 f'ptrimer_full_imp_samp_w_tetramer_params_patched_{imp_samp_walkers[i]}_' +
                                 f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
                a = -9.129961343400107E-002 * har2wave
                energies_imp3[i, j] = np.mean(Energy[5000:]) - a
            elif system == 'tetramer':
                Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
                                f'ptetramer_full_imp_samp_patched_{imp_samp_walkers[i]}_' +
                                f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
                a = -0.122146858971399 * har2wave
                energies_imp[i, j] = np.mean(Energy[avg_beg:avg_end]) - a
                Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_imp_samp_waters/' +
                                 f'ptetramer_imp_samp_waters_{imp_samp_walkers[i]}_' +
                                 f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
                energies_imp2[i, j] = np.mean(Energy[avg_beg:avg_end]) - a
                Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_w_trimer_params/' +
                                 f'ptetramer_full_imp_samp_w_trimer_params_{imp_samp_walkers[i]}_' +
                                 f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
                a = -0.122146858971399 * har2wave
                energies_imp3[i, j] = np.mean(Energy[avg_beg:avg_end]) - a
            elif system == 'monomer':
                Energy = np.load(f'Trial_wvfn_testing/results/pmonomer_full_imp_samp/' +
                                 f'pmonomer_full_imp_samp_{imp_samp_walkers[i]}_' +
                                 f'Walkers_Test_{j+1}.npz')['Eref'] * har2wave
                a = 0 * har2wave
                energies_imp2[i, j] = np.mean(Energy[5000:]) - a
                Energy = np.load(f'Trial_wvfn_testing/results/pmonomer_full_imp_samp_water/' +
                                 f'pmonomer_full_imp_samp_water_{imp_samp_walkers[i]}_' +
                                 f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
                a = 0 * har2wave
                energies_imp[i, j] = np.mean(Energy[5000:]) - a

    if extra is True:
        for i in range(len(walk)):
            for j in range(trials_i):
                Energy = np.load(f'Trial_wvfn_testing/results/imp_sampled_discrete_analytic_5H_ts_1/' +
                                 f'imp_sampled_discrete_analytic_5H_ts_1_{walk[i]}_' +
                                 f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
                energies_imp3[i, j] = np.mean(Energy[avg_beg:avg_end])
                Energy = np.load(f'Trial_wvfn_testing/results/imp_sampled_analytic_5H_ts_1/' +
                                 f'imp_sampled_analytic_5H_ts_1_{walk[i]}_' +
                                 f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
                energies_imp2[i, j] = np.mean(Energy[avg_beg:avg_end])
    if system == 'tetramer':
        for j in range(trials_i):
            Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_imp_samp_waters/' +
                             f'ptetramer_imp_samp_waters_{30000}_' +
                             f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
            energies_imp2[-1, j] = np.mean(Energy[avg_beg:avg_end]) - a
            Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
                             f'ptetramer_full_imp_samp_patched_{30000}_' +
                             f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
            a = -0.122146858971399 * har2wave
            energies_imp[-2, j] = np.mean(Energy[avg_beg:avg_end]) - a
            Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_full_imp_samp_patched/' +
                             f'ptetramer_full_imp_samp_patched_{40000}_' +
                             f'Walkers_Test_{j + 1}.npz')['Eref'] * har2wave
            a = -0.122146858971399 * har2wave
            energies_imp[-1, j] = np.mean(Energy[avg_beg:avg_end]) - a
        #
        energies_non2 = np.zeros((6, 5))
        walkers = [10000, 20000, 40000, 50000, 60000, 75000]
        for i in range(len(walkers)):
            for j in range(trials_ni):
                Energy = np.load(f'Trial_wvfn_testing/results/ptetramer_non_imp_samp_ts_10/' +
                                 f'ptetramer_non_imp_samp_ts_10_{walkers[i]}_' +
                                 f'Walkers_Test_{j + 1}.npz')
                Energy = Energy['Eref'] * har2wave
                a = -0.122146858971399 * har2wave
                energies_non2[i, j] = np.mean(Energy[avg_beg:avg_end]) - a
        print(energies_non2)
        avg_non_2 = np.mean(energies_non2, axis=1)
        std_non_2 = np.std(energies_non2, axis=1)

    avg_imp = np.mean(energies_imp, axis=1)
    if system == 'tetramer':
        imp_samp_walkers = np.append(imp_samp_walkers, 30000)

    if system == 'trimer':
        avg_imp = np.append(avg_imp, (17998.764572511674))
        imp_samp_walkers = np.append(imp_samp_walkers, (30000))



    print('average energy full importance sampling ' + str(avg_imp))
    std_imp = np.std(energies_imp, axis=1)


    if system == 'trimer':
        std_imp = np.append(std_imp, (3.350799897507758))


    print('std full importance sampling ' + str(std_imp))

    avg_imp2 = np.mean(energies_imp2, axis=1)
    std_imp2 = np.std(energies_imp2, axis=1)


    if system == 'trimer':
        avg_imp2 = np.append(avg_imp2, (18002.15435066399))
        std_imp2 = np.append(std_imp2, (6.0249577448096945))



    print('average energy imp samp waters ' + str(avg_imp2))
    print('std imp samp waters ' + str(std_imp2))

    avg_non = np.mean(energies_non, axis=1)
    if system == 'dimer':
        avg_non = np.append(avg_non, 12359.101235609602-11792.22559467+11804.25054877)
        non_imp_samp_walkers = np.append(non_imp_samp_walkers, 20000)


    print('average energy non importance sampling ' + str(avg_non))

    std_non = np.std(energies_non, axis=1)
    if system == 'dimer':
        std_non = np.append(std_non, 4.1243809112563055)

    print('std non importance sampling ' + str(std_non))

    #trimer w tetramer params
    if system == 'trimer' or system == 'tetramer':
        walk = [2000, 5000, 10000, 15000, 20000]
    if extra:
        energies_imp3 = energies_imp3[:len(walk)]
    avg = np.mean(energies_imp3, axis=1)
    std = np.std(energies_imp3, axis=1)
    print(avg)
    print(std)

    fig, axes = plt.subplots()
    if system == 'CH5':
        axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [ens[isotop]] * 5000, yerr=ens_std[isotop],
                      color='grey', alpha=0.1)
        axes.errorbar(np.linspace(0, non_imp_samp_walkers[-1], 5000), [ens[isotop]] * 5000, yerr=0, color='black', label="Literature")

    elif system == 'monomer':
        axes.errorbar(np.linspace(10000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]]*5000, yerr=[std_imp[-1]]*5000, color='orange', alpha=0.01)
        axes.plot(np.linspace(10000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]] * 5000, color='black', linestyle='dotted', linewidth=2)
        axes.plot(np.linspace(10000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]] * 5000, color='orange',
                  linestyle='dotted', linewidth=2)

    elif system == 'dimer':
        axes.errorbar(np.linspace(5000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]] * 5000, yerr=[std_imp[-1]] * 5000,
                      color='orange', alpha=0.01)
        axes.plot(np.linspace(5000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]] * 5000, color='black',
                  linestyle='dotted', linewidth=2)
        axes.plot(np.linspace(5000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]] * 5000, color='orange',
                  linestyle='dotted', linewidth=2)

    # elif system == 'trimer':
        # axes.errorbar(np.linspace(30000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]] * 5000, yerr=[std_imp[-1]] * 5000,
        #               color='blue', alpha=0.01)
        # axes.plot(np.linspace(30000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]] * 5000, color='black',
        #           linestyle='dotted', linewidth=2)
        # axes.plot(np.linspace(30000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]] * 5000, color='blue',
        #           linestyle='dotted', linewidth=2)

    # elif system == 'tetramer':
    #     axes.errorbar(np.linspace(40000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]] * 5000, yerr=[std_imp[-1]] * 5000,
    #                   color='purple', alpha=0.01)
    #     axes.plot(np.linspace(40000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]] * 5000, color='black',
    #               linestyle='dotted', linewidth=2)
    #     axes.plot(np.linspace(40000, non_imp_samp_walkers[-1], 5000), [avg_imp[-1]] * 5000, color='purple',
    #               linestyle='dotted', linewidth=2)


    axes.errorbar(non_imp_samp_walkers, avg_non, yerr=std_non, marker='s', markerfacecolor='none', color='red', label=r'Unguided $\Delta\tau$ = 1')

    if system == 'trimer' or system == 'tetramer':
        axes.errorbar(imp_samp_walkers, avg_imp2, yerr=std_imp2, color='orange', label='Guided (water)', marker='v')

    # if system == 'trimer':
        # axes.errorbar(imp_samp_walkers, avg_imp, yerr=std_imp, color='blue', label=r'Guided (H$^+($H$_2$O)$_3$)', marker='o')
    # if system == 'tetramer':
    #     axes.errorbar(np.append(imp_samp_walkers, (40000)), avg_imp, yerr=std_imp, color='purple',
    #                   label=r'Guided (H$^+($H$_2$O)$_4$)', marker='o')
    if system == 'monomer' or system == 'dimer':
        axes.errorbar(imp_samp_walkers, avg_imp, yerr=std_imp, color='orange', label='Guided (water)', marker='v')

    if system == 'CH5':
        axes.errorbar(imp_samp_walkers, avg_imp, yerr=std_imp, color='blue', label='Guided', marker='o')
        if extra:
            # walk = [1000, 2000, 3000, 4000, 5000]
            axes.errorbar(walk, avg_imp2, yerr=std_imp2, color='orange', label='Guided (harmonic)', marker='o')
            axes.errorbar(walk, avg, yerr=std, color='purple', label='Guided (discrete harmonic)', marker='o')


    # if system == 'trimer':
    #     axes.errorbar(walk, avg, yerr=std, color='purple', label=r'Guided (H$^+($H$_2$O)$_4$)', marker='o')
    # if system == 'tetramer':
    #     axes.errorbar(walk, avg, yerr=std, color='blue', label=r'Guided (H$^+($H$_2$O)$_3$)', marker='o')
    #
    #     axes.errorbar(walkers,  avg_non_2,
    #                   yerr=std_non_2, marker='s', markerfacecolor='none', color='magenta',
    #                   label=r'Unguided $\Delta\tau$ = 10')


    axes.set_xlabel(r'N$_{\rmw}$', fontsize=28)
    axes.set_ylabel(r'E$_0$ (cm$^{-1}$)', fontsize=28)
    if system == 'CH5':
        axes.set_ylim(lowlim[isotop], highlim[isotop])
    axes.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                     bottom=True, top=False, left=True, right=False, labelsize=20)
    if system == 'tetramer':
       axes.set_ylim(23342, 23542)
    # axes.set_ylim(24301, 24800)
    if system == 'monomer':
        axes.set_ylim(7430, 7530)
    if system == 'trimer':
        axes.set_ylim(17970, 18070)
    if system == 'dimer':
        axes.set_ylim(12350, 12450)
    leg = plt.legend(loc='upper right', fontsize=12)
    leg.get_frame().set_edgecolor('white')
    plt.tight_layout()


walkers1 = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000]
walkers10 = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
walkers12 = [100, 200, 500, 1000, 2000, 5000, 10000]
walkers11 = [100, 200, 500, 2000, 5000]
# walkers11 = [100, 200, 500, 1000, 2000, 5000, 15000]
# walkers67 = [100, 200, 500, 1000, 2000, 5000, 20000]
walkers89 = [2000, 5000, 10000, 15000, 20000]
walkers68 = [2000, 5000, 10000, 15000, 20000, 30000]
# walkers67 = [2000, 5000, 10000, 20000]
walkers73 = [10000, 20000, 40000]
walkers74 = [10000, 20000, 30000, 40000, 50000, 75000, 100000]
walkers75 = [10000, 20000, 30000, 40000, 50000, 100000]
# walkers_dis = [40000]
walkers99 = [50, 100, 200, 500, 2000, 5000]
# walkers3 = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
# walkers5 = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 60000]
# walkers6 = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 40000, 50000, 60000]
# walkers2 = [500, 1000, 2000, 5000, 10000]
# walkers4 = [100, 200, 500, 1000, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
#             5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 20000]
# for i in range(6):
lets_get_some_energies(walkers74, walkers89, 5, 5, 5, system='tetramer', extra=False)
plt.show()
plt.close()



