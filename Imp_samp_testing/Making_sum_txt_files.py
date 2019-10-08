import numpy as np
import CH5pot

har2wave = 219474.6
psi_t = np.load('Fits_CH_stretch_wvfns/Average_min_no_fit.npy')
np.savetxt('Fits_CH_stretch_wvfns/Average_min_trial_wvfn.txt', psi_t.T, delimiter=' ')

walkers = np.load('Trial_wvfn_testing/Avg_wvfn/coords/Imp_samp_DMC_CH5_coords_avg_20000_walkers_5.npy')
np.savetxt('walker_coords_20000.txt', walkers.reshape((len(walkers), 18)), delimiter=' ')

weights = np.load('Trial_wvfn_testing/Avg_wvfn/weights/Imp_samp_DMC_CH5_weights_avg_20000_walkers_5.npy')
np.savetxt('walker_weights_20000.txt', weights[0, :], delimiter=' ')

V = np.array(CH5pot.mycalcpot(walkers))


energies = np.load('Trial_wvfn_testing/Avg_wvfn/energies/Imp_samp_CH5_energy_avg_20000_walkers_5.npy')
# print(energies[1, -250]*har2wave)

walkers2 = np.load('Non_imp_sampled/DMC_CH5_coords_20000_walkers_5.npy')
np.savetxt('walker_coords_ni_20000.txt', walkers2.reshape((len(walkers2), 18)), delimiter=' ')

weights2 = np.load('Non_imp_sampled/DMC_CH5_weights_20000_walkers_5.npy')
np.savetxt('walker_weights_ni_20000.txt', weights2[0, :], delimiter=' ')
print(np.sum(weights2[1, :]))

energies = np.load('Non_imp_sampled/DMC_CH5_Energy_20000_walkers_5.npy')
en = energies[1, :]*har2wave
print(energies[0, -250])
print(energies[1, -250]*har2wave)

V = np.array(CH5pot.mycalcpot(walkers2))
Vref = sum(weights2[0, :]*V)/sum(weights2[0, :]) - 0.5 * np.log(sum(weights2[0, :])/20000.)
print(Vref*har2wave)



