import numpy as np


blah = np.load('Trial_wvfn_testing/results/HH_to_rCHrCD_5H_GSW2/HH_to_rCHrCD_5H_GSW2_5000_Walkers_Test_2.npz')
a = blah['accept']

print(a)



