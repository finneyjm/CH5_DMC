import numpy as np
from CH5_funcs.Dev_dep_fd_imp_samp import *

blah = np.load(f'Trial_wvfn_testing/results/HH_to_rCHrCD_{5}H_GSW2/' +
                                 f'HH_to_rCHrCD_{5}H_GSW2_{10000}_' +
                                 f'Walkers_Test_{1}.npz')['coords']

coords = blah[-1]

the_real_thing = coords[:5120]

hh_relate = np.load('../lets_go_girls/jobs/params/sigma_hh_to_rch_exp_relationship_params.npy')
anharm_ch_wvfn = np.load('../lets_go_girls/jobs/params/min_wvfns/new_GSW_min_CH_2.npy')

anharm_interp_filler = interpolate.splrep(anharm_ch_wvfn[0], anharm_ch_wvfn[1], s=0)
anharm_interp = []
for i in range(5):
    anharm_interp.append(anharm_interp_filler)

dx = 1e-3
dtau = 1
sigmaH = np.sqrt(dtau / m_H)
sigmaC = np.sqrt(dtau / m_C)
sigmaCH = np.zeros((6, 3))
sigmaCH[0] = np.array([[sigmaC] * 3])
for i in np.arange(1, 6):
    sigmaCH[i] = np.array([[sigmaH] * 3])

psi = Walkers(5120)
psi.coords = the_real_thing

psi.psit = psi_t(ch_dist(psi.coords), anharm_interp, 'dev_dep', psi.coords, interp_exp=hh_relate)
kin = local_kinetic(psi, sigmaCH, dtau)

def easy_Parr_Potential(coords):
    crds = np.array_split(coords, mp.cpu_count()-1)
    V = pool.map(get_pot, crds)
    new_V = np.concatenate(V)
    return new_V

v = easy_Parr_Potential(psi.coords)

elocal = v + kin


stop = True



