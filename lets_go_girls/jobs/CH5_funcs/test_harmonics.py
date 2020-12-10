from CH5_funcs.Dev_dep_fd_imp_samp import *

psi = Walkers(20)
m_C = 12.000000000 / (Avo_num*me*1000)
m_H = 1.00782503223 / (Avo_num*me*1000)
dtau = 1
sigmaH = np.sqrt(dtau/m_H)
sigmaC = np.sqrt(dtau/m_C)
sigmaCH = np.zeros((6, 3))
sigmaCH[0] = np.array([[sigmaC] * 3])
for i in np.arange(1, 6):
    sigmaCH[i] = np.array([[sigmaH] * 3])

# disp = np.random.normal(0.0, sigmaCH, (20, 6, 3))

psi.coords = np.load('test_coords_CH5.npy')
disp = np.load('test_displacements.npy')
disp_coords = psi.coords + disp
# np.save('test_disp_coords', disp_coords)
# np.save('test_displacements', disp)


anharm_ch_wvfn = np.load('../params/min_wvfns/new_GSW_min_CH_2.npy')
anharm_interp_filler = interpolate.splrep(anharm_ch_wvfn[0], anharm_ch_wvfn[1], s=0)
anharm_interp = []
for i in range(5):
    anharm_interp.append(anharm_interp_filler)

# harm_ch_wvfn_filler = np.load('../params/min_wvfns/rch_params_GSW2.npy')
# harm_ch_wvfn = []
# for i in range(5):
#     harm_ch_wvfn.append(harm_ch_wvfn_filler)
#
# harm_disc_ch_wvfn = np.load('../params/min_wvfns/GSW_min_CH_2_harm.npy')
# harm_interp_filler = interpolate.splrep(anharm_ch_wvfn[0], harm_disc_ch_wvfn, s=0)
# harm_interp = []
# for i in range(5):
#     harm_interp.append(harm_interp_filler)

hh = np.load('../params/sigma_hh_to_rch_exp_relationship_params.npy')


psi.V = np.load('test_pots.npy')
dr1, psi1 = drift(None, psi.coords, anharm_interp, 'dev_dep', interp_exp=hh, multicore=False, analytic_rch=None)
disp_coords += dr1*sigmaCH**2/2
dr2, psi2 = drift(None, disp_coords, anharm_interp, 'dev_dep', interp_exp=hh, multicore=False, analytic_rch=None)

a = metropolis(dr1, dr2, psi.coords, disp_coords, sigmaCH, psi1, psi2)

new_coords = disp_coords + disp
new_coords += dr2*sigmaCH**2/2

dr3, psi3 = drift(None, new_coords, anharm_interp, 'dev_dep', interp_exp=hh, multicore=False, analytic_rch=None)

a2 = metropolis(dr2, dr3, disp_coords, new_coords, sigmaCH, psi2, psi3)

newer_coords = new_coords + disp
newer_coords += dr3*sigmaCH**2/2

dr4, psi4 = drift(None, newer_coords, anharm_interp, 'dev_dep', interp_exp=hh, multicore=False, analytic_rch=None)

a3 = metropolis(dr3, dr4, new_coords, newer_coords, sigmaCH, psi3, psi4)

psi.psit=psi1
psi = E_loc(psi, sigmaCH, dtau)
el1 = psi.El
kin1 = el1-psi.V





stop = True