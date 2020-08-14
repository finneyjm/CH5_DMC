import numpy as np

CH_wvfn = np.load('min_wvfns/GSW_min_CH_2.npy')
CD_wvfn = np.load('min_wvfns/GSW_min_CD_2.npy')

OH_wvfn = np.load('wvfns/free_oh_wvfn.npy')

x = np.linspace(0.4, 6, 5000)
np.savetxt('ch_stretch_wvfn', np.hstack((x, CH_wvfn)))
np.savetxt('cd_stretch_wvfn', np.hstack((x, CD_wvfn)))
np.savetxt('oh_stretch_wvfn', OH_wvfn)

# np.savetxt('x', x)











