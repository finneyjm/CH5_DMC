
def changy_change(filepath_pyvib, filename_intended, walkers, wvfn_1, wvfn_minus_1, wvfn_increment):
    import numpy as np
    import pyvibdmc as pv
    sim = pv.SimInfo(f'{filepath_pyvib}')
    energy = sim.get_zpe(ret_cm=False)
    wvfn_list = np.arange(wvfn_1, wvfn_minus_1+wvfn_increment, wvfn_increment)
    cds, dws, weights = sim.get_wfns(wvfn_list)
    cds = cds.reshape((len(wvfn_list), walkers, cds.shape[-2], cds.shape[-1]))
    dws = dws.reshape((len(wvfn_list), walkers))
    weights = weights.reshape((len(wvfn_list), walkers))
    np.savez(f'{filename_intended}', coords=cds, weights=weights, d=dws, Eref=energy)
