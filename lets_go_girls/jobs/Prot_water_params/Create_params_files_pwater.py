walkers = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000]
# walkers = [6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]
# walkers = [50000]
# bro_str = ['5', '10']
# bro = [5, 10]
atoms = ['H', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O', 'H', 'H', 'O']
ts = 10
# for j in range(len(bro)):
# for ts in bro:
# for j in range(6):
for i in range(5):
    for x in range(len(walkers)):
        with open(f'../params/params_ptetramer_non_imp_samp_ts_{ts}_{walkers[x]}_walkers_test_{i+1}.py', 'w') as myfile:
            myfile.write('import numpy as np\n\n')
            # myfile.write('from scipy import interpolate\n')
            # for hs in range(j):
            #     atoms[hs+1] = 'H'
            myfile.write(f'atoms = {atoms}\n')
            # myfile.write('wvfn = np.load("params/wvfns/free_oh_wvfn.npy")\n')
            # myfile.write('hbond_wvfn = np.load("Prot_water_params/wvfns/shared_prot_moveable_wvfn.npy")\n')
            # myfile.write('free_oh_wvfn = interpolate.splrep(wvfn[:, 0], wvfn[:, 1], s=0)\n')
            # myfile.write('hbond_wvfn = interpolate.splrep(hbond_wvfn[:, 0], hbond_wvfn[:, 1], s=0)\n')
            # myfile.write('trial_wvfn = {\n')
            # myfile.write('    "reg_oh": free_oh_wvfn,\n')
            # myfile.write('    "ang": None, \n')
            # myfile.write('    "hbond": hbond_wvfn,\n')
            # myfile.write('    "OO_shift": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
            #              'bowman_h7o3_Re_Polynomials")),\n')
            # myfile.write('    "OO_scale": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
            #              'bowman_h7o3_Std_Polynomials"))\n')
            # myfile.write('}\n\n')
            # myfile.write(f'wvfn_D = np.load("params/min_wvfns/GSW_min_CD_2.npy")\n')
            # myfile.write(f'wvfn_H = np.load("params/min_wvfns/GSW_min_CH_2.npy")\n\n')
            # myfile.write('wvfn = np.zeros((5, 5000))\n')
            # myfile.write('for i in range(5):\n')
            # myfile.write('    if atoms[i+1] == "H":\n')
            # myfile.write('        wvfn[i] = wvfn_H\n')
            # myfile.write('    else:\n')
            # myfile.write('        wvfn[i] = wvfn_D\n')
            # myfile.write(f'hh = np.load("params/sigma_hh_to_rch_exp_relationship_params.npy")\n\n')
            # myfile.write(f'wvfn[0, :] = (wvfn[0, :] - wvfn[0, np.argmax(wvfn[1, :])])*{bro[j]}'
            #              f' + wvfn[0, np.argmax(wvfn[1, :])]\n\n')
            # myfile.write('for CH in range(5):\n')
            # myfile.write(f'    wvfn[CH] = np.vstack((x, np.load(f"params/min_wvfns/GSW_min_CH_{{CH+1}}.npy")))\n\n')
            myfile.write('pars = {\n')
            myfile.write(f'    "N_0": {walkers[x]},\n')
            myfile.write('    "system": "ptetramer",\n')
            myfile.write('    "time_steps": 20000,\n')
            myfile.write(f'    "dtau": {ts},\n')
            myfile.write('    "multicore": True,\n')
            myfile.write('    "equilibration": 5000,\n')
            myfile.write('    "wait_time": 500,\n')
            myfile.write(f'    "output": "ptetramer_non_imp_samp_ts_{ts}_{walkers[x]}_Walkers_Test_{i+1}",\n')
            myfile.write('    "atoms": atoms,\n')
            # myfile.write('    "imp_samp": True,\n')
            # myfile.write('    "trial_wvfn": trial_wvfn,\n')
            # myfile.write('    "imp_samp_type": "dev_dep",\n')
            # myfile.write('    "hh_relate": hh,\n')
            # myfile.write('    "rand_samp": False,\n')
            myfile.write('}\n\n')
            myfile.write(f'output_dir = "ptetramer_non_imp_samp_ts_{ts}"\n')
            myfile.close()
