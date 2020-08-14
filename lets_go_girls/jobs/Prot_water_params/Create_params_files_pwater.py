walkers = [10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 75000, 100000]
# walkers = [100000]
# walkers = [6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]
# walkers = [5000]
# walkers = [50000]
# bro_str = ['5', '10']
# bro = [5, 10]
atoms = ['H', 'H', 'H', 'O']
ts = 10
thresh = None
thresh_num = 0.1
system = 'pmonomer'
type_of_sim = f'non_imp_samp_ts_10_full'
max_thresh = None
imp_samp = False
weighting = None
params = 'waters'
shift = False
patch = True
deuterated = False
bare = True
# for j in range(len(bro)):
# for ts in bro:
# for j in range(6):
for i in range(5):
    for x in range(len(walkers)):
        with open(f'../params/params_{system}_{type_of_sim}_{walkers[x]}_walkers_test_{i+1}.py', 'w') as myfile:
            myfile.write('import numpy as np\n')
            myfile.write('from scipy import interpolate\n')
            # for hs in range(j):
            #     atoms[hs+1] = 'H'
            myfile.write(f'atoms = {atoms}\n\n')
            if imp_samp:
                if system == 'pmonomer':
                    if shift:
                        myfile.write('wvfn = np.load("Prot_water_params/wvfns/hydronium_shifted_oh_wvfn.npy")\n')
                    elif params == 'water':
                        myfile.write('wvfn = np.load("Prot_water_params/wvfns/free_oh_wvfn.npy")\n')
                    else:
                        myfile.write('wvfn = np.load("Prot_water_params/wvfns/hydronium_oh_wvfn.npy")\n')
                    myfile.write('free_oh_wvfn = interpolate.splrep(wvfn[:, 0], wvfn[:, 1], s=0)\n\n')
                    myfile.write('trial_wvfn = {\n')
                    myfile.write('    "reg_oh": free_oh_wvfn,\n')
                    myfile.write('}\n\n')
                else:
                    if deuterated:
                        myfile.write('wvfn = np.load("Prot_water_params/wvfns/free_od_wvfn.npy")\n')
                    else:
                        myfile.write('wvfn = np.load("Prot_water_params/wvfns/free_oh_wvfn.npy")\n')
                    myfile.write('free_oh_wvfn = interpolate.splrep(wvfn[:, 0], wvfn[:, 1], s=0)\n\n')
                    if params == 'waters':
                        myfile.write('trial_wvfn = {\n')
                        myfile.write('    "reg_oh": free_oh_wvfn,\n')
                        myfile.write('    "ang": None, \n')
                        myfile.write('    "hbond": None,\n')
                        myfile.write('    "OO_shift": None, \n')
                        myfile.write('    "OO_scale": None, \n')
                        myfile.write('}\n\n')
                    elif params == 'tetramer':
                        if deuterated:
                            myfile.write(
                                'hbond_wvfn = np.load("Prot_water_params/wvfns/shared_deuterium_moveable_wvfn.npy")\n')

                        else:
                            myfile.write('hbond_wvfn = np.load("Prot_water_params/wvfns/shared_prot_moveable_wvfn.npy")\n')
                        myfile.write('hbond_wvfn = interpolate.splrep(hbond_wvfn[:, 0], hbond_wvfn[:, 1], s=0)\n\n')
                        myfile.write('trial_wvfn = {\n')
                        myfile.write('    "reg_oh": free_oh_wvfn,\n')
                        myfile.write('    "ang": None, \n')
                        myfile.write('    "hbond": hbond_wvfn,\n')
                        if patch and deuterated is False:
                            myfile.write('    "OO_shift": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                         'bowman_patched_h9o4_Re_Polynomials")),\n')
                            myfile.write('    "OO_scale": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                         'bowman_patched_h9o4_Std_Polynomials"))\n')
                        elif patch and deuterated:
                            myfile.write('    "OO_shift": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                     'bowman_patched_D9O4_Re_Polynomials")),\n')
                            myfile.write('    "OO_scale": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                     'bowman_patched_D9O4_Std_Polynomials"))\n')
                        else:
                            myfile.write('    "OO_shift": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                     'bowman_h9o4_Re_Polynomials")),\n')
                            myfile.write('    "OO_scale": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                     'bowman_h9o4_Std_Polynomials"))\n')
                        myfile.write('}\n\n')
                    elif params == 'trimer':
                        if deuterated:
                            myfile.write(
                                'hbond_wvfn = np.load("Prot_water_params/wvfns/shared_deuterium_moveable_wvfn.npy")\n')

                        else:
                            myfile.write(
                                'hbond_wvfn = np.load("Prot_water_params/wvfns/shared_prot_moveable_wvfn.npy")\n')
                        myfile.write('hbond_wvfn = interpolate.splrep(hbond_wvfn[:, 0], hbond_wvfn[:, 1], s=0)\n\n')
                        myfile.write('trial_wvfn = {\n')
                        myfile.write('    "reg_oh": free_oh_wvfn,\n')
                        myfile.write('    "ang": None, \n')
                        myfile.write('    "hbond": hbond_wvfn,\n')
                        if patch and deuterated is False:
                            myfile.write('    "OO_shift": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                         'bowman_patched_h7o3_Re_Polynomials")),\n')
                            myfile.write('    "OO_scale": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                         'bowman_patched_h7o3_Std_Polynomials"))\n')
                        elif patch and deuterated:
                            myfile.write('    "OO_shift": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                         'bowman_patched_D7O3_Re_Polynomials")),\n')
                            myfile.write('    "OO_scale": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                         'bowman_patched_D7O3_Std_Polynomials"))\n')
                        else:
                            myfile.write('    "OO_shift": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                     'bowman_h7o3_Re_Polynomials")),\n')
                            myfile.write('    "OO_scale": np.array(np.loadtxt("Prot_water_params/shared_prot_params/'
                                     'bowman_h7o3_Std_Polynomials"))\n')
                        myfile.write('}\n\n')

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
            myfile.write(f'    "system": "{system}",\n')
            myfile.write('    "time_steps": 18000,\n')
            myfile.write(f'    "dtau": {ts},\n')
            myfile.write('    "multicore": True,\n')
            myfile.write('    "equilibration": 5000,\n')
            myfile.write('    "wait_time": 500,\n')
            myfile.write(f'    "output": "{system}_{type_of_sim}_{walkers[x]}_Walkers_Test_{i+1}",\n')
            myfile.write('    "atoms": atoms,\n')
            if imp_samp:
                myfile.write('    "imp_samp": True,\n')
                myfile.write('    "trial_wvfn": trial_wvfn,\n')
            if thresh is not None:
                myfile.write(f'    "threshold": {thresh_num/walkers[x]}\n')
            if weighting == 'discrete':
                myfile.write(f'    "weighting": "discrete"\n')
            if max_thresh is not None:
                myfile.write(f'    "max_thesh": {max_thresh}\n')
            # if bare:
            #     myfile.write(f'    "bare_dimer": {bare}\n')
            # myfile.write('    "imp_samp_type": "dev_dep",\n')
            # myfile.write('    "hh_relate": hh,\n')
            # myfile.write('    "rand_samp": False,\n')
            myfile.write('}\n\n')
            myfile.write(f'output_dir = "{system}_{type_of_sim}"\n')
            myfile.close()
