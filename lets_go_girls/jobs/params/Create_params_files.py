walkers = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
# walkers = [6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]
bro_str = ['5', '10']
bro = [5, 10]

# for j in range(len(bro)):
for i in range(5):
    for x in range(len(walkers)):
        with open(f'params_DD_to_rCD_4H_{walkers[x]}_walkers_test_{i+1}.py', 'w') as myfile:
            myfile.write('import numpy as np\n\n')
            myfile.write('atoms = ["C", "H", "H", "H", "H", "D"]\n')
            myfile.write(f'wvfn_D = np.load("params/min_wvfns/Min_CD_average.npy")\n')
            myfile.write(f'wvfn_H = np.load("params/min_wvfns/Min_CH_average.npy")\n\n')
            myfile.write('wvfn = np.zeros((5, 2, 5000))\n')
            myfile.write('for i in range(5):\n')
            myfile.write('    if atoms[i+1] == "H":\n')
            myfile.write('        wvfn[i] = wvfn_H\n')
            myfile.write('    else:\n')
            myfile.write('        wvfn[i] = wvfn_D\n')
            myfile.write(f'hh = np.load("params/sigma_hh_to_rch_cub_relationship.npy")\n\n')
            # myfile.write(f'wvfn[0, :] = (wvfn[0, :] - wvfn[0, np.argmax(wvfn[1, :])])*{bro[j]}'
            #              f' + wvfn[0, np.argmax(wvfn[1, :])]\n\n')
            # myfile.write('for CH in range(5):\n')
            # myfile.write(f'    wvfn[CH] = np.vstack((x, np.load(f"params/min_wvfns/GSW_min_CH_{{CH+1}}.npy")))\n\n')
            myfile.write('pars = {\n')
            myfile.write(f'    "N_0": {walkers[x]},\n')
            myfile.write('    "time_steps": 20000,\n')
            myfile.write('    "dtau": 1,\n')
            myfile.write('    "equilibration": 5000,\n')
            myfile.write('    "wait_time": 500,\n')
            myfile.write(f'    "output": "DD_to_rCD_4H_{walkers[x]}_Walkers_Test_{i+1}",\n')
            myfile.write('    "atoms": atoms,\n')
            myfile.write('    "imp_samp": True,\n')
            myfile.write('    "trial_wvfn": wvfn,\n')
            myfile.write('    "imp_samp_type": "dev_dep",\n')
            myfile.write('    "hh_relate": hh,\n')
            # myfile.write('    "rand_samp": False,\n')
            myfile.write('}\n\n')
            myfile.write(f'output_dir = "DD_to_rCD_4H"\n')
            myfile.close()
