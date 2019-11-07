walkers = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
bro_str = ['1_2', '1_3', '1_4', '1_5']
bro = [1.2, 1.3, 1.4, 1.5]

for i in range(5):
    for x in range(len(walkers)):
        with open(f'params_average_{walkers[x]}_walkers_test_{i+1}.py', 'w') as myfile:
            myfile.write('import numpy as np\n\n')
            myfile.write(f'wvfn = np.load(f"params/min_wvfns/Average_min_broadening_1.0x.npy")\n\n')
            myfile.write('pars = {\n')
            myfile.write(f'    "N_0": {walkers[x]},\n')
            myfile.write('    "time_steps": 20000,\n')
            myfile.write('    "dtau": 1,\n')
            myfile.write('    "equilibration": 5000,\n')
            myfile.write('    "wait_time": 500,\n')
            myfile.write(f'    "output": "average_wvfn_old_way_{walkers[x]}_Walkers_Test_{i+1}",\n')
            myfile.write('    "imp_samp": True,\n')
            myfile.write('    "trial_wvfn": wvfn')
            myfile.write('}\n\n')
            myfile.write(f'output_dir = "average_wvfn"\n')
            myfile.close()
