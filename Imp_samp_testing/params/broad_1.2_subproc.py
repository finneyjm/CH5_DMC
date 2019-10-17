import subprocess as proc
walkers = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]

for i in range(5):
    for j in walkers:
        proc.call(['python', 'runDMC.py', f'params_min_broad_1_2x_{j}_walkers_test_{i+1}'])
