walkers = [[100, 200, 500, 1000, 2000, 5000], [10000], [20000]]
walkers = [[6000, 6500, 7000], [7500, 8000, 8500], [9000, 9500]]
size = ['small', 'med', 'large']
bro_str = ['5', '10']
bro = [5, 10]

for j in range(len(bro)):
    # for i in range(5):
    for a, b in zip(walkers, size):
        with open(f'HH_to_rCH_min_wvfn_subproc_more_extras_{b}.py', 'w') as myfile:
            myfile.write('import subprocess as proc\n')
            myfile.write(f'walkers = {a}\n\n')
            myfile.write('for j in walkers:\n')
            myfile.write('    for i in range(5):\n')
            myfile.write(f'        proc.call(["python", "runDMC.py", f"params_HH_to_rCH_min_wvfn_{{j}}_walkers_test_{{i+1}}"])\n')
            myfile.close()

