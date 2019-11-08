walkers = [[100, 200, 500, 1000, 2000, 5000], [10000], [20000]]
size = ['small', 'med', 'large']

for i in range(5):
    for a, b in zip(walkers, size):
        with open(f'Switch_subproc_{i+1}_{b}.py', 'w') as myfile:
            myfile.write('import subprocess as proc\n')
            myfile.write(f'walkers = {a}\n\n')
            myfile.write('for j in walkers:\n')
            myfile.write(f'    proc.call(["python", "runDMC.py", f"params_switch_min_speed_1_{{j}}_walkers_test_{i+1}"])\n')
            myfile.close()

