walkers = [[100, 200, 500, 1000, 2000, 5000], [10000], [20000]]
size = ['small', 'med', 'large']
bro_str = ['1_01', '1_05', '1_10', '1_50']
bro = [1.01, 1.05, 1.10, 1.50]

# for j in range(len(bro)):
for i in range(5):
    for a, b in zip(walkers, size):
        with open(f'Average_min_fd_subproc_{i+1}_{b}.py', 'w') as myfile:
            myfile.write('import subprocess as proc\n')
            myfile.write(f'walkers = {a}\n\n')
            myfile.write('for j in walkers:\n')
            myfile.write(f'    proc.call(["python", "runDMC.py", f"params_average_fd_{{j}}_walkers_test_{i+1}"])\n')
            myfile.close()

