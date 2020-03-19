walkers = [[100, 200, 500, 1000, 2000], [5000, 10000], [20000]]
# walkers = [[6000, 6500, 7000], [7500, 8000, 8500], [9000, 9500]]
walkers = [[15000], [25000]]
walkers = [[1000], [2000], [5000]]
# walkers = [[30000]]
size = ['small', 'med', 'large']
# size = ['med', 'large']
# size = ['biggo']
bro_str = ['5', '10']
bro = [5, 10]

# for ts in bro:
# for k in range(4):
#         for j in range(len(bro)):
            # for i in range(5):
for j in range(6):
    # for i in range(5):
        for a, b in zip(walkers, size):
            with open(f'Non_imp_sampled_{j}H_ts_10_subproc_{b}.py', 'w') as myfile:
                myfile.write('import subprocess as proc\n')
                myfile.write(f'walkers = {a}\n\n')
                myfile.write('for j in walkers:\n')
                myfile.write('    for i in range(5):\n')
                myfile.write(f'        proc.call(["python", "runDMC.py", '
                             f'f"params_Non_imp_sampled_{j}H_ts_10_{{j}}_walkers_test_{{i+1}}"])\n')
                myfile.close()

