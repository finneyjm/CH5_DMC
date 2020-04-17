walkers = [[100, 200, 500, 1000, 2000], [10000, 5000], [20000]]
# walkers = [[20000]]
size = ['small', 'med', 'large']
# size = ['med']
system = 'pdimer'
type_of_sim = 'non_imp_samp'
# thresh = ['half', 'one', 'five', 'ten', 'twenty']
# for i in thresh:
for a, b in zip(walkers, size):
    with open(f'{system}_{type_of_sim}_subproc_{b}.py', 'w') as myfile:
        myfile.write('import subprocess as proc\n')
        myfile.write(f'walkers = {a}\n\n')
        myfile.write('for j in walkers:\n')
        myfile.write('    for i in range(5):\n')
        myfile.write(f'        proc.call(["python", "runDMC.py", '
                     f'f"params_{system}_{type_of_sim}_{{j}}_walkers_test_{{i+1}}"])\n')
        myfile.close()

