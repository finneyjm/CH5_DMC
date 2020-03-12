walkers = [[1000], [2000], [5000]]

size = ['small', 'med', 'large']

system = 'ptetramer'
type_of_sim = 'non_imp_samp'


for a, b in zip(walkers, size):
    with open(f'{system}_{type_of_sim}_subproc_{b}.py', 'w') as myfile:
        myfile.write('import subprocess as proc\n')
        myfile.write(f'walkers = {a}\n\n')
        myfile.write('for j in walkers:\n')
        myfile.write('    for i in range(5):\n')
        myfile.write(f'        proc.call(["python", "runDMC.py", '
                     f'f"params_{system}_{type_of_sim}_{{j}}_walkers_test_{{i+1}}"])\n')
        myfile.close()

