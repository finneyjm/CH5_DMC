size = ['small', 'med', 'large']
s = ['s', 'm', 'l']
# size = ['med']
# s = ['m']
system = 'pmonomer'
type_of_sim = 'full_imp_samp_water'
# thresh = ['half', 'one', 'five', 'ten', 'twenty']

# for i in thresh:
# for a, b in zip(size, s):
#     with open(f'DMC_{system}_{type_of_sim}_{a}.sh', 'w') as myfile:
#         myfile.write('#!/bin/bash\n\n')
#         myfile.write('## Job Name\n\n')
#         myfile.write(f'#SBATCH --job-name={system}_{type_of_sim}_{b}\n\n')
#         myfile.write('## Allocation Definition\n\n')
#         myfile.write('## Which queue should we use?\n\n')
#         if b == 's' or b == 'm' or b == 'l':
#             myfile.write('#SBATCH --partition=ilahie\n\n')
#             myfile.write('#SBATCH --account=ilahie\n\n')
#             myfile.write('## Number of cores\n\n')
#             myfile.write('#SBATCH --tasks=28\n\n')
#         else:
#             myfile.write('#SBATCH --partition=chem\n\n')
#             myfile.write('#SBATCH --account=chem\n\n')
#         # myfile.write('#SBATCH --partition=ckpt\n\n')
#         # myfile.write('#SBATCH --account=chem-ckpt\n\n')
#             myfile.write('## Number of cores\n\n')
#             myfile.write('#SBATCH --tasks=40\n\n')
#         myfile.write('## Number of nodes\n\n')
#         myfile.write('#SBATCH --nodes=1\n\n')
#         myfile.write('#SBATCH --exclude=n2023\n\n')
#         myfile.write('## Time needed (days-hours:minutes:seconds\n\n')
#         myfile.write('#SBATCH --time=00-08:00:00\n\n')
#         myfile.write('## Memory per node\n\n')
#         myfile.write('#SBATCH --mem=122G\n\n')
#         myfile.write('## Where is the working directory of this job?\n\n')
#         myfile.write('#SBATCH --chdir=.\n\n')
#         myfile.write('## Where should the output go?\n\n')
#         myfile.write('#SBATCH -o check1.dat\n\n')
#         myfile.write('module load contrib/python/3.6.3\n')
#         myfile.write('START=$(date +%s.%N)\n\n')
#         myfile.write(f'python {system}_{type_of_sim}_subproc_{a}.py\n')
#         myfile.write('wait\n')
#         myfile.write('END=$(date +%s.%N)\n')
#         myfile.write('DIFF=$(echo "$END - $START" | bc)\n')
#         myfile.write(f'echo "{system}_{type_of_sim}_{b} took: $DIFF"')

size = ['large']
s = ['l']
system = 'water'

type_of_sim = 'non_imp_samp_ts_10'

for i in range(5):
    for a, b in zip(size, s):
        with open(f'DMC_{system}_{type_of_sim}_{a}_{i+1}.sh', 'w') as myfile:
            myfile.write('#!/bin/bash\n\n')
            myfile.write('## Job Name\n\n')
            myfile.write(f'#SBATCH --job-name={system}_{type_of_sim}_{b}_{i+1}\n\n')
            myfile.write('## Allocation Definition\n\n')
            myfile.write('## Which queue should we use?\n\n')
            # if b == 's' or b == 'm':
            if i > 3:
                myfile.write('#SBATCH --partition=ckpt\n\n')
                myfile.write('#SBATCH --account=chem-ckpt\n\n')
                myfile.write('## Number of cores\n\n')
                myfile.write('#SBATCH --tasks=28\n\n')

            elif i < 1:
                myfile.write('#SBATCH --partition=ckpt\n\n')
                myfile.write('#SBATCH --account=chem-ckpt\n\n')
                myfile.write('## Number of cores\n\n')
                myfile.write('#SBATCH --tasks=28\n\n')

            elif i == 3:
                myfile.write('#SBATCH --partition=ckpt\n\n')
                myfile.write('#SBATCH --account=chem-ckpt\n\n')
                myfile.write('## Number of cores\n\n')
                myfile.write('#SBATCH --tasks=28\n\n')
            else:
                myfile.write('#SBATCH --partition=ckpt\n\n')
                myfile.write('#SBATCH --account=chem-ckpt\n\n')
                myfile.write('## Number of cores\n\n')
                myfile.write('#SBATCH --tasks=28\n\n')
            myfile.write('## Number of nodes\n\n')
            myfile.write('#SBATCH --nodes=1\n\n')
            # myfile.write('#SBATCH --exclude=n2023\n\n')
            myfile.write('## Time needed (days-hours:minutes:seconds\n\n')
            if b == 's':
                myfile.write('#SBATCH --time=0-12:00:00\n\n')
            elif b == 'm':
                myfile.write('#SBATCH --time=0-12:00:00\n\n')
            elif b == 'l':
                myfile.write('#SBATCH --time=0-04:00:00\n\n')
            elif b == 'b':
                myfile.write('#SBATCH --time=10-00:00:00\n\n')
            elif b == 'vb':
                myfile.write('#SBATCH --time=3-00:00:00\n\n')
            myfile.write('## Memory per node\n\n')
            myfile.write('#SBATCH --mem=122G\n\n')
            myfile.write('## Where is the working directory of this job?\n\n')
            myfile.write('#SBATCH --chdir=.\n\n')
            myfile.write('## Where should the output go?\n\n')
            myfile.write(f'#SBATCH -o check_{system}_{type_of_sim}_{b}_{i+1}.dat\n\n')
            myfile.write('module load contrib/python/3.6.3\n')
            myfile.write('START=$(date +%s.%N)\n\n')
            if b == 's':
                # myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_1000_walkers_test_{i+1}\n')
                # myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_100_walkers_test_{i+1}\n')
                # myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_200_walkers_test_{i+1}\n')
                # myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_500_walkers_test_{i+1}\n')
                myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_20000_walkers_test_{i+1}\n')
            elif b == 'm':
                # myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_2000_walkers_test_{i+1}\n')
                myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_25000_walkers_test_{i+1}\n')
            elif b == 'l':
                myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_10000_walkers_test_{i+1}\n')
            elif b == 'b':
                myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_75000_walkers_test_{i + 1}\n')
            elif b == 'vb':
                myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_40000_walkers_test_{i + 1}\n')

            myfile.write('wait\n')
            myfile.write('END=$(date +%s.%N)\n')
            myfile.write('DIFF=$(echo "$END - $START" | bc)\n')
            myfile.write(f'echo "{system}_{type_of_sim}_{b}_{i+1} took: $DIFF"')