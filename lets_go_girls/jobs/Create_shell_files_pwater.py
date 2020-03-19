size = ['med']
s = ['m']

system = 'ptetramer'
type_of_sim = 'non_imp_samp_ts_10_thresh_pone'
# thresh = ['half', 'one', 'five', 'ten', 'twenty']

# for i in thresh:
for a, b in zip(size, s):
    with open(f'DMC_{system}_{type_of_sim}_{a}.sh', 'w') as myfile:
        myfile.write('#!/bin/bash\n\n')
        myfile.write('## Job Name\n\n')
        myfile.write(f'#SBATCH --job-name={system}_{type_of_sim}_{b}\n\n')
        myfile.write('## Allocation Definition\n\n')
        myfile.write('## Which queue should we use?\n\n')
        if b == 's' or b == 'm':
            myfile.write('#SBATCH --partition=chem\n\n')
            myfile.write('#SBATCH --account=chem\n\n')
            myfile.write('## Number of cores\n\n')
            myfile.write('#SBATCH --tasks=40\n\n')
        else:
            myfile.write('#SBATCH --partition=ilahie\n\n')
            myfile.write('#SBATCH --account=ilahie\n\n')
        # myfile.write('#SBATCH --partition=ckpt\n\n')
        # myfile.write('#SBATCH --account=chem-ckpt\n\n')
            myfile.write('## Number of cores\n\n')
            myfile.write('#SBATCH --tasks=28\n\n')
        myfile.write('## Number of nodes\n\n')
        myfile.write('#SBATCH --nodes=1\n\n')
        myfile.write('#SBATCH --exclude=n2023\n\n')
        myfile.write('## Time needed (days-hours:minutes:seconds\n\n')
        myfile.write('#SBATCH --time=1-12:00:00\n\n')
        myfile.write('## Memory per node\n\n')
        myfile.write('#SBATCH --mem=122G\n\n')
        myfile.write('## Where is the working directory of this job?\n\n')
        myfile.write('#SBATCH --chdir=.\n\n')
        myfile.write('## Where should the output go?\n\n')
        myfile.write('#SBATCH -o check1.dat\n\n')
        myfile.write('module load contrib/python/3.6.3\n')
        myfile.write('START=$(date +%s.%N)\n\n')
        myfile.write(f'python {system}_{type_of_sim}_subproc_{a}.py\n')
        myfile.write('wait\n')
        myfile.write('END=$(date +%s.%N)\n')
        myfile.write('DIFF=$(echo "$END - $START" | bc)\n')
        myfile.write(f'echo "{system}_{type_of_sim}_{b} took: $DIFF"')

size = ['large']
s = ['l']

type_of_sim = 'non_imp_samp_ts_10'

# for i in range(5):
#     for a, b in zip(size, s):
#         with open(f'DMC_{system}_{type_of_sim}_{a}_{i+1}.sh', 'w') as myfile:
#             myfile.write('#!/bin/bash\n\n')
#             myfile.write('## Job Name\n\n')
#             myfile.write(f'#SBATCH --job-name={system}_{type_of_sim}_{b}_{i+1}\n\n')
#             myfile.write('## Allocation Definition\n\n')
#             myfile.write('## Which queue should we use?\n\n')
#             # if b == 's' or b == 'm':
#             myfile.write('#SBATCH --partition=ilahie\n\n')
#             myfile.write('#SBATCH --account=ilahie\n\n')
#             myfile.write('## Number of cores\n\n')
#             myfile.write('#SBATCH --tasks=28\n\n')
#             myfile.write('## Number of nodes\n\n')
#             myfile.write('#SBATCH --nodes=1\n\n')
#             myfile.write('#SBATCH --exclude=n2023\n\n')
#             myfile.write('## Time needed (days-hours:minutes:seconds\n\n')
#             if b == 's':
#                 myfile.write('#SBATCH --time=0-10:00:00\n\n')
#             elif b == 'm':
#                 myfile.write('#SBATCH --time=0-12:00:00\n\n')
#             elif b == 'l':
#                 myfile.write('#SBATCH --time=0-06:00:00\n\n')
#             myfile.write('## Memory per node\n\n')
#             myfile.write('#SBATCH --mem=122G\n\n')
#             myfile.write('## Where is the working directory of this job?\n\n')
#             myfile.write('#SBATCH --chdir=.\n\n')
#             myfile.write('## Where should the output go?\n\n')
#             myfile.write('#SBATCH -o check1.dat\n\n')
#             myfile.write('module load contrib/python/3.6.3\n')
#             myfile.write('START=$(date +%s.%N)\n\n')
#             if b == 's':
#                 # myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_100_walkers_test_{i+1}\n')
#                 # myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_200_walkers_test_{i+1}\n')
#                 # myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_500_walkers_test_{i+1}\n')
#                 myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_6000_walkers_test_{i+1}\n')
#             elif b == 'm':
#                 myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_7000_walkers_test_{i+1}\n')
#                 # myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_5000_walkers_test_{i+1}\n')
#             elif b == 'l':
#                 myfile.write(f'python runDMC.py params_{system}_{type_of_sim}_40000_walkers_test_{i+1}\n')
#
#             myfile.write('wait\n')
#             myfile.write('END=$(date +%s.%N)\n')
#             myfile.write('DIFF=$(echo "$END - $START" | bc)\n')
#             myfile.write(f'echo "{system}_{type_of_sim}_{b}_{i+1} took: $DIFF"')