
sim = ['asym_left_1d', 'asym_right_1d', 'XH_left_2d', 'XH_right_2d']
sim = ['asym_left_full', 'asym_right_full', 'XH_left_full', 'XH_right_full']
# sim = ['ground_1d', 'ground_2d']
# sim = ['ground_full']
for i in range(5):
    for type_of_sim in sim:
        with open(f'DMC_h3o2_{type_of_sim}_{i+1}.sh', 'w') as myfile:
            myfile.write('#!/bin/bash\n\n')
            myfile.write('## Job Name\n\n')
            myfile.write(f'#SBATCH --job-name=h3o2_{type_of_sim}_{i+1}\n\n')
            myfile.write('## Allocation Definition\n\n')
            myfile.write('## Which queue should we use?\n\n')
            myfile.write('#SBATCH --partition=ilahie\n\n')
            myfile.write('#SBATCH --account=ilahie\n\n')
            myfile.write('## Number of cores\n\n')
            myfile.write('#SBATCH --tasks=28\n\n')
            myfile.write('## Number of nodes\n\n')
            myfile.write('#SBATCH --nodes=1\n\n')
            # myfile.write('#SBATCH --exclude=n2023\n\n')
            myfile.write('## Time needed (days-hours:minutes:seconds\n\n')
            myfile.write('#SBATCH --time=0-12:00:00\n\n')
            myfile.write('## Memory per node\n\n')
            myfile.write('#SBATCH --mem=122G\n\n')
            myfile.write('## Where is the working directory of this job?\n\n')
            myfile.write('#SBATCH --chdir=.\n\n')
            myfile.write('## Where should the output go?\n\n')
            myfile.write(f'#SBATCH -o check_h3o2_{type_of_sim}_{ i +1}.dat\n\n')
            myfile.write('module load contrib/python/3.6.3\n')
            myfile.write('START=$(date +%s.%N)\n\n')
            myfile.write(f'python chain_rule_h3o2_dmc_new_drift_{type_of_sim}_{i + 1}_param.py\n')

            myfile.write('wait\n')
            myfile.write('END=$(date +%s.%N)\n')
            myfile.write('DIFF=$(echo "$END - $START" | bc)\n')
            myfile.write(f'echo "h3o2_{type_of_sim}_{ i +1} took: $DIFF"')