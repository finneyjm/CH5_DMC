
sim = ['ground']
for i in range(10):
    for type_of_sim in sim:
        with open(f'DMC_h3o2_{type_of_sim}_{ i +1}.sh', 'w') as myfile:
            myfile.write('#!/bin/bash\n\n')
            myfile.write('## Job Name\n\n')
            myfile.write(f'#SBATCH --job-name=h3o2_{type_of_sim}_{ i +1}\n\n')
            myfile.write('## Allocation Definition\n\n')
            myfile.write('## Which queue should we use?\n\n')
            myfile.write('#SBATCH --partition=ckpt\n\n')
            myfile.write('#SBATCH --account=chem-ckpt\n\n')
            myfile.write('## Number of cores\n\n')
            myfile.write('#SBATCH --tasks=28\n\n')
            myfile.write('## Number of nodes\n\n')
            myfile.write('#SBATCH --nodes=1\n\n')
            # myfile.write('#SBATCH --exclude=n2023\n\n')
            myfile.write('## Time needed (days-hours:minutes:seconds\n\n')
            myfile.write('#SBATCH --time=0-04:00:00\n\n')
            myfile.write('## Memory per node\n\n')
            myfile.write('#SBATCH --mem=122G\n\n')
            myfile.write('## Where is the working directory of this job?\n\n')
            myfile.write('#SBATCH --chdir=.\n\n')
            myfile.write('## Where should the output go?\n\n')
            myfile.write(f'#SBATCH -o check_h3o2_{type_of_sim}_{ i +1}.dat\n\n')
            myfile.write('module load contrib/python/3.6.3\n')
            myfile.write('START=$(date +%s.%N)\n\n')
            myfile.write(f'python brute_force_h3o2_dmc_{type_of_sim}_{i + 1}_param.py\n')

            myfile.write('wait\n')
            myfile.write('END=$(date +%s.%N)\n')
            myfile.write('DIFF=$(echo "$END - $START" | bc)\n')
            myfile.write(f'echo "h3o2_{type_of_sim}_{ i +1} took: $DIFF"')