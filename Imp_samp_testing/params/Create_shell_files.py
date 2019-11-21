size = ['small', 'med', 'large']
s = ['s', 'm', 'l']
bro_str = ['1_01', '1_05', '1_10', '1_50']
bro = [1.01, 1.05, 1.10, 1.50]

# for j in range(len(bro)):
for i in range(5):
    for a, b in zip(size, s):
        with open(f'DMC_average_fd_{a}_{i+1}.sh', 'w') as myfile:
            myfile.write('#!/bin/bash\n\n')
            myfile.write('## Job Name\n\n')
            myfile.write(f'#SBATCH --job-name=Min_wvfn_fd_{b}_{i+1}\n\n')
            myfile.write('## Allocation Definition\n\n')
            myfile.write('## Which queue should we use?\n\n')
            myfile.write('#SBATCH --partition=ckpt\n\n')
            myfile.write('#SBATCH --account=chem-ckpt\n\n')
            myfile.write('## Number of cores\n\n')
            myfile.write('#SBATCH --tasks=28\n\n')
            myfile.write('## Number of nodes\n\n')
            myfile.write('#SBATCH --nodes=1\n\n')
            myfile.write('#SBATCH --exclude=n2023\n\n')
            myfile.write('## Time needed (days-hours:minutes:seconds\n\n')
            myfile.write('#SBATCH --time=5-00:00:00\n\n')
            myfile.write('## Memory per node\n\n')
            myfile.write('#SBATCH --mem=122G\n\n')
            myfile.write('## Where is the working directory of this job?\n\n')
            myfile.write('#SBATCH --chdir=.\n\n')
            myfile.write('## Where should the output go?\n\n')
            myfile.write('#SBATCH -o check1.dat\n\n')
            myfile.write('module load contrib/python/3.6.3\n')
            myfile.write('START=$(date +%s.%N)\n\n')
            myfile.write(f'python Average_min_fd_subproc_{i+1}_{a}.py\n\n')
            myfile.write('wait\n')
            myfile.write('END=$(date +%s.%N)\n')
            myfile.write('DIFF=$(echo "$END - $START" | bc)\n')
            myfile.write('echo $DIFF')


