size = ['small', 'med', 'large']
s = ['s', 'm', 'l']
bro_str = ['5', '10']
bro = [5, 10]

# for j in range(len(bro)):
    # for i in range(5):
for a, b in zip(size, s):
    with open(f'DMC_HH_to_rCH_min_wvfn_{a}.sh', 'w') as myfile:
        myfile.write('#!/bin/bash\n\n')
        myfile.write('## Job Name\n\n')
        myfile.write(f'#SBATCH --job-name=HH_to_rch_{b}\n\n')
        myfile.write('## Allocation Definition\n\n')
        myfile.write('## Which queue should we use?\n\n')
        if b == 's':
            myfile.write('#SBATCH --partition=stf\n\n')
            myfile.write('#SBATCH --account=stf\n\n')
        else:
            myfile.write('#SBATCH --partition=chem\n\n')
            myfile.write('#SBATCH --account=chem\n\n')
        myfile.write('## Number of cores\n\n')
        myfile.write('#SBATCH --tasks=28\n\n')
        myfile.write('## Number of nodes\n\n')
        myfile.write('#SBATCH --nodes=1\n\n')
        myfile.write('#SBATCH --exclude=n2023\n\n')
        myfile.write('## Time needed (days-hours:minutes:seconds\n\n')
        myfile.write('#SBATCH --time=7-00:00:00\n\n')
        myfile.write('## Memory per node\n\n')
        myfile.write('#SBATCH --mem=122G\n\n')
        myfile.write('## Where is the working directory of this job?\n\n')
        myfile.write('#SBATCH --chdir=.\n\n')
        myfile.write('## Where should the output go?\n\n')
        myfile.write('#SBATCH -o check1.dat\n\n')
        myfile.write('module load contrib/python/3.6.3\n')
        myfile.write('START=$(date +%s.%N)\n\n')
        myfile.write(f'python HH_to_rCH_min_wvfn_subproc_more_extras_{a}.py\n\n')
        myfile.write('wait\n')
        myfile.write('END=$(date +%s.%N)\n')
        myfile.write('DIFF=$(echo "$END - $START" | bc)\n')
        myfile.write('echo $DIFF')


