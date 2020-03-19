size = ['small', 'med', 'large']
s = ['s', 'm', 'l']
# size = ['med', 'large']
# s = ['m', 'l']
# size = ['biggo']
# s = ['b']
bro_str = ['5', '10']
bro = [5, 10]

# for ts in bro:
# for k in range(2):
for j in range(6):
    # for i in range(5):
        for a, b in zip(size, s):
            with open(f'DMC_Non_imp_sampled_{j}H_ts_10_{a}.sh', 'w') as myfile:
                myfile.write('#!/bin/bash\n\n')
                myfile.write('## Job Name\n\n')
                myfile.write(f'#SBATCH --job-name=Non_imp_sampled_{j}H_ts_10_{b}\n\n')
                myfile.write('## Allocation Definition\n\n')
                myfile.write('## Which queue should we use?\n\n')
                # if b == 's':
                #     myfile.write('#SBATCH --partition=stf\n\n')
                #     myfile.write('#SBATCH --account=stf\n\n')
                # else:
                myfile.write('#SBATCH --partition=chem\n\n')
                myfile.write('#SBATCH --account=chem\n\n')
                # myfile.write('#SBATCH --partition=ckpt\n\n')
                # myfile.write('#SBATCH --account=chem-ckpt\n\n')
                myfile.write('## Number of cores\n\n')
                myfile.write('#SBATCH --tasks=40\n\n')
                myfile.write('## Number of nodes\n\n')
                myfile.write('#SBATCH --nodes=1\n\n')
                myfile.write('#SBATCH --exclude=n2023\n\n')
                myfile.write('## Time needed (days-hours:minutes:seconds\n\n')
                myfile.write('#SBATCH --time=1-00:00:00\n\n')
                myfile.write('## Memory per node\n\n')
                myfile.write('#SBATCH --mem=122G\n\n')
                myfile.write('## Where is the working directory of this job?\n\n')
                myfile.write('#SBATCH --chdir=.\n\n')
                myfile.write('## Where should the output go?\n\n')
                myfile.write('#SBATCH -o check1.dat\n\n')
                myfile.write('module load contrib/python/3.6.3\n')
                myfile.write('START=$(date +%s.%N)\n\n')
                myfile.write(f'python Non_imp_sampled_{j}H_ts_10_subproc_{a}.py\n')
                # myfile.write(f'python Non_imp_sampled_CD_subproc_{size[1]}.py\n')
                # myfile.write(f'python Non_imp_sampled_CD_subproc_large.py\n')
                # myfile.write(f'python Non_imp_sampled_CD_subproc_med.py\n\n')
                # myfile.write(f'python Non_imp_sampled_CD_subproc_large.py\n\n')
                myfile.write('wait\n')
                myfile.write('END=$(date +%s.%N)\n')
                myfile.write('DIFF=$(echo "$END - $START" | bc)\n')
                myfile.write(f'echo "Non_imp_sampled_{j}H_ts_10_{b} took: $DIFF"')


