size = ['biggo']
s = ['b']
# size = ['med', 'large']
# s = ['m', 'l']
# size = ['biggo']
# s = ['b']
bro_str = ['5', '10']
bro = [5, 10]

# for ts in bro:
# for k in range(2):
# for j in range(6):
    # for i in range(5):
for i in range(5):
    for a, b in zip(size, s):
        with open(f'DMC_imp_sampled_discrete_analytic_5H_ts_1_{a}_{i+1}.sh', 'w') as myfile:
            myfile.write('#!/bin/bash\n\n')
            myfile.write('## Job Name\n\n')
            myfile.write(f'#SBATCH --job-name=Imp_sampled_disc_analytic_5H_ts_1_{b}_{i+1}\n\n')
            myfile.write('## Allocation Definition\n\n')
            myfile.write('## Which queue should we use?\n\n')
            # if b == 's':
            #     myfile.write('#SBATCH --partition=stf\n\n')
            #     myfile.write('#SBATCH --account=stf\n\n')
            # else:
            # myfile.write('#SBATCH --partition=chem\n\n')
            # myfile.write('#SBATCH --account=chem\n\n')
            myfile.write('#SBATCH --partition=ckpt\n\n')
            myfile.write('#SBATCH --account=chem-ckpt\n\n')
            myfile.write('## Number of cores\n\n')
            myfile.write('#SBATCH --tasks=28\n\n')
            myfile.write('## Number of nodes\n\n')
            myfile.write('#SBATCH --nodes=1\n\n')
            myfile.write('## Time needed (days-hours:minutes:seconds\n\n')
            myfile.write('#SBATCH --time=0-05:00:00\n\n')
            myfile.write('## Memory per node\n\n')
            myfile.write('#SBATCH --mem=122G\n\n')
            myfile.write('## Where is the working directory of this job?\n\n')
            myfile.write('#SBATCH --chdir=.\n\n')
            myfile.write('## Where should the output go?\n\n')
            myfile.write(f'#SBATCH -o check_imp_sampled_analytic_5H_ts_1_{b}_{i+1}.dat\n\n')
            myfile.write('module load contrib/python/3.6.3\n')
            myfile.write('START=$(date +%s.%N)\n\n')
            if b == 'l':
                myfile.write(f'python runDMC.py params_imp_sampled_analytic_5H_ts_1_6000_walkers_test_{i+1}\n')
            elif b == 'b':
                myfile.write(f'python runDMC.py params_imp_sampled_discrete_analytic_5H_ts_1_10000_walkers_test_{i+1}\n')
            # myfile.write(f'python Imp_sampled_discrete_analytic_5H_ts_1_subproc_{a}.py\n')
            # myfile.write(f'python Non_imp_sampled_CD_subproc_{size[1]}.py\n')
            # myfile.write(f'python Non_imp_sampled_CD_subproc_large.py\n')
            # myfile.write(f'python Non_imp_sampled_CD_subproc_med.py\n\n')
            # myfile.write(f'python Non_imp_sampled_CD_subproc_large.py\n\n')
            myfile.write('wait\n')
            myfile.write('END=$(date +%s.%N)\n')
            myfile.write('DIFF=$(echo "$END - $START" | bc)\n')
            myfile.write(f'echo "Imp_sampled_disc_analytic_5H_ts_1_{b}_{i+1} took: $DIFF"')


