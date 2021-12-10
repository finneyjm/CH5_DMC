
def create_sim_py_file(filename, number_of_sims, folder_name, type_of_sim, excite, walkers):
    for i in range(number_of_sims):
        with open(f'{filename}_{i+1}.py', 'w') as myfile:
            myfile.write('import pyvibdmc as pv\n')
            myfile.write('import numpy as np\n')
            myfile.write('from pyvibdmc.simulation_utilities import potential_manager as pm\n')
            myfile.write('from scipy import interpolate\n')
            myfile.write('import matplotlib.pyplot as plt\n')
            myfile.write('import multiprocessing as mp\n\n')

            myfile.write("pot_dir = '.'\n")
            myfile.write("py_file = 'h3o2_wave_function_and_pot.py'\n")
            myfile.write("pot_func = 'get_pot'\n\n")

            if type_of_sim == 'asym_right' or type_of_sim == 'XH_right':
                myfile.write('test_structure = np.array([\n')
                myfile.write('        [ 2.75704662,  0.05115356, -0.2381117 ],\n')
                myfile.write('        [ 0.24088235, -0.09677082,  0.09615192],\n')
                myfile.write('        [-0.08502706, -1.66894299, -0.79579001],\n')
                myfile.write('        [ 5.02836896, -0.06798562, -0.30434529],\n')
                myfile.write('        [ 5.84391277,  0.14767547,  1.4669121 ],\n')
                myfile.write('])\n\n')
            elif type_of_sim == 'asym_left' or type_of_sim == 'XH_left':
                myfile.write('test_structure = np.array([\n')
                myfile.write('        [ 2.45704662,  0.05115356, -0.2381117 ],\n')
                myfile.write('        [ 0.24088235, -0.09677082,  0.09615192],\n')
                myfile.write('        [-0.47502706, -1.46894299, -0.69579001],\n')
                myfile.write('        [ 5.02836896, -0.06798562, -0.30434529],\n')
                myfile.write('        [ 5.84391277,  0.14767547,  1.4669121 ],\n')
                myfile.write('])\n\n')
            elif type_of_sim == 'ground' and i < 5:
                myfile.write('test_structure = np.array([\n')
                myfile.write('        [ 2.75704662,  0.05115356, -0.2381117 ],\n')
                myfile.write('        [ 0.24088235, -0.09677082,  0.09615192],\n')
                myfile.write('        [-0.08502706, -1.66894299, -0.79579001],\n')
                myfile.write('        [ 5.02836896, -0.06798562, -0.30434529],\n')
                myfile.write('        [ 5.84391277,  0.14767547,  1.4669121 ],\n')
                myfile.write('])\n\n')
            else:
                myfile.write('test_structure = np.array([\n')
                myfile.write('        [ 2.45704662,  0.05115356, -0.2381117 ],\n')
                myfile.write('        [ 0.24088235, -0.09677082,  0.09615192],\n')
                myfile.write('        [-0.47502706, -1.46894299, -0.69579001],\n')
                myfile.write('        [ 5.02836896, -0.06798562, -0.30434529],\n')
                myfile.write('        [ 5.84391277,  0.14767547,  1.4669121 ],\n')
                myfile.write('])\n\n')

            myfile.write("wvfn_stuff = np.load('h3o2_2d_wvfn.npz')\n")
            myfile.write('sp_grid = wvfn_stuff["gridz"][0]\n')
            myfile.write('Roo_grid = wvfn_stuff["gridz"][1]\n')
            myfile.write('grid_points = len(sp_grid)\n\n')

            if type_of_sim == 'XH_right' or type_of_sim == 'XH_left':
                myfile.write('no_der = interpolate.interp2d(sp_grid, Roo_grid, '
                             'wvfn_stuff["wvfns"][:, 0].reshape((grid_points, grid_points)).T, kind="cubic"\n')
                myfile.write('z_dx1 = np.load("z_xh_dx1_2d_h3o2.npy")\n')
                myfile.write('z_dy1 = np.load("z_xh_dy1_2d_h3o2.npy")\n')
                myfile.write('z_dx2 = np.load("z_xh_dx2_2d_h3o2.npy")\n')
                myfile.write('z_dy2 = np.load("z_xh_dy2_2d_h3o2.npy")\n')
                myfile.write('z_dx1_dy1 = np.load("z_xh_dx1_dy1_2d_h3o2.npy")\n\n')
            else:
                myfile.write('no_der = interpolate.interp2d(sp_grid, Roo_grid, '
                             'wvfn_stuff["wvfns"][:, 2].reshape((grid_points, grid_points)).T, kind="cubic")\n')
                myfile.write('z_dx1 = np.load("z_ground_dx1_2d_h3o2.npy")\n')
                myfile.write('z_dy1 = np.load("z_ground_dy1_2d_h3o2.npy")\n')
                myfile.write('z_dx2 = np.load("z_ground_dx2_2d_h3o2.npy")\n')
                myfile.write('z_dy2 = np.load("z_ground_dy2_2d_h3o2.npy")\n')
                myfile.write('z_dx1_dy1 = np.load("z_ground_dx1_dy1_2d_h3o2.npy")\n\n')

            myfile.write('dx1 = interpolate.interp2d(sp_grid, Roo_grid, z_dx1.T, kind="cubic")\n')
            myfile.write('dy1 = interpolate.interp2d(sp_grid, Roo_grid, z_dy1.T, kind="cubic")\n')
            myfile.write('dx2 = interpolate.interp2d(sp_grid, Roo_grid, z_dx2.T, kind="cubic")\n')
            myfile.write('dy2 = interpolate.interp2d(sp_grid, Roo_grid, z_dy2.T, kind="cubic")\n')
            myfile.write('dx1_dy1 = interpolate.interp2d(sp_grid, Roo_grid, z_dx1_dy1.T, kind="cubic")\n\n')

            myfile.write('kwargs = {\n')
            myfile.write(f'    "excite": "{excite}",\n')
            myfile.write('    "interp": no_der,\n')
            myfile.write('    "dx1": dx1,\n')
            myfile.write('    "dy1": dy1,\n')
            myfile.write('    "dx2": dx2,\n')
            myfile.write('    "dy2": dy2,\n')
            myfile.write('    "dx1_dy1": dx1_dy1,\n')
            myfile.write('}\n\n')

            myfile.write('pot = pv.Potential(potential_function=pot_func,\n')
            myfile.write('                   python_file=py_file,\n')
            myfile.write('                   potential_directory=pot_dir,\n')
            myfile.write('                   num_cores=11)\n\n')

            myfile.write('imp_smplr = pv.ImpSampManager(trial_function="psi_t",\n')
            myfile.write('                              trial_directory=pot_dir,\n')
            myfile.write('                              python_file=py_file,\n')
            myfile.write('                              pot_manager=pot,\n')
            myfile.write('                              deriv_function="derivatives",\n')
            myfile.write('                              trial_kwargs=kwargs,\n')
            myfile.write('                              deriv_kwargs=kwargs,\n')
            myfile.write('                              )\n\n')

            myfile.write(f'sim = pv.DMC_Sim(sim_name="{filename}_{i+1}",\n')
            myfile.write(f'                 output_folder="{folder_name}",\n')
            myfile.write('                 weighting="continuous",\n')
            myfile.write('                 cont_wt_thresh=[0.01, 20],\n')
            myfile.write(f'                 num_walkers={walkers},\n')
            myfile.write('                 num_timesteps=20000,\n')
            myfile.write('                 equil_steps=500,\n')
            myfile.write('                 chkpt_every=500,\n')
            myfile.write('                 wfn_every=500,\n')
            myfile.write('                 desc_wt_steps=250,\n')
            myfile.write('                 atoms=["H", "O", "H", "O", "H"],\n')
            myfile.write('                 delta_t=1,\n')
            myfile.write('                 potential=pot,\n')
            myfile.write('                 start_structures=test_structure[None, :, :],\n')
            myfile.write('                 imp_samp=imp_smplr)\n\n')

            myfile.write('sim.run()')


def create_sim_sh_file(filename, number_of_sims, job_name, py_filename):
    for i in range(number_of_sims):
        with open(f'{filename}_{i+1}.sh', 'w') as myfile:
            myfile.write('#!/bin/bash\n\n')
            myfile.write('## Job Name\n\n')
            myfile.write(f'#SBATCH --job-name=h3o2_{job_name}_{i+1}\n\n')
            myfile.write('## Allocation Definition\n\n')
            myfile.write('## Which queue should we use?\n\n')
            myfile.write('#SBATCH --partition=ilahie\n\n')
            myfile.write('#SBATCH --account=ilahie\n\n')
            myfile.write('## Number of cores\n\n')
            myfile.write('#SBATCH --tasks=28\n\n')
            myfile.write('## Number of nodes\n\n')
            myfile.write('#SBATCH --nodes=1\n\n')
            myfile.write('## Time needed (days-hours:minutes:seconds\n\n')
            myfile.write('#SBATCH --time=0-10:00:00\n\n')
            myfile.write('## Memory per node\n\n')
            myfile.write('#SBATCH --mem=122G\n\n')
            myfile.write('## Where is the working directory of this job?\n\n')
            myfile.write('#SBATCH --chdir=.\n\n')
            myfile.write('## Where should the output go?\n\n')
            myfile.write(f'#SBATCH -o check_h3o2_{job_name}_{i+1}.dat\n\n')
            myfile.write('START=$(date +%s.%N)\n\n')

            myfile.write(f'python {py_filename}_{i + 1}.py\n')

            myfile.write('wait\n')
            myfile.write('END=$(date +%s.%N)\n')
            myfile.write('DIFF=$(echo "$END - $START" | bc)\n')
            myfile.write(f'echo "h3o2_{job_name}_{i+1} took: $DIFF"')


create_sim_py_file('h3o2_ground_chain_rule', 10, 'h3o2_sims', 'ground', None, 20000)
create_sim_sh_file('DMC_h3o2_ground', 10, 'ground', 'h3o2_ground_chain_rule')

create_sim_py_file('h3o2_asym_right_chain_rule', 5, 'h3o2_sims', 'asym_right', 'a', 20000)
create_sim_sh_file('DMC_h3o2_asym_right', 5, 'asym_right', 'h3o2_asym_right_chain_rule')

create_sim_py_file('h3o2_asym_left_chain_rule', 5, 'h3o2_sims', 'asym_left', 'a', 20000)
create_sim_sh_file('DMC_h3o2_asym_left_right', 5, 'asym_left', 'h3o2_asym_left_chain_rule')

create_sim_py_file('h3o2_XH_right_chain_rule', 5, 'h3o2_sims', 'XH_right', 'sp', 20000)
create_sim_sh_file('DMC_h3o2_XH_right', 5, 'XH_right', 'h3o2_XH_right_chain_rule')

create_sim_py_file('h3o2_XH_left_chain_rule', 5, 'h3o2_sims', 'XH_left', 'sp', 20000)
create_sim_sh_file('DMC_h3o2_XH_left', 5, 'XH_left', 'h3o2_XH_left_chain_rule')
