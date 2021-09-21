sim = ['XH_left', 'XH_right']
sim = ['ground']
# sim = ['asym_left', 'asym_right']
mod = 'full'
for i in range(10):
    for type_of_sim in sim:
        with open(f'chain_rule_h3o2_dmc_new_drift_bigger_{type_of_sim}_{mod}_{i + 1}_param.py', 'w') as myfile:
            myfile.write('import numpy as np\n\n')
            myfile.write('import os\n\n')
            myfile.write('os.chdir("Imp_samp_testing")\n\n')
            myfile.write('from Imp_samp_testing import *\n\n')
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
            if mod == '1d':
                myfile.write('coords, weights, time, Eref_array, sum_weights, accept, des = run1(\n')
            elif mod == '2d':
                myfile.write('coords, weights, time, Eref_array, sum_weights, accept, des = run2(\n')
            else:
                myfile.write('coords, weights, time, Eref_array, sum_weights, accept, des = run(\n')
            if type_of_sim == 'asym_left' or type_of_sim == 'asym_right':
                myfile.write('        40000, 20000, 250, 500, 500, "a", test_structure, [0, 2.5721982410729867], [0, 0]\n')
            elif type_of_sim == 'XH_left' or type_of_sim == 'XH_right':
                myfile.write('        40000, 20000, 250, 500, 500, "sp", test_structure, [0, 2.5721982410729867], [0, 0]\n')
            elif type_of_sim == 'ground':
                myfile.write('        40000, 20000, 250, 500, 500, None, test_structure, [0, 2.5721982410729867], [0, 0]\n')
            myfile.write('    )\n')
            myfile.write(f'np.savez(f"{type_of_sim}_excite_state_chain_rule2_bigger_{mod}_h3o2_{i+1}", coords=coords, weights=weights, time=time, Eref=Eref_array,\n')
            myfile.write('             sum_weights=sum_weights, accept=accept, d=des)')
