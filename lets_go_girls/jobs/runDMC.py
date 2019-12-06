"""
This is a script template to run jobs based on a parameters file for a directory
structure that looks like:

main_dir

  jobs
    + DMC.py
    + run_script.py
    params
      + params_no_imp_samp.py
      + ParamsComplicated.py
      + ...

  results
    ParamsComplicated
      my_result_1.npz
      ...

"""

import argparse, sys, os
# load/define run_function
from .CH5_DMC import *

parser = argparse.ArgumentParser(prog='runDMC')
parser.add_argument('job_name',
                    type=str,
                    dest='job_name',
                    help='the name of the file with parameters')
args = parser.parse_args()
job_name = args.job_name
job_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(job_dir)


# alternate loading method from modules
sys.path.insert(0, os.path.join(job_dir, "params"))
exec(f"from {job_name} import pars")


# make sure all results get exported into correct directory
res_dir = os.path.join(main_dir, "results")
my_res_dir = os.path.join(res_dir, job_name)
try:
  os.mkdir(my_res_dir)
except OSError:
  pass

os.chdir(my_res_dir)

# run the code for real now guys truly this time
run(**pars)
