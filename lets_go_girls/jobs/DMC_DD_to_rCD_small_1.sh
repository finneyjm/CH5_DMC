#!/bin/bash

## Job Name

#SBATCH --job-name=DD_to_rCD_s

## Allocation Definition

## Which queue should we use?

#SBATCH --partition=ckpt

#SBATCH --account=chem-ckpt

## Number of cores

#SBATCH --tasks=28

## Number of nodes

#SBATCH --nodes=1

#SBATCH --exclude=n2023

## Time needed (days-hours:minutes:seconds

#SBATCH --time=7-00:00:00

## Memory per node

#SBATCH --mem=122G

## Where is the working directory of this job?

#SBATCH --chdir=.

## Where should the output go?

#SBATCH -o check1.dat

module load contrib/python/3.6.3
START=$(date +%s.%N)

python DD_to_rCD_subproc_small_1.py

wait
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF