#!/bin/sh

#SBATCH --account=pi-gavoth
#SBATCH --partition=broadwl-lc
#SBATCH --job-name=water
#SBATCH --time=3:00:00
#SBATCH --verbose
#SBATCH --exclusive
#SBATCH --ntasks-per-node=28
export OMP_NUM_THREADS=1
source /etc/profile
module load intel
module load intelmpi
module unload python
module load python

mpirun -np 28 python generate.py ../../step2/short.lammpstrj
