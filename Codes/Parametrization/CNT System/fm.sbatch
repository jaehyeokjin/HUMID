#!/bin/sh

#SBATCH --account=pi-gavoth
#SBATCH --partition=broadwl-lc
#SBATCH --job-name=water
#SBATCH --time=3:00:00
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --ntasks=1

export OMP_NUM_THREADS=1
source /etc/profile
module load mkl/10.3
 module unload gcc
 module unload intel
 module unload gsl
 module load intel/16.0
 module load gcc
 module load gsl/2.2.1+intel-16.0

./newfm.x -l ../../cg.lammpstrj
