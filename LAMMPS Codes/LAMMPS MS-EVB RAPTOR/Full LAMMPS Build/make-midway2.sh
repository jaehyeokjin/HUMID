#!/bin/bash

module unload intelmpi fftw3 mkl

module load intelmpi/5.1+intel-16.0
module load fftw3/3.3.5+intelmpi-5.1+intel-16.0
module load mkl/11.3
module load libmatheval/1.1
module load boost/1.61
module load gcc/6.1
export LD_LIBRARY_PATH=/home/xinyouma/local/lib:$LD_LIBRARY_PATH
export CPATH=/home/xinyouma/plumed-experiment/include:$CPATH
export PATH=/home/xinyouma/plumed-experiment/bin:$PATH
if [ "$PWD" == '/home/xinyouma/lammps-1Mar16-raptor' ]
then
  cd src
  #make clean-midway
  make midway
fi
