/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   
   splitted for MS-EVB by Yuxing Peng
   
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS

KSpaceStyle(evb_pppm/crhydroxide,EVB_PPPMCRHYDROXIDE)

#else

#ifndef EVB_PPPM_CRHYDROXIDE_H
#define EVB_PPPM_CRHYDROXIDE_H

#include <mpi.h>
#ifdef FFT_SINGLE
typedef float FFT_SCALAR;
#define  MPI_FFT_SCALAR MPI_FLOAT
#else
typedef double FFT_SCALAR;
#define  MPI_FFT_SCALAR MPI_DOUBLE
#endif

#include "EVB_pppm.h"

#define MAX_CR_RING 40
#define COEF_L2P 0.1

namespace LAMMPS_NS {

class EVB_PPPMCRHYDROXIDE : public EVB_PPPM {
  public:
    EVB_PPPMCRHYDROXIDE(class LAMMPS *, int, char **);
    ~EVB_PPPMCRHYDROXIDE();

  public:
    void find_Ring(int);
    void map_force_Ring(int);
    
    void map2density_one(int);
    void field2force_one_ik(int, bool);
    void field2force_one_ad(int, bool);
    
    void make_rho();
    void fieldforce_ik();
    void fieldforce_ad();

    int debug_q;

    double **xRing;
    double **fRing;

    double vector_norm(double [3]);
    double vector_dot(double [3], double [3]);
    double vector_dist(double [3], double [3]);
    void vector_unit(double [3]);
    void vector_unit(double [3], double [3]);
    void vector_cross(double [3], double [3], double [3]);

    void map_force_analytic(double [3], double [3], double [3], double [3]);
};

}

#endif
#endif
