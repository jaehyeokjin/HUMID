/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   
   Written by Chris Knight
   Based on pppm_electrode written by Yuxing Peng 
   
------------------------------------------------------------------------- */

#if defined (_OPENMP)

#ifdef KSPACE_CLASS

KSpaceStyle(evb_pppm/electrode/omp,EVB_PPPMELECTRODEOMP)

#else

#ifndef EVB_PPPM_ELECTRODE_OMP_H
#define EVB_PPPM_ELECTRODE_OMP_H

#include "EVB_pppm_electrode.h"
#include "thr_omp.h"

namespace LAMMPS_NS {
  
  class EVB_PPPMELECTRODEOMP : public EVB_PPPMELECTRODE, public ThrOMP {
  public:
    EVB_PPPMELECTRODEOMP(class LAMMPS *, int, char **);
    virtual ~EVB_PPPMELECTRODEOMP();

    double * virial_omp; // Temp array for virial; should really conform to USER-OMP

    virtual void poisson_energy(int);
};

}

#endif
#endif

#endif
