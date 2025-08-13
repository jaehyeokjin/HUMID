/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS

KSpaceStyle(pppm/acc,PPPMACC)

#else

#ifndef LMP_PPPM_ACC_H
#define LMP_PPPM_ACC_H

#include "pppm.h"

namespace LAMMPS_NS {
  
  class PPPMACC : public PPPM {
  public:
    PPPMACC(class LAMMPS *, int, char **);
    virtual ~PPPMACC();
  };
}
    
#endif
#endif
