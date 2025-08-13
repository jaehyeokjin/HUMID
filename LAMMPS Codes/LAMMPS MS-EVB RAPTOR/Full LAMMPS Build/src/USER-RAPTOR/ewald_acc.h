/* ----------------------------------------------------------------------
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

KSpaceStyle(ewald/acc,EwaldACC)

#else

#ifndef LMP_EWALD_ACC_H
#define LMP_EWALD_ACC_H

#include "ewald.h"

namespace LAMMPS_NS {

class EwaldACC : public Ewald {
  public:
    EwaldACC(class LAMMPS *, int, char **);
    virtual ~EwaldACC();
  };
}

#endif
#endif
