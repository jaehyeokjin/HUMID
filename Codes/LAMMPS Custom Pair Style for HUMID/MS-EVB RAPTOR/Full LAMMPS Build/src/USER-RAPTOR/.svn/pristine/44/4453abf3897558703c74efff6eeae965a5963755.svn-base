/* -------------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Package: MULTIPRO
   Purpose: Improve the parallel effeciency of PPPM in MD simulations
   Authors: Yuxing Peng and Chris Knight 
            Voth Group, Department of Chemistry, University of Chicago
------------------------------------------------------------------------- */


#ifdef COMMAND_CLASS

CommandStyle(multipro,MultiPro)

#else

#ifndef LMP_MULTIPRO_H
#define LMP_MULTIPRO_H

#include "pointers.h"

namespace LAMMPS_NS {

class MultiPro : protected Pointers {
 public:
  MultiPro(class LAMMPS *);
  void command(int, char **);
};

}

#endif
#endif
