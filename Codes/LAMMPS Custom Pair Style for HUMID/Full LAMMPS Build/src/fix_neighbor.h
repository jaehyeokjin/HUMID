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

#ifdef FIX_CLASS

FixStyle(neighbor,FixNeighbor)

#else

#ifndef LMP_FIX_NEIGHBOR_H
#define LMP_FIX_NEIGHBOR_H

#include "fix.h"

namespace LAMMPS_NS {

class FixNeighbor : public Fix {
 public:
   FixNeighbor(class LAMMPS *, int, char **);
   ~FixNeighbor();
  void init();
  void setup(int);
  int setmask();
  void init_list(int, class NeighList *);
  void post_force(int);
  double memory_usage();


  /*
  void init_style();
  */

private:
  int nlevels_respa;
  class NeighList *list;
  double nmax;

 protected:
  double sigma_cutoff;


};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
