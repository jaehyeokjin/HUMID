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

FixStyle(intcoeff,FixIntCoeff)

#else

#ifndef LMP_FIX_INT_COEFF_H
#define LMP_FIX_INT_COEFF_H

#include "fix.h"

namespace LAMMPS_NS {

class FixIntCoeff : public Fix {
 public:
  FixIntCoeff(class LAMMPS *, int, char **);
  ~FixIntCoeff();
  void init();
  void setup(int);
  void init_list(int, class NeighList *);
  int setmask();
  double memory_usage();
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_scalar();


 private:
  double dist_cutoff, e_step;
  int nmax;
  int nmax_pair;
  int nlevels_respa;
  double **statecoeff;
  int *countneigh;

  int ncount;
  double *vector;
  double **array;
  class NeighList *list;
  void reallocate(int);

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix setforce does not exist

Self-explanatory.

E: Variable name for fix setforce does not exist

Self-explanatory.

E: Variable for fix setforce is invalid style

Only equal-style variables can be used.

E: Cannot use non-zero forces in an energy minimization

Fix setforce cannot be used in this manner.  Use fix addforce
instead.

*/
