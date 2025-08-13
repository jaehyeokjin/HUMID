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

#ifdef PAIR_CLASS

PairStyle(msucg_neigh_noiter,PairMSUCG_NEIGH_NOITER)

#else

#ifndef LMP_PAIR_MSUCG_NEIGH_NOITER_H
#define LMP_PAIR_MSUCG_NEIGH_NOITER_H

#include "pair.h"

namespace LAMMPS_NS {

class PairMSUCG_NEIGH_NOITER : public Pair {
 public:
  PairMSUCG_NEIGH_NOITER(class LAMMPS *);
  virtual ~PairMSUCG_NEIGH_NOITER();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  void init_list(int, class NeighList *);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

  void compute_inner();
  void compute_middle();
  void compute_outer(int, int);

 protected:
  double cut_global;
  double sigma_cutoff;
  double p_constant;
  double **cut;
  double **epsilon,**sigma;
  double **lj1,**lj2,**lj3,**lj4,**offset;
  double *cut_respa;

  virtual void allocate();


  /*Comments:  All the contents above are from pair_lj_cut.* source
  /*Comments:  In-house codes starts from here */

  double T, kT;     /*Comments:  target temperature */
  int *type_linked; /*Comments:  Add one more paramters for each type
                    /*Comments:  to store the linked types (that represent
		    /*Comments:  other states for the same particle) */
 /* Force check parameter */
  int countiter;
  int countneigh;
  double P(int, int, double*, double*, double, double*, int); /*Comments:  function of calculating P(i,_a)
                                  /*Comments:  and its partial derivative  */


  double *W, **dW, *U;
  double **subforce_1, **subforce_2, **subforce_3, **subforce_4, **totalforce, **pureljforce;
  double nmax;

  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  int pack_forward_comm(int, int *, double *);
  void unpack_forward_comm(int, int, double *);

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
