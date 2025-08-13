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

PairStyle(msucg_mix1d_domain,PairMSUCG_MIX1D_DOMAIN)

#else

#ifndef LMP_PAIR_MSUCG_MIX1D_DOMAIN_H
#define LMP_PAIR_MSUCG_MIX1D_DOMAIN_H

#include "pair.h"

namespace LAMMPS_NS {

class PairMSUCG_MIX1D_DOMAIN : public Pair {
 public:
  PairMSUCG_MIX1D_DOMAIN(class LAMMPS *);
  virtual ~PairMSUCG_MIX1D_DOMAIN();
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
  double **cut;
  double **epsilon,**sigma;
  double **lj1,**lj2,**lj3,**lj4,**offset;
  double *cut_respa;

  virtual void allocate();
  
  
  /*---YP--- All the contents above are from pair_lj_cut.* source
  /*---YP--- In-house codes starts from here */
  
  double T, kT;     /*---YP--- target temperature */
  
  int *type_linked; /*---YP--- Add one more paramters for each type 
                    /*---YP--- to store the linked types (that represent 
		    /*---YP--- other states for the same particle) */
  
  int countiter;
  double forcecomp1, forcecomp2, forcecomp3; 
  double fcomp1, fcomp2, fcomp3;
  int formchk,cross_check;
  double P(int, int, double*, double*); /*---YP--- function of calculating P(i,_a)
                                  /*---YP--- and its partial derivative  */

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
