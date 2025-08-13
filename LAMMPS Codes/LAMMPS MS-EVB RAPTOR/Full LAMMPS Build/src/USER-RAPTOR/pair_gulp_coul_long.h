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

#ifdef PAIR_CLASS

PairStyle(gulp/coul/long,PairGulpCoulLong)

#else

#ifndef LMP_PAIR_GULP_COUL_LONG_H
#define LMP_PAIR_GULP_COUL_LONG_H

#include "pair.h"

namespace LAMMPS_NS {

class PairGulpCoulLong : public Pair {

 public:
  PairGulpCoulLong(class LAMMPS *);
  virtual ~PairGulpCoulLong();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  void init_list(int, class NeighList *);
  virtual double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);

  // SCI-MS-EVB functions
  virtual double single_ener_noljcoul(int, int, int, int, double, double);
  virtual double single_fpair_noljcoul(int, int, int, int, double, double);

  void compute_inner() {};
  void compute_middle() {};
  virtual void compute_outer(int, int) {};
  virtual void *extract(const char *, int &);

  // protected:

  void allocate();

  double cut_global, cut_inner_global;
  double cut_coul, cut_coulsq;
  double *cut_respa;
  double qdist;             // TIP4P distance from O site to negative charge
  double g_ewald;

  // lj/cut
  double **cut_ljcut, **cutsq_ljcut;
  double **epsilon_ljcut, **sigma_ljcut;
  int **setflag_ljcut;
  double **lj1_ljcut, **lj2_ljcut, **lj3_ljcut, **lj4_ljcut, **offset_ljcut;

  // lj/gulp
  double **cut_ljgulp, **cutsq_ljgulp;
  double **cut_inner_ljgulp, **cutsq_inner_ljgulp;
  double **epsilon_ljgulp, **sigma_ljgulp;
  int **setflag_ljgulp;
  double **lj1_ljgulp, **lj2_ljgulp, **lj3_ljgulp, **lj4_ljgulp;

  // len/gulp
  double **cut_lengulp, **cutsq_lengulp;
  double **cut_inner_lengulp, **cutsq_inner_lengulp;
  double **a_lengulp, **b_lengulp;
  int **setflag_lengulp;
  double **lj1_lengulp, **lj2_lengulp, **lj3_lengulp, **lj4_lengulp;

  // buck/gulp
  double **cut_buckgulp, **cutsq_buckgulp;
  double **cut_inner_buckgulp, **cutsq_inner_buckgulp;
  double **a_buckgulp, **rho_buckgulp, **c_buckgulp;
  int **setflag_buckgulp;
  double **rhoinv_buckgulp, **buck1_buckgulp, **buck2_buckgulp, **offset_buckgulp;

  // table
  enum{LOOKUP,LINEAR,SPLINE,BITMAP};

  int tabstyle,tablength;
  struct Table {
    int ninput,rflag,fpflag,match,ntablebits;
    int nshiftbits,nmask;
    double rlo,rhi,fplo,fphi,cut;
    double *rfile,*efile,*ffile;
    double *e2file,*f2file;
    double innersq,delta,invdelta,deltasq6;
    double *rsq,*drsq,*e,*de,*f,*df,*e2,*f2;
  };
  int ntables;
  Table *tables;

  int do_tables;
  int **setflag_table;
  double **cut_table;
  double **cutsq_table;
  int **tabindex;
  double **tab_scale;

  void read_table(Table *, char *, char *);
  void param_extract(Table *, char *);
  void bcast_table(Table *);
  void spline_table(Table *);
  void compute_table(Table *);
  void null_table(Table *);
  void free_table(Table *);
  void spline(double *, double *, int, double, double, double *);
  double splint(double *, double *, double *, int, double);
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

E: Pair style lj/cut/coul/long requires atom attribute q

The atom style defined does not have this attribute.

E: Pair style requires a KSpace style

No kspace style is defined.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
