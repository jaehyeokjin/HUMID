
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

PairStyle(electrode,PairElectrode)

#else

#ifndef LMP_PAIR_ELECTRODE_H
#define LMP_PAIR_ELECTRODE_H

#define RAPTOR_SUPPORT

#include "pair.h"

namespace LAMMPS_NS {

class PairElectrode : public Pair {

 public:
 
  PairElectrode(class LAMMPS *);
  virtual ~PairElectrode();
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
  virtual void *extract(const char *, int &);

  double cut_lj_global;
  double **cut_lj,**cut_ljsq;
  double cut_coul,cut_coulsq;
  double **epsilon,**sigma;
  double **lj1,**lj2,**lj3,**lj4,**offset;
  double *cut_respa;
  double qdist;             // TIP4P distance from O site to negative charge
  double g_ewald;

  void allocate();
  
 /* ----- For electrode model ----- */

 public:
  
  int nlocal, nall;
  double eimage;
  
  void et_create();
  void et_destroy();
  void et_init();
  void et_setup();
  void et_compute();
  void et_compute_pln();
  void et_compute_pln_noQ0();
  double et_compute_pln_eng();
  void et_compute_pln_frc();

  void et_compute_one(int, double*, double*, double, int);
  void et_compute_one_noQ0(int, double*, double*, double, int);
  double et_compute_one_eng(int, double *, double, int);
  void et_compute_one_frc(int, double *, double *, double, int);

  void et_compute_3body(int, int, double, double, double, double);
  double et_compute_3body_eng(int, int, double, double, double, double);

  void et_compute_setup(); // Setup image charges for compute
                           // Needed when partition doesn't call compute() for ENV in RAPTOR simulations.
  
  double pos_lo, pos_hi, D, rD, volt, qext_lo, charge, D0, Q0, field, qE2f;
  double et_cut, et_cutsq, et_sigma, et_epsnl;
  double cell_x1, cell_x2, cell_y2, skin;
  double rcell_x1, rcell_y2;
  double * Qc_proc; // Accumulated charge on each processor
  
  double single_coul(double *xi, double* xj, double qi, double qj, double* fi);
  double single_coul_eng(double *xi, double* xj, double qi, double qj);
  void single_coul_frc(double *xi, double* xj, double qi, double qj, double * ff);
  
  /* PLANE structures */
  
  #define MAXPLN 10
  int nplanes;
  double planes[MAXPLN][3];
  
  /* Replica of image charges */
  
  int max_image;
  double **x_image, **x_image2;
  
  /* Table of electrode crystal */
  
  int cryst_size;
  double *cryst, *et_lj1, *et_lj2, *et_lj3, *et_lj4;   
  
  /* Three body term */
  
  int npar_3body, nbond_3body, *ipar_3body;
  struct parameters_3body { int type; double rcut, alpha, beta, A, B, C, D, E, F; };
  parameters_3body *par_3body;
  double **bonds_3body;
  
  /* Tabulated long-range force */
  
  void locate3d(double*, int*, double*);
  double interpolate3d(double***, int*, double*);
  
  bool et_table;
  double ***tbl_force_x, ***tbl_force_y, ***tbl_force_z;
  double ***tbl_energy_const, ***tbl_energy_polar;
  int tbl_nx, tbl_ny, tbl_nz, tbl_np;
  double tbl_dz, tbl_di, tbl_dj, tbl_djx, tbl_djy;
  double rtbl_dz, rtbl_di, rtbl_djy, rtbl_np;

  /* CGIS force */
  
  void   _cgis_init(double cut);
  double _cgis_single(double *rij, double qiqj, double *fi, double *fj);
  double _cgis_single_eng(double * rij, double qiqj);
  double _cgis_cut, _cgis_cut_sq, _cgis_cut_inv;
  double _cgis_A, _cgis_B, _cgis_C;
  double _cgis_eng_core;
  
  /* RAPTOR support */
  
  double compute_exch(int);
  double compute_exch_image_eng(int);
  
  long last_step;
  double A_Rq;
  int *is_exch_chg;
  int *complex_atom;
  class FixEVB* fixevb;

  // Functions used by SCI code
  virtual double single_ener_noljcoul(int, int, int, int, double, double);
  virtual void single_fpair_noljcoul(int, int, double *, double *);

};

}

#endif
#endif
