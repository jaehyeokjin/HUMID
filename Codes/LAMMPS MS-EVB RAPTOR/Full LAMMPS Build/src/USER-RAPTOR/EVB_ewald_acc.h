/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   splitted for MS-EVB by Tianying Yan, Chris and Yuxing Peng

------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS

KSpaceStyle(evb_ewald/acc,EVB_EwaldACC)

#else

#ifndef EVB_EWALD_ACC_H
#define EVB_EWALD_ACC_H

#include "EVB_ewald.h"

namespace LAMMPS_NS {
    
class EVB_EwaldACC : public EVB_Ewald {
 public:
  EVB_EwaldACC(class LAMMPS *, int, char **);
  ~EVB_EwaldACC();

  public: 
  void setup();
  void evb_setup();
  void compute_env(int);
  void compute_cplx(int);
  void compute_exch(int);
  void compute_eff(int);
  void eik_dot_r_env_acc();
  void eik_dot_r_cplx_acc();
  void eik_dot_r_exch();
  void coeffs_acc();
  void allocate_acc();
  void deallocate_acc();
  double memory_usage();

  public:
  int kxmax_acc, kymax_acc, kzmax_acc;
  int kmax_acc;
  int kmax3d_acc;
  int kcount_acc;

  int *kxvecs_acc, *kyvecs_acc, *kzvecs_acc;
  double *ug_acc;
  double **eg_acc, **vg_acc;

  double *sfacrl_env_acc,   *sfacim_env_acc;       /* Env part */

  double **eikrrl_acc,**eikrim_acc;
};

}

#endif
#endif

