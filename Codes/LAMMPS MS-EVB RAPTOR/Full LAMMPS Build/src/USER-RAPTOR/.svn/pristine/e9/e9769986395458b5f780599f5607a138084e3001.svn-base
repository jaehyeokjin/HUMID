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

KSpaceStyle(evb_ewald,EVB_Ewald)

#else

#ifndef EVB_EWALD_H
#define EVB_EWALD_H

#include "EVB_kspace.h"

namespace LAMMPS_NS {
    
class EVB_Ewald : public EVB_KSpace {
 public:
  EVB_Ewald(class LAMMPS *, int narg, char **arg);
  ~EVB_Ewald();

 protected:
  
  /************************************
   ****** Original vars ***************
   ************************************/
  
  int kxmax,kymax,kzmax;
  int kcount,kmax,kmax3d,kmax_created;
  double qqrd2e;
  double gsqmx,qsum,qsqsum,q2,volume;
  int nmax;

  double unitk[3];
  int *kxvecs,*kyvecs,*kzvecs;
  double *ug;
  double **eg,**vg;
  double **ek;
  double *sfacrl,*sfacim;
  double ***cs,***sn;

  double rms(int, double, bigint, double);
  void init();
  void setup();
  double memory_usage();
  void coeffs();
  void allocate();
  void deallocate();
  
  /************************************
   ****** EVB related *****************
   ************************************/

  double *sfacrl_env,*sfacim_env;       /* Env part */
  double *sfacrl_cplx,*sfacim_cplx;     /* Complex part */
  double *sfacrl_exch,*sfacim_exch;     /* Exchange Charge part */
  double *sfacrl_inter,*sfacim_inter;   /* Interaction env part */
  
  int nmax_cplx;
  double **eikrrl,**eikrim;
  
  double qsum_all,qsqsum_all;
  double qsum_cplx,qsqsum_cplx;

  double dipole_env; // z-component of dipole of environment for slab correction
  double dipole_r2_env;
  
  void evb_setup();                /* This function is used to increase the memory
                                    * for one-atom-space. If [atom->nlocal] is larger
                                    * than [nmax], [ek],[cs] and [sn] need to be
                                    * reallocated. If maximum of [evb_complex(i)->
                                    * nlocal_cplx] is larger than [nmax_cplx],
                                    * [eik***] need to be reallocated. This function
                                    * only need to be called once, at very beginning
                                    * of each MD time-steup. */
  
  void compute_env(int);           /* This function calculates the energy oflong-range
                                    * interaction. [sfacrl(cim)_env] will be stored 
                                    * for the following steps. Force are not computed
                                    * in this function, but will be in compute_eff()
                                    * procedure. This function will be called at
                                    * begining for each EVB_Complex in EVB_Engine->setup()
                                    * and once before each EVB_Complex in EVB_Engine
                                    * ->sci_setup(). */

  virtual void compute_env_density(int);
  
  void compute_cplx(int);          /* This function calculates the interaction of
                                    * cplx <-> cplx+env. [sfacrl(cim)_env] is added
                                    * for each time of calculation. Only force on
                                    * complex(i)-atoms is calculated and stored. It
                                    * will be called in EVB_Engine->setup() and EVB_
                                    * Engine->sci_setup for each complex in all dia-
                                    * gonal elements. */
  
  void compute_exch(int);
  void compute_eff(int);
  
  void eik_dot_r_env();
  void eik_dot_r_cplx();
  void eik_dot_r_exch();

  void slabcorr_cplx();
  void slabcorr_exch();
  void slabcorr_eff();
  void slabcorr_sci_cplx();
  
  void sci_setup_iteration();
  void sci_setup_init();
  void sci_compute_env(int);
  void sci_compute_cplx(int);
  void sci_compute_exch(int);
  void sci_compute_eff(int);
  void compute_eff_mp(int);
  void sci_compute_eff_mp(int);
};

}

#endif
#endif

