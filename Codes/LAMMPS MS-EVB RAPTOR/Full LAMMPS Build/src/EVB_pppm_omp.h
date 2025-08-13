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

#if defined (_OPENMP)

#ifdef KSPACE_CLASS

KSpaceStyle(evb_pppm/omp,EVB_PPPMOMP)

#else

#ifndef LMP_EVB_PPPM_OMP_H
#define LMP_EVB_PPPM_OMP_H

#include "fft3d_wrap.h"
#include "EVB_engine.h"
#include "EVB_complex.h"
#include "EVB_offdiag.h"
#include "EVB_pppm.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

class EVB_PPPMOMP : public EVB_PPPM, public ThrOMP {
 public:
  EVB_PPPMOMP(class LAMMPS *, int, char **);
  virtual ~EVB_PPPMOMP ();
  virtual void setup();
  virtual void compute(int, int);

  // Extra densities
  FFT_SCALAR ***density_brick1, ***density_brick2, ***density_brick3;
  FFT_SCALAR ***vdx_brick2,***vdy_brick2,***vdz_brick2;
  FFT_SCALAR ***vdx_brick3,***vdy_brick3,***vdz_brick3;
  FFT_SCALAR ***u_brick2, ***u_brick3;
  FFT_SCALAR *density_fft1;
  FFT_SCALAR *density_fft2;
  FFT_SCALAR *density_fft3;
  FFT_SCALAR *work11;
  FFT_SCALAR *work12;
  FFT_SCALAR *work13;
  FFT_SCALAR *work22;
  FFT_SCALAR *work23;
  FFT_SCALAR *bbuf1, *bbuf2; // DELETE ME

  // Extra FFT data
  class FFT3d *fft1_2,*fft2_2;
  class Remap *remap_2;
  class FFT3d *fft1_3,*fft2_3;
  class Remap *remap_3;

  // ** AWGL: Functions for MS-EVB not in the usual pppm_omp.h ** //
  virtual void compute_env(int);
  virtual void compute_cplx(int);
  virtual void compute_cplx_eff(int);
  virtual void compute_exch(int);
  virtual void map2density_one(int);
  virtual void map2density_one(int, int);
  virtual void field2force_one(int,bool);
  virtual void compute_rho1d(const FFT_SCALAR &, const FFT_SCALAR &,const FFT_SCALAR &);
  virtual void compute_drho1d(const FFT_SCALAR &, const FFT_SCALAR &,const FFT_SCALAR &);

  void map2density_one_subtract(int); 
  void field2force_one_subtract(int);
  
  double qscale;

 protected:
  virtual void allocate();
  virtual void deallocate();
  virtual void deallocate_omp();
  virtual void fieldforce();
  virtual void fieldforce_ik();
  virtual void fieldforce_ad();
  virtual void fieldforce_peratom();
  virtual void make_rho();
  virtual void brick2fft();

  // Special extra density routines
  virtual void brick2fft_all3();
  virtual void poisson_all3(int, double[3], double[3][6]);
  virtual void poisson_all3_ik(int, double[3], double[3][6]);
  virtual void poisson_all3_ad(int, double[3], double[3][6]);

  void compute_rho1d_thr(FFT_SCALAR * const * const, const FFT_SCALAR &,
			 const FFT_SCALAR &, const FFT_SCALAR &);
  void compute_drho1d_thr(FFT_SCALAR * const * const, const FFT_SCALAR &,
			 const FFT_SCALAR &, const FFT_SCALAR &);
  // void map2density_one_thr(int id, FFT_SCALAR * const * const r1d, FFT_SCALAR * const * const * const db);

  // void compute_exch_split1(int vflag);
  // void compute_exch_split2(int vflag);
  // void compute_exch_split3(int vflag);

  // template<int AFLAG>
  // void field2force_one_thr(int id, FFT_SCALAR * const * const r1d, double** ft);

//  void compute_rho_coeff();
//  void slabcorr(int);

  virtual void compute_gf_ik();
  virtual void compute_gf_ad();

  // Communication functions (based on new CommGrid logic)
  MPI_Request request;
  MPI_Status status;
  
  void forward_comm3(int);
  void reverse_comm3(int);
  
  virtual void pack_forward3(int, FFT_SCALAR *, int, int *);
  virtual void unpack_forward3(int, FFT_SCALAR *, int, int *);
  virtual void pack_reverse3(int, FFT_SCALAR *, int, int *);
  virtual void unpack_reverse3(int, FFT_SCALAR *, int, int *);

};

}

#endif
#endif

#endif
