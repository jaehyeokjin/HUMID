/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   
   Written by Chris Knight
   Based on pppm_electrode written by Yuxing Peng 
   
------------------------------------------------------------------------- */

#if defined (_OPENMP)

#include "lmptype.h"
#include "mpi.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "atom.h"
#include "comm.h"
#include "gridcomm.h"
#include "neighbor.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "domain.h"
#include "fft3d_wrap.h"
#include "remap_wrap.h"
#include "memory.h"
#include "error.h"

#include "EVB_pppm_electrode_omp.h"
#include "EVB_engine.h"
#include "EVB_effpair.h"
#include "EVB_offdiag.h"
#include "EVB_complex.h"
#include "EVB_timer.h"

#include "math_const.h"
#include "math_special.h"

#include "suffix.h"
using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#define MAXORDER 7
#define OFFSET 16384
#define SMALL 0.00001
#define LARGE 10000.0
#define EPS_HOC 1.0e-7

enum{REVERSE_RHO};
enum{FORWARD_IK,FORWARD_AD,FORWARD_IK_PERATOM,FORWARD_AD_PERATOM};

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Electrode Model */
#include "pair_electrode.h"
#define ZPRD_CORR() (pair_et->D * 3.0 * slab_volfactor)
/* End */

#define Q_ATOM 0
#define Q_EFFECTIVE 1

#define KSPACE_DEFAULT    0 // Hellman-Feynman forces for Ewald
#define PPPM_HF_FORCES    1 // Hellman-Feynman forces for PPPM
#define PPPM_ACC_FORCES   2 // Approximate (acc) forces for PPPM. 
#define PPPM_POLAR_FORCES 3 // ACC forces plus an additional polarization force on complex atoms for PPPM.

EVB_PPPMELECTRODEOMP::EVB_PPPMELECTRODEOMP(LAMMPS *lmp, int narg, char **arg) : 
  EVB_PPPMELECTRODE(lmp, narg, arg), ThrOMP(lmp, THR_KSPACE)
{
  suffix_flag |= Suffix::OMP;

  memory->create(virial_omp, 6*comm->nthreads, "EVB_PPPM:virial_omp");
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

EVB_PPPMELECTRODEOMP::~EVB_PPPMELECTRODEOMP()
{
  memory->destroy(virial_omp);
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODEOMP::poisson_energy(int vflag)
{
#pragma omp parallel default(none) shared(vflag)
  {

  // transform charge density (r -> k)
#pragma omp parallel for
    for (int i=0; i<nfft; i++) {
      int const n = i * 2;
      work1[n] = density_fft[i];
      work1[n+1] = ZEROF;
    }

#pragma omp master
    {
      fft1->compute(work1,work1,1);
    }

    /* Electrode Model */
#pragma omp barrier
#pragma omp parallel for
    for(int i = 0; i < nfft; i++)  {
      int const n = i * 2;
      int const off = greensfn_rev[i];
      
      work1_img[n]    = img1_rl[i]*work1[off]   - img1_im[i]*work1[off+1];
      work1_img[n+1]  = img1_rl[i]*work1[off+1] + img1_im[i]*work1[off];
      
      work1_img[n]   += img2_rl[i]*work1[off]   - img2_im[i]*work1[off+1];
      work1_img[n+1] += img2_rl[i]*work1[off+1] + img2_im[i]*work1[off];
    }
    /* End */

    double e, sqr;
    double eng = 0.0;
    double const scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
    double const s2 = scaleinv*scaleinv;
    
    // compute energy and virial contribution

    if (vflag) {      
      
      int const tid = omp_get_thread_num();
      int const v_indx = tid * 6;
      for(int j=0; j<6; j++) virial_omp[v_indx+j] = 0.0;

#pragma omp parallel for reduction(+:eng) default(none) private(sqr,e)
      for (int i=0; i<nfft; ++i) {
	int const n = i * 2;
	sqr  = work1[n]*work1[n]     + work1[n+1]*work1[n+1];
	sqr += work1[n]*work1_img[n] + work1[n+1]*work1_img[n+1];
	e = s2 * greensfn[i] * sqr;
	for (int j=0; j<6; ++j) virial_omp[v_indx+j] += e * vg[i][j];
	eng += e;
      }
    } else {
#pragma omp parallel for reduction(+:eng) default(none) private(sqr)
      for (int i=0; i<nfft; ++i) {
	int const n = i * 2;
	sqr  = work1[n]*work1[n]     + work1[n+1]*work1[n+1];
	sqr += work1[n]*work1_img[n] + work1[n+1]*work1_img[n+1];
	eng += greensfn[i] * sqr;
      }
      eng *= s2;
    }

#pragma omp master
    {
      energy += eng;
      for(int i=0; i<6; i++) for(int j=0; j<comm->nthreads; j++) virial[i] += virial_omp[j*6+i];
    }
  }

}

#endif
