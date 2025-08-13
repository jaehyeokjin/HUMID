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

/* ----------------------------------------------------------------------
   Contributing author: Chris Knight
         
     OMP support derived from EVB_pppm_omp
------------------------------------------------------------------------- */

#if defined (_OPENMP)

#include "atom.h"
#include "comm.h"
#define _CRACKER_GRIDCOMM
#include "EVB_cracker.h"
#undef _CRACKER_GRIDCOMM
#include "neighbor.h"
#include "domain.h"
#include "force.h"
#include "memory.h"
#include "remap_wrap.h"

#include "EVB_pppm_acc_omp.h"

#include <string.h>
#include <math.h>

#include "suffix.h"
#include "math_const.h"
#include "math_special.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#define MAXORDER 7
#define OFFSET 4096
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

/* ---------------------------------------------------------------------- */

EVB_PPPMACCOMP::EVB_PPPMACCOMP(LAMMPS *lmp, int narg, char **arg) :
  EVB_PPPM(lmp, narg, arg), ThrOMP(lmp, THR_KSPACE)
{
  suffix_flag |= Suffix::OMP;

  density_brick_acc = vdx_brick_acc = vdy_brick_acc = vdz_brick_acc = NULL;
  density_fft_acc = NULL;
  u_brick_acc = NULL;
  greensfn_acc = NULL;
  work1_acc = work2_acc = NULL;
  vg_acc = NULL;
  fkx_acc = fky_acc = fkz_acc = NULL;

  sf_precoeff1_acc = sf_precoeff2_acc = sf_precoeff3_acc = 
    sf_precoeff4_acc = sf_precoeff5_acc = sf_precoeff6_acc = NULL;

  gf_b_acc = NULL;
  rho1d_acc = rho_coeff_acc = drho1d_acc = drho_coeff_acc = NULL;

  fft1_acc = fft2_acc = NULL;
  remap_acc = NULL;
  cg_acc = NULL;

  part2grid_acc = NULL;
  
  /***************************************/
  /***************************************/
  env_density_brick_acc = NULL;
  part2grid_dr_acc = NULL;

  // GridComm
  cg_buf1_acc = cg_buf2_acc = NULL;

  // This is true so that compute_eff() will calculate full-grid PPPM at end of MD step
  bEff = true;
}

/* ----------------------------------------------------------------------
   free all memory 
------------------------------------------------------------------------- */

EVB_PPPMACCOMP::~EVB_PPPMACCOMP()
{
  //  deallocate_acc();
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors and order 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::allocate()
{
  EVB_PPPM::allocate();

  const int nthreads = comm->nthreads;

#if defined(_OPENMP)
#pragma omp parallel default(none)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif

    ThrData *thr = fix->get_thr(tid);
    thr->init_pppm(order,memory);
  }

  const int nzend = (nzhi_out-nzlo_out+1)*nthreads + nzlo_out - 1;

  // reallocate density brick, so it fits our needs
  memory->destroy3d_offset(density_brick,nzlo_out,nylo_out,nxlo_out);
  memory->create3d_offset(density_brick,nzlo_out,nzend,nylo_out,nyhi_out,
			  nxlo_out,nxhi_out,"evb_pppm_omp:density_brick");
  /*******************************************************/
  /*******************************************************/
  memory->destroy3d_offset(env_density_brick,nzlo_out,nylo_out,nxlo_out);
  memory->create3d_offset(env_density_brick,nzlo_out,nzend,nylo_out,nyhi_out,
			  nxlo_out,nxhi_out,"evb_pppm_omp:env_density_brick");
}

/* ----------------------------------------------------------------------
   free memory that depends on # of K-vectors and order 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::deallocate()
{
  EVB_PPPM::deallocate();
  for (int i=0; i < comm->nthreads; ++i) {
    ThrData * thr = fix->get_thr(i);
    FFT_SCALAR ** rho1d_thr = static_cast<FFT_SCALAR **>(thr->get_rho1d());
    memory->destroy2d_offset(rho1d_thr,-order/2);
  }
}

/* ----------------------------------------------------------------------
   called once before run 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::init()
{
  EVB_PPPM::init(); // full-grid PPPM initialization

  if (me == 0) {
    if (screen) fprintf(screen,"\nEVB_PPPMACCOMP initialization ...\n");
    if (logfile) fprintf(logfile,"\nEVB_PPPMACCOMP initialization ...\n");
  }

  // set accuracy (force units) from accuracy_relative or accuracy_absolute
  
  if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  else accuracy = accuracy_relative * two_charge_force;

  g_ewald_acc = g_ewald;
  
  // free all arrays previously allocated

  deallocate_acc();

  // setup FFT grid resolution and g_ewald
  // normally one iteration thru while loop is all that is required
  // if grid stencil does not extend beyond neighbor proc
  //   or overlap is allowed, then done
  // else reduce order and try again

  int (*procneigh)[2] = comm->procneigh;

  GridComm *cgtmp = NULL;
  int iteration = 0;

  order_acc = 5; // Default value in kspace.cpp
  while (order >= minorder) {
    if (iteration && me == 0)
      error->warning(FLERR,"Reducing EVB_PPPMACC order b/c stencil extends "
		     "beyond nearest neighbor processor");

    set_grid_global_acc();
    set_grid_local_acc();
    if(overlap_allowed) break;

    cgtmp = new GridComm(lmp_pointer,world,1,1,
                         nxlo_in_acc,nxhi_in_acc,nylo_in_acc,nyhi_in_acc,nzlo_in_acc,nzhi_in_acc,
                         nxlo_out_acc,nxhi_out_acc,nylo_out_acc,nyhi_out_acc,nzlo_out_acc,nzhi_out_acc,
                         procneigh[0][0],procneigh[0][1],procneigh[1][0],
                         procneigh[1][1],procneigh[2][0],procneigh[2][1]);
    cgtmp->ghost_notify();
    if (!cgtmp->ghost_overlap()) break;
    delete cgtmp;

    order_acc--;
    iteration++;
  }

  if (order < minorder) error->all(FLERR,"EVB_PPPMACC order < minimum allowed order");
  if (!overlap_allowed && cgtmp->ghost_overlap())
    error->all(FLERR,"EVB_PPPMACC grid stencil extends beyond nearest neighbor processor");
  if (cgtmp) delete cgtmp;

  // adjust g_ewald

  if(!gewaldflag) adjust_gewald_acc();

  // calculate the final accuracy

  double estimated_accuracy = final_accuracy_acc();

  // print stats

  int ngrid_max,nfft_both_max,nbuf_max;
  MPI_Allreduce(&ngrid_acc,&ngrid_max,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(&nfft_both_acc,&nfft_both_max,1,MPI_INT,MPI_MAX,world);

  if (me == 0) {

#ifdef FFT_SINGLE
    const char fft_prec[] = "single";
#else
    const char fft_prec[] = "double";
#endif

    if (screen) {
      fprintf(screen,"  G vector (1/distance)= %g\n",g_ewald_acc);
      fprintf(screen,"  grid = %d %d %d\n",nx_pppm_acc,ny_pppm_acc,nz_pppm_acc);
      fprintf(screen,"  stencil order = %d\n",order_acc);
      if(differentiation_flag == 1) fprintf(screen,"  differentiation = ad (1 FFT energies + 1 FFT forces)\n");
      else fprintf(screen,"  differentiation = ik (1 FFT energies + 3 FFT forces)\n");
      fprintf(screen,"  estimated absolute RMS force accuracy = %g\n",
              estimated_accuracy);
      fprintf(screen,"  estimated relative force accuracy = %g\n",
              estimated_accuracy/two_charge_force);
      fprintf(screen,"  using %s precision FFTs\n",fft_prec);
      fprintf(screen,"  3d grid and FFT values/proc = %d %d\n",
              ngrid_max,nfft_both_max);
    }
    if (logfile) {
      fprintf(logfile,"  G vector (1/distance) = %g\n",g_ewald_acc);
      fprintf(logfile,"  grid = %d %d %d\n",nx_pppm_acc,ny_pppm_acc,nz_pppm_acc);
      fprintf(logfile,"  stencil order = %d\n",order_acc);
      if(differentiation_flag == 1) fprintf(logfile,"  differentiation = ad (1 FFT energies + 1 FFT forces)\n");
      else fprintf(logfile,"  differentiation = ik (1 FFT energies + 3 FFT forces)\n");
      fprintf(logfile,"  estimated absolute RMS force accuracy = %g\n",
              estimated_accuracy);
      fprintf(logfile,"  estimated relative force accuracy = %g\n",
              estimated_accuracy/two_charge_force);
      fprintf(logfile,"  using %s precision FFTs\n",fft_prec);
      fprintf(logfile,"  3d grid and FFT values/proc = %d %d\n",
              ngrid_max,nfft_both_max);
    }
  }

  // allocate K-space dependent memory

  allocate_acc();
  cg_acc->ghost_notify();
  cg_acc->setup();

  // GridComm;  cg->nbuf is defined in cg->setup().
  memory->create(cg_buf1_acc, 3*cg_acc->nbuf, "EVB_PPPMACC:cg_buf1_acc");
  memory->create(cg_buf2_acc, 3*cg_acc->nbuf, "EVB_PPPMACC:cg_buf2_acc");

  // pre-compute Green's function denomiator expansion
  // pre-compute 1d charge distribution coefficients

  compute_gf_denom_acc();
  if (differentiation_flag == 1) compute_sf_precoeff_acc();
  compute_rho_coeff_acc();
}

/* ---------------------------------------------------------------------- */

// NOTE: special version of reduce_data for FFT_SCALAR data type.
// reduce per thread data into the first part of the data
// array that is used for the non-threaded parts and reset
// the temporary storage to 0.0. this routine depends on
// multi-dimensional arrays like force stored in this order
// x1,y1,z1,x2,y2,z2,...
// we need to post a barrier to wait until all threads are done
// writing to the array.

/* ---------------------------------------------------------------------- */

static void data_reduce_fft(FFT_SCALAR *dall, int nall, int nthreads, int ndim, int tid)
{
#if defined(_OPENMP)
  // NOOP in non-threaded execution.
  if (nthreads == 1) return;
#pragma omp barrier
  {
    const int nvals = ndim*nall;
    const int idelta = nvals/nthreads + 1;
    const int ifrom = tid*idelta;
    const int ito   = ((ifrom + idelta) > nvals) ? nvals : (ifrom + idelta);

    // ** AWGL : swap order of loops for better memory access ** //
    // this if protects against having more threads than atoms
    const int iter = ito - ifrom;
    FFT_SCALAR *dall_off = dall + ifrom;
    if (ifrom < nall) { 
      for (int n = 1; n < nthreads; ++n) {
        for (int m = 0; m < iter; ++m) {
	  dall_off[m] += dall_off[n*nvals + m];
	  dall_off[n*nvals + m] = 0.0;
	}
      }
    }
  }
#else
  // NOOP in non-threaded execution.
  return;
#endif
}

/*************************************************************************/

void EVB_PPPMACCOMP::evb_setup()
{
  int nlocal = atom->nlocal;

  // extend size of per-atom arrays if necessary

  if (nlocal > nmax) {
    memory->destroy(part2grid);
    memory->destroy(part2grid_dr);
    memory->destroy(part2grid_acc);
    memory->destroy(part2grid_dr_acc);

    nmax = atom->nmax;
    memory->create(part2grid,nmax,3,"EVB_PPPMACCOMP:part2grid");
    memory->create(part2grid_dr,nmax,3,"EVB_PPPMACCOMP:part2grid_dr");
    memory->create(part2grid_acc,nmax,3,"EVB_PPPMACCOMP:part2grid_acc");
    memory->create(part2grid_dr_acc,nmax,3,"EVB_PPPMACCOMP:part2grid_dr_acc");
  }
}

/* ----------------------------------------------------------------------
   adjust PPPM coeffs, called initially and whenever volume has changed 
------------------------------------------------------------------------- */
void EVB_PPPMACCOMP::setup()
{
  int i,j,k,n;
  double *prd;

  // volume-dependent factors
  // adjust z dimension for 2d slab PPPM
  // z dimension for 3d PPPM is zprd since slab_volfactor = 1.0

  if (triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;
    
  delxinv = nx_pppm/xprd;
  delyinv = ny_pppm/yprd;
  delzinv = nz_pppm/zprd_slab;

  delvolinv = delxinv*delyinv*delzinv;

  const double unitkx = (2.0*MY_PI/xprd);
  const double unitky = (2.0*MY_PI/yprd);
  const double unitkz = (2.0*MY_PI/zprd_slab);

  // fkx,fky,fkz for my FFT grid pts

  double per;

  for (i = nxlo_fft; i <= nxhi_fft; i++) {
    per = i - nx_pppm*(2*i/nx_pppm);
    fkx[i] = unitkx*per;
  }

  for (i = nylo_fft; i <= nyhi_fft; i++) {
    per = i - ny_pppm*(2*i/ny_pppm);
    fky[i] = unitky*per;
  }

  for (i = nzlo_fft; i <= nzhi_fft; i++) {
    per = i - nz_pppm*(2*i/nz_pppm);
    fkz[i] = unitkz*per;
  }

  // virial coefficients

  double sqk,vterm;

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++) {
    for (j = nylo_fft; j <= nyhi_fft; j++) {
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
	sqk = fkx[i]*fkx[i] + fky[j]*fky[j] + fkz[k]*fkz[k];
	if (sqk == 0.0) {
	  vg[n][0] = 0.0;
	  vg[n][1] = 0.0;
	  vg[n][2] = 0.0;
	  vg[n][3] = 0.0;
	  vg[n][4] = 0.0;
	  vg[n][5] = 0.0;
	} else {
	  vterm = -2.0 * (1.0/sqk + 0.25/(g_ewald*g_ewald));
	  vg[n][0] = 1.0 + vterm*fkx[i]*fkx[i];
	  vg[n][1] = 1.0 + vterm*fky[j]*fky[j];
	  vg[n][2] = 1.0 + vterm*fkz[k]*fkz[k];
	  vg[n][3] = vterm*fkx[i]*fky[j];
	  vg[n][4] = vterm*fkx[i]*fkz[k];
	  vg[n][5] = vterm*fky[j]*fkz[k];
	}
	n++;
      }
    }
  }

  if (differentiation_flag == 1) compute_gf_ad();
  else compute_gf_ik();

  // Setup for coarse grid
  setup_acc();
}

/* ----------------------------------------------------------------------
   pre-compute modified (Hockney-Eastwood) Coulomb Green's function
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::compute_gf_ik()
{
  const double * const prd = (triclinic==0) ? domain->prd : domain->prd_lamda;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  const int nbx = static_cast<int> ((g_ewald*xprd/(MY_PI*nx_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nby = static_cast<int> ((g_ewald*yprd/(MY_PI*ny_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nbz = static_cast<int> ((g_ewald*zprd_slab/(MY_PI*nz_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int numk = nxhi_fft - nxlo_fft + 1;
  const int numl = nyhi_fft - nylo_fft + 1;

  const int twoorder = 2*order;

#if defined(_OPENMP)
#pragma omp parallel default(none)
#endif
  {
    double snx,sny,snz;
    double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
    double sum1,dot1,dot2;
    double numerator,denominator;
    double sqk;

    int k,l,m,nx,ny,nz,kper,lper,mper,n,nfrom,nto,tid;

    loop_setup_thr(nfrom, nto, tid, nfft, comm->nthreads);

    for (n = nfrom; n < nto; ++n) {
      m = n / (numl*numk);
      l = (n - m*numl*numk) / numk;
      k = n - m*numl*numk - l*numk;
      m += nzlo_fft;
      l += nylo_fft;
      k += nxlo_fft;

      mper = m - nz_pppm*(2*m/nz_pppm);
      snz = square(sin(0.5*unitkz*mper*zprd_slab/nz_pppm));

      lper = l - ny_pppm*(2*l/ny_pppm);
      sny = square(sin(0.5*unitky*lper*yprd/ny_pppm));

      kper = k - nx_pppm*(2*k/nx_pppm);
      snx = square(sin(0.5*unitkx*kper*xprd/nx_pppm));

      sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);

      if (sqk != 0.0) {
        numerator = 12.5663706/sqk;
        denominator = gf_denom(snx,sny,snz);
        sum1 = 0.0;

        for (nx = -nbx; nx <= nbx; nx++) {
          qx = unitkx*(kper+nx_pppm*nx);
          sx = exp(-0.25*square(qx/g_ewald));
          argx = 0.5*qx*xprd/nx_pppm;
          wx = powsinxx(argx,twoorder);

          for (ny = -nby; ny <= nby; ny++) {
            qy = unitky*(lper+ny_pppm*ny);
            sy = exp(-0.25*square(qy/g_ewald));
            argy = 0.5*qy*yprd/ny_pppm;
            wy = powsinxx(argy,twoorder);

            for (nz = -nbz; nz <= nbz; nz++) {
              qz = unitkz*(mper+nz_pppm*nz);
              sz = exp(-0.25*square(qz/g_ewald));
              argz = 0.5*qz*zprd_slab/nz_pppm;
              wz = powsinxx(argz,twoorder);

              dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
              dot2 = qx*qx+qy*qy+qz*qz;
              sum1 += (dot1/dot2) * sx*sy*sz * wx*wy*wz;
            }
          }
        }
        greensfn[n] = numerator*sum1/denominator;
      } else greensfn[n] = 0.0;
    }
  } // end of parallel region
}

/* ----------------------------------------------------------------------
   compute optimized Green's function for energy calculation
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::compute_gf_ad()
{

  const double * const prd = (triclinic==0) ? domain->prd : domain->prd_lamda;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  const int numk = nxhi_fft - nxlo_fft + 1;
  const int numl = nyhi_fft - nylo_fft + 1;

  const int twoorder = 2*order;
  double sf0=0.0,sf1=0.0,sf2=0.0,sf3=0.0,sf4=0.0,sf5=0.0;

#if defined(_OPENMP)
#pragma omp parallel default(none) reduction(+:sf0,sf1,sf2,sf3,sf4,sf5)
#endif
  {
    double snx,sny,snz,sqk;
    double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
    double numerator,denominator;
    int k,l,m,kper,lper,mper,n,nfrom,nto,tid;

    loop_setup_thr(nfrom, nto, tid, nfft, comm->nthreads);

    for (n = nfrom; n < nto; ++n) {

      m = n / (numl*numk);
      l = (n - m*numl*numk) / numk;
      k = n - m*numl*numk - l*numk;
      m += nzlo_fft;
      l += nylo_fft;
      k += nxlo_fft;

      mper = m - nz_pppm*(2*m/nz_pppm);
      qz = unitkz*mper;
      snz = square(sin(0.5*qz*zprd_slab/nz_pppm));
      sz = exp(-0.25*square(qz/g_ewald));
      argz = 0.5*qz*zprd_slab/nz_pppm;
      wz = powsinxx(argz,twoorder);

      lper = l - ny_pppm*(2*l/ny_pppm);
      qy = unitky*lper;
      sny = square(sin(0.5*qy*yprd/ny_pppm));
      sy = exp(-0.25*square(qy/g_ewald));
      argy = 0.5*qy*yprd/ny_pppm;
      wy = powsinxx(argy,twoorder);

      kper = k - nx_pppm*(2*k/nx_pppm);
      qx = unitkx*kper;
      snx = square(sin(0.5*qx*xprd/nx_pppm));
      sx = exp(-0.25*square(qx/g_ewald));
      argx = 0.5*qx*xprd/nx_pppm;
      wx = powsinxx(argx,twoorder);

      sqk = qx*qx + qy*qy + qz*qz;

      if (sqk != 0.0) {
        numerator = MY_4PI/sqk;
        denominator = gf_denom(snx,sny,snz);
        greensfn[n] = numerator*sx*sy*sz*wx*wy*wz/denominator;
        sf0 += sf_precoeff1[n]*greensfn[n];
        sf1 += sf_precoeff2[n]*greensfn[n];
        sf2 += sf_precoeff3[n]*greensfn[n];
        sf3 += sf_precoeff4[n]*greensfn[n];
        sf4 += sf_precoeff5[n]*greensfn[n];
        sf5 += sf_precoeff6[n]*greensfn[n];
      } else {
        greensfn[n] = 0.0;
        sf0 += sf_precoeff1[n]*greensfn[n];
        sf1 += sf_precoeff2[n]*greensfn[n];
        sf2 += sf_precoeff3[n]*greensfn[n];
        sf3 += sf_precoeff4[n]*greensfn[n];
        sf4 += sf_precoeff5[n]*greensfn[n];
        sf5 += sf_precoeff6[n]*greensfn[n];
      }
    }
  } // end of parallel region
  
  // compute the coefficients for the self-force correction

  double prex, prey, prez, tmp[6];
  prex = prey = prez = MY_PI/volume;
  prex *= nx_pppm/xprd;
  prey *= ny_pppm/yprd;
  prez *= nz_pppm/zprd_slab;
  tmp[0] = sf0 * prex;
  tmp[1] = sf1 * prex*2;
  tmp[2] = sf2 * prey;
  tmp[3] = sf3 * prey*2;
  tmp[4] = sf4 * prez;
  tmp[5] = sf5 * prez*2;

  // communicate values with other procs

  MPI_Allreduce(tmp,sf_coeff,6,MPI_DOUBLE,MPI_SUM,world);
}

/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::make_rho()
{
  const double * const q = atom->q;
  const double * const * const x = atom->x;
  const int nthreads = comm->nthreads;
  const int nlocal = atom->nlocal;

#if defined(_OPENMP)
#pragma omp parallel default(none) 
#endif
  {  
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int tid = omp_get_thread_num();
    const int inum = nlocal;
    const int idelta = 1 + inum/nthreads;
    const int ifrom = tid*idelta;
    const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
#else
    const int tid = 0;
    const int ifrom = 0;
    const int ito = nlocal;
#endif

    // set up clear 3d density array
    const int nzoffs = (nzhi_out-nzlo_out+1)*tid;
    FFT_SCALAR * const * const * const db = &(density_brick[nzoffs]);
    memset(&(db[nzlo_out][nylo_out][nxlo_out]),0,ngrid*sizeof(FFT_SCALAR));

    ThrData *thr = fix->get_thr(tid);
    FFT_SCALAR * const * const r1d = static_cast<FFT_SCALAR **>(thr->get_rho1d());

    // loop over my charges, add their contribution to nearby grid points
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // (dx,dy,dz) = distance to "lower left" grid pt
    // (mx,my,mz) = global coords of moving stencil pt
    
    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) { 
     if (order == 5) {
      for (int i = ifrom; i < ito; i++) {

        const double ddx = (x[i][0]-boxlo[0])*delxinv;
        const double ddy = (x[i][1]-boxlo[1])*delyinv;
        const double ddz = (x[i][2]-boxlo[2])*delzinv;
        const int nx = static_cast<int> (ddx+shift) - OFFSET;
        const int ny = static_cast<int> (ddy+shift) - OFFSET;
        const int nz = static_cast<int> (ddz+shift) - OFFSET;
        part2grid[i][0] = nx;
        part2grid[i][1] = ny;
        part2grid[i][2] = nz;
	const FFT_SCALAR dx = nx+shiftone - ddx;
	const FFT_SCALAR dy = ny+shiftone - ddy;
	const FFT_SCALAR dz = nz+shiftone - ddz; 
        part2grid_dr[i][0] = dx;
        part2grid_dr[i][1] = dy;
        part2grid_dr[i][2] = dz; 

        // Code specific to order = 5
	//compute_rho1d_thr(r1d,dx,dy,dz);
	// completely unrolled loop
        const FFT_SCALAR dx2 = dx*dx;
        const FFT_SCALAR dx3 = dx2*dx;
        const FFT_SCALAR dx4 = dx2*dx2;
        const FFT_SCALAR dy2 = dy*dy;
        const FFT_SCALAR dy3 = dy2*dy;
        const FFT_SCALAR dy4 = dy2*dy2;
        const FFT_SCALAR dz2 = dz*dz;
        const FFT_SCALAR dz3 = dz2*dz;
        const FFT_SCALAR dz4 = dz2*dz2;
	int k = -2;
        r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
        r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
        r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
        k = -1;
        r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
        r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
        r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
        k = 0;
        r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
        r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
        r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
        k = 1;
        r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
        r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
        r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
        k = 2;
        r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
        r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
        r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;

	const FFT_SCALAR z0 = delvolinv * q[i];
	for (int n = nlower; n <= nupper; n++) {
	  const FFT_SCALAR y0 = z0*r1d[2][n];
	  for (int m = nlower; m <= nupper; m++) {
	    const FFT_SCALAR x0 = y0*r1d[1][m];
	    for (int l = nlower; l <= nupper; l++) {
	      db[n+nz][m+ny][l+nx] += x0*r1d[0][l];
	    }
	  }
	}
      }
     } else {
      for (int i = ifrom; i < ito; i++) {

        const double ddx = (x[i][0]-boxlo[0])*delxinv;
        const double ddy = (x[i][1]-boxlo[1])*delyinv;
        const double ddz = (x[i][2]-boxlo[2])*delzinv;
        const int nx = static_cast<int> (ddx+shift) - OFFSET;
        const int ny = static_cast<int> (ddy+shift) - OFFSET;
        const int nz = static_cast<int> (ddz+shift) - OFFSET;
        part2grid[i][0] = nx;
        part2grid[i][1] = ny;
        part2grid[i][2] = nz;
	const FFT_SCALAR dx = nx+shiftone - ddx;
	const FFT_SCALAR dy = ny+shiftone - ddy;
	const FFT_SCALAR dz = nz+shiftone - ddz; 
        part2grid_dr[i][0] = dx;
        part2grid_dr[i][1] = dy;
        part2grid_dr[i][2] = dz; 

        // General order code 
	compute_rho1d_thr(r1d,dx,dy,dz);

	const FFT_SCALAR z0 = delvolinv * q[i];
	for (int n = nlower; n <= nupper; n++) {
	  const FFT_SCALAR y0 = z0*r1d[2][n];
	  for (int m = nlower; m <= nupper; m++) {
	    const FFT_SCALAR x0 = y0*r1d[1][m];
	    for (int l = nlower; l <= nupper; l++) {
	      db[n+nz][m+ny][l+nx] += x0*r1d[0][l];
	    }
	  }
	}
      }
     }
    }
#if defined(_OPENMP)
    // reduce 3d density array
    if (nthreads > 1) {
      data_reduce_fft(&(density_brick[nzlo_out][nylo_out][nxlo_out]),ngrid,nthreads,1,tid);
    }
#endif
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::fieldforce()
{
  if (differentiation_flag == 1) fieldforce_ad();
  else fieldforce_ik();
}

void EVB_PPPMACCOMP::fieldforce_ik()
{
  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  const double * const q = atom->q;
  const double * const * const x = atom->x;
  const int nthreads = comm->nthreads;
  const int nlocal = atom->nlocal;

#if defined(_OPENMP)
#pragma omp parallel default(none)
#endif
  {  
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int tid = omp_get_thread_num();
    const int inum = nlocal;
    const int idelta = 1 + inum/nthreads;
    const int ifrom = tid*idelta;
    const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
#else
    const int ifrom = 0;
    const int ito = nlocal;
    const int tid = 0;
#endif
    ThrData *thr = fix->get_thr(tid);
    double * const * const f = thr->get_f();
    FFT_SCALAR * const * const r1d =  static_cast<FFT_SCALAR **>(thr->get_rho1d());
    
    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) { 
     if (order == 5) {
      for (int i = ifrom; i < ito; ++i) {

	const int nx = part2grid[i][0];
	const int ny = part2grid[i][1];
	const int nz = part2grid[i][2];

	//compute_rho1d_thr(r1d,part2grid_dr[i][0],part2grid_dr[i][1],part2grid_dr[i][2]);
	// order = 5 specific code
	const FFT_SCALAR dx = part2grid_dr[i][0];
	const FFT_SCALAR dy = part2grid_dr[i][1];
	const FFT_SCALAR dz = part2grid_dr[i][2];
        const FFT_SCALAR dx2 = dx*dx;
        const FFT_SCALAR dx3 = dx2*dx;
        const FFT_SCALAR dx4 = dx2*dx2;
        const FFT_SCALAR dy2 = dy*dy;
        const FFT_SCALAR dy3 = dy2*dy;
        const FFT_SCALAR dy4 = dy2*dy2;
        const FFT_SCALAR dz2 = dz*dz;
        const FFT_SCALAR dz3 = dz2*dz;
        const FFT_SCALAR dz4 = dz2*dz2;
	int k = -2;
        r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
        r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
        r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
        k = -1;
        r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
        r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
        r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
        k = 0;
        r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
        r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
        r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
        k = 1;
        r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
        r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
        r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
        k = 2;
        r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
        r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
        r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;

        FFT_SCALAR ekx, eky, ekz;
        ekx = eky = ekz = ZEROF;
	for (int n = nlower; n <= nupper; ++n) {
	  const FFT_SCALAR z0 = r1d[2][n];
	  for (int m = nlower; m <= nupper; ++m) {
	    const FFT_SCALAR y0 = z0*r1d[1][m];
	    for (int l = nlower; l <= nupper; ++l) {
	      const FFT_SCALAR x0 = y0*r1d[0][l];
	      ekx -= x0*vdx_brick[n+nz][m+ny][l+nx];
	      eky -= x0*vdy_brick[n+nz][m+ny][l+nx];
	      ekz -= x0*vdz_brick[n+nz][m+ny][l+nx];
	    }
	  }
	}

	// convert E-field to force
	const double qfactor = qscale*q[i]; 
	f[i][0] += qfactor * ekx;
	f[i][1] += qfactor * eky;
	f[i][2] += qfactor * ekz;
      }
     } else {
      for (int i = ifrom; i < ito; ++i) {

	const int nx = part2grid[i][0];
	const int ny = part2grid[i][1];
	const int nz = part2grid[i][2];

        // General order code
	compute_rho1d_thr(r1d,part2grid_dr[i][0],part2grid_dr[i][1],part2grid_dr[i][2]);

        FFT_SCALAR ekx, eky, ekz;
        ekx = eky = ekz = ZEROF;
	for (int n = nlower; n <= nupper; ++n) {
	  const FFT_SCALAR z0 = r1d[2][n];
	  for (int m = nlower; m <= nupper; ++m) {
	    const FFT_SCALAR y0 = z0*r1d[1][m];
	    for (int l = nlower; l <= nupper; ++l) {
	      const FFT_SCALAR x0 = y0*r1d[0][l];
	      ekx -= x0*vdx_brick[n+nz][m+ny][l+nx];
	      eky -= x0*vdy_brick[n+nz][m+ny][l+nx];
	      ekz -= x0*vdz_brick[n+nz][m+ny][l+nx];
	    }
	  }
	}

	// convert E-field to force
	const double qfactor = qscale*q[i]; 
	f[i][0] += qfactor * ekx;
	f[i][1] += qfactor * eky;
	f[i][2] += qfactor * ekz;
      }
     }
    }
  }
}

void EVB_PPPMACCOMP::fieldforce_ad()
{
  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  // no local atoms => nothing to do

  if (nlocal == 0) return;

  const double *prd = (triclinic == 0) ? domain->prd : domain->prd_lamda;
  const double hx_inv = nx_pppm / prd[0];
  const double hy_inv = ny_pppm / prd[1];
  const double hz_inv = nz_pppm / prd[2];

  const double * _noalias const q = atom->q;
  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  const int3_t * _noalias const p2g = (int3_t *) part2grid[0];
  const dbl3_t * _noalias const p2g_dr = (dbl3_t *) part2grid_dr[0];
  const int nthreads = comm->nthreads;
  const int nlocal = atom->nlocal;


#if defined(_OPENMP)
#pragma omp parallel default(none) shared(stdout)
#endif
  {  
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int tid = omp_get_thread_num();
    const int inum = nlocal;
    const int idelta = 1 + inum/nthreads;
    const int ifrom = tid*idelta;
    const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
#else
    const int ifrom = 0;
    const int ito = nlocal;
    const int tid = 0;
#endif

    double s1, s2, s3, sf;
    FFT_SCALAR ekx, eky, ekz;
    int mx, my, mz;
    
    ThrData *thr = fix->get_thr(tid);
    dbl3_t * _noalias const f = (dbl3_t *) thr->get_f()[0];
    FFT_SCALAR * const * const r1d =  static_cast<FFT_SCALAR **>(thr->get_rho1d());
    FFT_SCALAR * const * const d1d =  static_cast<FFT_SCALAR **>(thr->get_drho1d());
    
    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) { 
     if (order == 5) {
      for (int i = ifrom; i < ito; ++i) {

	const int nx = p2g[i].a;
	const int ny = p2g[i].b;
	const int nz = p2g[i].t;
	
	// order = 5 specific code
	const FFT_SCALAR dx = p2g_dr[i].x;
	const FFT_SCALAR dy = p2g_dr[i].y;
	const FFT_SCALAR dz = p2g_dr[i].z;
	const FFT_SCALAR dx2 = dx*dx;
	const FFT_SCALAR dx3 = dx2*dx;
	const FFT_SCALAR dx4 = dx2*dx2;
	const FFT_SCALAR dy2 = dy*dy;
	const FFT_SCALAR dy3 = dy2*dy;
	const FFT_SCALAR dy4 = dy2*dy2;
	const FFT_SCALAR dz2 = dz*dz;
	const FFT_SCALAR dz3 = dz2*dz;
	const FFT_SCALAR dz4 = dz2*dz2;
	
	// compute_rho1d_thr
	int k = -2;
	r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
	r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
	r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
	k = -1;
	r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
	r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
	r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
	k = 0;
	r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
	r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
	r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
	k = 1;
	r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
	r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
	r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
	k = 2;
	r1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
	r1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
	r1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;

	// compute_drho1d_thr
	k = -2;
	d1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
	d1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
	d1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
	k = -1;
	d1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
	d1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
	d1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
	k = 0;
	d1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
	d1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
	d1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
	k = 1;
	d1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
	d1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
	d1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
	k = 2;
	d1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
	d1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
	d1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;

	ekx = eky = ekz = ZEROF;
	for (int n = nlower; n <= nupper; ++n) {
	  mz = n + nz;
	  for (int m = nlower; m <= nupper; ++m) {
	    my = m + ny;
	    for (int l = nlower; l <= nupper; ++l) {
	      mx = m + nx;
	      ekx += d1d[0][l] * r1d[1][m] * r1d[2][n] * u_brick[mz][my][mx];
	      eky += r1d[0][l] * d1d[1][m] * r1d[2][n] * u_brick[mz][my][mx];
	      ekz += r1d[0][l] * r1d[1][m] * d1d[2][n] * u_brick[mz][my][mx];
	    }
	  }
	}
	ekx *= hx_inv;
	eky *= hy_inv;
	ekz *= hz_inv;

	// convert E-field to force and subtract self forces

	const double qi = q[i];
	const double qfactor = qscale * q[i]; 

	s1 = x[i].x*hx_inv;
	sf = sf_coeff[0]*sin(MY_2PI*s1);
	sf += sf_coeff[1]*sin(MY_4PI*s1);
	sf *= 2.0*qi;
	f[i].x += qfactor*(ekx - sf);
	
	s2 = x[i].y*hy_inv;
	sf = sf_coeff[2]*sin(MY_2PI*s2);
	sf += sf_coeff[3]*sin(MY_4PI*s2);
	sf *= 2.0*qi;
	f[i].y += qfactor*(eky - sf);
	
	if (slabflag != 2) {
	  s3 = x[i].z*hz_inv;
	  sf = sf_coeff[4]*sin(MY_2PI*s3);
	  sf += sf_coeff[5]*sin(MY_4PI*s3);
	  sf *= 2.0*qi;
	  f[i].z += qfactor*(ekz - sf);
	}

      }
     } else {
      for (int i = ifrom; i < ito; ++i) {

	const int nx = p2g[i].a;
	const int ny = p2g[i].b;
	const int nz = p2g[i].t;

        // General order code
	compute_rho1d_thr( r1d, p2g_dr[i].x, p2g_dr[i].y, p2g_dr[i].z);
	compute_drho1d_thr(d1d, p2g_dr[i].x, p2g_dr[i].y, p2g_dr[i].z);

        ekx = eky = ekz = ZEROF;

	for (int n = nlower; n <= nupper; ++n) {
	  mz = n + nz;
	  for (int m = nlower; m <= nupper; ++m) {
	    my = m + ny;
	    for (int l = nlower; l <= nupper; ++l) {
	      mx = m + nx;
	      ekx += d1d[0][l] * r1d[1][m] * r1d[2][n] * u_brick[mz][my][mx];
	      eky += r1d[0][l] * d1d[1][m] * r1d[2][n] * u_brick[mz][my][mx];
	      ekz += r1d[0][l] * r1d[1][m] * d1d[2][n] * u_brick[mz][my][mx];
	    }
	  }
	}
	ekx *= hx_inv;
	eky *= hy_inv;
	ekz *= hz_inv;

	// convert E-field to force and subtract self forces

	const double qi = q[i];
	const double qfactor = qscale * q[i]; 

	s1 = x[i].x*hx_inv;
	sf = sf_coeff[0]*sin(MY_2PI*s1);
	sf += sf_coeff[1]*sin(MY_4PI*s1);
	sf *= 2.0*qi;
	f[i].x += qfactor*(ekx - sf);
	
	s2 = x[i].y*hy_inv;
	sf = sf_coeff[2]*sin(MY_2PI*s2);
	sf += sf_coeff[3]*sin(MY_4PI*s2);
	sf *= 2.0*qi;
	f[i].y += qfactor*(eky - sf);
	
	if (slabflag != 2) {
	  s3 = x[i].z*hz_inv;
	  sf = sf_coeff[4]*sin(MY_2PI*s3);
	  sf += sf_coeff[5]*sin(MY_4PI*s3);
	  sf *= 2.0*qi;
	  f[i].z += qfactor*(ekz - sf);
	}

      }
     }
    }
  }
}

/* ----------------------------------------------------------------------
 interpolate from grid to get per-atom energy/virial
 ------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::fieldforce_peratom()
{
  // loop over my charges, interpolate from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt


  const double * const q = atom->q;
  const double * const * const x = atom->x;
  const int nthreads = comm->nthreads;
  const int nlocal = atom->nlocal;

#if defined(_OPENMP)
#pragma omp parallel default(none)
#endif
  {
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int tid = omp_get_thread_num();
    const int inum = nlocal;
    const int idelta = 1 + inum/nthreads;
    const int ifrom = tid*idelta;
    const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
#else
    const int ifrom = 0;
    const int ito = nlocal;
    const int tid = 0;
#endif
    ThrData *thr = fix->get_thr(tid);
    FFT_SCALAR * const * const r1d =  static_cast<FFT_SCALAR **>(thr->get_rho1d());

    int i,l,m,n,nx,ny,nz,mx,my,mz;
    FFT_SCALAR dx,dy,dz,x0,y0,z0;
    FFT_SCALAR u,v0,v1,v2,v3,v4,v5;

    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) {
      for (int i = ifrom; i < ito; i++) {

	nx = part2grid[i][0];
	ny = part2grid[i][1];
	nz = part2grid[i][2];
	dx = nx+shiftone - (x[i][0]-boxlo[0])*delxinv;
	dy = ny+shiftone - (x[i][1]-boxlo[1])*delyinv;
	dz = nz+shiftone - (x[i][2]-boxlo[2])*delzinv;

	compute_rho1d_thr(r1d,dx,dy,dz);

	u = v0 = v1 = v2 = v3 = v4 = v5 = ZEROF;
	for (n = nlower; n <= nupper; n++) {
	  mz = n+nz;
	  z0 = r1d[2][n];
	  for (m = nlower; m <= nupper; m++) {
	    my = m+ny;
	    y0 = z0*r1d[1][m];
	    for (l = nlower; l <= nupper; l++) {
	      mx = l+nx;
	      x0 = y0*r1d[0][l];
	      if (eflag_atom) u += x0*u_brick[mz][my][mx];
	      if (vflag_atom) {
		v0 += x0*v0_brick[mz][my][mx];
		v1 += x0*v1_brick[mz][my][mx];
		v2 += x0*v2_brick[mz][my][mx];
		v3 += x0*v3_brick[mz][my][mx];
		v4 += x0*v4_brick[mz][my][mx];
		v5 += x0*v5_brick[mz][my][mx];
	      }
	    }
	  }
	}

	if (eflag_atom) eatom[i] += q[i]*u;
	if (vflag_atom) {
	  vatom[i][0] += v0;
	  vatom[i][1] += v1;
	  vatom[i][2] += v2;
	  vatom[i][3] += v3;
	  vatom[i][4] += v4;
	  vatom[i][5] += v5;
	}
      }
    }
  }
}


/* ----------------------------------------------------------------------
   charge assignment into rho1d
   dx,dy,dz = distance of particle from "lower left" grid point 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::compute_rho1d(const FFT_SCALAR &dx, const FFT_SCALAR &dy, const FFT_SCALAR &dz)
{
  if (order == 5) {
    // order = 5 case, completely unrolled loops
    const double dx2 = dx*dx;
    const double dx3 = dx2*dx;
    const double dx4 = dx2*dx2;
    const double dy2 = dy*dy;
    const double dy3 = dy2*dy;
    const double dy4 = dy2*dy2;
    const double dz2 = dz*dz;
    const double dz3 = dz2*dz;
    const double dz4 = dz2*dz2;
    int k = -2;
    rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
    rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
    rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
    k = -1;
    rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
    rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
    rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
    k = 0;
    rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
    rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
    rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
    k = 1;
    rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
    rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
    rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
    k = 2;
    rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
    rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
    rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
  } else {
    // general case
    FFT_SCALAR r[3];
    for (int k = (1-order)/2; k <= order/2; ++k) {
      r[0] = r[1] = r[2] = ZEROF;
      for (int l = order-1; l >= 0; --l) {
        r[0] = rho_coeff[l][k] + r[0]*dx;
        r[1] = rho_coeff[l][k] + r[1]*dy;
        r[2] = rho_coeff[l][k] + r[2]*dz;
      }
      rho1d[0][k] = r[0];
      rho1d[1][k] = r[1];
      rho1d[2][k] = r[2];
    }
  }
}

/* ----------------------------------------------------------------------
   charge assignment into rho1d
   dx,dy,dz = distance of particle from "lower left" grid point 
------------------------------------------------------------------------- */

inline void EVB_PPPMACCOMP::compute_rho1d_thr(FFT_SCALAR * const * const r1d, const FFT_SCALAR &dx,
                                           const FFT_SCALAR &dy, const FFT_SCALAR &dz)
{
  FFT_SCALAR r[3];
  for (int k = (1-order)/2; k <= order/2; ++k) {
    r[0] = r[1] = r[2] = ZEROF;
    for (int l = order-1; l >= 0; --l) {
      r[0] = rho_coeff[l][k] + r[0]*dx;
      r[1] = rho_coeff[l][k] + r[1]*dy;
      r[2] = rho_coeff[l][k] + r[2]*dz;
    }
    r1d[0][k] = r[0];
    r1d[1][k] = r[1];
    r1d[2][k] = r[2];
  }
}

/* ----------------------------------------------------------------------
   charge assignment into drho1d
   dx,dy,dz = distance of particle from "lower left" grid point 
------------------------------------------------------------------------- */

inline void EVB_PPPMACCOMP::compute_drho1d_thr(FFT_SCALAR * const * const d1d, const FFT_SCALAR &dx,
			      const FFT_SCALAR &dy, const FFT_SCALAR &dz)
{
  FFT_SCALAR r1,r2,r3;

  for (int k = (1-order)/2; k <= order/2; k++) {
    r1 = r2 = r3 = ZEROF;

    for (int l = order-2; l >= 0; l--) {
      r1 = drho_coeff[l][k] + r1*dx;
      r2 = drho_coeff[l][k] + r2*dy;
      r3 = drho_coeff[l][k] + r3*dz;
    }
    d1d[0][k] = r1;
    d1d[1][k] = r2;
    d1d[2][k] = r3;
  }
}

/* ----------------------------------------------------------------------
   compute self force coefficients for ad-differentiation scheme
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::compute_sf_precoeff_acc()
{
  int i,k,l,m,n;
  int nx,ny,nz,kper,lper,mper;
  double wx0[5],wy0[5],wz0[5],wx1[5],wy1[5],wz1[5],wx2[5],wy2[5],wz2[5];
  double qx0,qy0,qz0,qx1,qy1,qz1,qx2,qy2,qz2;
  double u0,u1,u2,u3,u4,u5,u6;
  double sum1,sum2,sum3,sum4,sum5,sum6;

  n = 0;
  for (m = nzlo_fft_acc; m <= nzhi_fft_acc; m++) {
    mper = m - nz_pppm_acc*(2*m/nz_pppm_acc);

    for (l = nylo_fft_acc; l <= nyhi_fft_acc; l++) {
      lper = l - ny_pppm_acc*(2*l/ny_pppm_acc);

      for (k = nxlo_fft_acc; k <= nxhi_fft_acc; k++) {
        kper = k - nx_pppm_acc*(2*k/nx_pppm_acc);

        sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = 0.0;
        for (i = 0; i < 5; i++) {

          qx0 = MY_2PI*(kper+nx_pppm_acc*(i-2));
          qx1 = MY_2PI*(kper+nx_pppm_acc*(i-1));
          qx2 = MY_2PI*(kper+nx_pppm_acc*(i  ));
          wx0[i] = powsinxx(0.5*qx0/nx_pppm_acc,order_acc);
          wx1[i] = powsinxx(0.5*qx1/nx_pppm_acc,order_acc);
          wx2[i] = powsinxx(0.5*qx2/nx_pppm_acc,order_acc);

          qy0 = MY_2PI*(lper+ny_pppm_acc*(i-2));
          qy1 = MY_2PI*(lper+ny_pppm_acc*(i-1));
          qy2 = MY_2PI*(lper+ny_pppm_acc*(i  ));
          wy0[i] = powsinxx(0.5*qy0/ny_pppm_acc,order_acc);
          wy1[i] = powsinxx(0.5*qy1/ny_pppm_acc,order_acc);
          wy2[i] = powsinxx(0.5*qy2/ny_pppm_acc,order_acc);

          qz0 = MY_2PI*(mper+nz_pppm_acc*(i-2));
          qz1 = MY_2PI*(mper+nz_pppm_acc*(i-1));
          qz2 = MY_2PI*(mper+nz_pppm_acc*(i  ));

          wz0[i] = powsinxx(0.5*qz0/nz_pppm_acc,order_acc);
          wz1[i] = powsinxx(0.5*qz1/nz_pppm_acc,order_acc);
          wz2[i] = powsinxx(0.5*qz2/nz_pppm_acc,order_acc);
        }

        for (nx = 0; nx < 5; nx++) {
          for (ny = 0; ny < 5; ny++) {
            for (nz = 0; nz < 5; nz++) {
              u0 = wx0[nx]*wy0[ny]*wz0[nz];
              u1 = wx1[nx]*wy0[ny]*wz0[nz];
              u2 = wx2[nx]*wy0[ny]*wz0[nz];
              u3 = wx0[nx]*wy1[ny]*wz0[nz];
              u4 = wx0[nx]*wy2[ny]*wz0[nz];
              u5 = wx0[nx]*wy0[ny]*wz1[nz];
              u6 = wx0[nx]*wy0[ny]*wz2[nz];

              sum1 += u0*u1;
              sum2 += u0*u2;
              sum3 += u0*u3;
              sum4 += u0*u4;
              sum5 += u0*u5;
              sum6 += u0*u6;
            }
          }
        }

        // store values

        sf_precoeff1_acc[n] = sum1;
        sf_precoeff2_acc[n] = sum2;
        sf_precoeff3_acc[n] = sum3;
        sf_precoeff4_acc[n] = sum4;
        sf_precoeff5_acc[n] = sum5;
        sf_precoeff6_acc[n++] = sum6;
      }
    }
  }
}

/*************************************************************************/

void EVB_PPPMACCOMP::map2density_one_subtract(int id)
{
  double *q = atom->q;

  // Subtracts the contrinution for a given id
  const int nx = part2grid[id][0];
  const int ny = part2grid[id][1];
  const int nz = part2grid[id][2];
  
  // (dx,dy,dz) = distance to "lower left" grid pt
  
  compute_rho1d(part2grid_dr[id][0], part2grid_dr[id][1], part2grid_dr[id][2]);
  
  // (mx,my,mz) = global coords of moving stencil pt
  
  const FFT_SCALAR z0 = delvolinv * q[id];
  for (int n = nlower; n <= nupper; n++) 
    {
      const FFT_SCALAR y0 = z0*rho1d[2][n];
      for (int m = nlower; m <= nupper; m++) 
	{
	  const FFT_SCALAR x0 = y0*rho1d[1][m];
	  for (int l = nlower; l <= nupper; l++) 
	    {
	      density_brick[n+nz][m+ny][l+nx] -= x0*rho1d[0][l];
	    } // Loop mx
	} // Loop my
    } // Loop mz
}

/*************************************************************************/

void EVB_PPPMACCOMP::map2density_one(int id)
{
  double *q = atom->q;

  const int nx = part2grid[id][0];
  const int ny = part2grid[id][1];
  const int nz = part2grid[id][2];
  
  // (dx,dy,dz) = distance to "lower left" grid pt
  
  compute_rho1d(part2grid_dr[id][0], part2grid_dr[id][1], part2grid_dr[id][2]);
  
  // (mx,my,mz) = global coords of moving stencil pt
  
  const FFT_SCALAR z0 = delvolinv * q[id];
  for (int n = nlower; n <= nupper; n++) 
    {
      const FFT_SCALAR y0 = z0*rho1d[2][n];
      for (int m = nlower; m <= nupper; m++) 
	{
	  const FFT_SCALAR x0 = y0*rho1d[1][m];
	  for (int l = nlower; l <= nupper; l++) 
	    {
	      density_brick[n+nz][m+ny][l+nx] += x0*rho1d[0][l];
	    } // Loop mx
	} // Loop my
    } // Loop mz
}

/* ----------------------------------------------------------------------
   ghost-swap to accumulate full density in brick decomposition 
   remap density from 3d brick decomposition to FFT decomposition
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::brick2fft()
{

  int i,n,ix,iy,iz;

  // remap from 3d brick decomposition to FFT decomposition
  // copy grabs inner portion of density from 3d brick
  // remap could be done as pre-stage of FFT,
  //   but this works optimally on only double values, not complex values

  n = 0;
  for (iz = nzlo_in; iz <= nzhi_in; iz++)
    for (iy = nylo_in; iy <= nyhi_in; iy++)
      for (ix = nxlo_in; ix <= nxhi_in; ix++)
	density_fft[n++] = density_brick[iz][iy][ix];

  remap->perform(density_fft,density_fft,work1);
}

/*************************************************************************/
/*************************************************************************/

/* The #_acc() functions below operate on the coarse k-space grid.

/*************************************************************************/
/*************************************************************************/

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors and order 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::allocate_acc()
{
  const int nthreads = comm->nthreads;

  const int nzend = (nzhi_out_acc - nzlo_out_acc + 1)*nthreads + nzlo_out_acc - 1;

  /*******************************************************/
  /*******************************************************/
  memory->create3d_offset(env_density_brick_acc,nzlo_out_acc,nzend,nylo_out_acc,
			  nyhi_out_acc,nxlo_out_acc,nxhi_out_acc,
			  "EVB_PPPMACCOMP:env_density_brick_acc");

  /*******************************************************/
  /*******************************************************/
  
  memory->create3d_offset(density_brick_acc,nzlo_out_acc,nzend,nylo_out_acc,
			  nyhi_out_acc,nxlo_out_acc,nxhi_out_acc,
			  "EVB_PPPMACCOMP:density_brick_acc");

  memory->create(density_fft_acc,nfft_both_acc,"EVB_PPPM:density_fft_acc");
  memory->create(greensfn_acc,nfft_both_acc,"EVB_PPPM:greensfn_acc");
  memory->create(work1_acc,2*nfft_both_acc,"EVB_PPPM:work1_acc");
  memory->create(work2_acc,2*nfft_both_acc,"EVB_PPPM:work2_acc");
  memory->create(vg_acc,nfft_both_acc,6,"EVB_PPPM:vg_acc");

  memory->create1d_offset(fkx_acc,nxlo_fft_acc,nxhi_fft_acc,"EVB_PPPMACCOMP:fkx_acc");
  memory->create1d_offset(fky_acc,nylo_fft_acc,nyhi_fft_acc,"EVB_PPPMACCOMP:fky_acc");
  memory->create1d_offset(fkz_acc,nzlo_fft_acc,nzhi_fft_acc,"EVB_PPPMACCOMP:fkz_acc");

  if(differentiation_flag == 1) {
    memory->create3d_offset(u_brick_acc,nzlo_out_acc,nzhi_out_acc,nylo_out_acc,nyhi_out_acc,
			    nxlo_out_acc,nxhi_out_acc,"EVB_PPPM:u_brick_acc");
    
    memory->create(sf_precoeff1_acc,nfft_both_acc,"EVB_PPPM:sf_precoeff1_acc");
    memory->create(sf_precoeff2_acc,nfft_both_acc,"EVB_PPPM:sf_precoeff2_acc");
    memory->create(sf_precoeff3_acc,nfft_both_acc,"EVB_PPPM:sf_precoeff3_acc");
    memory->create(sf_precoeff4_acc,nfft_both_acc,"EVB_PPPM:sf_precoeff4_acc");
    memory->create(sf_precoeff5_acc,nfft_both_acc,"EVB_PPPM:sf_precoeff5_acc");
    memory->create(sf_precoeff6_acc,nfft_both_acc,"EVB_PPPM:sf_precoeff6_acc");

  } else {
    memory->create3d_offset(vdx_brick_acc,nzlo_out_acc,nzhi_out_acc,nylo_out_acc,
			    nyhi_out_acc,nxlo_out_acc,nxhi_out_acc,
			    "EVB_PPPM:vdx_brick_acc");
    memory->create3d_offset(vdy_brick_acc,nzlo_out_acc,nzhi_out_acc,nylo_out_acc,
			    nyhi_out_acc,nxlo_out_acc,nxhi_out_acc,
			    "EVB_PPPM:vdy_brick_acc");
    memory->create3d_offset(vdz_brick_acc,nzlo_out_acc,nzhi_out_acc,nylo_out_acc,
			    nyhi_out_acc,nxlo_out_acc,nxhi_out_acc,
			    "EVB_PPPM:vdz_brick_acc");
  }

  // summation coeffs

  memory->create(gf_b_acc,order_acc,"EVB_PPPMACCOMP:gf_b_acc");
  memory->create2d_offset(rho1d_acc,3,-order_acc/2,order_acc/2,"EVB_PPPMACCOMP:rho1d_acc");
  memory->create2d_offset(drho1d_acc,3,-order_acc/2,order_acc/2,"EVB_PPPMACCOMP:drho1d_acc");
  memory->create2d_offset(rho_coeff_acc,order_acc,(1-order_acc)/2,order_acc/2,"EVB_PPPMACCOMP:rho_coeff_acc");
  memory->create2d_offset(drho_coeff_acc,order_acc,(1-order_acc)/2,order_acc/2,"EVB_PPPMACCOMP:drho_coeff_acc");

  // create 2 FFTs and a Remap
  // 1st FFT keeps data in FFT decompostion
  // 2nd FFT returns data in 3d brick decomposition
  // remap takes data from 3d brick to FFT decomposition

  int tmp;

  fft1_acc = new FFT3d(lmp_pointer,world,nx_pppm_acc,ny_pppm_acc,nz_pppm_acc,
		       nxlo_fft_acc,nxhi_fft_acc,nylo_fft_acc,nyhi_fft_acc,nzlo_fft_acc,nzhi_fft_acc,
		       nxlo_fft_acc,nxhi_fft_acc,nylo_fft_acc,nyhi_fft_acc,nzlo_fft_acc,nzhi_fft_acc,
		       0,0,&tmp,collective_flag);

  fft2_acc = new FFT3d(lmp_pointer,world,nx_pppm_acc,ny_pppm_acc,nz_pppm_acc,
		       nxlo_fft_acc,nxhi_fft_acc,nylo_fft_acc,nyhi_fft_acc,nzlo_fft_acc,nzhi_fft_acc,
		       nxlo_in_acc,nxhi_in_acc,nylo_in_acc,nyhi_in_acc,nzlo_in_acc,nzhi_in_acc,
		       0,0,&tmp,collective_flag);

  remap_acc = new Remap(lmp_pointer,world,
			nxlo_in_acc,nxhi_in_acc,nylo_in_acc,nyhi_in_acc,nzlo_in_acc,nzhi_in_acc,
			nxlo_fft_acc,nxhi_fft_acc,nylo_fft_acc,nyhi_fft_acc,nzlo_fft_acc,nzhi_fft_acc,
			1,0,0,FFT_PRECISION,collective_flag);

  // create ghost grid object for rho and electric field communication

  int (*procneigh)[2] = comm->procneigh;

  if (differentiation_flag == 1)
    cg_acc = new GridComm(lmp_pointer,world,1,1,
			  nxlo_in_acc,nxhi_in_acc,nylo_in_acc,nyhi_in_acc,nzlo_in_acc,nzhi_in_acc,
			  nxlo_out_acc,nxhi_out_acc,nylo_out_acc,nyhi_out_acc,nzlo_out_acc,nzhi_out_acc,
			  procneigh[0][0],procneigh[0][1],procneigh[1][0],
			  procneigh[1][1],procneigh[2][0],procneigh[2][1]);
  else
    cg_acc = new GridComm(lmp_pointer,world,3,1,
			  nxlo_in_acc,nxhi_in_acc,nylo_in_acc,nyhi_in_acc,nzlo_in_acc,nzhi_in_acc,
			  nxlo_out_acc,nxhi_out_acc,nylo_out_acc,nyhi_out_acc,nzlo_out_acc,nzhi_out_acc,
			  procneigh[0][0],procneigh[0][1],procneigh[1][0],
			  procneigh[1][1],procneigh[2][0],procneigh[2][1]);
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors and order 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::deallocate_acc()
{
  /*******************************************************/
  /*******************************************************/
  memory->destroy3d_offset(env_density_brick_acc,nzlo_out_acc,nylo_out_acc,nxlo_out_acc);
  /*******************************************************/
  /*******************************************************/

  memory->destroy3d_offset(density_brick_acc,nzlo_out_acc,nylo_out_acc,nxlo_out_acc);


  if(differentiation_flag == 1) {
    memory->destroy3d_offset(u_brick_acc,nzlo_out_acc,nylo_out_acc,nxlo_out_acc);
    memory->destroy(sf_precoeff1_acc);
    memory->destroy(sf_precoeff2_acc);
    memory->destroy(sf_precoeff3_acc);
    memory->destroy(sf_precoeff4_acc);
    memory->destroy(sf_precoeff5_acc);
    memory->destroy(sf_precoeff6_acc);
  } else {
    memory->destroy3d_offset(vdx_brick_acc,    nzlo_out_acc,nylo_out_acc,nxlo_out_acc);
    memory->destroy3d_offset(vdy_brick_acc,    nzlo_out_acc,nylo_out_acc,nxlo_out_acc);
    memory->destroy3d_offset(vdz_brick_acc,    nzlo_out_acc,nylo_out_acc,nxlo_out_acc);
  }

  memory->sfree(density_fft_acc);
  memory->sfree(greensfn_acc);
  memory->sfree(work1_acc);
  memory->sfree(work2_acc);
  memory->destroy(vg_acc);

  memory->destroy1d_offset(fkx_acc,nxlo_fft_acc);
  memory->destroy1d_offset(fky_acc,nylo_fft_acc);
  memory->destroy1d_offset(fkz_acc,nzlo_fft_acc);

  memory->destroy(gf_b_acc);
  memory->destroy2d_offset(rho1d_acc,-order_acc/2);
  memory->destroy2d_offset(drho1d_acc,-order_acc/2);
  memory->destroy2d_offset(rho_coeff_acc,(1-order_acc)/2);
  memory->destroy2d_offset(drho_coeff_acc,(1-order_acc)/2);

  delete fft1_acc;
  delete fft2_acc;
  delete remap_acc;
  delete cg_acc;
}


/* ----------------------------------------------------------------------
   adjust EVB_PPPM coeffs, called initially and whenever volume has changed 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::setup_acc()
{
  evb_engine->flag_ACC = 1;

  int i,j,k,l,m,n;
  double *prd;

  // volume-dependent factors
  // adjust z dimension for 2d slab EVB_PPPM
  // z dimension for 3d EVB_PPPM is zprd since slab_volfactor = 1.0

  if (triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;
    
  delxinv_acc = nx_pppm_acc / xprd;
  delyinv_acc = ny_pppm_acc / yprd;
  delzinv_acc = nz_pppm_acc / zprd_slab;

  delvolinv_acc = delxinv_acc*delyinv_acc*delzinv_acc;

  const double unitkx = (2.0*MY_PI/xprd);
  const double unitky = (2.0*MY_PI/yprd);
  const double unitkz = (2.0*MY_PI/zprd_slab);

  // fkx,fky,fkz for my FFT grid pts

  double per;

  for (i = nxlo_fft_acc; i <= nxhi_fft_acc; i++) {
    per = i - nx_pppm_acc*(2*i/nx_pppm_acc);
    fkx_acc[i] = unitkx*per;
  }

  for (i = nylo_fft_acc; i <= nyhi_fft_acc; i++) {
    per = i - ny_pppm_acc*(2*i/ny_pppm_acc);
    fky_acc[i] = unitky*per;
  }

  for (i = nzlo_fft_acc; i <= nzhi_fft_acc; i++) {
    per = i - nz_pppm_acc*(2*i/nz_pppm_acc);
    fkz_acc[i] = unitkz*per;
  }

  // virial coefficients

  double sqk,vterm;

  n = 0;
  for (k = nzlo_fft_acc; k <= nzhi_fft_acc; k++) {
    for (j = nylo_fft_acc; j <= nyhi_fft_acc; j++) {
      for (i = nxlo_fft_acc; i <= nxhi_fft_acc; i++) {
	sqk = fkx_acc[i]*fkx_acc[i] + fky_acc[j]*fky_acc[j] + fkz_acc[k]*fkz_acc[k];
	if (sqk == 0.0) {
	  vg_acc[n][0] = 0.0;
	  vg_acc[n][1] = 0.0;
	  vg_acc[n][2] = 0.0;
	  vg_acc[n][3] = 0.0;
	  vg_acc[n][4] = 0.0;
	  vg_acc[n][5] = 0.0;
	} else {
	  vterm = -2.0 * (1.0/sqk + 0.25/(g_ewald*g_ewald));
	  vg_acc[n][0] = 1.0 + vterm*fkx_acc[i]*fkx_acc[i];
	  vg_acc[n][1] = 1.0 + vterm*fky_acc[j]*fky_acc[j];
	  vg_acc[n][2] = 1.0 + vterm*fkz_acc[k]*fkz_acc[k];
	  vg_acc[n][3] = vterm*fkx_acc[i]*fky_acc[j];
	  vg_acc[n][4] = vterm*fkx_acc[i]*fkz_acc[k];
	  vg_acc[n][5] = vterm*fky_acc[j]*fkz_acc[k];
	}
	n++;
      }
    }
  }

  if (differentiation_flag == 1) compute_gf_ad_acc();
  else compute_gf_ik_acc();
}

/* ----------------------------------------------------------------------
   pre-compute modified (Hockney-Eastwood) Coulomb Green's function
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::compute_gf_ik_acc()
{
  const double * const prd = (triclinic==0) ? domain->prd : domain->prd_lamda;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double snx,sny,snz;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double sum1,dot1,dot2;
  double numerator,denominator;
  double sqk;

  int k,l,m,n,nx,ny,nz,kper,lper,mper;

  const int nbx = static_cast<int> ((g_ewald_acc*xprd/(MY_PI*nx_pppm_acc)) * 
				    pow(-log(EPS_HOC),0.25));
  const int nby = static_cast<int> ((g_ewald_acc*yprd/(MY_PI*ny_pppm_acc)) * 
				    pow(-log(EPS_HOC),0.25));
  const int nbz = static_cast<int> ((g_ewald_acc*zprd_slab/(MY_PI*nz_pppm_acc)) * 
				    pow(-log(EPS_HOC),0.25));
  const int twoorder = 2*order_acc;

  n = 0;
  const double gew2 = -4.0 * g_ewald_acc * g_ewald_acc;
  
  for (m = nzlo_fft_acc; m <= nzhi_fft_acc; m++) {
    mper = m - nz_pppm_acc*(2*m/nz_pppm_acc);
    snz = square(sin(0.5*unitkz*mper*zprd_slab/nz_pppm_acc));

    for (l = nylo_fft_acc; l <= nyhi_fft_acc; l++) {
      lper = l - ny_pppm_acc*(2*l/ny_pppm_acc);
      sny = square(sin(0.5*unitky*lper*yprd/ny_pppm_acc));

      for (k = nxlo_fft_acc; k <= nxhi_fft_acc; k++) {
        kper = k - nx_pppm_acc*(2*k/nx_pppm_acc);
        snx = square(sin(0.5*unitkx*kper*xprd/nx_pppm_acc));

        sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);

	if (sqk != 0.0) {
	  numerator = 12.5663706/sqk;
	  denominator = gf_denom_acc(snx,sny,snz);  
	  sum1 = 0.0;

	  for (nx = -nbx; nx <= nbx; nx++) {
            qx = unitkx*(kper+nx_pppm_acc*nx);
            sx = exp(-0.25*square(qx/g_ewald_acc));
            argx = 0.5*qx*xprd/nx_pppm_acc;
            wx = powsinxx(argx,twoorder);

            for (ny = -nby; ny <= nby; ny++) {
              qy = unitky*(lper+ny_pppm_acc*ny);
              sy = exp(-0.25*square(qy/g_ewald_acc));
              argy = 0.5*qy*yprd/ny_pppm_acc;
              wy = powsinxx(argy,twoorder);

              for (nz = -nbz; nz <= nbz; nz++) {
                qz = unitkz*(mper+nz_pppm_acc*nz);
                sz = exp(-0.25*square(qz/g_ewald_acc));
                argz = 0.5*qz*zprd_slab/nz_pppm_acc;
                wz = powsinxx(argz,twoorder);

                dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
                dot2 = qx*qx+qy*qy+qz*qz;
                sum1 += (dot1/dot2) * sx*sy*sz * wx*wy*wz;
              }
            }
	  }
	  greensfn_acc[n++] = numerator*sum1/denominator;
	} else greensfn_acc[n++] = 0.0;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   compute optimized Green's function for energy calculation
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::compute_gf_ad_acc()
{
  const double * const prd = (triclinic==0) ? domain->prd : domain->prd_lamda;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double snx,sny,snz,sqk;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double numerator,denominator;
  int k,l,m,n,kper,lper,mper;

  const int twoorder = 2*order_acc;

  for (int i = 0; i < 6; i++) sf_coeff_acc[i] = 0.0;

  n = 0;
  for (m = nzlo_fft_acc; m <= nzhi_fft_acc; m++) {
    mper = m - nz_pppm_acc*(2*m/nz_pppm_acc);
    qz = unitkz*mper;
    snz = square(sin(0.5*qz*zprd_slab/nz_pppm_acc));
    sz = exp(-0.25*square(qz/g_ewald_acc));
    argz = 0.5*qz*zprd_slab/nz_pppm_acc;
    wz = powsinxx(argz,twoorder);

    for (l = nylo_fft_acc; l <= nyhi_fft_acc; l++) {
      lper = l - ny_pppm_acc*(2*l/ny_pppm_acc);
      qy = unitky*lper;
      sny = square(sin(0.5*qy*yprd/ny_pppm_acc));
      sy = exp(-0.25*square(qy/g_ewald_acc));
      argy = 0.5*qy*yprd/ny_pppm_acc;
      wy = powsinxx(argy,twoorder);

      for (k = nxlo_fft_acc; k <= nxhi_fft_acc; k++) {
        kper = k - nx_pppm_acc*(2*k/nx_pppm_acc);
        qx = unitkx*kper;
        snx = square(sin(0.5*qx*xprd/nx_pppm_acc));
        sx = exp(-0.25*square(qx/g_ewald_acc));
        argx = 0.5*qx*xprd/nx_pppm_acc;
        wx = powsinxx(argx,twoorder);

        sqk = qx*qx + qy*qy + qz*qz;

        if (sqk != 0.0) {
          numerator = MY_4PI/sqk;
          denominator = gf_denom_acc(snx,sny,snz);
          greensfn_acc[n] = numerator*sx*sy*sz*wx*wy*wz/denominator;
          sf_coeff_acc[0] += sf_precoeff1_acc[n]*greensfn_acc[n];
          sf_coeff_acc[1] += sf_precoeff2_acc[n]*greensfn_acc[n];
          sf_coeff_acc[2] += sf_precoeff3_acc[n]*greensfn_acc[n];
          sf_coeff_acc[3] += sf_precoeff4_acc[n]*greensfn_acc[n];
          sf_coeff_acc[4] += sf_precoeff5_acc[n]*greensfn_acc[n];
          sf_coeff_acc[5] += sf_precoeff6_acc[n]*greensfn_acc[n];
          n++;
        } else {
          greensfn_acc[n] = 0.0;
          sf_coeff_acc[0] += sf_precoeff1_acc[n]*greensfn_acc[n];
          sf_coeff_acc[1] += sf_precoeff2_acc[n]*greensfn_acc[n];
          sf_coeff_acc[2] += sf_precoeff3_acc[n]*greensfn_acc[n];
          sf_coeff_acc[3] += sf_precoeff4_acc[n]*greensfn_acc[n];
          sf_coeff_acc[4] += sf_precoeff5_acc[n]*greensfn_acc[n];
          sf_coeff_acc[5] += sf_precoeff6_acc[n]*greensfn_acc[n];
          n++;
        }
      }
    }
  }

  // compute the coefficients for the self-force correction

  double prex, prey, prez;
  prex = prey = prez = MY_PI/volume;
  prex *= nx_pppm_acc/xprd;
  prey *= ny_pppm_acc/yprd;
  prez *= nz_pppm_acc/zprd_slab;
  sf_coeff_acc[0] *= prex;
  sf_coeff_acc[1] *= prex*2;
  sf_coeff_acc[2] *= prey;
  sf_coeff_acc[3] *= prey*2;
  sf_coeff_acc[4] *= prez;
  sf_coeff_acc[5] *= prez*2;

  // communicate values with other procs

  double tmp[6];
  MPI_Allreduce(sf_coeff_acc,tmp,6,MPI_DOUBLE,MPI_SUM,world);
  for (n = 0; n < 6; n++) sf_coeff_acc[n] = tmp[n];
}


/* ----------------------------------------------------------------------
   set global size of PPPM grid = nx,ny,nz_pppm
   used for charge accumulation, FFTs, and electric field interpolation 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::set_grid_global_acc()
{
  // use xprd,yprd,zprd even if triclinic so grid size is the same
  // adjust z dimension for 2d slab EVB_PPPM
  // 3d EVB_PPPM just uses zprd since slab_volfactor = 1.0

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  
  // make initial g_ewald estimate
  // based on desired error and real space cutoff
  // fluid-occupied volume used to estimate real-space error
  // zprd used rather than zprd_slab

  double h,h_x,h_y,h_z;
  bigint natoms = atom->natoms;

  if (!gewaldflag) {
    if(accuracy <= 0.0) error->all(FLERR,"KSpace accuaracy must be > 0");
    g_ewald_acc = accuracy*sqrt(natoms*cutoff*xprd*yprd*zprd) / (2.0*q2);
    if (g_ewald_acc >= 1.0) g_ewald_acc = (1.35 - 0.15*log(accuracy)) / cutoff;
    else g_ewald_acc = sqrt(-log(g_ewald)) / cutoff;
  }

  // set optimal nx_pppm,ny_pppm,nz_pppm based on order and accuracy
  // nz_pppm uses extended zprd_slab instead of zprd
  // reduce it until precision target is met

  if (!gridflag) {

    if (differentiation_flag == 1) {

      h = h_x = h_y = h_z = 4.0/g_ewald_acc;
      int count = 0;
      while (1) {

        // set grid dimension

	nx_pppm_acc = nx_pppm / 2; // Factor of 2 or could be input parameter
	ny_pppm_acc = ny_pppm / 2;
	nz_pppm_acc = nz_pppm / 2;

        if (nx_pppm_acc <= 1) nx_pppm_acc = 2;
        if (ny_pppm_acc <= 1) ny_pppm_acc = 2;
        if (nz_pppm_acc <= 1) nz_pppm_acc = 2;

        //set local grid dimension
        int npey_fft,npez_fft;
        if (nz_pppm_acc >= nprocs) {
          npey_fft = 1;
          npez_fft = nprocs;
        } else procs2grid2d(nprocs,ny_pppm_acc,nz_pppm_acc,&npey_fft,&npez_fft);

        int me_y = me % npey_fft;
        int me_z = me / npey_fft;

        nxlo_fft_acc = 0;
        nxhi_fft_acc = nx_pppm_acc - 1;
        nylo_fft_acc = me_y*ny_pppm_acc/npey_fft;
        nyhi_fft_acc = (me_y+1)*ny_pppm_acc/npey_fft - 1;
        nzlo_fft_acc = me_z*nz_pppm_acc/npez_fft;
        nzhi_fft_acc = (me_z+1)*nz_pppm_acc/npez_fft - 1;

        double df_kspace = compute_df_kspace_acc();

        count++;

        // break loop if the accuracy has been reached or
        // too many loops have been performed

	break; // grid size fixed based on full grid

        if (df_kspace <= accuracy) break;
        if (count > 500) error->all(FLERR, "Could not compute grid size");
        h *= 0.95;
        h_x = h_y = h_z = h;
      }

    } else {

      double err;
      h_x = h_y = h_z = 1/g_ewald_acc;

      nx_pppm_acc = nx_pppm / 2; // Factor of 2 or could be input parameter
      ny_pppm_acc = ny_pppm / 2;
      nz_pppm_acc = nz_pppm / 2;
    }
  }

  // boost grid size until it is factorable

  while (!factorable(nx_pppm_acc)) nx_pppm_acc++;
  while (!factorable(ny_pppm_acc)) ny_pppm_acc++;
  while (!factorable(nz_pppm_acc)) nz_pppm_acc++;

  if (nx_pppm_acc >= OFFSET || ny_pppm_acc >= OFFSET || nz_pppm_acc >= OFFSET)
    error->all(FLERR,"PPPM grid is too large");
}

/* ----------------------------------------------------------------------
   set local subset of PPPM/FFT grid that I own
   n xyz lo/hi in = 3d brick that I own (inclusive)
   n xyz lo/hi out = 3d brick + ghost cells in 6 directions (inclusive)
   n xyz lo/hi fft = FFT columns that I own (all of x dim, 2d decomp in yz)
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::set_grid_local_acc()
{
  // global indices of PPPM grid range from 0 to N-1
  // nlo_in,nhi_in = lower/upper limits of the 3d sub-brick of
  //   global PPPM grid that I own without ghost cells
  // for slab PPPM, assign z grid as if it were not extended

  nxlo_in_acc = static_cast<int> (comm->xsplit[comm->myloc[0]] * nx_pppm_acc);
  nxhi_in_acc = static_cast<int> (comm->xsplit[comm->myloc[0]+1] * nx_pppm_acc) - 1;

  nylo_in_acc = static_cast<int> (comm->ysplit[comm->myloc[1]] * ny_pppm_acc);
  nyhi_in_acc = static_cast<int> (comm->ysplit[comm->myloc[1]+1] * ny_pppm_acc) - 1;

  nzlo_in_acc = static_cast<int>
      (comm->zsplit[comm->myloc[2]] * nz_pppm_acc/slab_volfactor);
  nzhi_in_acc = static_cast<int>
      (comm->zsplit[comm->myloc[2]+1] * nz_pppm_acc/slab_volfactor) - 1;

  // nlower,nupper = stencil size for mapping particles to PPPM grid

  nlower_acc = -(order_acc-1)/2;
  nupper_acc = order_acc/2;

  // shift values for particle <-> grid mapping
  // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

  if (order_acc % 2) shift_acc = OFFSET + 0.5;
  else shift_acc = OFFSET;
  if (order_acc % 2) shiftone_acc = 0.0;
  else shiftone_acc = 0.5;

  // nlo_out,nhi_out = lower/upper limits of the 3d sub-brick of
  //   global PPPM grid that my particles can contribute charge to
  // effectively nlo_in,nhi_in + ghost cells
  // nlo,nhi = global coords of grid pt to "lower left" of smallest/largest
  //           position a particle in my box can be at
  // dist[3] = particle position bound = subbox + skin/2.0 + qdist
  //   qdist = offset due to TIP4P fictitious charge
  //   convert to triclinic if necessary
  // nlo_out,nhi_out = nlo,nhi + stencil size for particle mapping
  // for slab PPPM, assign z grid as if it were not extended

  double *prd,*sublo,*subhi;

  if (triclinic == 0) {
    prd = domain->prd;
    boxlo = domain->boxlo;
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    prd = domain->prd_lamda;
    boxlo = domain->boxlo_lamda;
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;

  double dist[3];
  double cuthalf = 0.5*neighbor->skin + qdist;
  if (triclinic == 0) dist[0] = dist[1] = dist[2] = cuthalf;
  else {
    dist[0] = cuthalf/domain->prd[0];
    dist[1] = cuthalf/domain->prd[1];
    dist[2] = cuthalf/domain->prd[2];
  }

  int nlo,nhi;

  nlo = static_cast<int> ((sublo[0]-dist[0]-boxlo[0]) * nx_pppm_acc/xprd + shift_acc) - OFFSET;
  nhi = static_cast<int> ((subhi[0]+dist[0]-boxlo[0]) * nx_pppm_acc/xprd + shift_acc) - OFFSET;
  nxlo_out_acc = nlo + nlower_acc;
  nxhi_out_acc = nhi + nupper_acc;
  
  nlo = static_cast<int> ((sublo[1]-dist[1]-boxlo[1]) * ny_pppm_acc/yprd + shift_acc) - OFFSET;
  nhi = static_cast<int> ((subhi[1]+dist[1]-boxlo[1]) * ny_pppm_acc/yprd + shift_acc) - OFFSET;
  nylo_out_acc = nlo + nlower_acc;
  nyhi_out_acc = nhi + nupper_acc;

  nlo = static_cast<int> ((sublo[2]-dist[2]-boxlo[2]) * nz_pppm_acc/zprd_slab + shift_acc) - OFFSET;
  nhi = static_cast<int> ((subhi[2]+dist[2]-boxlo[2]) * nz_pppm_acc/zprd_slab + shift_acc) - OFFSET;
  nzlo_out_acc = nlo + nlower_acc;
  nzhi_out_acc = nhi + nupper_acc;

  // for slab PPPM, change the grid boundary for processors at +z end
  //   to include the empty volume between periodically repeating slabs
  // for slab PPPM, want charge data communicated from -z proc to +z proc,
  //   but not vice versa, also want field data communicated from +z proc to
  //   -z proc, but not vice versa
  // this is accomplished by nzhi_in = nzhi_out on +z end (no ghost cells)
  // also insure no other procs use ghost cells beyond +z limit

  if (slabflag) {
    if (comm->myloc[2] == comm->procgrid[2]-1)
      nzhi_in_acc = nzhi_out_acc = nz_pppm_acc - 1;
    nzhi_out_acc = MIN(nzhi_out_acc,nz_pppm_acc-1);
  }
    
  // decomposition of FFT mesh
  // global indices range from 0 to N-1
  // proc owns entire x-dimension, clumps of columns in y,z dimensions
  // npey_fft,npez_fft = # of procs in y,z dims
  // if nprocs is small enough, proc can own 1 or more entire xy planes,
  //   else proc owns 2d sub-blocks of yz plane
  // me_y,me_z = which proc (0-npe_fft-1) I am in y,z dimensions
  // nlo_fft,nhi_fft = lower/upper limit of the section
  //   of the global FFT mesh that I own

  int npey_fft,npez_fft;
  if (nz_pppm_acc >= nprocs) {
    npey_fft = 1;
    npez_fft = nprocs;
  } else procs2grid2d(nprocs,ny_pppm_acc,nz_pppm_acc,&npey_fft,&npez_fft);

  int me_y = me % npey_fft;
  int me_z = me / npey_fft;

  nxlo_fft_acc = 0;
  nxhi_fft_acc = nx_pppm_acc - 1;
  nylo_fft_acc = me_y*ny_pppm_acc/npey_fft;
  nyhi_fft_acc = (me_y+1)*ny_pppm_acc/npey_fft - 1;
  nzlo_fft_acc = me_z*nz_pppm_acc/npez_fft;
  nzhi_fft_acc = (me_z+1)*nz_pppm_acc/npez_fft - 1;

  // PPPM grid pts owned by this proc, including ghosts

  ngrid_acc = (nxhi_out_acc-nxlo_out_acc+1) * (nyhi_out_acc-nylo_out_acc+1) *
    (nzhi_out_acc-nzlo_out_acc+1);

  // FFT grids owned by this proc, without ghosts
  // nfft = FFT points in FFT decomposition on this proc
  // nfft_brick = FFT points in 3d brick-decomposition on this proc
  // nfft_both = greater of 2 values

  nfft_acc = (nxhi_fft_acc-nxlo_fft_acc+1) * (nyhi_fft_acc-nylo_fft_acc+1) *
    (nzhi_fft_acc-nzlo_fft_acc+1);
  int nfft_brick = (nxhi_in_acc-nxlo_in_acc+1) * (nyhi_in_acc-nylo_in_acc+1) *
    (nzhi_in_acc-nzlo_in_acc+1);
  nfft_both_acc = MAX(nfft_acc,nfft_brick);
}

/* ----------------------------------------------------------------------
   estimate kspace force error for ik method
------------------------------------------------------------------------- */

double EVB_PPPMACCOMP::estimate_ik_error_acc(double h, double prd, bigint natoms)
{
  double sum = 0.0;
  for (int m = 0; m < order_acc; m++)
    sum += acons[order_acc][m] * pow(h*g_ewald_acc,2.0*m);
  double value = q2 * pow(h*g_ewald_acc,(double)order_acc) *
    sqrt(g_ewald_acc*prd*sqrt(MY_2PI)*sum/natoms) / (prd*prd);

  return value;
}

/* ----------------------------------------------------------------------
   compute estimated kspace force error
------------------------------------------------------------------------- */

double EVB_PPPMACCOMP::compute_df_kspace_acc()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  bigint natoms = atom->natoms;
  double df_kspace = 0.0;
  if (differentiation_flag == 1) {
    double qopt = compute_qopt_acc();
    df_kspace = sqrt(qopt/natoms)*q2/(xprd*yprd*zprd_slab);
  } else {
    double lprx = estimate_ik_error_acc(xprd/nx_pppm_acc,xprd,natoms);
    double lpry = estimate_ik_error_acc(yprd/ny_pppm_acc,yprd,natoms);
    double lprz = estimate_ik_error_acc(zprd_slab/nz_pppm_acc,zprd_slab,natoms);
    df_kspace = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
  }
  return df_kspace;
}

/* ----------------------------------------------------------------------
   compute qopt
------------------------------------------------------------------------- */

double EVB_PPPMACCOMP::compute_qopt_acc()
{
  double qopt = 0.0;
  double *prd = (triclinic==0) ? domain->prd : domain->prd_lamda;
  
  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double u1, u2, sqk;
  double sum1,sum2,sum3,sum4,dot2;

  int k,l,m,nx,ny,nz;
  const int twoorder = 2*order_acc;

  for (m = nzlo_fft_acc; m <= nzhi_fft_acc; m++) {
    const int mper = m - nz_pppm_acc*(2*m/nz_pppm_acc);

    for (l = nylo_fft_acc; l <= nyhi_fft_acc; l++) {
      const int lper = l - ny_pppm_acc*(2*l/ny_pppm_acc);

      for (k = nxlo_fft_acc; k <= nxhi_fft_acc; k++) {
        const int kper = k - nx_pppm_acc*(2*k/nx_pppm_acc);

        sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);

        if (sqk != 0.0) {

          sum1 = 0.0;
          sum2 = 0.0;
          sum3 = 0.0;
          sum4 = 0.0;
          for (nx = -2; nx <= 2; nx++) {
            qx = unitkx*(kper+nx_pppm_acc*nx);
            sx = exp(-0.25*square(qx/g_ewald_acc));
            argx = 0.5*qx*xprd/nx_pppm_acc;
            wx = powsinxx(argx,twoorder);
            qx *= qx;

            for (ny = -2; ny <= 2; ny++) {
              qy = unitky*(lper+ny_pppm_acc*ny);
              sy = exp(-0.25*square(qy/g_ewald_acc));
              argy = 0.5*qy*yprd/ny_pppm_acc;
              wy = powsinxx(argy,twoorder);
              qy *= qy;

              for (nz = -2; nz <= 2; nz++) {
                qz = unitkz*(mper+nz_pppm_acc*nz);
                sz = exp(-0.25*square(qz/g_ewald_acc));
                argz = 0.5*qz*zprd_slab/nz_pppm_acc;
                wz = powsinxx(argz,twoorder);
                qz *= qz;

                dot2 = qx+qy+qz;
                u1   = sx*sy*sz;
                u2   = wx*wy*wz;
                sum1 += u1*u1/dot2*MY_4PI*MY_4PI;
                sum2 += u1 * u2 * MY_4PI;
                sum3 += u2;
                sum4 += dot2*u2;
              }
            }
          }
          sum2 *= sum2;
          qopt += sum1 - sum2/(sum3*sum4);
        }
      }
    }
  }
  double qopt_all;
  MPI_Allreduce(&qopt,&qopt_all,1,MPI_DOUBLE,MPI_SUM,world);
  return qopt_all;
}

/* ----------------------------------------------------------------------
   adjust the g_ewald parameter to near its optimal value
   using a Newton-Raphson solver
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::adjust_gewald_acc()
{
  double dx;

  for (int i = 0; i < LARGE; i++) {
    dx = newton_raphson_f_acc() / derivf_acc();
    g_ewald_acc -= dx;
    if (fabs(newton_raphson_f_acc()) < SMALL) return;
  }

  char str[128];
  sprintf(str, "Could not compute g_ewald");
  error->all(FLERR, str);
}

/* ----------------------------------------------------------------------
 Calculate f(x) using Newton-Raphson solver
 ------------------------------------------------------------------------- */

double EVB_PPPMACCOMP::newton_raphson_f_acc()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  bigint natoms = atom->natoms;

  double df_rspace = 2.0*q2*exp(-g_ewald_acc*g_ewald_acc*cutoff*cutoff) /
       sqrt(natoms*cutoff*xprd*yprd*zprd);

  double df_kspace = compute_df_kspace_acc();

  return df_rspace - df_kspace;
}

/* ----------------------------------------------------------------------
 Calculate numerical derivative f'(x) using forward difference
 [f(x + h) - f(x)] / h
 ------------------------------------------------------------------------- */

double EVB_PPPMACCOMP::derivf_acc()
{
  double h = 0.000001;  //Derivative step-size
  double df,f1,f2,g_ewald_old;

  f1 = newton_raphson_f_acc();
  g_ewald_old = g_ewald_acc;
  g_ewald_acc += h;
  f2 = newton_raphson_f_acc();
  g_ewald_acc = g_ewald_old;
  df = (f2 - f1)/h;

  return df;
}

/* ----------------------------------------------------------------------
   Calculate the final estimate of the accuracy
------------------------------------------------------------------------- */

double EVB_PPPMACCOMP::final_accuracy_acc()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  bigint natoms = atom->natoms;

  double df_kspace = compute_df_kspace_acc();
  double q2_over_sqrt = q2 / sqrt(natoms*cutoff*xprd*yprd*zprd_slab);
  double df_rspace = 2.0 * q2_over_sqrt * exp(-g_ewald_acc*g_ewald_acc*cutoff*cutoff);
  double df_table = estimate_table_accuracy(q2_over_sqrt,df_rspace);
  double estimated_accuracy = sqrt(df_kspace*df_kspace + df_rspace*df_rspace +
   df_table*df_table);

  return estimated_accuracy;
}

/* ----------------------------------------------------------------------
   pre-compute Green's function denominator expansion coeffs, Gamma(2n) 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::compute_gf_denom_acc()
{
  int k,l,m;
  
  for (l = 1; l < order_acc; l++) gf_b_acc[l] = 0.0;
  gf_b_acc[0] = 1.0;
  
  for (m = 1; m < order_acc; m++) {
    for (l = m; l > 0; l--) 
      gf_b_acc[l] = 4.0 * (gf_b_acc[l]*(l-m)*(l-m-0.5)-gf_b_acc[l-1]*(l-m-1)*(l-m-1));
    gf_b_acc[0] = 4.0 * (gf_b_acc[0]*(l-m)*(l-m-0.5));
  }

  int ifact = 1;
  for (k = 1; k < 2*order_acc; k++) ifact *= k;
  double gaminv = 1.0/ifact;
  for (l = 0; l < order_acc; l++) gf_b_acc[l] *= gaminv;
}

/* ----------------------------------------------------------------------
   denominator for Hockney-Eastwood Green's function
------------------------------------------------------------------------- */

double EVB_PPPMACCOMP::gf_denom_acc(double x, double y, double z)
{
  double sx,sy,sz;
  sz = sy = sx = 0.0;
  for (int l = order_acc-1; l >= 0; l--) {
    sx = gf_b_acc[l] + sx*x;
    sy = gf_b_acc[l] + sy*y;
    sz = gf_b_acc[l] + sz*z;
  }
  double s = sx*sy*sz;
  return s*s;
}

/* ----------------------------------------------------------------------
   generate coeffients for the weight function of order n
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::compute_rho_coeff_acc()
{
  int j,k,l,m;
  FFT_SCALAR s;

  FFT_SCALAR **a;
  memory->create2d_offset(a,order_acc,-order_acc,order_acc,"EVB_PPPM:a");

  for (k = -order_acc; k <= order_acc; k++) 
    for (l = 0; l < order_acc; l++)
      a[l][k] = 0.0;
        
  a[0][0] = 1.0;
  for (j = 1; j < order_acc; j++) {
    for (k = -j; k <= j; k += 2) {
      s = 0.0;
      for (l = 0; l < j; l++) {
	a[l+1][k] = (a[l][k+1]-a[l][k-1]) / (l+1);
#ifdef FFT_SINGLE
	s += powf(0.5,(float) l+1) *
	  (a[l][k-1] + powf(-1.0,(float) l) * a[l][k+1]) / (l+1);
#else
	s += pow(0.5,(double) l+1) * 
	  (a[l][k-1] + pow(-1.0,(double) l) * a[l][k+1]) / (l+1);
#endif
      }
      a[0][k] = s;
    }
  }

  m = (1-order_acc)/2;
  for (k = -(order_acc-1); k < order_acc; k += 2) {
    for (l = 0; l < order_acc; l++)
      rho_coeff_acc[l][m] = a[l][k];
    for (l = 1; l < order_acc; l++)
      drho_coeff_acc[l-1][m] = l*a[l][k];
    m++;
  }

  memory->destroy2d_offset(a,-order_acc);
}

/* ----------------------------------------------------------------------
   ghost-swap to accumulate full density in brick decomposition 
   remap density from 3d brick decomposition to FFT decomposition
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::brick2fft_acc()
{
  int n,ix,iy,iz;

  // remap from 3d brick decomposition to FFT decomposition
  // copy grabs inner portion of density from 3d brick
  // remap could be done as pre-stage of FFT,
  //   but this works optimally on only double values, not complex values

  n = 0;
  for (iz = nzlo_in_acc; iz <= nzhi_in_acc; iz++)
    for (iy = nylo_in_acc; iy <= nyhi_in_acc; iy++)
      for (ix = nxlo_in_acc; ix <= nxhi_in_acc; ix++)
	density_fft_acc[n++] = density_brick_acc[iz][iy][ix];

  remap_acc->perform(density_fft_acc,density_fft_acc,work1_acc);
}

/*************************************************************************/

void EVB_PPPMACCOMP::clear_density_acc()
{
  FFT_SCALAR *vec = &density_brick_acc[nzlo_out_acc][nylo_out_acc][nxlo_out_acc];
  memset(vec, ZEROF, sizeof(FFT_SCALAR)*ngrid_acc);
}

/*************************************************************************/

void EVB_PPPMACCOMP::load_env_density_acc()
{
  memcpy(&density_brick_acc[nzlo_out_acc][nylo_out_acc][nxlo_out_acc],
         &env_density_brick_acc[nzlo_out_acc][nylo_out_acc][nxlo_out_acc],
		 sizeof(FFT_SCALAR)*ngrid_acc);
}

/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::make_rho_acc()
{
  const double * const q = atom->q;
  const double * const * const x = atom->x;
  const int nthreads = comm->nthreads;
  const int nlocal = atom->nlocal;

#if defined(_OPENMP)
#pragma omp parallel default(none) 
#endif
  {  
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int tid = omp_get_thread_num();
    const int inum = nlocal;
    const int idelta = 1 + inum/nthreads;
    const int ifrom = tid*idelta;
    const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
#else
    const int tid = 0;
    const int ifrom = 0;
    const int ito = nlocal;
#endif

    // set up clear 3d density array
    const int nzoffs = (nzhi_out_acc-nzlo_out_acc+1)*tid;
    FFT_SCALAR * const * const * const db_acc = &(density_brick_acc[nzoffs]);
    memset(&(db_acc[nzlo_out_acc][nylo_out_acc][nxlo_out_acc]),0,ngrid_acc*sizeof(FFT_SCALAR));

    const double boxlox = boxlo[0];
    const double boxloy = boxlo[1];
    const double boxloz = boxlo[2];

    ThrData *thr = fix->get_thr(tid);
    FFT_SCALAR * const * const r1d = static_cast<FFT_SCALAR **>(thr->get_rho1d());

    // loop over my charges, add their contribution to nearby grid points
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // (dx,dy,dz) = distance to "lower left" grid pt
    // (mx,my,mz) = global coords of moving stencil pt
    
    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) { 
     if (order_acc == 5) {
      for (int i = ifrom; i < ito; i++) {

        const double ddx = (x[i][0] - boxlox) * delxinv_acc;
        const double ddy = (x[i][1] - boxloy) * delyinv_acc;
        const double ddz = (x[i][2] - boxloz) * delzinv_acc;
        const int nx = static_cast<int> (ddx+shift_acc) - OFFSET;
        const int ny = static_cast<int> (ddy+shift_acc) - OFFSET;
        const int nz = static_cast<int> (ddz+shift_acc) - OFFSET;
        part2grid_acc[i][0] = nx;
        part2grid_acc[i][1] = ny;
        part2grid_acc[i][2] = nz;
	const FFT_SCALAR dx = nx+shiftone_acc - ddx;
	const FFT_SCALAR dy = ny+shiftone_acc - ddy;
	const FFT_SCALAR dz = nz+shiftone_acc - ddz; 
        part2grid_dr_acc[i][0] = dx;
        part2grid_dr_acc[i][1] = dy;
        part2grid_dr_acc[i][2] = dz; 

        // Code specific to order = 5
	// compute_rho1d_thr(r1d,dx,dy,dz);
	// completely unrolled loop
        const FFT_SCALAR dx2 = dx*dx;
        const FFT_SCALAR dx3 = dx2*dx;
        const FFT_SCALAR dx4 = dx2*dx2;
        const FFT_SCALAR dy2 = dy*dy;
        const FFT_SCALAR dy3 = dy2*dy;
        const FFT_SCALAR dy4 = dy2*dy2;
        const FFT_SCALAR dz2 = dz*dz;
        const FFT_SCALAR dz3 = dz2*dz;
        const FFT_SCALAR dz4 = dz2*dz2;
	int k = -2;
        r1d[0][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dx + rho_coeff_acc[2][k]*dx2 + rho_coeff_acc[3][k]*dx3 + rho_coeff_acc[4][k]*dx4;
        r1d[1][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dy + rho_coeff_acc[2][k]*dy2 + rho_coeff_acc[3][k]*dy3 + rho_coeff_acc[4][k]*dy4;
        r1d[2][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dz + rho_coeff_acc[2][k]*dz2 + rho_coeff_acc[3][k]*dz3 + rho_coeff_acc[4][k]*dz4;
        k = -1;
        r1d[0][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dx + rho_coeff_acc[2][k]*dx2 + rho_coeff_acc[3][k]*dx3 + rho_coeff_acc[4][k]*dx4;
        r1d[1][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dy + rho_coeff_acc[2][k]*dy2 + rho_coeff_acc[3][k]*dy3 + rho_coeff_acc[4][k]*dy4;
        r1d[2][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dz + rho_coeff_acc[2][k]*dz2 + rho_coeff_acc[3][k]*dz3 + rho_coeff_acc[4][k]*dz4;
        k = 0;
        r1d[0][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dx + rho_coeff_acc[2][k]*dx2 + rho_coeff_acc[3][k]*dx3 + rho_coeff_acc[4][k]*dx4;
        r1d[1][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dy + rho_coeff_acc[2][k]*dy2 + rho_coeff_acc[3][k]*dy3 + rho_coeff_acc[4][k]*dy4;
        r1d[2][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dz + rho_coeff_acc[2][k]*dz2 + rho_coeff_acc[3][k]*dz3 + rho_coeff_acc[4][k]*dz4;
        k = 1;
        r1d[0][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dx + rho_coeff_acc[2][k]*dx2 + rho_coeff_acc[3][k]*dx3 + rho_coeff_acc[4][k]*dx4;
        r1d[1][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dy + rho_coeff_acc[2][k]*dy2 + rho_coeff_acc[3][k]*dy3 + rho_coeff_acc[4][k]*dy4;
        r1d[2][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dz + rho_coeff_acc[2][k]*dz2 + rho_coeff_acc[3][k]*dz3 + rho_coeff_acc[4][k]*dz4;
        k = 2;
        r1d[0][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dx + rho_coeff_acc[2][k]*dx2 + rho_coeff_acc[3][k]*dx3 + rho_coeff_acc[4][k]*dx4;
        r1d[1][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dy + rho_coeff_acc[2][k]*dy2 + rho_coeff_acc[3][k]*dy3 + rho_coeff_acc[4][k]*dy4;
        r1d[2][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dz + rho_coeff_acc[2][k]*dz2 + rho_coeff_acc[3][k]*dz3 + rho_coeff_acc[4][k]*dz4;

	const FFT_SCALAR z0 = delvolinv_acc * q[i];
	for (int n = nlower_acc; n <= nupper_acc; n++) {
	  const FFT_SCALAR y0 = z0*r1d[2][n];
	  for (int m = nlower_acc; m <= nupper_acc; m++) {
	    const FFT_SCALAR x0 = y0*r1d[1][m];
	    for (int l = nlower_acc; l <= nupper_acc; l++) {
	      db_acc[n+nz][m+ny][l+nx] += x0*r1d[0][l];
	    }
	  }
	}
      }
     } else {
      for (int i = ifrom; i < ito; i++) {

        const double ddx = (x[i][0] - boxlox) * delxinv_acc;
        const double ddy = (x[i][1] - boxloy) * delyinv_acc;
        const double ddz = (x[i][2] - boxloz) * delzinv_acc;
        const int nx = static_cast<int> (ddx+shift_acc) - OFFSET;
        const int ny = static_cast<int> (ddy+shift_acc) - OFFSET;
        const int nz = static_cast<int> (ddz+shift_acc) - OFFSET;
        part2grid_acc[i][0] = nx;
        part2grid_acc[i][1] = ny;
        part2grid_acc[i][2] = nz;
	const FFT_SCALAR dx = nx+shiftone_acc - ddx;
	const FFT_SCALAR dy = ny+shiftone_acc - ddy;
	const FFT_SCALAR dz = nz+shiftone_acc - ddz; 
        part2grid_dr_acc[i][0] = dx;
        part2grid_dr_acc[i][1] = dy;
        part2grid_dr_acc[i][2] = dz; 

        // General order code 
	compute_rho1d_thr_acc(r1d,dx,dy,dz);

	const FFT_SCALAR z0 = delvolinv_acc * q[i];
	for (int n = nlower_acc; n <= nupper_acc; n++) {
	  const FFT_SCALAR y0 = z0*r1d[2][n];
	  for (int m = nlower_acc; m <= nupper_acc; m++) {
	    const FFT_SCALAR x0 = y0*r1d[1][m];
	    for (int l = nlower_acc; l <= nupper_acc; l++) {
	      db_acc[n+nz][m+ny][l+nx] += x0*r1d[0][l];
	    }
	  }
	}
      }
     }
    }

#if defined(_OPENMP)
    // reduce 3d density array
    if (nthreads > 1) {
      data_reduce_fft(&(density_brick_acc[nzlo_out_acc][nylo_out_acc][nxlo_out_acc]),ngrid_acc,nthreads,1,tid);
    }
#endif
  }
}

/*************************************************************************/

void EVB_PPPMACCOMP::map2density_one_subtract_acc(int id)
{
  double *q = atom->q;

  // Subtracts the contrinution for a given id
  const int nx = part2grid_acc[id][0];
  const int ny = part2grid_acc[id][1];
  const int nz = part2grid_acc[id][2];
  
  // (dx,dy,dz) = distance to "lower left" grid pt
  
  compute_rho1d_acc(part2grid_dr_acc[id][0], part2grid_dr_acc[id][1], part2grid_dr_acc[id][2]);
  
  // (mx,my,mz) = global coords of moving stencil pt
  
  const FFT_SCALAR z0 = delvolinv_acc * q[id];
  for (int n = nlower_acc; n <= nupper_acc; n++) {
    const FFT_SCALAR y0 = z0*rho1d_acc[2][n];
    for (int m = nlower_acc; m <= nupper_acc; m++) {
      const FFT_SCALAR x0 = y0*rho1d_acc[1][m];
      for (int l = nlower_acc; l <= nupper_acc; l++) {
	density_brick_acc[n+nz][m+ny][l+nx] -= x0*rho1d_acc[0][l];
      } // Loop mx
    } // Loop my
  } // Loop mz
}

/*************************************************************************/

void EVB_PPPMACCOMP::map2density_one_acc(int id)
{	
  nx = part2grid_acc[id][0];
  ny = part2grid_acc[id][1];
  nz = part2grid_acc[id][2];
  
  // (dx,dy,dz) = distance to "lower left" grid pt
  
  compute_rho1d_acc(part2grid_dr_acc[id][0], part2grid_dr_acc[id][1], part2grid_dr_acc[id][2]);
  
  // (mx,my,mz) = global coords of moving stencil pt
  
  const FFT_SCALAR z0 = delvolinv_acc * q[id];
  for (int n = nlower_acc; n <= nupper_acc; n++) {
    const FFT_SCALAR y0 = z0*rho1d_acc[2][n];
    for (int m = nlower_acc; m <= nupper_acc; m++) {
      const FFT_SCALAR x0 = y0*rho1d_acc[1][m];
      for (int l = nlower_acc; l <= nupper_acc; l++) {
	density_brick_acc[n+nz][m+ny][l+nx] += x0*rho1d_acc[0][l];
      } // Loop mx
    } // Loop my
  } // Loop mz
  
}

/*************************************************************************/

/* ----------------------------------------------------------------------
   charge assignment into rho1d
   dx,dy,dz = distance of particle from "lower left" grid point 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::compute_rho1d_acc(const FFT_SCALAR &dx, const FFT_SCALAR &dy, const FFT_SCALAR &dz)
{
  if (order_acc == 5) {
    // order = 5 case, completely unrolled loops
    const double dx2 = dx*dx;
    const double dx3 = dx2*dx;
    const double dx4 = dx2*dx2;
    const double dy2 = dy*dy;
    const double dy3 = dy2*dy;
    const double dy4 = dy2*dy2;
    const double dz2 = dz*dz;
    const double dz3 = dz2*dz;
    const double dz4 = dz2*dz2;
    int k = -2;
    rho1d_acc[0][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dx + rho_coeff_acc[2][k]*dx2 + rho_coeff_acc[3][k]*dx3 + rho_coeff_acc[4][k]*dx4;
    rho1d_acc[1][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dy + rho_coeff_acc[2][k]*dy2 + rho_coeff_acc[3][k]*dy3 + rho_coeff_acc[4][k]*dy4;
    rho1d_acc[2][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dz + rho_coeff_acc[2][k]*dz2 + rho_coeff_acc[3][k]*dz3 + rho_coeff_acc[4][k]*dz4;
    k = -1;
    rho1d_acc[0][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dx + rho_coeff_acc[2][k]*dx2 + rho_coeff_acc[3][k]*dx3 + rho_coeff_acc[4][k]*dx4;
    rho1d_acc[1][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dy + rho_coeff_acc[2][k]*dy2 + rho_coeff_acc[3][k]*dy3 + rho_coeff_acc[4][k]*dy4;
    rho1d_acc[2][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dz + rho_coeff_acc[2][k]*dz2 + rho_coeff_acc[3][k]*dz3 + rho_coeff_acc[4][k]*dz4;
    k = 0;
    rho1d_acc[0][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dx + rho_coeff_acc[2][k]*dx2 + rho_coeff_acc[3][k]*dx3 + rho_coeff_acc[4][k]*dx4;
    rho1d_acc[1][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dy + rho_coeff_acc[2][k]*dy2 + rho_coeff_acc[3][k]*dy3 + rho_coeff_acc[4][k]*dy4;
    rho1d_acc[2][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dz + rho_coeff_acc[2][k]*dz2 + rho_coeff_acc[3][k]*dz3 + rho_coeff_acc[4][k]*dz4;
    k = 1;
    rho1d_acc[0][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dx + rho_coeff_acc[2][k]*dx2 + rho_coeff_acc[3][k]*dx3 + rho_coeff_acc[4][k]*dx4;
    rho1d_acc[1][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dy + rho_coeff_acc[2][k]*dy2 + rho_coeff_acc[3][k]*dy3 + rho_coeff_acc[4][k]*dy4;
    rho1d_acc[2][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dz + rho_coeff_acc[2][k]*dz2 + rho_coeff_acc[3][k]*dz3 + rho_coeff_acc[4][k]*dz4;
    k = 2;
    rho1d_acc[0][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dx + rho_coeff_acc[2][k]*dx2 + rho_coeff_acc[3][k]*dx3 + rho_coeff_acc[4][k]*dx4;
    rho1d_acc[1][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dy + rho_coeff_acc[2][k]*dy2 + rho_coeff_acc[3][k]*dy3 + rho_coeff_acc[4][k]*dy4;
    rho1d_acc[2][k] = rho_coeff_acc[0][k] + rho_coeff_acc[1][k]*dz + rho_coeff_acc[2][k]*dz2 + rho_coeff_acc[3][k]*dz3 + rho_coeff_acc[4][k]*dz4;
  } else {
    // general case
    FFT_SCALAR r[3];
    for (int k = (1-order_acc)/2; k <= order_acc/2; ++k) {
      r[0] = r[1] = r[2] = ZEROF;
      for (int l = order_acc-1; l >= 0; --l) {
        r[0] = rho_coeff_acc[l][k] + r[0]*dx;
        r[1] = rho_coeff_acc[l][k] + r[1]*dy;
        r[2] = rho_coeff_acc[l][k] + r[2]*dz;
      }
      rho1d_acc[0][k] = r[0];
      rho1d_acc[1][k] = r[1];
      rho1d_acc[2][k] = r[2];
    }
  }
}

inline void EVB_PPPMACCOMP::compute_rho1d_thr_acc(FFT_SCALAR * const * const r1d, const FFT_SCALAR &dx,
                                           const FFT_SCALAR &dy, const FFT_SCALAR &dz)
{
  FFT_SCALAR r[3];
  for (int k = (1-order_acc)/2; k <= order_acc/2; ++k) {
    r[0] = r[1] = r[2] = ZEROF;
    for (int l = order_acc-1; l >= 0; --l) {
      r[0] = rho_coeff_acc[l][k] + r[0]*dx;
      r[1] = rho_coeff_acc[l][k] + r[1]*dy;
      r[2] = rho_coeff_acc[l][k] + r[2]*dz;
    }
    r1d[0][k] = r[0];
    r1d[1][k] = r[1];
    r1d[2][k] = r[2];
  }
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver 
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::poisson_energy_acc(int vflag)
{
  int n;
  double eng;

  // transform charge density (r -> k) 

  n = 0;
  for (int i=0; i<nfft_acc; i++) {
    work1_acc[n++] = density_fft_acc[i];
    work1_acc[n++] = ZEROF;
  }
  
  fft1_acc->compute(work1_acc,work1_acc,1);
  
  // if requested, compute energy and virial contribution

  double scaleinv = 1.0/(nx_pppm_acc*ny_pppm_acc*nz_pppm_acc);
  double s2 = scaleinv*scaleinv;

  n = 0;
  if (vflag) {
    for (int i=0; i<nfft_acc; ++i) {
      eng = s2 * greensfn_acc[i] * (work1_acc[n]*work1_acc[n] + work1_acc[n+1]*work1_acc[n+1]);
      for (int j=0; j<6; ++j) virial[j] += eng*vg_acc[i][j];
      energy += eng;
      n += 2;
    }
  } else {
    for (int i=0; i<nfft_acc; ++i) {
      eng = greensfn_acc[i] * (work1_acc[n]*work1_acc[n] + work1_acc[n+1]*work1_acc[n+1]);
      energy += eng;
      n += 2;
    }
    energy *= s2;
  }
}

/*************************************************************************/
/*************************************************************************/
/*************************************************************************/

void EVB_PPPMACCOMP::compute_env(int vflag)
{
  nlocal = atom->nlocal;
  
  q = atom->q;
  x = atom->x;
  f = atom->f;
  
  int* is_cplx_atom = evb_engine->complex_atom;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int has_cplx_atom = evb_engine->has_complex_atom;

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;

  // Calculate ENV density for coarse grid
  FFT_SCALAR ***save_density = density_brick_acc;
  density_brick_acc = env_density_brick_acc;
  clear_density_acc();
  
  make_rho_acc();
  if(has_cplx_atom) for(int i=0; i<nlocal_cplx; i++) map2density_one_subtract_acc(cplx_list[i]);  
  
  density_brick_acc = save_density;
  
  // Calculate the ENV density full grid;
  save_density = density_brick;
  density_brick = env_density_brick;
  clear_density();
  
  make_rho();
  if(has_cplx_atom) for(int i=0; i<nlocal_cplx; i++) map2density_one_subtract(cplx_list[i]);
  
  density_brick = save_density;
  
  load_env_density();

  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();
  
  poisson_energy(vflag);
  
  qsqsum = evb_engine->qsqsum_env = evb_engine->qsqsum_sys-evb_engine->evb_complex->qsqsum;
  reduce_ev(vflag,true);
  
  env_energy = energy;
  if (vflag) for (int i = 0; i < 6; i++) virial[i] = 0.0;

  // Environment contribution to dipole for slab correction
  if(slabflag) {
    double *q = atom->q;
    double **x = atom->x;
    
    double dipole = 0.0;
    for(int i=0; i<atom->nlocal; i++) dipole += q[i] * x[i][2];
    
    MPI_Allreduce(&dipole,&dipole_env,1,MPI_DOUBLE,MPI_SUM,world);
  }
}

/*************************************************************************/

void EVB_PPPMACCOMP::compute_cplx(int vflag)
{
  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
  load_env_density_acc();

  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  for(int i=0; i<nlocal_cplx; i++) map2density_one_acc(cplx_list[i]);
  
  reverse_comm_acc(REVERSE_RHO);
  brick2fft_acc();

  poisson_energy_acc(vflag);
  
  qsqsum = evb_engine->qsqsum_env + evb_engine->evb_complex->qsqsum;
  reduce_ev(vflag,true);

  if(slabflag) slabcorr_cplx();

  energy -= env_energy;
}

/*************************************************************************/

void EVB_PPPMACCOMP::compute_exch(int vflag)
{
  energy = 0.0;
  if(slabflag) slabcorr_exch();
}

/*************************************************************************/

void EVB_PPPMACCOMP::compute_eff(int vflag)
{
  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
  // load full-grid density for all atoms

  load_env_density();
  
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;

  for(int i=0; i<nlocal_cplx; i++) map2density_one(cplx_list[i]);
  
  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();
  
  poisson(true,vflag);

  if (differentiation_flag == 1) cg->forward_comm(this,FORWARD_AD);
  else cg->forward_comm(this,FORWARD_IK);
  qscale = force->qqrd2e;
  fieldforce();
}

/*************************************************************************/

/* ----------------------------------------------------------------------
   use swap list in reverse order to compute fully summed value
   for each owned grid pt that some other proc has copy of as a ghost grid pt
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::reverse_comm_acc(int which)
{
  int i,n;

  int nswap = cg_acc->nswap;
  for (int m = nswap-1; m >= 0; m--) {
    if (cg_acc->swap[m].recvproc == me) 
      pack_reverse_acc(which,cg_buf2_acc,cg_acc->swap[m].nunpack,cg_acc->swap[m].unpacklist);
    else
      pack_reverse_acc(which,cg_buf1_acc,cg_acc->swap[m].nunpack,cg_acc->swap[m].unpacklist);

    if (cg_acc->swap[m].recvproc != me) {
      MPI_Irecv(cg_buf2_acc,(cg_acc->nreverse)*(cg_acc->swap[m].npack),MPI_FFT_SCALAR,
                cg_acc->swap[m].sendproc,0,cg_acc->gridcomm,&request);
      MPI_Send(cg_buf1_acc,(cg_acc->nreverse)*(cg_acc->swap[m].nunpack),MPI_FFT_SCALAR,
               cg_acc->swap[m].recvproc,0,cg_acc->gridcomm);
      MPI_Wait(&request,&status);
    }
    
    unpack_reverse_acc(which,cg_buf2_acc,cg_acc->swap[m].npack,cg_acc->swap[m].packlist);
  }
}

/* ----------------------------------------------------------------------
   pack ghost values into buf to send to another proc
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::pack_reverse_acc(int flag, FFT_SCALAR *buf, int nlist, int *list)
{
  if (flag == REVERSE_RHO) {
    FFT_SCALAR *src = &density_brick_acc[nzlo_out_acc][nylo_out_acc][nxlo_out_acc];
    for (int i = 0; i < nlist; i++)
      buf[i] = src[list[i]];
  }
}

/* ----------------------------------------------------------------------
   unpack another proc's ghost values from buf and add to own values
------------------------------------------------------------------------- */

void EVB_PPPMACCOMP::unpack_reverse_acc(int flag, FFT_SCALAR *buf, int nlist, int *list)
{
  if (flag == REVERSE_RHO) {
    FFT_SCALAR *dest = &density_brick_acc[nzlo_out_acc][nylo_out_acc][nxlo_out_acc];
    for (int i = 0; i < nlist; i++)
      dest[list[i]] += buf[i];
  } 
}

#endif
