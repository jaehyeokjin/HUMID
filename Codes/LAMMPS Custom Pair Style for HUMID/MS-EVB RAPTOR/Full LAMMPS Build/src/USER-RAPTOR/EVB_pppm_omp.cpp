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
   Contributing author: Adrian Lange (ALCF)
------------------------------------------------------------------------- */

#if defined (_OPENMP)

#include "atom.h"
#include "comm.h"
#define _CRACKER_GRIDCOMM
#include "EVB_cracker.h"
#undef _CRACKER_GRIDCOMM
#include "domain.h"
#include "force.h"
#include "memory.h"
#include "math_const.h"
#include "math_special.h"
#include "remap_wrap.h"

#include "EVB_pppm_omp.h"
#include "EVB_effpair.h"
#include "EVB_timer.h"

#include <string.h>
#include <math.h>

#include "suffix.h"
using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif

#define EPS_HOC 1.0e-7
#define OFFSET 4096

enum{REVERSE_RHO};
enum{FORWARD_IK,FORWARD_AD,FORWARD_IK_PERATOM,FORWARD_AD_PERATOM};

#define Q_ATOM 0
#define Q_EFFECTIVE 1

#define KSPACE_DEFAULT    0 // Hellman-Feynman forces for Ewald
#define PPPM_HF_FORCES    1 // Hellman-Feynman forces for PPPM
#define PPPM_ACC_FORCES   2 // Approximate (acc) forces for PPPM. 
#define PPPM_POLAR_FORCES 3 // ACC forces plus an additional polarization force on complex atoms for PPPM.

#if defined(_OPENMP)
#include <omp.h>
#endif

/* ---------------------------------------------------------------------- */

EVB_PPPMOMP::EVB_PPPMOMP(LAMMPS *lmp, int narg, char **arg) :
  EVB_PPPM(lmp, narg, arg), ThrOMP(lmp, THR_KSPACE)
{
  suffix_flag |= Suffix::OMP;

  // *** Extra density stuff *** //
  density_brick1 = density_brick2 = density_brick3 = NULL;
  density_fft1 = density_fft2 = density_fft3 = NULL;
  work11 = work12 = work13 = work22 = work23 = NULL;

  vdx_brick2 = vdy_brick2 = vdz_brick2 = NULL;
  vdx_brick3 = vdy_brick3 = vdz_brick3 = NULL;
  u_brick2 = u_brick3 = NULL;

  // *** Extra FFT data *** //
  fft1_2 = fft2_2 = NULL;
  fft1_3 = fft2_3 = NULL;
  remap_2 = remap_3 = NULL;

  // GridComm
  cg_buf1 = cg_buf2 = NULL;
}

/* ----------------------------------------------------------------------
   free all memory 
------------------------------------------------------------------------- */

EVB_PPPMOMP::~EVB_PPPMOMP()
{
  deallocate_omp();
}


/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors and order 
------------------------------------------------------------------------- */

void EVB_PPPMOMP::allocate()
{
  EVB_PPPM::allocate();

  // ** Allocate extra FFT data ** //

  int tmp;

  fft1_2 = new FFT3d(lmp_pointer,world,nx_pppm,ny_pppm,nz_pppm,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
		     0,0,&tmp,collective_flag);

  fft2_2 = new FFT3d(lmp_pointer,world,nx_pppm,ny_pppm,nz_pppm,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
		     0,0,&tmp,collective_flag);

  remap_2 = new Remap(lmp_pointer,world,
                    nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                    nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
		      1,0,0,FFT_PRECISION,collective_flag);

  fft1_3 = new FFT3d(lmp_pointer,world,nx_pppm,ny_pppm,nz_pppm,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
		     0,0,&tmp,collective_flag);

  fft2_3 = new FFT3d(lmp_pointer,world,nx_pppm,ny_pppm,nz_pppm,
                   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                   nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
		     0,0,&tmp,collective_flag);

  remap_3 = new Remap(lmp_pointer,world,
                    nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                    nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
		      1,0,0,FFT_PRECISION,collective_flag);

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
  /*******************************************************/
  /*******************************************************/

  // *** Extra density stuff *** //
  // larger buffers
  memory->create3d_offset(density_brick1,nzlo_out,nzend,nylo_out,nyhi_out,
			  nxlo_out,nxhi_out,"evb_pppm_omp:density_brick1");
  memory->create3d_offset(density_brick2,nzlo_out,nzend,nylo_out,nyhi_out,
			  nxlo_out,nxhi_out,"evb_pppm_omp:density_brick2");
  memory->create3d_offset(density_brick3,nzlo_out,nzend,nylo_out,nyhi_out,
			  nxlo_out,nxhi_out,"evb_pppm_omp:density_brick3");

  density_fft1 = (FFT_SCALAR *) memory->smalloc(nfft_both*sizeof(FFT_SCALAR),"EVB_PPPMOMP:density_fft1");
  density_fft2 = (FFT_SCALAR *) memory->smalloc(nfft_both*sizeof(FFT_SCALAR),"EVB_PPPMOMP:density_fft2");
  density_fft3 = (FFT_SCALAR *) memory->smalloc(nfft_both*sizeof(FFT_SCALAR),"EVB_PPPMOMP:density_fft3");

  work11 = (FFT_SCALAR *) memory->smalloc(2*nfft_both*sizeof(FFT_SCALAR),"EVB_PPPMOMP:work11");
  work12 = (FFT_SCALAR *) memory->smalloc(2*nfft_both*sizeof(FFT_SCALAR),"EVB_PPPMOMP:work12");
  work13 = (FFT_SCALAR *) memory->smalloc(2*nfft_both*sizeof(FFT_SCALAR),"EVB_PPPMOMP:work13");
  work22 = (FFT_SCALAR *) memory->smalloc(2*nfft_both*sizeof(FFT_SCALAR),"EVB_PPPMOMP:work22");
  work23 = (FFT_SCALAR *) memory->smalloc(2*nfft_both*sizeof(FFT_SCALAR),"EVB_PPPMOMP:work23");

  if(differentiation_flag==1) {
    memory->create3d_offset(u_brick2,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"EVB_PPPMOMP:u_brick2");
    memory->create3d_offset(u_brick3,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"EVB_PPPMOMP:u_brick3");
  } else {
    memory->create3d_offset(vdx_brick2,nzlo_out,nzhi_out,nylo_out,nyhi_out,
			    nxlo_out,nxhi_out,"EVB_PPPMOMP:vdx_brick2");
    memory->create3d_offset(vdy_brick2,nzlo_out,nzhi_out,nylo_out,nyhi_out,
			    nxlo_out,nxhi_out,"EVB_PPPMOMP:vdy_brick2");
    memory->create3d_offset(vdz_brick2,nzlo_out,nzhi_out,nylo_out,nyhi_out,
			    nxlo_out,nxhi_out,"EVB_PPPMOMP:vdz_brick2");
    memory->create3d_offset(vdx_brick3,nzlo_out,nzhi_out,nylo_out,nyhi_out,
			    nxlo_out,nxhi_out,"EVB_PPPMOMP:vdx_brick3");
    memory->create3d_offset(vdy_brick3,nzlo_out,nzhi_out,nylo_out,nyhi_out,
			    nxlo_out,nxhi_out,"EVB_PPPMOMP:vdy_brick3");
    memory->create3d_offset(vdz_brick3,nzlo_out,nzhi_out,nylo_out,nyhi_out,
			    nxlo_out,nxhi_out,"EVB_PPPMOMP:vdz_brick3");
  }
}

/* ----------------------------------------------------------------------
   free memory that depends on # of K-vectors and order 
------------------------------------------------------------------------- */

void EVB_PPPMOMP::deallocate()
{
  EVB_PPPM::deallocate();
  deallocate_omp();
}

/* ----------------------------------------------------------------------
   free memory that depends on # of K-vectors and order 
------------------------------------------------------------------------- */

void EVB_PPPMOMP::deallocate_omp()
{
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
    thr->init_pppm(-order,memory);
  }

  // *** Extra density stuff *** //
  memory->destroy3d_offset(density_brick1,nzlo_out,nylo_out,nxlo_out);
  memory->destroy3d_offset(density_brick2,nzlo_out,nylo_out,nxlo_out);
  memory->destroy3d_offset(density_brick3,nzlo_out,nylo_out,nxlo_out);

  memory->sfree(density_fft1);
  memory->sfree(density_fft2);
  memory->sfree(density_fft3);

  memory->sfree(work11);
  memory->sfree(work12);
  memory->sfree(work13);
  memory->sfree(work22);
  memory->sfree(work23);

  if(differentiation_flag==1) {
    memory->destroy3d_offset(u_brick2,nzlo_out,nylo_out,nxlo_out);
    memory->destroy3d_offset(u_brick3,nzlo_out,nylo_out,nxlo_out);
  } else {
    memory->destroy3d_offset(vdx_brick2,nzlo_out,nylo_out,nxlo_out);
    memory->destroy3d_offset(vdy_brick2,nzlo_out,nylo_out,nxlo_out);
    memory->destroy3d_offset(vdz_brick2,nzlo_out,nylo_out,nxlo_out);
    memory->destroy3d_offset(vdx_brick3,nzlo_out,nylo_out,nxlo_out);
    memory->destroy3d_offset(vdy_brick3,nzlo_out,nylo_out,nxlo_out);
    memory->destroy3d_offset(vdz_brick3,nzlo_out,nylo_out,nxlo_out);
  }

  // *** Extra FFT data *** //
  delete fft1_2;
  delete fft2_2;
  delete remap_2;
  delete fft1_3;
  delete fft2_3;
  delete remap_3;
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


/* ----------------------------------------------------------------------
   adjust PPPM coeffs, called initially and whenever volume has changed 
------------------------------------------------------------------------- */

void EVB_PPPMOMP::setup()
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

  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

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
}

/* ----------------------------------------------------------------------
   run the regular toplevel compute method from plain PPPM 
   which will have individual methods replaced by our threaded
   versions and then call the obligatory force reduction.
------------------------------------------------------------------------- */

void EVB_PPPMOMP::compute(int eflag, int vflag)
{
  return;
}

/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid 
------------------------------------------------------------------------- */

void EVB_PPPMOMP::make_rho()
{
  const double * _noalias const q = atom->q;
  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
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

    const double boxlox = boxlo[0];
    const double boxloy = boxlo[1];
    const double boxloz = boxlo[2];

    // loop over my charges, add their contribution to nearby grid points
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // (dx,dy,dz) = distance to "lower left" grid pt
    // (mx,my,mz) = global coords of moving stencil pt
    
    // this if protects against having more threads than local atoms
    if (ifrom < nlocal) { 
     if (order == 5) {
      for (int i = ifrom; i < ito; i++) {

        const double ddx = (x[i].x - boxlox) * delxinv;
        const double ddy = (x[i].y - boxloy) * delyinv;
        const double ddz = (x[i].z - boxloz) * delzinv;
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

        const double ddx = (x[i].x-boxlox)*delxinv;
        const double ddy = (x[i].y-boxloy)*delyinv;
        const double ddz = (x[i].z-boxloz)*delzinv;
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

void EVB_PPPMOMP::fieldforce()
{
  if (differentiation_flag == 1) fieldforce_ad();
  else fieldforce_ik();
}

void EVB_PPPMOMP::fieldforce_ik()
{
  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  // no local atoms => nothing to do

  if (nlocal == 0) return;

  const double * _noalias const q = atom->q;
  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  const int3_t * _noalias const p2g = (int3_t *) part2grid[0];
  const dbl3_t * _noalias const p2g_dr = (dbl3_t *) part2grid_dr[0];
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

    FFT_SCALAR ekx, eky, ekz;
    
    ThrData *thr = fix->get_thr(tid);
    dbl3_t * _noalias const f = (dbl3_t *) thr->get_f()[0];
    FFT_SCALAR * const * const r1d =  static_cast<FFT_SCALAR **>(thr->get_rho1d());
    
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
	f[i].x += qfactor*ekx;
	f[i].y += qfactor*eky;
	if(slabflag != 2) f[i].z += qfactor*ekz;
      }
     } else {
      for (int i = ifrom; i < ito; ++i) {

	const int nx = p2g[i].a;
	const int ny = p2g[i].b;
	const int nz = p2g[i].t;

        // General order code
	compute_rho1d_thr(r1d, p2g_dr[i].x, p2g_dr[i].y, p2g_dr[i].z);

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
	f[i].x += qfactor*ekx;
	f[i].y += qfactor*eky;
	if(slabflag != 2) f[i].z += qfactor*ekz;
      }
     }
    }
  }
}

void EVB_PPPMOMP::fieldforce_ad()
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

void EVB_PPPMOMP::fieldforce_peratom()
{
  // loop over my charges, interpolate from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  // no local atoms => nothing to do

  if (nlocal == 0) return;

  const double * _noalias const q = atom->q;
  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  const int nthreads = comm->nthreads;
  const int nlocal = atom->nlocal;

  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

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
	dx = nx+shiftone - (x[i].x-boxlox)*delxinv;
	dy = ny+shiftone - (x[i].y-boxloy)*delyinv;
	dz = nz+shiftone - (x[i].z-boxloz)*delzinv;

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

	const double qi = q[i];
	if (eflag_atom) eatom[i] += qi*u;
	if (vflag_atom) {
	  vatom[i][0] += qi * v0;
	  vatom[i][1] += qi * v1;
	  vatom[i][2] += qi * v2;
	  vatom[i][3] += qi * v3;
	  vatom[i][4] += qi * v4;
	  vatom[i][5] += qi * v5;
	}
      }
    }
  }
}

/* ----------------------------------------------------------------------
   charge assignment into rho1d
   dx,dy,dz = distance of particle from "lower left" grid point 
------------------------------------------------------------------------- */

void EVB_PPPMOMP::compute_rho1d(const FFT_SCALAR &dx, const FFT_SCALAR &dy, const FFT_SCALAR &dz)
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
    FFT_SCALAR r1, r2, r3;
    for (int k = (1-order)/2; k <= order/2; ++k) {
      r1 = r2 = r3 = ZEROF;
      for (int l = order-1; l >= 0; --l) {
        r1 = rho_coeff[l][k] + r1 * dx;
        r2 = rho_coeff[l][k] + r2 * dy;
        r3 = rho_coeff[l][k] + r3 * dz;
      }
      rho1d[0][k] = r1;
      rho1d[1][k] = r2;
      rho1d[2][k] = r3;
    }
  }
}

/* ----------------------------------------------------------------------
   charge assignment into rho1d
   dx,dy,dz = distance of particle from "lower left" grid point 
------------------------------------------------------------------------- */
inline void EVB_PPPMOMP::compute_rho1d_thr(FFT_SCALAR * const * const r1d, const FFT_SCALAR &dx,
                                           const FFT_SCALAR &dy, const FFT_SCALAR &dz)
{
  FFT_SCALAR r1, r2, r3;
  for (int k = (1-order)/2; k <= order/2; ++k) {
    r1 = r2 = r3 = ZEROF;
    for (int l = order-1; l >= 0; --l) {
      r1 = rho_coeff[l][k] + r1 * dx;
      r2 = rho_coeff[l][k] + r2 * dy;
      r3 = rho_coeff[l][k] + r3 * dz;
    }
    drho1d[0][k] = r1;
    drho1d[1][k] = r2;
    drho1d[2][k] = r3;
  }
}

/* ----------------------------------------------------------------------
   charge assignment into drho1d
   dx,dy,dz = distance of particle from "lower left" grid point 
------------------------------------------------------------------------- */

void EVB_PPPMOMP::compute_drho1d(const FFT_SCALAR &dx, const FFT_SCALAR &dy, const FFT_SCALAR &dz)
{
  if (order == 5) {
    // order = 5 case, completely unrolled loops
    const double dx2 = dx*dx;
    const double dx3 = dx2*dx;
    const double dy2 = dy*dy;
    const double dy3 = dy2*dy;
    const double dz2 = dz*dz;
    const double dz3 = dz2*dz;

    int k = -2;
    drho1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
    drho1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
    drho1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
    k = -1;
    drho1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
    drho1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
    drho1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
    k = 0;
    drho1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
    drho1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
    drho1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
    k = 1;
    drho1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
    drho1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
    drho1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
    k = 2;
    drho1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
    drho1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
    drho1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
  } else {
    FFT_SCALAR r1,r2,r3;
    for (int k = (1-order)/2; k <= order/2; k++) {
      r1 = r2 = r3 = ZEROF;
      
      for (int l = order-2; l >= 0; l--) {
	r1 = drho_coeff[l][k] + r1*dx;
	r2 = drho_coeff[l][k] + r2*dy;
	r3 = drho_coeff[l][k] + r3*dz;
      }
      drho1d[0][k] = r1;
      drho1d[1][k] = r2;
      drho1d[2][k] = r3;
    }
  }
}

/* ----------------------------------------------------------------------
   charge assignment into drho1d
   dx,dy,dz = distance of particle from "lower left" grid point 
------------------------------------------------------------------------- */

inline void EVB_PPPMOMP::compute_drho1d_thr(FFT_SCALAR * const * const d1d, const FFT_SCALAR &dx,
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


/*************************************************************************/

void EVB_PPPMOMP::compute_env(int vflag)
{
  TIMER_STAMP(EVB_PPPM, compute_env);
  
  nlocal = atom->nlocal;
  
  q = atom->q;
  x = atom->x;
  f = atom->f;
  
  int* is_cplx_atom = evb_engine->complex_atom;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int has_cplx_atom = evb_engine->has_complex_atom;

  // Calculate the ENV density map;
  FFT_SCALAR ***save_density = density_brick;
  density_brick = env_density_brick;
  clear_density();

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;

  // Make the density all at once
  make_rho(); 
  
  if (has_cplx_atom) for(int i=0; i<nlocal_cplx; ++i) map2density_one_subtract(cplx_list[i]);
  
  density_brick = save_density;
  
  load_env_density();

  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();

  poisson_energy(vflag);
  
  qsqsum = evb_engine->qsqsum_env = evb_engine->qsqsum_sys - evb_engine->evb_complex->qsqsum;
  reduce_ev(vflag,true);
  
  env_energy = energy;
  if (vflag) for (int i = 0; i < 6; i++) virial[i] = 0.0;

  // Environment contribution to dipole for slab correction
  if(slabflag) {
    double *q = atom->q;
    double **x = atom->x;
    
    double dipole    = 0.0;
    double dipole_r2 = 0.0;
    for(int i=0; i<atom->nlocal; i++) if(!is_cplx_atom[i]) {
	dipole    += q[i] * x[i][2];
	dipole_r2 += q[i] * x[i][2] * x[i][2];
      }
    
    MPI_Allreduce(&dipole,    &dipole_env,    1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&dipole_r2, &dipole_r2_env, 1, MPI_DOUBLE, MPI_SUM, world);
  }
 
  TIMER_CLICK(EVB_PPPM, compute_env); 
}

/*************************************************************************/

void EVB_PPPMOMP::compute_cplx(int vflag)
{
  TIMER_STAMP(EVB_PPPM, compute_cplx);

  f = atom->f;

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
  load_env_density();
  
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  for(int i=0; i<nlocal_cplx; i++) map2density_one(cplx_list[i]);
  
  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();

  // Set appropriate scaling factor
  qscale = force->qqrd2e;

  // Don't calculate forces here in SCI simulations during compute(), only initialize().
  if(evb_engine->ncomplex == 1) {
    poisson(true,vflag);
    
    if (differentiation_flag == 1) cg->forward_comm(this,FORWARD_AD);
    else cg->forward_comm(this,FORWARD_IK);
    fieldforce();

  } if( (evb_engine->SCI_KSPACE_flag == KSPACE_DEFAULT || evb_engine->SCI_KSPACE_flag == PPPM_HF_FORCES) &&
	evb_engine->engine_indicator == ENGINE_INDICATOR_INITIALIZE) {
    poisson(true,vflag);
    
    if (differentiation_flag == 1) {
      cg->forward_comm(this,FORWARD_AD);
      for(int i=0; i<nlocal_cplx; i++) field2force_one_ad(cplx_list[i],false);
    } else {
      cg->forward_comm(this,FORWARD_IK);
      for(int i=0; i<nlocal_cplx; i++) field2force_one_ik(cplx_list[i],false);
    }

  } else poisson_energy(vflag);
  
  qsqsum = evb_engine->qsqsum_env + evb_engine->evb_complex->qsqsum;
  reduce_ev(vflag,true); 
  
  // Don't calculate slab correction here in SCI simulations
  if(slabflag && evb_engine->ncomplex == 1) slabcorr_cplx();

  energy -= env_energy;

  TIMER_CLICK(EVB_PPPM, compute_cplx);
}

/*************************************************************************/

void EVB_PPPMOMP::compute_cplx_eff(int vflag)
{
  TIMER_STAMP(EVB_PPPM, compute_cplx_eff);

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  f = atom->f;

  int *is_cplx_atom = evb_engine->complex_atom;
  int cplx_id = evb_engine->evb_complex->id;  

  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;

  clear_density();
  for(int i=0; i<nlocal; i++) {
      if(is_cplx_atom[i] == cplx_id) map2density_one(i,Q_ATOM);
      else map2density_one(i,Q_EFFECTIVE);
    }
  
  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();
 
  poisson(true,vflag);

  // Only accumulate forces on complex atoms
  if (differentiation_flag == 1) {
    cg->forward_comm(this,FORWARD_AD);
    for(int i=0; i<nlocal_cplx; i++) field2force_one_ad(cplx_list[i],false);
  } else {
    cg->forward_comm(this,FORWARD_IK);
    for(int i=0; i<nlocal_cplx; i++) field2force_one_ik(cplx_list[i],false);
  }

 
  TIMER_CLICK(EVB_PPPM, compute_cplx_eff);
}

/*************************************************************************/

void EVB_PPPMOMP::compute_exch(int vflag)
{
  double qsum_save = qsum;

  f = atom->f;
  double save_energy = energy;
  double save_virial[6];
  memcpy(save_virial,virial,sizeof(double)*6);

  // Check if we have any special atoms
  int has_exch_chg  = evb_engine->has_exch_chg;
  int * exch_list   = evb_engine->evb_offdiag->exch_list;
  int n_exch_chg    = evb_engine->evb_offdiag->n_exch_chg;
  int* is_cplx_atom = evb_engine->complex_atom;
  int nlocal_cplx   = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list    = evb_engine->evb_complex->cplx_list;
  int has_cplx_atom = evb_engine->has_complex_atom;

  // save pointers
  FFT_SCALAR *** orig_vdx_brick, *** orig_vdy_brick, *** orig_vdz_brick;
  FFT_SCALAR *** orig_u_brick;
  if(differentiation_flag == 1) orig_u_brick = u_brick;
  else {
    orig_vdx_brick = vdx_brick;
    orig_vdy_brick = vdy_brick;
    orig_vdz_brick = vdz_brick;
  }

  /***************************************************/
  // Build all three densities
  /***************************************************/

  // ** Build overall density ** //
  load_env_density();
  for(int i=0; i<nlocal_cplx; i++) map2density_one(cplx_list[i]);
  // save to density_brick1
  memcpy(&density_brick1[nzlo_out][nylo_out][nxlo_out],
         &density_brick [nzlo_out][nylo_out][nxlo_out],
         sizeof(FFT_SCALAR)*ngrid);

  // ** Build exch_chg->non_exch_chg density ** //
  clear_density();
  if (has_exch_chg) {
    for(int i=0; i<n_exch_chg; i++) {
      if (exch_list[i] < nlocal) map2density_one(exch_list[i]);
    }
  }
  // save to density_brick2
  memcpy(&density_brick2[nzlo_out][nylo_out][nxlo_out],
         &density_brick [nzlo_out][nylo_out][nxlo_out],
         sizeof(FFT_SCALAR)*ngrid);

  // ** Build non_exch_chg->exch_chg density ** //
  load_env_density();
  if (has_cplx_atom) {
    if (has_exch_chg) {
      if(evb_engine->ncomplex == 1) {
	for(int i=0; i<nlocal_cplx; ++i) if(!is_exch_chg[cplx_list[i]]) map2density_one(cplx_list[i]);
      } else {
	for(int i=0; i<nlocal; i++) if(is_cplx_atom[i] && !is_exch_chg[i]) map2density_one(i);
      }
    } else {
      if(evb_engine->ncomplex == 1) {
	for(int i=0; i<nlocal_cplx; ++i) map2density_one(cplx_list[i]);
      } else {
	for(int i=0; i<nlocal; i++) if(is_cplx_atom[i]) map2density_one(i);
      }
    }
  }
  // save to density_brick3
  memcpy(&density_brick3[nzlo_out][nylo_out][nxlo_out],
         &density_brick [nzlo_out][nylo_out][nxlo_out],
         sizeof(FFT_SCALAR)*ngrid);

  // Set up energy and virial
  double energy3[3];
  double virial3[3][6];
  energy3[0] = energy3[1] = energy3[2] = 0.0;
  if (vflag) {
    for (int i=0; i<6; i++) { 
      virial3[0][i] = 0.0;
      virial3[1][i] = 0.0;
      virial3[2][i] = 0.0;
    }
  }

  // ******** Do all 3 brick2ffts ********* //
  reverse_comm3(REVERSE_RHO);
  brick2fft_all3();
  // ******** Do all 3 poissons ********* //
  poisson_all3(vflag, energy3, virial3);
  // ******** Do all 3 fillbricks ********* //
  if(differentiation_flag==1) forward_comm3(FORWARD_AD);
  else forward_comm3(FORWARD_IK);

  // ******** Finish off density1 stuff ******* //
  energy = energy3[0];
  if (vflag) memcpy(virial,&(virial3[0][0]),sizeof(double)*6);
  qsum = qsqsum = 0.0;
  reduce_ev(vflag,true);
  off_diag_energy = energy;
  if (vflag) memcpy(off_diag_virial,virial,sizeof(double)*6);

  // ******** Finish off density2 stuff ******* //
  energy = energy3[1];
  if (vflag) memcpy(virial,&(virial3[1][0]),sizeof(double)*6);

  if(differentiation_flag == 1) u_brick = u_brick2;
  else {
    vdx_brick = vdx_brick2;
    vdy_brick = vdy_brick2;
    vdz_brick = vdz_brick2;
  }
  // Set appropriate scaling factor
  qscale = force->qqrd2e * A_Rq;
  // fieldforce();
  // if(has_exch_chg) {
  //   for(int i=0; i<n_exch_chg; i++) {
  //     if (exch_list[i] < nlocal) field2force_one_subtract(exch_list[i]);
  //   }
  // }
  reduce_ev(vflag,true);
  off_diag_energy -= energy;
  if (vflag) for(int i=0; i<6; i++) off_diag_virial[i]-=virial[i];

  // ******** Finish off density3 stuff ******* //
  energy = energy3[2];
  if (vflag) memcpy(virial,&(virial3[2][0]),sizeof(double)*6);
  if(differentiation_flag == 1) u_brick = u_brick3;
  else {
    vdx_brick = vdx_brick3;
    vdy_brick = vdy_brick3;
    vdz_brick = vdz_brick3;
  }
  // if (has_exch_chg) {
  //   // Set appropriate scaling factor
  //   qscale = force->qqrd2e * A_Rq;
  //   for(int i=0; i<n_exch_chg; i++) {
  //     if (exch_list[i] < nlocal) field2force_one(exch_list[i],true);
  //   }
  // }
  reduce_ev(vflag,true);
  off_diag_energy -= energy;
  if (vflag) for(int i=0; i<6; ++i) off_diag_virial[i]-=virial[i];

  /***************************************************/

  double energy_all;
  MPI_Allreduce(&off_diag_energy,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
  off_diag_energy = energy_all;

  energy = save_energy;
  memcpy(virial, save_virial,sizeof(double)*6);

  qsum = qsum_save;

  // restore pointers
  if(differentiation_flag == 1) u_brick = orig_u_brick;
  else {
    vdx_brick = orig_vdx_brick;
    vdy_brick = orig_vdy_brick;
    vdz_brick = orig_vdz_brick;
  }

  qscale = force->qqrd2e; // Reset to default value
}

/*************************************************************************/

// void EVB_PPPMOMP::compute_exch_split1(int vflag)
// {
 
//   /***************************************************/
//   /******* Overall                             *******/
//   /***************************************************/
  
//   energy = 0.0;
//   if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
//   load_env_density();

//   int* is_cplx_atom = evb_engine->complex_atom;
//   int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
//   int* cplx_list = evb_engine->evb_complex->cplx_list;

//   for(int i=0; i<nlocal_cplx; i++) map2density_one(cplx_list[i]);

//   brick2fft();
//   poisson_energy(vflag);
//   if (differentiation_flag == 1) cg->forward_comm(this,FORWARD_AD);
//   else cg->forward_comm(this,FORWARD_IK);

//   qsum = qsqsum = 0.0;
//   reduce_ev(vflag,true);
  
//   off_diag_energy = energy;
//   if (vflag) memcpy(off_diag_virial,virial,sizeof(double)*6);
// }
 
// void EVB_PPPMOMP:compute_exch_split2(int vflag)
// {
//   // Of the three splits, this one is taking the longest and needs threaded

//   /***************************************************/
//   /******* Mesh(Exch_chg)->Point(Non_exch_chg) *******/
//   /***************************************************/
  
//   energy = 0.0;
//   if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
//   clear_density();

//   for (int i=0; i<nlocal; ++i) if(is_exch_chg[i]) map2density_one(i);
  
//   brick2fft();
//   qscale = force->qqrd2e;
//   poisson(true, vflag);
//   if (differentiation_flag == 1) cg->forward_comm(this,FORWARD_AD);
//   else cg->forward_comm(this,FORWARD_IK);

//   // This appears to be the expensive loop
//   //for (int i=0; i<nlocal; i++) if(!is_exch_chg[i]) field2force_one(i,true);
//   // ** AWGL : thread it ** //
//   int i, t;
// #if defined(_OPENMP)
// #pragma omp parallel default(none) private(i,t)
// #endif
//   {
// #if defined(_OPENMP)
//     // each thread works on a fixed chunk of atoms.
//     const int nthreads = comm->nthreads;
//     const int tid = omp_get_thread_num();
//     const int inum = nlocal;
//     const int idelta = 1 + inum/nthreads;
//     const int ifrom = tid*idelta;
//     const int ito = ((ifrom + idelta) > inum) ? inum : ifrom + idelta;
// #else
//     const int ifrom = 0;
//     const int ito = nlocal;
//     const int tid = 0;
// #endif
//     ThrData *thr = fix->get_thr(tid);
//     double** fthread = thr->get_f();
//     FFT_SCALAR * const * const r1d =  static_cast<FFT_SCALAR **>(thr->get_rho1d());

//     // ** The loop ** //
//     if (ifrom < nlocal) {
//       for (i=ifrom; i<ito; i++) if(!is_exch_chg[i]) field2force_one_thr<1>(i, r1d, fthread);
//     }
//   } // close OpenMP bracket  


//   reduce_ev(vflag,true);

//   off_diag_energy -= energy;
//   if (vflag) for(int i=0; i<6; i++) off_diag_virial[i]-=virial[i];

// }

// void EVB_PPPMOMP::compute_exch_split3(int vflag)
// {

//   /***************************************************/
//   /******* Mesh(Non_exch_chg)->Point(Exch_chg) *******/
//   /***************************************************/
//   int* is_cplx_atom = evb_engine->complex_atom;
//   int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
//   int* cplx_list = evb_engine->evb_complex->cplx_list;
  
//   energy = 0.0;
//   if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
//   load_env_density();
  
//   for (int i=0; i<nlocal; i++) if(is_cplx_atom[i] && !is_exch_chg[i]) map2density_one(i);
  
//   brick2fft();
//   qscale = force->qqrd2e;
//   poisson(true, vflag);
//   if (differentiation_flag == 1) cg->forward_comm(this,FORWARD_AD);
//   else cg->forward_comm(this,FORWARD_IK);
  
//   // Set appropriate scaling factor
//   qscale = force->qqrd2e * A_Rq;
//   for (int i=0; i<nlocal; i++) if(is_exch_chg[i]) field2force_one(i,true);
  
//   reduce_ev(vflag,true);
  
//   off_diag_energy -= energy;
//   if (vflag) for(int i=0; i<6; i++) off_diag_virial[i]-=virial[i];

//   /***************************************************************/
//   /***************************************************************/
  
//   double energy_all;
//   MPI_Allreduce(&off_diag_energy,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
//   off_diag_energy = energy_all;
  
//   qscale = force->qqrd2e; // Reset to default value
// }


/*************************************************************************/

void EVB_PPPMOMP::map2density_one_subtract(int id)
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

void EVB_PPPMOMP::map2density_one(int id)
{
  double *q = atom->q;

  const int nx = part2grid[id][0];
  const int ny = part2grid[id][1];
  const int nz = part2grid[id][2];
  
  // (dx,dy,dz) = distance to "lower left" grid pt
  
  compute_rho1d(part2grid_dr[id][0], part2grid_dr[id][1], part2grid_dr[id][2]);
  
  // (mx,my,mz) = global coords of moving stencil pt
  
  const FFT_SCALAR z0 = delvolinv * q[id];
  for (int n = nlower; n <= nupper; n++) {
    const FFT_SCALAR y0 = z0*rho1d[2][n];
    for (int m = nlower; m <= nupper; m++) {
      const FFT_SCALAR x0 = y0*rho1d[1][m];
      for (int l = nlower; l <= nupper; l++) {
	density_brick[n+nz][m+ny][l+nx] += x0*rho1d[0][l];
      } // Loop mx
    } // Loop my
  } // Loop mz
}

/*************************************************************************/

void EVB_PPPMOMP::map2density_one(int id, int WHICH)
{
  double * q;
  if(WHICH == Q_ATOM) q = atom->q;
  else if(WHICH == Q_EFFECTIVE) q = evb_engine->evb_effpair->q;
  
  nx = part2grid[id][0];
  ny = part2grid[id][1];
  nz = part2grid[id][2];
  
  // (dx,dy,dz) = distance to "lower left" grid pt
  
  compute_rho1d(part2grid_dr[id][0], part2grid_dr[id][1], part2grid_dr[id][2]);
  
  // (mx,my,mz) = global coords of moving stencil pt   

  const FFT_SCALAR z0 = delvolinv * q[id];
  for (int n = nlower; n <= nupper; n++) {
    const int mz = n+nz;
    const FFT_SCALAR y0 = z0*rho1d[2][n];
    
    for (int m = nlower; m <= nupper; m++) {
      const int my = m+ny;
      const FFT_SCALAR x0 = y0*rho1d[1][m];
      
      for (int l = nlower; l <= nupper; l++) {
	const int mx = l+nx;
	density_brick[mz][my][mx] += x0*rho1d[0][l];
      } // Loop mx
    } // Loop my
  } // Loop mz
}

/*************************************************************************/

// void EVB_PPPMOMP::map2density_one_thr(int id, FFT_SCALAR * const * const r1d, FFT_SCALAR * const * const * const db)
// {
//   double *q = atom->q;

//   // ** AWGL : ensure scope ** //
//   const int nx = part2grid[id][0];
//   const int ny = part2grid[id][1];
//   const int nz = part2grid[id][2];
  
//   // (dx,dy,dz) = distance to "lower left" grid pt
  
//   compute_rho1d_thr(r1d, part2grid_dr[id][0], part2grid_dr[id][1], part2grid_dr[id][2]);
  
//   // (mx,my,mz) = global coords of moving stencil pt
  
//   const FFT_SCALAR z0 = delvolinv * q[id];
//   for (int n = nlower; n <= nupper; ++n) 
//     {
//       const FFT_SCALAR y0 = z0*r1d[2][n];
//       for (int m = nlower; m <= nupper; ++m) 
// 	{
// 	  const FFT_SCALAR x0 = y0*r1d[1][m];
// 	  for (int l = nlower; l <= nupper; ++l) 
// 	    {
// 	      db[n+nz][m+ny][l+nx] += x0*r1d[0][l];
// 	    } // Loop mx
// 	} // Loop my
//     } // Loop mz
  
// }

/*************************************************************************/

void EVB_PPPMOMP::field2force_one_subtract(int id)
{
  if(differentiation_flag == 1) {
    double s1, s2, s3;
    double sf = 0.0;

    const double *prd = (triclinic == 0) ? domain->prd : domain->prd_lamda;
    const double hx_inv = nx_pppm / prd[0];
    const double hy_inv = ny_pppm / prd[1];
    const double hz_inv = nz_pppm / prd[2];

    const int nx = part2grid[id][0];
    const int ny = part2grid[id][1];
    const int nz = part2grid[id][2];
    
    compute_rho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
    compute_drho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
    
    FFT_SCALAR ekx = eky = ekz = ZEROF;
    for (int n = nlower; n <= nupper; n++) {
      const int mz = n + nz;
      for (int m = nlower; m <= nupper; m++) {
	const int my = m + ny;
	for (int l = nlower; l <= nupper; l++) {
	  const int mx = m + nx;
	  ekx += drho1d[0][l] * rho1d[1][m]  * rho1d[2][n]  * u_brick[mz][my][mx];
	  eky += rho1d[0][l]  * drho1d[1][m] * rho1d[2][n]  * u_brick[mz][my][mx];
	  ekz += rho1d[0][l]  * rho1d[1][m]  * drho1d[2][n] * u_brick[mz][my][mx];
	}
      }
    }
    ekx *= hx_inv;
    eky *= hy_inv;
    ekz *= hz_inv;

    const double qi = q[id];
    const double qfactor = qscale * qi;

    s1 = x[id][0]*hx_inv;
    sf = sf_coeff[0]*sin(MY_2PI*s1);
    sf += sf_coeff[1]*sin(MY_4PI*s1);
    sf *= 2.0*qi;
    f[id][0] -= qfactor*(ekx - sf);
	
    s2 = x[id][1]*hy_inv;
    sf = sf_coeff[2]*sin(MY_2PI*s2);
    sf += sf_coeff[3]*sin(MY_4PI*s2);
    sf *= 2.0*qi;
    f[id][1] -= qfactor*(eky - sf);
	
    if (slabflag != 2) {
      s3 = x[id][2]*hz_inv;
      sf = sf_coeff[4]*sin(MY_2PI*s3);
      sf += sf_coeff[5]*sin(MY_4PI*s3);
      sf *= 2.0*qi;
      f[id][2] -= qfactor*(ekz - sf);
    }
  } else {
    const int nx = part2grid[id][0];
    const int ny = part2grid[id][1];
    const int nz = part2grid[id][2];
    
    compute_rho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
    
    FFT_SCALAR ekx = eky = ekz = ZEROF;
    for (int n = nlower; n <= nupper; n++) {
      const FFT_SCALAR z0 = rho1d[2][n];
      for (int m = nlower; m <= nupper; m++) {
	const FFT_SCALAR y0 = z0*rho1d[1][m];
	for (int l = nlower; l <= nupper; l++) {
	  const FFT_SCALAR x0 = y0*rho1d[0][l];
	  ekx -= x0*vdx_brick[n+nz][m+ny][l+nx];
	  eky -= x0*vdy_brick[n+nz][m+ny][l+nx];
	  ekz -= x0*vdz_brick[n+nz][m+ny][l+nx];
	}
      }
    }
    
    // convert E-field to force
    const double pre_factor = qscale * q[id];
    f[id][0] -= pre_factor * ekx;
    f[id][1] -= pre_factor * eky;
    f[id][2] -= pre_factor * ekz;
  }
}

/*************************************************************************/

void EVB_PPPMOMP::field2force_one(int id, bool Aflag)
{
  if(differentiation_flag == 1) {
    double s1, s2, s3;
    double sf = 0.0;

    const double *prd = (triclinic == 0) ? domain->prd : domain->prd_lamda;
    const double hx_inv = nx_pppm / prd[0];
    const double hy_inv = ny_pppm / prd[1];
    const double hz_inv = nz_pppm / prd[2];

    const int nx = part2grid[id][0];
    const int ny = part2grid[id][1];
    const int nz = part2grid[id][2];
    
    compute_rho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
    compute_drho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
    
    FFT_SCALAR ekx = eky = ekz = ZEROF;
    for (int n = nlower; n <= nupper; n++) {
      const int mz = n + nz;
      for (int m = nlower; m <= nupper; m++) {
	const int my = m + ny;
	for (int l = nlower; l <= nupper; l++) {
	  const int mx = m + nx;
	  ekx += drho1d[0][l] * rho1d[1][m]  * rho1d[2][n]  * u_brick[mz][my][mx];
	  eky += rho1d[0][l]  * drho1d[1][m] * rho1d[2][n]  * u_brick[mz][my][mx];
	  ekz += rho1d[0][l]  * rho1d[1][m]  * drho1d[2][n] * u_brick[mz][my][mx];
	}
      }
    }
    ekx *= hx_inv;
    eky *= hy_inv;
    ekz *= hz_inv;

    const double qi = q[id];
    const double qfactor = qscale * qi;

    s1 = x[id][0]*hx_inv;
    sf = sf_coeff[0]*sin(MY_2PI*s1);
    sf += sf_coeff[1]*sin(MY_4PI*s1);
    sf *= 2.0*qi;
    f[id][0] += qfactor*(ekx - sf);
	
    s2 = x[id][1]*hy_inv;
    sf = sf_coeff[2]*sin(MY_2PI*s2);
    sf += sf_coeff[3]*sin(MY_4PI*s2);
    sf *= 2.0*qi;
    f[id][1] += qfactor*(eky - sf);
	
    if (slabflag != 2) {
      s3 = x[id][2]*hz_inv;
      sf = sf_coeff[4]*sin(MY_2PI*s3);
      sf += sf_coeff[5]*sin(MY_4PI*s3);
      sf *= 2.0*qi;
      f[id][2] += qfactor*(ekz - sf);
    }
  } else {
    const int nx = part2grid[id][0];
    const int ny = part2grid[id][1];
    const int nz = part2grid[id][2];
    
    compute_rho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
    
    FFT_SCALAR ekx = eky = ekz = ZEROF;
    for (int n = nlower; n <= nupper; n++) {
      const FFT_SCALAR z0 = rho1d[2][n];
      for (int m = nlower; m <= nupper; m++) {
	const FFT_SCALAR y0 = z0*rho1d[1][m];
	for (int l = nlower; l <= nupper; l++) {
	  const FFT_SCALAR x0 = y0*rho1d[0][l];
	  ekx -= x0*vdx_brick[n+nz][m+ny][l+nx];
	  eky -= x0*vdy_brick[n+nz][m+ny][l+nx];
	  ekz -= x0*vdz_brick[n+nz][m+ny][l+nx];
	}
      }
    }
    
    // convert E-field to force
    const double pre_factor = qscale * q[id];
    f[id][0] += pre_factor * ekx;
    f[id][1] += pre_factor * eky;
    f[id][2] += pre_factor * ekz;
  }
}

/*************************************************************************/

// template <int AFLAG>
// void EVB_PPPMOMP::field2force_one_thr(int id, FFT_SCALAR * const * const r1d, double** ft)
// {
//   // Subtract force and energy of id, opposite of field2force_one
//   const int nx = part2grid[id][0];
//   const int ny = part2grid[id][1];
//   const int nz = part2grid[id][2];
  
//   compute_rho1d_thr(r1d, part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);

//   FFT_SCALAR ekx, eky, ekz;  
//   ekx = eky = ekz = ZEROF;

//   for (int n = nlower; n <= nupper; n++) 
//     {
//       const FFT_SCALAR z0 = r1d[2][n];
//       for (int m = nlower; m <= nupper; m++) 
// 	{
// 	  const FFT_SCALAR y0 = z0*r1d[1][m];
// 	  for (int l = nlower; l <= nupper; l++)
// 	    {
// 	      const FFT_SCALAR x0 = y0*r1d[0][l];
// 	      ekx -= x0*vdx_brick[n+nz][m+ny][l+nx];
// 	      eky -= x0*vdy_brick[n+nz][m+ny][l+nx];
// 	      ekz -= x0*vdz_brick[n+nz][m+ny][l+nx];
// 	    }
// 	}
//     }
  
//   // convert E-field to force
//   if(AFLAG) { 
//     const double pre_factor = A_Rq * qqrd2e * q[id];
//     ft[id][0] += pre_factor * ekx;
//     ft[id][1] += pre_factor * eky;
//     ft[id][2] += pre_factor * ekz;
//   } else {
//     const double pre_factor = qqrd2e * q[id];
//     ft[id][0] += pre_factor * ekx;
//     ft[id][1] += pre_factor * eky;
//     ft[id][2] += pre_factor * ekz;
//   }
// }
/*************************************************************************/

/* ----------------------------------------------------------------------
   ghost-swap to accumulate full density in brick decomposition 
   remap density from 3d brick decomposition to FFT decomposition
------------------------------------------------------------------------- */


void EVB_PPPMOMP::brick2fft()
{
  int i,n,ix,iy,iz;

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

/* ------------------------------------------------------------------------- */

void EVB_PPPMOMP::brick2fft_all3()
{
  int n,ix,iy,iz;

  // copy grabs inner portion of density from 3d brick
  // remap could be done as pre-stage of FFT,
  //   but this works optimally on only double values, not complex values

  n = 0;
  int m = 0;
  int l = 0;
  for (iz = nzlo_in; iz <= nzhi_in; iz++)
    for (iy = nylo_in; iy <= nyhi_in; iy++)
      for (ix = nxlo_in; ix <= nxhi_in; ix++)
      {
	density_fft1[n++] = density_brick1[iz][iy][ix];
	density_fft2[m++] = density_brick2[iz][iy][ix];
	density_fft3[l++] = density_brick3[iz][iy][ix];
      }

  remap->perform(density_fft1,density_fft1,work11);
  remap->perform(density_fft2,density_fft2,work12);
  remap->perform(density_fft3,density_fft3,work13);

}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver 
------------------------------------------------------------------------- */

void EVB_PPPMOMP::poisson_all3(int vflag, double energy3[3], double virial3[3][6])
{
  if (differentiation_flag == 1) poisson_all3_ad(vflag, energy3, virial3);
  else poisson_all3_ik(vflag, energy3, virial3);
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik
   All 3 exch at once
------------------------------------------------------------------------- */

void EVB_PPPMOMP::poisson_all3_ik(int vflag, double energy3[3], double virial3[3][6])
{
  int i,j,k,n;

  // transform charge density (r -> k) 

  n = 0;
  for (i = 0; i < nfft; i++) {
    work11[n]   = density_fft1[i];
    work12[n]   = density_fft2[i];
    work13[n]   = density_fft3[i];
    work11[n+1] = ZEROF;
    work12[n+1] = ZEROF;
    work13[n+1] = ZEROF;
    n += 2;
  }

  // Can we do the thread partitioned FFT call?
  int doit = 1;
#if defined (_OPENMP)
  int provided = 0;
  MPI_Query_thread(&provided);
  if (provided != MPI_THREAD_MULTIPLE) doit = 0;
#endif
 
  if (comm->nthreads >= 3 && doit) {
#if defined(_OPENMP)
    #pragma omp parallel default(none)
    {
      const int tid = omp_get_thread_num();
      if      (tid == 0) fft1->compute(work11,work11,1);
      else if (tid == 1) fft1_2->compute(work12,work12,1);
      else if (tid == 2) fft1_3->compute(work13,work13,1);
    }
#endif
  } else {
    fft1->compute(work11,work11,1);
    fft1->compute(work12,work12,1);
    fft1->compute(work13,work13,1);
  }

  // if requested, compute virial contribution
  // always compute the energy

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  if (vflag) {
    n = 0;
    for (i = 0; i < nfft; i++) {
      double eng1, eng2, eng3;
      double s2g = s2 * greensfn[i];
      eng1 = s2g * (work11[n]*work11[n] + work11[n+1]*work11[n+1]);
      eng2 = s2g * (work12[n]*work12[n] + work12[n+1]*work12[n+1]);
      eng3 = s2g * (work13[n]*work13[n] + work13[n+1]*work13[n+1]);
      for (j = 0; j < 6; j++) {
        virial3[0][j] += eng1*vg[i][j];
        virial3[1][j] += eng2*vg[i][j];
        virial3[2][j] += eng3*vg[i][j];
      }
      energy3[0] += eng1;
      energy3[1] += eng2;
      energy3[2] += eng3;
      n += 2;
    }
  } else {
    n = 0;
    double e3_0, e3_1, e3_2;
    e3_0 = e3_1 = e3_2 = 0.0;
    #pragma omp parallel for default(none) private(i) reduction(+:e3_0,e3_1,e3_2) 
    for (i = 0; i < nfft; i++) {
      e3_0 += greensfn[i] * (work11[2*i]*work11[2*i] + work11[2*i+1]*work11[2*i+1]);
      e3_1 += greensfn[i] * (work12[2*i]*work12[2*i] + work12[2*i+1]*work12[2*i+1]);
      e3_2 += greensfn[i] * (work13[2*i]*work13[2*i] + work13[2*i+1]*work13[2*i+1]);
    }
    energy3[0] += e3_0 * s2;
    energy3[1] += e3_1 * s2;
    energy3[2] += e3_2 * s2;
  }

  // *** Here after, we don't do any more for work11 b/c it only needs the energy *** //

  // ** Split the work across two threads, if we have two threads ** //
  if (comm->nthreads >= 2 && doit) {

    #pragma omp parallel default(none) shared(scaleinv) private(i,j,k,n)
    {
#if defined (_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      // scale by 1/total-grid-pts to get rho(k)
      // multiply by Green's function to get V(k)
      #pragma omp for
      for (i = 0; i < nfft; i++) {
        const double sg = scaleinv * greensfn[i];
        work12[2*i]   *= sg; 
        work12[2*i+1] *= sg;
        work13[2*i]   *= sg; 
        work13[2*i+1] *= sg;
      }

      // compute gradients of V(r) in each of 3 dims by transformimg -ik*V(k)
      // FFT leaves data in 3d brick decomposition
      // copy it into inner portion of vdx,vdy,vdz arrays

     // ** Split work here between thread 0 and 1 ** //
     if (tid == 0) {
      // x direction gradient
      n = 0;
      for (k = nzlo_fft; k <= nzhi_fft; k++)
        for (j = nylo_fft; j <= nyhi_fft; j++)
          for (i = nxlo_fft; i <= nxhi_fft; i++) {
	    work22[n]   =  fkx[i]*work12[n+1];
	    work22[n+1] = -fkx[i]*work12[n];
	    n += 2;
          }
      fft2_2->compute(work22,work22,-1);
      n = 0;
      for (k = nzlo_in; k <= nzhi_in; k++)
        for (j = nylo_in; j <= nyhi_in; j++)
          for (i = nxlo_in; i <= nxhi_in; i++) {
	    vdx_brick2[k][j][i] = work22[n];
	    n += 2;
          }
      // y direction gradient
      n = 0;
      for (k = nzlo_fft; k <= nzhi_fft; k++)
        for (j = nylo_fft; j <= nyhi_fft; j++) {
          const double fkyj = fky[j];
          for (i = nxlo_fft; i <= nxhi_fft; i++) {
	    work22[n]   =  fkyj*work12[n+1];
	    work22[n+1] = -fkyj*work12[n];
	    n += 2;
          }
        }
      fft2_2->compute(work22,work22,-1);
      n = 0;
      for (k = nzlo_in; k <= nzhi_in; k++)
        for (j = nylo_in; j <= nyhi_in; j++)
          for (i = nxlo_in; i <= nxhi_in; i++) {
	    vdy_brick2[k][j][i] = work22[n];
	    n += 2;
          }

      // z direction gradient
      n = 0;
      for (k = nzlo_fft; k <= nzhi_fft; k++) {
        const double fkzk = fkz[k];
        for (j = nylo_fft; j <= nyhi_fft; j++)
          for (i = nxlo_fft; i <= nxhi_fft; i++) {
	    work22[n]   =  fkzk*work12[n+1];
	    work22[n+1] = -fkzk*work12[n];
	    n += 2;
          }
      }
      fft2_2->compute(work22,work22,-1);
      n = 0;
      for (k = nzlo_in; k <= nzhi_in; k++)
        for (j = nylo_in; j <= nyhi_in; j++)
          for (i = nxlo_in; i <= nxhi_in; i++) {
	    vdz_brick2[k][j][i] = work22[n];
 	    n += 2;
          }

     } else if (tid == 1) {

      // x direction gradient
      n = 0;
      for (k = nzlo_fft; k <= nzhi_fft; k++)
        for (j = nylo_fft; j <= nyhi_fft; j++)
          for (i = nxlo_fft; i <= nxhi_fft; i++) {
	    work23[n]   =  fkx[i]*work13[n+1];
	    work23[n+1] = -fkx[i]*work13[n];
	    n += 2;
          }
      fft2_3->compute(work23,work23,-1);
      n = 0;
      for (k = nzlo_in; k <= nzhi_in; k++)
        for (j = nylo_in; j <= nyhi_in; j++)
          for (i = nxlo_in; i <= nxhi_in; i++) {
	    vdx_brick3[k][j][i] = work23[n];
	    n += 2;
          }
      // y direction gradient
      n = 0;
      for (k = nzlo_fft; k <= nzhi_fft; k++)
        for (j = nylo_fft; j <= nyhi_fft; j++) {
          const double fkyj = fky[j];
          for (i = nxlo_fft; i <= nxhi_fft; i++) {
	    work23[n]   =  fkyj*work13[n+1];
	    work23[n+1] = -fkyj*work13[n];
	    n += 2;
          }
        }
      fft2_3->compute(work23,work23,-1);
      n = 0;
      for (k = nzlo_in; k <= nzhi_in; k++)
        for (j = nylo_in; j <= nyhi_in; j++)
          for (i = nxlo_in; i <= nxhi_in; i++) {
	    vdy_brick3[k][j][i] = work23[n];
	    n += 2;
          }

      // z direction gradient
      n = 0;
      for (k = nzlo_fft; k <= nzhi_fft; k++) {
        const double fkzk = fkz[k];
        for (j = nylo_fft; j <= nyhi_fft; j++)
          for (i = nxlo_fft; i <= nxhi_fft; i++) {
	    work23[n]   =  fkzk*work13[n+1];
	    work23[n+1] = -fkzk*work13[n];
	    n += 2;
          }
      }
      fft2_3->compute(work23,work23,-1);
      n = 0;
      for (k = nzlo_in; k <= nzhi_in; k++)
        for (j = nylo_in; j <= nyhi_in; j++)
          for (i = nxlo_in; i <= nxhi_in; i++) {
 	    vdz_brick3[k][j][i] = work23[n];
 	    n += 2;
          }

     }

    } // close OMP parallel

  } else {

    // ** Only one thread or not doit ** //
    for (i = 0; i < nfft; i++) {
      const double sg = scaleinv * greensfn[i];
      work12[2*i]   *= sg; 
      work12[2*i+1] *= sg;
      work13[2*i]   *= sg; 
      work13[2*i+1] *= sg;
    }

    // compute gradients of V(r) in each of 3 dims by transformimg -ik*V(k)
    // FFT leaves data in 3d brick decomposition
    // copy it into inner portion of vdx,vdy,vdz arrays

    // x direction gradient
    n = 0;
    for (k = nzlo_fft; k <= nzhi_fft; k++)
      for (j = nylo_fft; j <= nyhi_fft; j++)
        for (i = nxlo_fft; i <= nxhi_fft; i++) {
	  work22[n]   =  fkx[i]*work12[n+1];
	  work22[n+1] = -fkx[i]*work12[n];
	  work23[n]   =  fkx[i]*work13[n+1];
	  work23[n+1] = -fkx[i]*work13[n];
	  n += 2;
        }
    fft2->compute(work22,work22,-1);
    fft2->compute(work23,work23,-1);
    n = 0;
    for (k = nzlo_in; k <= nzhi_in; k++)
      for (j = nylo_in; j <= nyhi_in; j++)
        for (i = nxlo_in; i <= nxhi_in; i++) {
	  vdx_brick2[k][j][i] = work22[n];
	  vdx_brick3[k][j][i] = work23[n];
	  n += 2;
        }

    // y direction gradient
    n = 0;
    for (k = nzlo_fft; k <= nzhi_fft; k++)
      for (j = nylo_fft; j <= nyhi_fft; j++)
        for (i = nxlo_fft; i <= nxhi_fft; i++) {
	  work22[n]   =  fky[j]*work12[n+1];
	  work22[n+1] = -fky[j]*work12[n];
	  work23[n]   =  fky[j]*work13[n+1];
	  work23[n+1] = -fky[j]*work13[n];
	  n += 2;
        }
    fft2->compute(work22,work22,-1);
    fft2->compute(work23,work23,-1);
    n = 0;
    for (k = nzlo_in; k <= nzhi_in; k++)
      for (j = nylo_in; j <= nyhi_in; j++)
        for (i = nxlo_in; i <= nxhi_in; i++) {
	  vdy_brick2[k][j][i] = work22[n];
	  vdy_brick3[k][j][i] = work23[n];
	  n += 2;
        }

    // z direction gradient
    n = 0;
    for (k = nzlo_fft; k <= nzhi_fft; k++)
      for (j = nylo_fft; j <= nyhi_fft; j++)
        for (i = nxlo_fft; i <= nxhi_fft; i++) {
	  work22[n]   =  fkz[k]*work12[n+1];
	  work22[n+1] = -fkz[k]*work12[n];
	  work23[n]   =  fkz[k]*work13[n+1];
	  work23[n+1] = -fkz[k]*work13[n];
	  n += 2;
        }
    fft2->compute(work22,work22,-1);
    fft2->compute(work23,work23,-1);
    n = 0;
    for (k = nzlo_in; k <= nzhi_in; k++)
      for (j = nylo_in; j <= nyhi_in; j++)
        for (i = nxlo_in; i <= nxhi_in; i++) {
	  vdz_brick2[k][j][i] = work22[n];
 	  vdz_brick3[k][j][i] = work23[n];
 	  n += 2;
        }

  } // close regular

}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik
   All 3 exch at once
------------------------------------------------------------------------- */

void EVB_PPPMOMP::poisson_all3_ad(int vflag, double energy3[3], double virial3[3][6])
{
  int i,j,k,n;

  // transform charge density (r -> k) 

  n = 0;
  for (i = 0; i < nfft; i++) {
    work11[n]   = density_fft1[i];
    work12[n]   = density_fft2[i];
    work13[n]   = density_fft3[i];
    work11[n+1] = ZEROF;
    work12[n+1] = ZEROF;
    work13[n+1] = ZEROF;
    n += 2;
  }

  // Can we do the thread partitioned FFT call?
  int doit = 1;
#if defined (_OPENMP)
  int provided = 0;
  MPI_Query_thread(&provided);
  if (provided != MPI_THREAD_MULTIPLE) doit = 0;
#endif
 
  if (comm->nthreads >= 3 && doit) {
#if defined(_OPENMP)
    #pragma omp parallel default(none)
    {
      const int tid = omp_get_thread_num();
      if      (tid == 0) fft1->compute(work11,work11,1);
      else if (tid == 1) fft1_2->compute(work12,work12,1);
      else if (tid == 2) fft1_3->compute(work13,work13,1);
    }
#endif
  } else {
    fft1->compute(work11,work11,1);
    fft1->compute(work12,work12,1);
    fft1->compute(work13,work13,1);
  }

  // if requested, compute virial contribution
  // always compute the energy

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  if (vflag) {
    n = 0;
    for (i = 0; i < nfft; i++) {
      double eng1, eng2, eng3;
      double s2g = s2 * greensfn[i];
      eng1 = s2g * (work11[n]*work11[n] + work11[n+1]*work11[n+1]);
      eng2 = s2g * (work12[n]*work12[n] + work12[n+1]*work12[n+1]);
      eng3 = s2g * (work13[n]*work13[n] + work13[n+1]*work13[n+1]);
      for (j = 0; j < 6; j++) {
        virial3[0][j] += eng1*vg[i][j];
        virial3[1][j] += eng2*vg[i][j];
        virial3[2][j] += eng3*vg[i][j];
      }
      energy3[0] += eng1;
      energy3[1] += eng2;
      energy3[2] += eng3;
      n += 2;
    }
  } else {
    n = 0;
    double e3_0, e3_1, e3_2;
    e3_0 = e3_1 = e3_2 = 0.0;
    #pragma omp parallel for default(none) private(i) reduction(+:e3_0,e3_1,e3_2) 
    for (i = 0; i < nfft; i++) {
      e3_0 += greensfn[i] * (work11[2*i]*work11[2*i] + work11[2*i+1]*work11[2*i+1]);
      e3_1 += greensfn[i] * (work12[2*i]*work12[2*i] + work12[2*i+1]*work12[2*i+1]);
      e3_2 += greensfn[i] * (work13[2*i]*work13[2*i] + work13[2*i+1]*work13[2*i+1]);
    }
    energy3[0] += e3_0 * s2;
    energy3[1] += e3_1 * s2;
    energy3[2] += e3_2 * s2;
  }

  // *** Here after, we don't do any more for work11 b/c it only needs the energy *** //

  // ** Split the work across two threads, if we have two threads ** //
  if (comm->nthreads >= 2 && doit) {

#pragma omp parallel default(none) shared(scaleinv) private(i,j,k,n)
    {
#if defined (_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    // scale by 1/total-grid-pts to get rho(k)
    // multiply by Green's function to get V(k)
#pragma omp for
    for (i = 0; i < nfft; i++) {
      const double sg = scaleinv * greensfn[i];
      work12[2*i]   *= sg; 
      work12[2*i+1] *= sg;
      work13[2*i]   *= sg; 
      work13[2*i+1] *= sg;
    }
    
    // ** Split work here between thread 0 and 1 ** //
    if (tid == 0) {
      n = 0;
      for (i = 0; i<nfft; i++) {
	work22[n]   =  work12[n];
	work22[n+1] =  work12[n+1];
	n += 2;
      }

      fft2_2->compute(work22,work22,-1);

      n = 0;
      for (k = nzlo_in; k <= nzhi_in; k++)
        for (j = nylo_in; j <= nyhi_in; j++)
          for (i = nxlo_in; i <= nxhi_in; i++) {
	    u_brick2[k][j][i] = work22[n];
	    n += 2;
	  }

     } else if (tid == 1) {

      n = 0;
      for (i = 0; i<nfft; i++) {
	work23[n]   =  work13[n];
	work23[n+1] =  work13[n+1];
	n += 2;
      }

      fft2_3->compute(work23,work23,-1);

      n = 0;
      for (k = nzlo_in; k <= nzhi_in; k++)
        for (j = nylo_in; j <= nyhi_in; j++)
          for (i = nxlo_in; i <= nxhi_in; i++) {
	    u_brick3[k][j][i] = work23[n];
	    n += 2;
          }
     }

    } // close OMP parallel

  } else {

    // ** Only one thread or not doit ** //
    for (i = 0; i < nfft; i++) {
      const double sg = scaleinv * greensfn[i];
      work12[2*i]   *= sg; 
      work12[2*i+1] *= sg;
      work13[2*i]   *= sg; 
      work13[2*i+1] *= sg;
    }

    n = 0;
    for (i = 0; i<nfft; i++) {
      work22[n]   = work12[n];
      work22[n+1] = work12[n+1];
      work23[n]   = work13[n];
      work23[n+1] = work13[n+1];
      n += 2;
    }

    fft2->compute(work22,work22,-1);
    fft2->compute(work23,work23,-1);

    n = 0;
    for (k = nzlo_in; k <= nzhi_in; k++)
      for (j = nylo_in; j <= nyhi_in; j++)
        for (i = nxlo_in; i <= nxhi_in; i++) {
	  u_brick2[k][j][i] = work22[n];
	  u_brick3[k][j][i] = work23[n];
	  n += 2;
        }

  } // close regular
}

/* ----------------------------------------------------------------------
   pre-compute modified (Hockney-Eastwood) Coulomb Green's function
------------------------------------------------------------------------- */

void EVB_PPPMOMP::compute_gf_ik()
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

void EVB_PPPMOMP::compute_gf_ad()
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
   use swap list in forward order to acquire copy of all needed ghost grid pts
     based on GridComm::forward_comm().
------------------------------------------------------------------------- */

void EVB_PPPMOMP::forward_comm3(int which)
{
  int i,n;

  int nswap = cg->nswap;
  for (int m = 0; m < nswap; m++) {
    if (cg->swap[m].sendproc == me) 
      pack_forward3(which,cg_buf2,cg->swap[m].npack,cg->swap[m].packlist);
    else
      pack_forward3(which,cg_buf1,cg->swap[m].npack,cg->swap[m].packlist);

     if (cg->swap[m].sendproc != me) {
       MPI_Irecv(cg_buf2,2*(cg->nforward)*(cg->swap[m].nunpack),MPI_FFT_SCALAR,
                 cg->swap[m].recvproc,0,cg->gridcomm,&request);

       MPI_Send(cg_buf1,2*(cg->nforward)*(cg->swap[m].npack),MPI_FFT_SCALAR,
                (cg->swap[m].sendproc),0,cg->gridcomm);

       MPI_Wait(&request,&status);
     }

     unpack_forward3(which,cg_buf2,cg->swap[m].nunpack,cg->swap[m].unpacklist);
  }
}

/* ----------------------------------------------------------------------
   use swap list in reverse order to compute fully summed value
   for each owned grid pt that some other proc has copy of as a ghost grid pt
    based on GridComm::reverse_comm().
------------------------------------------------------------------------- */

void EVB_PPPMOMP::reverse_comm3(int which)
{
  int i,n;

  int nswap = cg->nswap;
  for (int m = nswap-1; m >= 0; m--) {
    if (cg->swap[m].recvproc == me) 
      pack_reverse3(which,cg_buf2,cg->swap[m].nunpack,cg->swap[m].unpacklist);
    else
      pack_reverse3(which,cg_buf1,cg->swap[m].nunpack,cg->swap[m].unpacklist);

    if (cg->swap[m].recvproc != me) {
      MPI_Irecv(cg_buf2,3*(cg->nreverse)*(cg->swap[m].npack),MPI_FFT_SCALAR,
		cg->swap[m].sendproc,0,cg->gridcomm,&request);

      MPI_Send(cg_buf1,3*(cg->nreverse)*(cg->swap[m].nunpack),MPI_FFT_SCALAR,
	       cg->swap[m].recvproc,0,cg->gridcomm);

      MPI_Wait(&request,&status);
    }
    
    unpack_reverse3(which,cg_buf2,cg->swap[m].npack,cg->swap[m].packlist);
  }
}


/* ----------------------------------------------------------------------
   pack own values to buf to send to another proc
------------------------------------------------------------------------- */

void EVB_PPPMOMP::pack_forward3(int flag, FFT_SCALAR *buf, int nlist, int *list)
{
  int n = 0;

  if (flag == FORWARD_IK) {
    FFT_SCALAR *xsrc = &vdx_brick2[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *ysrc = &vdy_brick2[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *zsrc = &vdz_brick2[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      buf[n++] = xsrc[list[i]];
      buf[n++] = ysrc[list[i]];
      buf[n++] = zsrc[list[i]];
    }
    
    xsrc = &vdx_brick3[nzlo_out][nylo_out][nxlo_out];
    ysrc = &vdy_brick3[nzlo_out][nylo_out][nxlo_out];
    zsrc = &vdz_brick3[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      buf[n++] = xsrc[list[i]];
      buf[n++] = ysrc[list[i]];
      buf[n++] = zsrc[list[i]];
    }
  } else if (flag == FORWARD_AD) {
    FFT_SCALAR *src = &u_brick2[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) buf[n++] = src[list[i]];

    src = &u_brick3[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) buf[n++] = src[list[i]];

  } else if (flag == FORWARD_IK_PERATOM) {
    error->all(FLERR,"FORWARD_IK_PERATOM not yet supported in EVB_PPPMOMP::pack_forward3()\n");
  } else if (flag == FORWARD_AD_PERATOM) {
    error->all(FLERR,"FORWARD_AD_PERATOM not yet supported in EVB_PPPMOMP::pack_forward3()\n");
  }
}

/* ----------------------------------------------------------------------
   unpack another proc's own values from buf and set own ghost values
------------------------------------------------------------------------- */

void EVB_PPPMOMP::unpack_forward3(int flag, FFT_SCALAR *buf, int nlist, int *list)
{
  int n = 0;

  if (flag == FORWARD_IK) {
    FFT_SCALAR *xdest = &vdx_brick2[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *ydest = &vdy_brick2[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *zdest = &vdz_brick2[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      xdest[list[i]] = buf[n++];
      ydest[list[i]] = buf[n++];
      zdest[list[i]] = buf[n++];
    }

    xdest = &vdx_brick3[nzlo_out][nylo_out][nxlo_out];
    ydest = &vdy_brick3[nzlo_out][nylo_out][nxlo_out];
    zdest = &vdz_brick3[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      xdest[list[i]] = buf[n++];
      ydest[list[i]] = buf[n++];
      zdest[list[i]] = buf[n++];
    }
  } else if (flag == FORWARD_AD) {
    FFT_SCALAR *dest = &u_brick2[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) dest[list[i]] = buf[n++];

    dest = &u_brick3[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) dest[list[i]] = buf[n++];

  } else if (flag == FORWARD_IK_PERATOM) {
    error->all(FLERR,"FORWARD_IK_PERATOM not yet supported in EVB_PPPMOMP::pack_forward3()\n");
  } else if (flag == FORWARD_AD_PERATOM) {
    error->all(FLERR,"FORWARD_AD_PERATOM not yet supported in EVB_PPPMOMP::pack_forward3()\n");
  }
}

/* ----------------------------------------------------------------------
   pack ghost values into buf to send to another proc
------------------------------------------------------------------------- */

void EVB_PPPMOMP::pack_reverse3(int flag, FFT_SCALAR *buf, int nlist, int *list)
{
  if (flag == REVERSE_RHO) {
    int n = 0;
    FFT_SCALAR *src = &density_brick1[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) buf[n++] = src[list[i]];

    src = &density_brick2[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) buf[n++] = src[list[i]];

    src = &density_brick3[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) buf[n++] = src[list[i]];
  }
}

/* ----------------------------------------------------------------------
   unpack another proc's ghost values from buf and add to own values
------------------------------------------------------------------------- */

void EVB_PPPMOMP::unpack_reverse3(int flag, FFT_SCALAR *buf, int nlist, int *list)
{
  if (flag == REVERSE_RHO) {
    int n = 0;
    FFT_SCALAR *dest = &density_brick1[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) dest[list[i]] += buf[n++];

    dest = &density_brick2[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) dest[list[i]] += buf[n++];

    dest = &density_brick3[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) dest[list[i]] += buf[n++];
  }
}


#endif
