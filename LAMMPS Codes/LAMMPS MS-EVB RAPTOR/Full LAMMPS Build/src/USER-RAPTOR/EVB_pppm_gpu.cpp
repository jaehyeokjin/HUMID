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
   Contributing authors: Chris Knight

    based on pppm_gpu.* and EVB_pppm.*
------------------------------------------------------------------------- */ 

#ifdef _RAPTOR_GPU

#include "lmptype.h"
#include "mpi.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "atom.h"
#include "comm.h"
#define _CRACKER_COMMGRID
#include "EVB_cracker.h"
#undef _CRACKER_COMMGRID
#include "neighbor.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "domain.h"
#include "fft3d_wrap.h"
#include "remap_wrap.h"
#include "gpu_extra.h"
#include "memory.h"
#include "error.h"
#include "timer.h"
#include "fix.h"

#include "EVB_pppm_gpu.h"
#include "EVB_engine.h"
#include "EVB_offdiag.h"
#include "EVB_complex.h"
#include "EVB_timer.h"

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

// external functions from cuda library for atom decomposition

#ifdef FFT_SINGLE
#define PPPM_GPU_API(api)  pppm_gpu_ ## api ## _f
#else
#define PPPM_GPU_API(api)  pppm_gpu_ ## api ## _d
#endif

FFT_SCALAR* PPPM_GPU_API(init)(const int nlocal, const int nall, FILE *screen,
                               const int order, const int nxlo_out,
                               const int nylo_out, const int nzlo_out,
                               const int nxhi_out, const int nyhi_out,
                               const int nzhi_out, FFT_SCALAR **rho_coeff,
                               FFT_SCALAR **_vd_brick,
                               const double slab_volfactor,
                               const int nx_pppm, const int ny_pppm,
                               const int nz_pppm, const bool split,
                               const bool respa, int &success);
void PPPM_GPU_API(clear)(const double poisson_time);
int PPPM_GPU_API(spread)(const int ago, const int nlocal, const int nall,
                      double **host_x, int *host_type, bool &success,
                      double *host_q, double *boxlo, const double delxinv,
                      const double delyinv, const double delzinv);
void PPPM_GPU_API(interp)(const FFT_SCALAR qqrd2e_scale);
double PPPM_GPU_API(bytes)();
void PPPM_GPU_API(forces)(double **f);

/* ---------------------------------------------------------------------- */

EVB_PPPMGPU::EVB_PPPMGPU(LAMMPS *lmp, int narg, char **arg) : EVB_PPPM(lmp, narg, arg)
{
  if (narg < 1) error->all(FLERR,"Illegal kspace_style EVB_PPPMGPU command");

  density_brick_gpu = vd_brick = NULL;
  kspace_split = false;
  im_real_space = false;
  old_nlocal = 0;

  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/* ----------------------------------------------------------------------
   free all memory 
------------------------------------------------------------------------- */

EVB_PPPMGPU::~EVB_PPPMGPU()
{
  PPPM_GPU_API(clear)(poisson_time);
}

/* ----------------------------------------------------------------------
   called once before run 
------------------------------------------------------------------------- */

void EVB_PPPMGPU::init()
{
  // PPPM init manages all arrays except density_brick_gpu and vd_brick
  //      thru its deallocate(), allocate()
  // NOTE: could free density_brick and vdxyz_brick after PPPM allocates them,
  //       before allocating db_gpu and vd_brick down below, if don't need,
  //       if do this, make sure to set them to NULL

  destroy_3d_offset(    density_brick_gpu,nzlo_out,nylo_out);
  destroy_3d_offset(vd_brick,nzlo_out,nylo_out);
  density_brick_gpu = vd_brick = NULL; 

  EVB_PPPM::init();  

  // insure no conflict with fix balance

  for (int i = 0; i < modify->nfix; i++)
    if (strcmp(modify->fix[i]->style,"balance") == 0)
      error->all(FLERR,"Cannot currently use evb_pppm/gpu with fix balance.");

  // unsupported option

  if (differentiation_flag == 1)
    error->all(FLERR,"Cannot (yet) do analytic differentiation with evb_pppm/gpu");

  // GPU precision specific init

  if (order>8)
    error->all(FLERR,"Cannot use order greater than 8 with pppm/gpu.");
  PPPM_GPU_API(clear)(poisson_time);

  bool respa_value = false;

  int success;
  FFT_SCALAR *data, *h_brick;
  h_brick = PPPM_GPU_API(init)(atom->nlocal, atom->nlocal+atom->nghost, screen,
                               order, nxlo_out, nylo_out, nzlo_out, nxhi_out,
                               nyhi_out, nzhi_out, rho_coeff, &data,
                               slab_volfactor,nx_pppm,ny_pppm,nz_pppm,
                               kspace_split,respa_value,success);

  GPU_EXTRA::check_flag(success,error,world);

  // allocate density_brick_gpu and vd_brick

  density_brick_gpu =
    create_3d_offset(nzlo_out,nzhi_out,nylo_out,nyhi_out,
                     nxlo_out,nxhi_out,"evb_pppm:density_brick_gpu",h_brick,1);

  vd_brick =
    create_3d_offset(nzlo_out,nzhi_out,nylo_out,nyhi_out,
                     nxlo_out,nxhi_out,"evb_pppm:vd_brick",data,4);

  poisson_time = 0.0;
}

/* ----------------------------------------------------------------------
   create array using offsets from pinned memory allocation
------------------------------------------------------------------------- */

FFT_SCALAR ***EVB_PPPMGPU::create_3d_offset(int n1lo, int n1hi, int n2lo, int n2hi,
					    int n3lo, int n3hi, const char *name,
					    FFT_SCALAR *data, int vec_length)
{
  int i,j;
  int n1 = n1hi - n1lo + 1;
  int n2 = n2hi - n2lo + 1;
  int n3 = n3hi - n3lo + 1;

  FFT_SCALAR **plane = (FFT_SCALAR **)
    memory->smalloc(n1*n2*sizeof(FFT_SCALAR *),name);
  FFT_SCALAR ***array = (FFT_SCALAR ***)
    memory->smalloc(n1*sizeof(FFT_SCALAR **),name);

  int n = 0;
  for (i = 0; i < n1; i++) {
    array[i] = &plane[i*n2];
    for (j = 0; j < n2; j++) {
      plane[i*n2+j] = &data[n];
      n += n3*vec_length;
    }
  }

  for (i = 0; i < n1*n2; i++) array[0][i] -= n3lo*vec_length;
  for (i = 0; i < n1; i++) array[i] -= n2lo;
  return array-n1lo;
}

/* ----------------------------------------------------------------------
   3d memory offsets
------------------------------------------------------------------------- */

void EVB_PPPMGPU::destroy_3d_offset(FFT_SCALAR ***array, int n1_offset,
				    int n2_offset)
{
  if (array == NULL) return;
  memory->sfree(&array[n1_offset][n2_offset]);
  memory->sfree(array + n1_offset);
}

/*************************************************************************/

/* ----------------------------------------------------------------------
   find center grid pt for each of my cplx particles
   check that full stencil for the particle will fit in my 3d brick
   store central grid pt indices in part2grid array 
------------------------------------------------------------------------- */

void EVB_PPPMGPU::particle_map_cplx()
{
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  double **x = atom->x;
  int nlocal = atom->nlocal;

  const int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  const int * cplx_list = evb_engine->evb_complex->cplx_list;

  int flag = 0;
  for (int i = 0; i < nlocal_cplx; i++) {
    const int iatm = cplx_list[i];
    
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // current particle coord can be outside global and local box
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

    const double ddx = (x[iatm][0]- boxlox) * delxinv;
    const double ddy = (x[iatm][1]- boxloy) * delyinv;
    const double ddz = (x[iatm][2]- boxloz) * delzinv;

    const int nx = static_cast<int> (ddx + shift) - OFFSET;
    const int ny = static_cast<int> (ddy + shift) - OFFSET;
    const int nz = static_cast<int> (ddz + shift) - OFFSET;

    part2grid[iatm][0] = nx;
    part2grid[iatm][1] = ny;
    part2grid[iatm][2] = nz;

    const FFT_SCALAR dx = nx+shiftone - ddx;
    const FFT_SCALAR dy = ny+shiftone - ddy;
    const FFT_SCALAR dz = nz+shiftone - ddz; 
    
    part2grid_dr[iatm][0] = dx;
    part2grid_dr[iatm][1] = dy;
    part2grid_dr[iatm][2] = dz; 
  }
  
  if (flag) error->all(FLERR,"Out of range atoms - cannot compute EVB_PPPM");
}

/*************************************************************************/

void EVB_PPPMGPU::map2density_one_subtract(int id)
{
  double *q = atom->q;

  // Subtracts the contrinution for a given id
  nx = part2grid[id][0];
  ny = part2grid[id][1];
  nz = part2grid[id][2];
  
  // (dx,dy,dz) = distance to "lower left" grid pt
  
  compute_rho1d(part2grid_dr[id][0], part2grid_dr[id][1], part2grid_dr[id][2]);
  
  // (mx,my,mz) = global coords of moving stencil pt
  
  const FFT_SCALAR z0 = delvolinv * q[id];
  for (int n = nlower; n <= nupper; n++) {
    const FFT_SCALAR y0 = z0*rho1d[2][n];
    for (int m = nlower; m <= nupper; m++) {
      const FFT_SCALAR x0 = y0*rho1d[1][m];
      for (int l = nlower; l <= nupper; l++) {
	density_brick_gpu[n+nz][m+ny][l+nx] -= x0*rho1d[0][l];
      } // Loop mx
    } // Loop my
  } // Loop mz

}

/*************************************************************************/

void EVB_PPPMGPU::compute_env(int vflag)
{
  TIMER_STAMP(EVB_PPPM, compute_env);

  nlocal = atom->nlocal;

  int nago;
  if(kspace_split) {
    if(im_real_space) return;
    if(nlocal > old_nlocal) {
      nago = 0;
      old_nlocal = nlocal;
    } else nago = 1;
  } else nago = neighbor->ago;
  
  q = atom->q;
  x = atom->x;
  f = atom->f;
  
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int has_cplx_atom = evb_engine->has_complex_atom;

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;

  // Make the density all at once
  bool success = true;
  int flag=PPPM_GPU_API(spread)(nago, atom->nlocal, atom->nlocal +
                             atom->nghost, atom->x, atom->type, success,
                             atom->q, domain->boxlo, delxinv, delyinv,
                             delzinv);
  if (!success) error->one(FLERR,"Insufficient memory on accelerator");
  if (flag != 0) error->one(FLERR,"Out of range atoms - cannot compute PPPM");

  // Subtract density from complex
  if(has_cplx_atom) {
    particle_map_cplx();
    for(int i=0; i<nlocal_cplx; ++i) map2density_one_subtract(cplx_list[i]);
  }

  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }

  double t3 = MPI_Wtime();

  // all procs communicate density values from their ghost cells
  //  to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition

  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();

  poisson_energy(vflag);

  poisson_time += MPI_Wtime() - t3;

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

  TIMER_CLICK(EVB_PPPM, compute_env); 
}

/*************************************************************************/

void EVB_PPPMGPU::compute_cplx(int vflag)
{
  TIMER_STAMP(EVB_PPPM, compute_cplx);

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;

  // Make the density all at once
  // Need to save ENV density on GPU instead recalculating entire density, otherwise
  //  forces need to be calculated on CPU.
  bool success = true;
  int nago = neighbor->ago;

  TIMER_STAMP(EVB_PPPM, compute_cplx_gpu_spread);

  int flag=PPPM_GPU_API(spread)(nago, atom->nlocal, atom->nlocal +
                             atom->nghost, atom->x, atom->type, success,
                             atom->q, domain->boxlo, delxinv, delyinv,
                             delzinv);
  if (!success) error->one(FLERR,"Insufficient memory on accelerator");
  if (flag != 0) error->one(FLERR,"Out of range atoms - cannot compute PPPM");

  TIMER_CLICK(EVB_PPPM, compute_cplx_gpu_spread);


  TIMER_STAMP(EVB_PPPM, compute_cplx_cpu_reverse_comm);
  cg->reverse_comm(this,REVERSE_RHO);
  TIMER_CLICK(EVB_PPPM, compute_cplx_cpu_reverse_comm);

  
  TIMER_STAMP(EVB_PPPM, compute_cplx_cpu_brick2fft);
  brick2fft();
  TIMER_CLICK(EVB_PPPM, compute_cplx_cpu_brick2fft);

  TIMER_STAMP(EVB_PPPM, compute_cplx_cpu_poisson);
  poisson(true,vflag);
  TIMER_CLICK(EVB_PPPM, compute_cplx_cpu_poisson);
 
  // Don't calculate forces here in SCI simulations
  if(evb_engine->ncomplex == 1) {

    TIMER_STAMP(EVB_PPPM, compute_cplx_cpu_forward_comm);
    if (differentiation_flag == 1) cg->forward_comm(this,FORWARD_AD);
    else cg->forward_comm(this,FORWARD_IK);
    TIMER_CLICK(EVB_PPPM, compute_cplx_cpu_forward_comm);

    TIMER_STAMP(EVB_PPPM, compute_cplx_gpu_interp);
    FFT_SCALAR qscale = force->qqrd2e * scale;
    PPPM_GPU_API(interp)(qscale);
    TIMER_CLICK(EVB_PPPM, compute_cplx_gpu_interp);

    if(kspace_split) PPPM_GPU_API(forces)(atom->f);
  }
  
  qsqsum = evb_engine->qsqsum_env + evb_engine->evb_complex->qsqsum;
  reduce_ev(vflag,true);

  if(slabflag) slabcorr_cplx();

  energy -= env_energy;

  TIMER_CLICK(EVB_PPPM, compute_cplx);
}

/*************************************************************************/

void EVB_PPPMGPU::compute_exch(int vflag)
{
  TIMER_STAMP(EVB_PPPM, compute_exch);
  
  off_diag_energy = 0.0;
  error->universe_all(FLERR,"Must use real space approx. for off-diagonals with PPPM on GPUs");
 
  TIMER_CLICK(EVB_PPPM, compute_exch); 
}

/* ----------------------------------------------------------------------
   ghost-swap to accumulate full density in brick decomposition 
   remap density from 3d brick decomposition to FFT decomposition
------------------------------------------------------------------------- */

void EVB_PPPMGPU::brick2fft()
{
  int i,n,ix,iy,iz;

  // copy grabs inner portion of density from 3d brick
  // remap could be done as pre-stage of FFT,
  //   but this works optimally on only double values, not complex values

  n = 0;
  for (iz = nzlo_in; iz <= nzhi_in; iz++)
    for (iy = nylo_in; iy <= nyhi_in; iy++)
      for (ix = nxlo_in; ix <= nxhi_in; ix++)
	density_fft[n++] = density_brick_gpu[iz][iy][ix];

  remap->perform(density_fft,density_fft,work1);
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik
------------------------------------------------------------------------- */

void EVB_PPPMGPU::poisson_ik(int vflag)
{
  int i,j,k,n;
  double eng;

  // transform charge density (r -> k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = ZEROF;
  }

  fft1->compute(work1,work1,1);

  // global energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  if (vflag) {
    n = 0;
    for (i = 0; i < nfft; i++) {
      eng = s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
      for (j = 0; j < 6; j++) virial[j] += eng*vg[i][j];
      energy += eng;
      n += 2;
    }
  } else {
    n = 0;
    for (i = 0; i < nfft; i++) {
      energy += greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
      n += 2;
    }
    energy *= s2;
  }

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] *= scaleinv * greensfn[i];
    work1[n++] *= scaleinv * greensfn[i];
  }

  // compute gradients of V(r) in each of 3 dims by transformimg -ik*V(k)
  // FFT leaves data in 3d brick decomposition
  // copy it into inner portion of vdx,vdy,vdz arrays

  // x direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = fkx[i]*work1[n+1];
        work2[n+1] = -fkx[i]*work1[n];
        n += 2;
      }

  fft2->compute(work2,work2,-1);

  n = 0;
  int x_hi = nxhi_in * 4 + 3;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in * 4; i < x_hi; i+=4) {
        vd_brick[k][j][i] = work2[n];
        n += 2;
      }
  
  // y direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = fky[j]*work1[n+1];
        work2[n+1] = -fky[j]*work1[n];
        n += 2;
      }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in * 4 + 1; i < x_hi; i+=4) {
        vd_brick[k][j][i] = work2[n];
        n += 2;
      }

  // z direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = fkz[k]*work1[n+1];
        work2[n+1] = -fkz[k]*work1[n];
        n += 2;
      }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in * 4 + 2; i < x_hi; i+=4) {
        vd_brick[k][j][i] = work2[n];
        n += 2;
      }
}

/* ----------------------------------------------------------------------
   pack own values to buf to send to another proc
------------------------------------------------------------------------- */

void EVB_PPPMGPU::pack_forward(int flag, FFT_SCALAR *buf, int nlist, int *list)
{
  int n = 0;

  if (flag == FORWARD_IK) {
    int offset;
    FFT_SCALAR *src = &vd_brick[nzlo_out][nylo_out][4*nxlo_out];
    for (int i = 0; i < nlist; i++) {
      offset = 4 * list[i];
      buf[n++] = src[offset++];
      buf[n++] = src[offset++];
      buf[n++] = src[offset];
    }
  } else if (flag == FORWARD_AD) {
    FFT_SCALAR *src = &u_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      buf[i] = src[list[i]];
  } else if (flag == FORWARD_IK_PERATOM) {
    FFT_SCALAR *esrc = &u_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      if (eflag_atom) buf[n++] = esrc[list[i]];
      if (vflag_atom) {
        buf[n++] = v0src[list[i]];
        buf[n++] = v1src[list[i]];
        buf[n++] = v2src[list[i]];
        buf[n++] = v3src[list[i]];
        buf[n++] = v4src[list[i]];
        buf[n++] = v5src[list[i]];
      }
    }
  } else if (flag == FORWARD_AD_PERATOM) {
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      buf[n++] = v0src[list[i]];
      buf[n++] = v1src[list[i]];
      buf[n++] = v2src[list[i]];
      buf[n++] = v3src[list[i]];
      buf[n++] = v4src[list[i]];
      buf[n++] = v5src[list[i]];
    }
  }
}

/* ----------------------------------------------------------------------
   unpack another proc's own values from buf and set own ghost values
------------------------------------------------------------------------- */

void EVB_PPPMGPU::unpack_forward(int flag, FFT_SCALAR *buf, int nlist, int *list)
{
  int n = 0;

  if (flag == FORWARD_IK) {
    int offset;
    FFT_SCALAR *dest = &vd_brick[nzlo_out][nylo_out][4*nxlo_out];
    for (int i = 0; i < nlist; i++) {
      offset = 4 * list[i];
      dest[offset++] = buf[n++];
      dest[offset++] = buf[n++];
      dest[offset]   = buf[n++];
    }
  } else if (flag == FORWARD_AD) {
    FFT_SCALAR *dest = &u_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      dest[list[i]] = buf[i];
  } else if (flag == FORWARD_IK_PERATOM) {
    FFT_SCALAR *esrc = &u_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      if (eflag_atom) esrc[list[i]] = buf[n++];
      if (vflag_atom) {
        v0src[list[i]] = buf[n++];
        v1src[list[i]] = buf[n++];
        v2src[list[i]] = buf[n++];
        v3src[list[i]] = buf[n++];
        v4src[list[i]] = buf[n++];
        v5src[list[i]] = buf[n++];
      }
    }
  } else if (flag == FORWARD_AD_PERATOM) {
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      v0src[list[i]] = buf[n++];
      v1src[list[i]] = buf[n++];
      v2src[list[i]] = buf[n++];
      v3src[list[i]] = buf[n++];
      v4src[list[i]] = buf[n++];
      v5src[list[i]] = buf[n++];
    }
  }
}

/* ----------------------------------------------------------------------
   pack ghost values into buf to send to another proc
------------------------------------------------------------------------- */

void EVB_PPPMGPU::pack_reverse(int flag, FFT_SCALAR *buf, int nlist, int *list)
{
  if (flag == REVERSE_RHO) {
    FFT_SCALAR *src = &density_brick_gpu[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      buf[i] = src[list[i]];
  }
}

/* ----------------------------------------------------------------------
   unpack another proc's ghost values from buf and add to own values
------------------------------------------------------------------------- */

void EVB_PPPMGPU::unpack_reverse(int flag, FFT_SCALAR *buf, int nlist, int *list)
{
  if (flag == REVERSE_RHO) {
    FFT_SCALAR *dest = &density_brick_gpu[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      dest[list[i]] += buf[i];
  } 
}

/* ----------------------------------------------------------------------
   perform and time the 1d FFTs required for N timesteps
------------------------------------------------------------------------- */

int EVB_PPPMGPU::timing_1d(int n, double &time1d)
{
  if (im_real_space) {
    time1d = 1.0;
    return 4;
  }
  EVB_PPPM::timing_1d(n,time1d);
  return 4;
}

/* ----------------------------------------------------------------------
   perform and time the 3d FFTs required for N timesteps
------------------------------------------------------------------------- */

int EVB_PPPMGPU::timing_3d(int n, double &time3d)
{
  if (im_real_space) {
    time3d = 1.0;
    return 4;
  }
  EVB_PPPM::timing_3d(n,time3d);
  return 4;
}

/* ----------------------------------------------------------------------
   adjust PPPM coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void EVB_PPPMGPU::setup()
{
  if (im_real_space) return;
  EVB_PPPM::setup();
}


/* ----------------------------------------------------------------------
   memory usage of local arrays 
------------------------------------------------------------------------- */

double EVB_PPPMGPU::memory_usage()
{
  double bytes = EVB_PPPM::memory_usage();

  // NOTE: add tallying here for density_brick_gpu and vd_brick
  //       could subtract out density_brick and vdxyz_brick if freed them above
  //       it the net efffect is zero, do nothing

  return bytes + PPPM_GPU_API(bytes)();
}

#endif
