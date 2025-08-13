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

#include "EVB_pppm_electrode.h"
#include "EVB_engine.h"
#include "EVB_effpair.h"
#include "EVB_offdiag.h"
#include "EVB_complex.h"
#include "EVB_timer.h"

#include "math_const.h"
#include "math_special.h"

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

EVB_PPPMELECTRODE::EVB_PPPMELECTRODE(LAMMPS *lmp, int narg, char **arg) : PPPM_Electrode(lmp, narg, arg)
{
  env_density_brick = NULL;
  part2grid_dr = NULL;

  do_sci_compute_cplx_other = NULL;
  energy_sci_compute_cplx_other = NULL;

  do_sci_compute_cplx_self = NULL;
  energy_sci_compute_cplx_self = NULL;
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

EVB_PPPMELECTRODE::~EVB_PPPMELECTRODE()
{
  memory->destroy(part2grid_dr);

  memory->destroy(do_sci_compute_cplx_other);
  memory->destroy(energy_sci_compute_cplx_other);

  memory->destroy(do_sci_compute_cplx_self);
  memory->destroy(energy_sci_compute_cplx_self);
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::compute_env(int vflag)
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

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;

  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }
  
  // Calculate the ENV density map;
  FFT_SCALAR ***save_density = density_brick;
  density_brick = env_density_brick;
  clear_density();  

  // Make the density all at once
  make_rho();
  
  if(has_cplx_atom) for(int i=0; i<nlocal_cplx; ++i) map2density_one_subtract(cplx_list[i]);

  density_brick = save_density;  
  load_env_density();

  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();

  poisson_energy(vflag);
  
  qsqsum = evb_engine->qsqsum_env = evb_engine->qsqsum_sys - evb_engine->evb_complex->qsqsum;
  qsqsum *= 3.0; // Electrode model

  reduce_ev(vflag,true);

  env_energy = energy;

  if (vflag) for (int i = 0; i < 6; i++) virial[i] = 0.0;

  // Environment contribution to dipole for slab correction
  if(slabflag) {
    double *q = atom->q;
    double **x = atom->x;
    
    double dipole = 0.0;
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

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::compute_env_density(int vflag)
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

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;

  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }
  
  // Calculate the ENV density map;
  FFT_SCALAR ***save_density = density_brick;
  density_brick = env_density_brick;
  clear_density();  

  // Make the density all at once
  make_rho();
  
  if(has_cplx_atom) for(int i=0; i<nlocal_cplx; ++i) map2density_one_subtract(cplx_list[i]);

  density_brick = save_density;  
  load_env_density();
  
  qsqsum = evb_engine->qsqsum_env = 0.0;
  env_energy = energy;

  // Environment contribution to dipole for slab correction
  if(slabflag) {
    double *q = atom->q;
    double **x = atom->x;
    
    double dipole = 0.0;
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

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::compute_cplx(int vflag)
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
 
  // Don't calculate forces here in SCI simulations during compute(), only initialize().
  if(evb_engine->ncomplex == 1) {
    poisson(vflag);

    if (differentiation_flag == 1) cg->forward_comm(this,FORWARD_AD);
    else cg->forward_comm(this,FORWARD_IK);
    fieldforce();

  } else if( (evb_engine->SCI_KSPACE_flag == KSPACE_DEFAULT || evb_engine->SCI_KSPACE_flag == PPPM_HF_FORCES) &&
	     evb_engine->engine_indicator == ENGINE_INDICATOR_INITIALIZE) {
    poisson(vflag);
    
    // Only accumulate forces on complex atoms
    if (differentiation_flag == 1) {
      cg->forward_comm(this,FORWARD_AD);
      for(int i=0; i<nlocal_cplx; i++) field2force_one_ad(cplx_list[i],false);
    } else {
      cg->forward_comm(this,FORWARD_IK);
      for(int i=0; i<nlocal_cplx; i++) field2force_one_ik(cplx_list[i],false);
    }

  } else poisson_energy(vflag);
  
  qsqsum = evb_engine->qsqsum_env + evb_engine->evb_complex->qsqsum;
  qsqsum *= 3.0; // Electrode model
  reduce_ev(vflag,true);

  // Don't calculate slab correction here in SCI simulations
  if(slabflag && evb_engine->ncomplex == 1) slabcorr_cplx();

  energy -= env_energy;
 
  TIMER_CLICK(EVB_PPPM, compute_cplx);
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::compute_cplx_eff(int vflag)
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
 
  poisson(vflag);
    
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

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::init()
{
  PPPM_Electrode::init();

  qqrd2e = force->qqrd2e;
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::allocate()
{
  PPPM_Electrode::allocate();

  memory->create3d_offset(env_density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
			  nxlo_out,nxhi_out,"EVB_PPPM:env_density_brick");
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::deallocate()
{
  PPPM_Electrode::deallocate();
  
  memory->destroy3d_offset(env_density_brick,nzlo_out,nylo_out,nxlo_out);
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::clear_density()
{
  FFT_SCALAR *vec = &density_brick[nzlo_out][nylo_out][nxlo_out];
  memset(vec, ZEROF, sizeof(FFT_SCALAR)*ngrid);
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::load_env_density()
{
  memcpy(&density_brick[nzlo_out][nylo_out][nxlo_out],
         &env_density_brick[nzlo_out][nylo_out][nxlo_out],
		 sizeof(FFT_SCALAR)*ngrid);
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::evb_setup()
{
  nlocal = atom->nlocal;

  // extend size of per-atom arrays if necessary

  if (nlocal > nmax) {
    memory->destroy(part2grid);
    memory->destroy(part2grid_dr);
    nmax = atom->nmax;
    memory->create(part2grid,nmax,3,"EVB_PPPM:part2grid");
    memory->create(part2grid_dr,nmax,3,"EVB_PPPM:part2grid_dr");
  }
}

/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid 
------------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::make_rho()
{
  const double * const q = atom->q;
  const double * const * const x = atom->x;
  const int nlocal = atom->nlocal;
  
  // set up clear 3d density array
  FFT_SCALAR * const * const * const db = &(density_brick[0]);
  memset(&(db[nzlo_out][nylo_out][nxlo_out]),0,ngrid*sizeof(FFT_SCALAR));

  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  if (order == 5) {
    for (int i = 0; i < nlocal; i++) {
      const double ddx = (x[i][0]- boxlox) * delxinv;
      const double ddy = (x[i][1]- boxloy) * delyinv;
      const double ddz = (x[i][2]- boxloz) * delzinv;
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
      
      const FFT_SCALAR z0 = delvolinv * q[i];
      for (int n = nlower; n <= nupper; n++) {
	const FFT_SCALAR y0 = z0*rho1d[2][n];
	for (int m = nlower; m <= nupper; m++) {
	  const FFT_SCALAR x0 = y0*rho1d[1][m];
	  for (int l = nlower; l <= nupper; l++) {
	    db[n+nz][m+ny][l+nx] += x0*rho1d[0][l];
	  }
	}
      }
    } // for(i<nlocal)
  } else {
    for (int i = 0; i < nlocal; i++) {
      const double ddx = (x[i][0]-boxlox)*delxinv;
      const double ddy = (x[i][1]-boxloy)*delyinv;
      const double ddz = (x[i][2]-boxloz)*delzinv;
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
      compute_rho1d(dx,dy,dz);
      
      const FFT_SCALAR z0 = delvolinv * q[i];
      for (int n = nlower; n <= nupper; n++) {
	const FFT_SCALAR y0 = z0*rho1d[2][n];
	for (int m = nlower; m <= nupper; m++) {
	  const FFT_SCALAR x0 = y0*rho1d[1][m];
	  for (int l = nlower; l <= nupper; l++) {
	    db[n+nz][m+ny][l+nx] += x0*rho1d[0][l];
	  }
	}
      }

    } // for (i<nlocal)
  } // if (order == 5)
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::map2density_one(int id)
{
  double *q = atom->q;

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
	density_brick[n+nz][m+ny][l+nx] += x0*rho1d[0][l];
      } // Loop mx
    } // Loop my
  } // Loop mz
}

/*************************************************************************/

void EVB_PPPMELECTRODE::map2density_one(int id, int WHICH)
{
  double * q;
  if(WHICH == Q_ATOM)  q = atom->q;
  else if(WHICH == Q_EFFECTIVE) q = evb_engine->evb_effpair->q;

  nx = part2grid[id][0];
  ny = part2grid[id][1];
  nz = part2grid[id][2];
  
  // (dx,dy,dz) = distance to "lower left" grid pt
  
  compute_rho1d(part2grid_dr[id][0], part2grid_dr[id][1], part2grid_dr[id][2]);
  
  // (mx,my,mz) = global coords of moving stencil pt   

  z0 = delvolinv * q[id];
  for (int n = nlower; n <= nupper; n++) {
    mz = n+nz;
    y0 = z0*rho1d[2][n];
    
    for (int m = nlower; m <= nupper; m++) {
      my = m+ny;
      x0 = y0*rho1d[1][m];
      
      for (int l = nlower; l <= nupper; l++) {
	mx = l+nx;
	density_brick[mz][my][mx] += x0*rho1d[0][l];
      } // Loop mx
    } // Loop my
  } // Loop mz
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::map2density_one_subtract(int id)
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
	density_brick[n+nz][m+ny][l+nx] -= x0*rho1d[0][l];
      } // Loop mx
    } // Loop my
  } // Loop mz
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::poisson_energy(int vflag)
{
  int n;
  double eng;

  // transform charge density (r -> k) 

  n = 0;
  for (int i=0; i<nfft; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = ZEROF;
  }
  
  fft1->compute(work1,work1,1);

  /* Electrode Model */
  n = 0;
  for(int i = 0; i < nfft; i++) {
    const int off = greensfn_rev[i];

    const double rl1 = img1_rl[i];
    const double rl2 = img2_rl[i];
    const double im1 = img1_im[i];
    const double im2 = img2_im[i];

    const double w1 = work1[off];
    const double w2 = work1[off+1];

    work1_img[n]  = rl1*w1 - im1*w2 + rl2*w1 - im2*w2;
    work1_img[n+1]= rl1*w2 + im1*w1 + rl2*w2 + im2*w1;

    n += 2;
  }
  
  /* End */

  // compute energy and virial contribution
  
  double sqr;
  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  n = 0;
  if (vflag) {
    for (int i=0; i<nfft; ++i) {
      const double w1 = work1[n];
      const double w2 = work1[n+1];

      eng = s2 * greensfn[i];
      eng *= (w1*w1 + w2*w2 + w1*work1_img[n] + w2*work1_img[n+1]);
      for (int j=0; j<6; ++j) virial[j] += eng*vg[i][j];
      energy += eng;
      n += 2;
    }
  } else {
    for (int i=0; i<nfft; ++i) {
      const double w1 = work1[n];
      const double w2 = work1[n+1];

      energy += greensfn[i] * (w1*w1 + w2*w2 +w1*work1_img[n] + w2*work1_img[n+1]);
      n += 2;
    }
    energy *= s2;
  }

}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver
------------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::poisson(int vflag)
{
  if (differentiation_flag == 1) {
    //poisson_ad(vflag);
    error->all(FLERR,"EVB_PPPMELECTRODE does not support ad.");
  } else poisson_ik(vflag);
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik
------------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::poisson_ik(int vflag)
{
  int i,j,k,n;

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);

  poisson_energy(vflag);

  /* Eletrode Model */
  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n] += work1_img[n];
    work1[n+1] += work1_img[n+1];
    n += 2;
  }
  /* End */

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
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdx_brick[k][j][i] = work2[n];
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
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdy_brick[k][j][i] = work2[n];
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
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdz_brick[k][j][i] = work2[n];
        n += 2;
      }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles
------------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::fieldforce()
{
  if (differentiation_flag == 1) {
    //fieldforce_ad();
    error->all(FLERR,"EVB_PPPMELECTRODE does not support ad.");
  } else fieldforce_ik();
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ik
------------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::fieldforce_ik()
{
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR ekx,eky,ekz;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;

  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++) {
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx+shiftone - (x[i][0] - boxlox) * delxinv;
    dy = ny+shiftone - (x[i][1] - boxloy) * delyinv;
    dz = nz+shiftone - (x[i][2] - boxloz) * delzinv;

    compute_rho1d(dx,dy,dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n+nz;
      z0 = rho1d[2][n];
      for (m = nlower; m <= nupper; m++) {
        my = m+ny;
        y0 = z0*rho1d[1][m];
        for (l = nlower; l <= nupper; l++) {
          mx = l+nx;
          x0 = y0*rho1d[0][l];
          ekx -= x0*vdx_brick[mz][my][mx];
          eky -= x0*vdy_brick[mz][my][mx];
          ekz -= x0*vdz_brick[mz][my][mx];
        }
      }
    }

    // convert E-field to force

    const double qfactor = force->qqrd2e * scale * q[i];
    f[i][0] += qfactor*ekx;
    f[i][1] += qfactor*eky;
    if (slabflag != 2) f[i][2] += qfactor*ekz;
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my ENV particles
------------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::fieldforce_env()
{
  if (differentiation_flag == 1) {
    //fieldforce_env_ad();
    error->all(FLERR,"EVB_PPPMELECTRODE does not support ad.");
  } else fieldforce_env_ik();
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ik
------------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::fieldforce_env_ik()
{
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR ekx,eky,ekz;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;

  int nlocal = atom->nlocal;

  int * is_cplx_atom = evb_engine->complex_atom;

  for (i = 0; i < nlocal; i++) {
    if(is_cplx_atom[i]) continue; // Skip complex atoms

    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx+shiftone - (x[i][0] - boxlox) * delxinv;
    dy = ny+shiftone - (x[i][1] - boxloy) * delyinv;
    dz = nz+shiftone - (x[i][2] - boxloz) * delzinv;

    compute_rho1d(dx,dy,dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n+nz;
      z0 = rho1d[2][n];
      for (m = nlower; m <= nupper; m++) {
        my = m+ny;
        y0 = z0*rho1d[1][m];
        for (l = nlower; l <= nupper; l++) {
          mx = l+nx;
          x0 = y0*rho1d[0][l];
          ekx -= x0*vdx_brick[mz][my][mx];
          eky -= x0*vdy_brick[mz][my][mx];
          ekz -= x0*vdz_brick[mz][my][mx];
        }
      }
    }

    // convert E-field to force

    const double qfactor = force->qqrd2e * scale * q[i];
    f[i][0] += qfactor*ekx;
    f[i][1] += qfactor*eky;
    if (slabflag != 2) f[i][2] += qfactor*ekz;
  }
}

/*************************************************************************/

void EVB_PPPMELECTRODE::field2force_one_ik(int id, bool Aflag)
{

  FFT_SCALAR ekx,eky,ekz;

  nx = part2grid[id][0];
  ny = part2grid[id][1];
  nz = part2grid[id][2];
  
  compute_rho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
  
  ekx = eky = ekz = ZEROF;
  for (int n = nlower; n <= nupper; n++) {
    mz = n+nz;
    z0 = rho1d[2][n];
    for (int m = nlower; m <= nupper; m++) {
      my = m+ny;
      y0 = z0*rho1d[1][m];
      for (int l = nlower; l <= nupper; l++) {
	mx = l+nx;
	x0 = y0*rho1d[0][l];
	ekx -= x0*vdx_brick[mz][my][mx];;
	eky -= x0*vdy_brick[mz][my][mx];;
	ekz -= x0*vdz_brick[mz][my][mx];;
      }
    }
  }
  
  // convert E-field to force
  double qfactor = qqrd2e * scale * q[id];
  if(Aflag) qfactor *= A_Rq;
  
  f[id][0] += qfactor * ekx;
  f[id][1] += qfactor * eky;
  if (slabflag != 2) f[id][2] += qfactor * ekz;
}

/*************************************************************************/

void EVB_PPPMELECTRODE::field2force_one_ad(int id, bool Aflag)
{
  error->all(FLERR,"EVB_PPPMELECTRODE does not support ad.");
  // double s1, s2, s3;
  // double sf = 0.0;
  // double *prd;

  // if(triclinic == 0) prd = domain->prd;
  // else prd = domain->prd_lamda;
  
  // const double hx_inv = nx_pppm / prd[0];
  // const double hy_inv = ny_pppm / prd[1];
  // const double hz_inv = nz_pppm / prd[2];

  // nx = part2grid[id][0];
  // ny = part2grid[id][1];
  // nz = part2grid[id][2];
  
  // compute_rho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
  // compute_drho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
  
  // ekx = eky = ekz = ZEROF;
  // for (int n = nlower; n <= nupper; n++) {
  //   mz = n+nz;
  //   for (int m = nlower; m <= nupper; m++) {
  //     my = m+ny;
  //     for (int l = nlower; l <= nupper; l++) {
  // 	mx = l+nx;
  // 	ekx += drho1d[0][l] *  rho1d[1][m] *  rho1d[2][n] * u_brick[mz][my][mx];
  // 	eky +=  rho1d[0][l] * drho1d[1][m] *  rho1d[2][n] * u_brick[mz][my][mx];
  // 	ekz +=  rho1d[0][l] *  rho1d[1][m] * drho1d[2][n] * u_brick[mz][my][mx];
  //     }
  //   }
  // }
  // ekx *= hx_inv;
  // eky *= hy_inv;
  // ekz *= hz_inv;
  
  // // convert E-field to force and subtract self forces
  // double qfactor = qqrd2e * scale;
  // if(Aflag) qfactor *= A_Rq;

  // s1 = x[id][0] * hx_inv;
  // s2 = x[id][1] * hy_inv;
  // s3 = x[id][2] * hz_inv;
  // sf = sf_coeff[0] * sin(2 * MY_PI * s1);
  // sf += sf_coeff[1] * sin(4 * MY_PI * s1);
  // sf *= 2 * q[id] * q[id];
  // f[id][0] += qfactor * (ekx * q[id] - sf);

  // sf = sf_coeff[2] * sin(2 * MY_PI * s2);
  // sf += sf_coeff[3] * sin(4 * MY_PI * s2);
  // sf *= 2 * q[id] * q[id];
  // f[id][1] += qfactor * (eky * q[id] - sf);

  // if (slabflag != 2) {
  //   sf = sf_coeff[4] * sin(2 * MY_PI * s3);
  //   sf += sf_coeff[5] * sin(4 * MY_PI * s3);
  //   sf *= 2 * q[id] * q[id];
  //   f[id][2] += qfactor * (ekz * q[id] - sf);
  // }
}

/* ---------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::reduce_ev(int vflag, bool Aflag)
{
  energy *= 0.5*volume;
  if(comm->me==0) energy -= g_ewald*qsqsum/3.0/MY_PIS + MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
  
  energy *= qqrd2e;

  // sum virial across procs

  if (vflag) {
    double virial_all[6];
    MPI_Allreduce(virial,virial_all,6,MPI_DOUBLE,MPI_SUM,world);
    double pre_factor = 0.5*qqrd2e*volume;
    if(Aflag) pre_factor *= A_Rq;
    for (int i=0; i<6; i++) virial[i] = pre_factor*virial_all[i];
  }
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if 
   adequate empty space is left between repeating slabs (J. Chem. Phys. 
   111, 3155).  Slabs defined here to be parallel to the xy plane. 
------------------------------------------------------------------------- */

void EVB_PPPMELECTRODE::slabcorr_cplx()
{
  // compute local cplx contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;

  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int * cplx_list = evb_engine->evb_complex->cplx_list;
  int * is_cplx_atom = evb_engine->complex_atom;

  double et_factor = 3.0;
  if(pair_et->et_table) et_factor = 2.0;

  double dipole_cplx    = 0.0;
  double dipole_r2_cplx = 0.0;
  for(int i=0; i<nlocal; i++) if(is_cplx_atom[i]) {
    dipole_cplx    += q[i] * x[i][2];
    dipole_r2_cplx += q[i] * x[i][2] * x[i][2];
  }
  
  // sum local contributions to get global dipole moment
  
  double dipole_all    = 0.0;
  double dipole_r2_all = 0.0;
  MPI_Allreduce(&dipole_cplx,    &dipole_all,    1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&dipole_r2_cplx, &dipole_r2_all, 1, MPI_DOUBLE, MPI_SUM, world);

  dipole_all    += dipole_env; // Total System Dipole
  dipole_r2_all += dipole_r2_env;

  dipole_all *= et_factor;

  // compute corrections

  const double e_slabcorr = MY_2PI * (dipole_all * dipole_all - qsum * dipole_r2_all - 
				      qsum * qsum * zprd * zprd / 12.0) / volume;
  
  energy += qqrd2e * e_slabcorr / comm->nprocs / et_factor;

  // add on force corrections

  double ffact = -4.0 * MY_PI * qqrd2e / volume;
  double **f = atom->f;

  for(int i=0; i<nlocal; i++) f[i][2] += ffact * q[i] * (dipole_all - qsum*x[i][2]);
}
