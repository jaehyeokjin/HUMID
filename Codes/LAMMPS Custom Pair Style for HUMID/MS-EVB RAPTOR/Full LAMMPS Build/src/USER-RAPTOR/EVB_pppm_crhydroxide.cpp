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
   Contributing authors: chris

   This is derived from the pppm/tip4p kspace style
------------------------------------------------------------------------- */

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
#include "memory.h"
#include "error.h"

#include "EVB_pppm.h"
#include "EVB_pppm_crhydroxide.h"
#include "EVB_engine.h"
#include "EVB_offdiag.h"
#include "EVB_complex.h"

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

/* ---------------------------------------------------------------------- */

EVB_PPPMCRHYDROXIDE::EVB_PPPMCRHYDROXIDE(LAMMPS *lmp, int narg, char **arg) : 
  EVB_PPPM(lmp, narg, arg)
{
  debug_q = 0;   // 0: False   1: True

  memory->create(xRing,MAX_CR_RING,3,"pppm:xRing");
  memory->create(fRing,MAX_CR_RING,3,"pppm:fRing");
}

/* ----------------------------------------------------------------------
   free all memory 
------------------------------------------------------------------------- */

EVB_PPPMCRHYDROXIDE::~EVB_PPPMCRHYDROXIDE()
{
  memory->destroy(xRing);
  memory->destroy(fRing);
}

/*************************************************************************/

void EVB_PPPMCRHYDROXIDE::map2density_one(int id)
{
  int maxI = 1;
  double qI = q[id];
  bool RingI_q = false;
  double *x1;

  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  if(atom->type[id]==typeO) {
    find_Ring(id);
    qI /= double(cr_N);
    maxI = cr_N;
    RingI_q = true;
  } 

  for(int iRing=0; iRing<maxI; iRing++) {

    if(!RingI_q) {
      x1 = x[id];
      nx = part2grid[id][0];
      ny = part2grid[id][1];
      nz = part2grid[id][2];
    } else {
      x1 = xRing[iRing];
      nx = static_cast<int> ((x1[0] - boxlox)*delxinv+shift) - OFFSET;
      ny = static_cast<int> ((x1[1] - boxloy)*delyinv+shift) - OFFSET;
      nz = static_cast<int> ((x1[2] - boxloz)*delzinv+shift) - OFFSET;
    }
	
    // (dx,dy,dz) = distance to "lower left" grid pt
    dx = nx+shiftone - (x1[0] - boxlox)*delxinv;
    dy = ny+shiftone - (x1[1] - boxloy)*delyinv;
    dz = nz+shiftone - (x1[2] - boxloz)*delzinv;
  
    compute_rho1d(dx, dy, dz);
  
    // (mx,my,mz) = global coords of moving stencil pt
    
    z0 = delvolinv * qI;
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
    
  } // Loop iRing
    
}

/*************************************************************************/

void EVB_PPPMCRHYDROXIDE::field2force_one_ik(int id, bool Aflag)
{
  int maxI = 1;
  double qI = q[id];
  bool RingI_q = false;
  double *x1;

  double const boxlox = boxlo[0];
  double const boxloy = boxlo[1];
  double const boxloz = boxlo[2];

  if(atom->type[id]==typeO) {
    find_Ring(id);
    qI /= double(cr_N);
    maxI = cr_N;
    RingI_q = true;
  }
  
  for(int iRing=0; iRing<maxI; iRing++) {
    
    if(!RingI_q) {
      x1 = x[id];
      nx = part2grid[id][0];
      ny = part2grid[id][1];
      nz = part2grid[id][2];
    } else {
      x1 = xRing[iRing];
      nx = static_cast<int> ((x1[0] - boxlox) * delxinv+shift) - OFFSET;
      ny = static_cast<int> ((x1[1] - boxloy) * delyinv+shift) - OFFSET;
      nz = static_cast<int> ((x1[2] - boxloz) * delzinv+shift) - OFFSET;
    }

    // (dx,dy,dz) = distance to "lower left" grid pt
    dx = nx+shiftone - (x1[0] - boxlox) * delxinv;
    dy = ny+shiftone - (x1[1] - boxloy) * delyinv;
    dz = nz+shiftone - (x1[2] - boxloz) * delzinv;
  
    compute_rho1d(dx,dy,dz);
  
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
    double pre_factor = qqrd2e*qI;
    if(Aflag) pre_factor *= A_Rq;
    
    if(!RingI_q) {
      f[id][0] += pre_factor * ekx;
      f[id][1] += pre_factor * eky;
      if(slabflag != 2) f[id][2] += pre_factor * ekz;
    } else {
      fRing[iRing][0] += pre_factor * ekx;
      fRing[iRing][1] += pre_factor * eky;
      if(slabflag != 2) fRing[iRing][2] += pre_factor * ekz;
    }

  } // Loop iRing
  if(RingI_q) map_force_Ring(id); // Map ring forces back onto hydroxide atoms
}


/*************************************************************************/

void EVB_PPPMCRHYDROXIDE::field2force_one_ad(int id, bool Aflag)
{
  int maxI = 1;
  double qI = q[id];
  bool RingI_q = false;
  double *x1;

  double s1, s2, s3;
  double sf = 0.0;
  double *prd;

  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  if(triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;
  
  const double hx_inv = nx_pppm / prd[0];
  const double hy_inv = ny_pppm / prd[1];
  const double hz_inv = nz_pppm / prd[2];

  if(atom->type[id]==typeO) {
    find_Ring(id);
    qI /= double(cr_N);
    maxI = cr_N;
    RingI_q = true;
  }
  
  for(int iRing=0; iRing<maxI; iRing++) {
    
    if(!RingI_q) {
      x1 = x[id];
      nx = part2grid[id][0];
      ny = part2grid[id][1];
      nz = part2grid[id][2];
    } else {
      x1 = xRing[iRing];
      nx = static_cast<int> ((x1[0] - boxlox) * delxinv+shift) - OFFSET;
      ny = static_cast<int> ((x1[1] - boxloy) * delyinv+shift) - OFFSET;
      nz = static_cast<int> ((x1[2] - boxloz) * delzinv+shift) - OFFSET;
    }

    // (dx,dy,dz) = distance to "lower left" grid pt
    dx = nx+shiftone - (x1[0] - boxlox) * delxinv;
    dy = ny+shiftone - (x1[1] - boxloy) * delyinv;
    dz = nz+shiftone - (x1[2] - boxloz) * delzinv;
  
    compute_rho1d(dx,dy,dz);
    compute_drho1d(dx,dy,dz);
  
    ekx = eky = ekz = ZEROF;
    for (int n = nlower; n <= nupper; n++) {
      mz = n+nz;
      for (int m = nlower; m <= nupper; m++) {
	my = m+ny;
	for (int l = nlower; l <= nupper; l++) {
	  mx = l+nx;
	  ekx += drho1d[0][l] *  rho1d[1][m] *  rho1d[2][n] * u_brick[mz][my][mx];
	  eky +=  rho1d[0][l] * drho1d[1][m] *  rho1d[2][n] * u_brick[mz][my][mx];
	  ekz +=  rho1d[0][l] *  rho1d[1][m] * drho1d[2][n] * u_brick[mz][my][mx];
	}
      }
    }
    ekx *= hx_inv;
    eky *= hy_inv;
    ekz *= hz_inv;
    
    // convert E-field to force
    double qfactor = qqrd2e*qI;
    if(Aflag) qfactor *= A_Rq;
    
    if(!RingI_q) {
      s1 = x[id][0] * hx_inv;
      s2 = x[id][1] * hy_inv;
      s3 = x[id][2] * hz_inv;
      sf = sf_coeff[0] * sin(2 * MY_PI * s1);
      sf += sf_coeff[1] * sin(4 * MY_PI * s1);
      sf *= 2 * qI * qI;
      f[id][0] += qfactor * (ekx * qI - sf);
      
      sf = sf_coeff[2] * sin(2 * MY_PI * s2);
      sf += sf_coeff[3] * sin(4 * MY_PI * s2);
      sf *= 2 * qI * qI;
      f[id][1] += qfactor * (eky * qI - sf);
      
      if (slabflag != 2) {
	sf = sf_coeff[4] * sin(2 * MY_PI * s3);
	sf += sf_coeff[5] * sin(4 * MY_PI * s3);
	sf *= 2 * qI * qI;
	f[id][2] += qfactor * (ekz * qI - sf);
      }
    } else {
      s1 = xRing[iRing][0] * hx_inv;
      s2 = xRing[iRing][1] * hy_inv;
      s3 = xRing[iRing][2] * hz_inv;
      sf = sf_coeff[0] * sin(2 * MY_PI * s1);
      sf += sf_coeff[1] * sin(4 * MY_PI * s1);
      sf *= 2 * qI * qI;
      fRing[iRing][0] += qfactor * (ekx * qI - sf);
      
      sf = sf_coeff[2] * sin(2 * MY_PI * s2);
      sf += sf_coeff[3] * sin(4 * MY_PI * s2);
      sf *= 2 * qI * qI;
      fRing[iRing][1] += qfactor * (eky * qI - sf);
      
      if (slabflag != 2) {
	sf = sf_coeff[4] * sin(2 * MY_PI * s3);
	sf += sf_coeff[5] * sin(4 * MY_PI * s3);
	sf *= 2 * qI * qI;
	fRing[iRing][2] += qfactor * (ekz * qI - sf);
      }
    }

  } // Loop iRing
  if(RingI_q) map_force_Ring(id); // Map ring forces back onto hydroxide atoms
}

/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid 
------------------------------------------------------------------------- */

void EVB_PPPMCRHYDROXIDE::make_rho()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;

  // clear 3d density array
  
  memset(&(density_brick[nzlo_out][nylo_out][nxlo_out]),0,ngrid*sizeof(FFT_SCALAR));
  
  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  double *q = atom->q;
  double **x = atom->x;
  int nlocal = atom->nlocal;

  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  for (int i = 0; i < nlocal; i++) {
    
    int maxI = 1;
    double qI = q[i];
    bool RingI_q = false;
    double *x1;

    if(atom->type[i]==typeO) {
      find_Ring(i);
      qI /= double(cr_N);
      maxI = cr_N;
      RingI_q = true;
    }

    for(int iRing=0; iRing<maxI; iRing++) {

      if(!RingI_q) {
	x1 = x[i];
	nx = part2grid[i][0];
	ny = part2grid[i][1];
	nz = part2grid[i][2];
      } else {
	x1 = xRing[iRing];
	nx = static_cast<int> ((x1[0] - boxlox) * delxinv+shift) - OFFSET;
	ny = static_cast<int> ((x1[1] - boxloy) * delyinv+shift) - OFFSET;
	nz = static_cast<int> ((x1[2] - boxloz) * delzinv+shift) - OFFSET;
      }
      
      // (dx,dy,dz) = distance to "lower left" grid pt
      dx = nx+shiftone - (x1[0] - boxlox) * delxinv;
      dy = ny+shiftone - (x1[1] - boxloy) * delyinv;
      dz = nz+shiftone - (x1[2] - boxloz) * delzinv;

      compute_rho1d(dx,dy,dz);

      z0 = delvolinv * qI;
      for (n = nlower; n <= nupper; n++) {
	mz = n+nz;
	y0 = z0*rho1d[2][n];
	for (m = nlower; m <= nupper; m++) {
	  my = m+ny;
	  x0 = y0*rho1d[1][m];
	  for (l = nlower; l <= nupper; l++) {
	    mx = l+nx;
	    density_brick[mz][my][mx] += x0*rho1d[0][l];
	  }
	}
      } // Loop mz
  
    } // Loop iRing

  } // Loop i
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ik
------------------------------------------------------------------------- */

void EVB_PPPMCRHYDROXIDE::fieldforce_ik()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR ekx, eky, ekz;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;
  int nlocal = atom->nlocal;

  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  for (i = 0; i < nlocal; i++) {

    int maxI = 1;
    double qI = q[i];
    bool RingI_q = false;
    double *x1;

    if(atom->type[i]==typeO) {
      find_Ring(i);
      qI /= double (cr_N);
      maxI = cr_N;
      RingI_q = true;
    }
    
    for(int iRing=0; iRing<maxI; iRing++) {

      if(!RingI_q) {
	x1 = x[i];
	nx = part2grid[i][0];
	ny = part2grid[i][1];
	nz = part2grid[i][2];
      } else {
	x1 = xRing[iRing];
	nx = static_cast<int> ((x1[0] - boxlox) * delxinv+shift) - OFFSET;
	ny = static_cast<int> ((x1[1] - boxloy) * delyinv+shift) - OFFSET;
	nz = static_cast<int> ((x1[2] - boxloz) * delzinv+shift) - OFFSET;
      }
      
      // (dx,dy,dz) = distance to "lower left" grid pt
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
	    ekx -= x0*vdx_brick[mz][my][mx];;
	    eky -= x0*vdy_brick[mz][my][mx];;
	    ekz -= x0*vdz_brick[mz][my][mx];;
	  }
	}
      } // Loop mz

      // convert E-field to force
      
      double prefactor = qqrd2e * qI;
      if(!RingI_q) {
	f[i][0] += prefactor * ekx;
	f[i][1] += prefactor * eky;
	if(slabflag != 2) f[i][2] += prefactor * ekz;
      } else {
	fRing[iRing][0]+= prefactor * ekx;
	fRing[iRing][1]+= prefactor * eky;
	if(slabflag != 2) fRing[iRing][2]+= prefactor * ekz;
      }

    } // Loop iRing
    if(RingI_q) map_force_Ring(i); // Map ring forces back onto hydroxide atoms

  } // Loop i
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ad
------------------------------------------------------------------------- */

void EVB_PPPMCRHYDROXIDE::fieldforce_ad()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR ekx, eky, ekz;

  double s1,s2,s3;
  double sf = 0.0;
  double *prd;

  if (triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];

  double hx_inv = nx_pppm/xprd;
  double hy_inv = ny_pppm/yprd;
  double hz_inv = nz_pppm/zprd;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;
  int nlocal = atom->nlocal;

  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  for (i = 0; i < nlocal; i++) {

    int maxI = 1;
    double qI = q[i];
    bool RingI_q = false;
    double *x1;

    if(atom->type[i]==typeO) {
      find_Ring(i);
      qI /= double (cr_N);
      maxI = cr_N;
      RingI_q = true;
    }
    
    for(int iRing=0; iRing<maxI; iRing++) {

      if(!RingI_q) {
	x1 = x[i];
	nx = part2grid[i][0];
	ny = part2grid[i][1];
	nz = part2grid[i][2];
      } else {
	x1 = xRing[iRing];
	nx = static_cast<int> ((x1[0] - boxlox) * delxinv+shift) - OFFSET;
	ny = static_cast<int> ((x1[1] - boxloy) * delyinv+shift) - OFFSET;
	nz = static_cast<int> ((x1[2] - boxloz) * delzinv+shift) - OFFSET;
      }
      
      // (dx,dy,dz) = distance to "lower left" grid pt
      dx = nx+shiftone - (x[i][0] - boxlox) * delxinv;
      dy = ny+shiftone - (x[i][1] - boxloy) * delyinv;
      dz = nz+shiftone - (x[i][2] - boxloz) * delzinv;

      compute_rho1d(dx,dy,dz);
      compute_drho1d(dx,dy,dz);

      ekx = eky = ekz = ZEROF;
      for (n = nlower; n <= nupper; n++) {
	mz = n+nz;
	for (m = nlower; m <= nupper; m++) {
	  my = m+ny;
	  for (l = nlower; l <= nupper; l++) {
	    mx = l+nx;
	    ekx += drho1d[0][l]*rho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
	    eky += rho1d[0][l]*drho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
	    ekz += rho1d[0][l]*rho1d[1][m]*drho1d[2][n]*u_brick[mz][my][mx];
	  }
	}
      }
      ekx *= hx_inv;
      eky *= hy_inv;
      ekz *= hz_inv;

      // convert E-field to force
      
      double qfactor = qqrd2e * qI;

      if(!RingI_q) {
	s1 = x[i][0] * hx_inv;
	s2 = x[i][1] * hy_inv;
	s3 = x[i][2] * hz_inv;
	sf = sf_coeff[0] * sin(2 * MY_PI * s1);
	sf += sf_coeff[1] * sin(4 * MY_PI * s1);
	sf *= 2 * qI * qI;
	f[i][0] += qfactor * (ekx * qI - sf);
	
	sf = sf_coeff[2] * sin(2 * MY_PI * s2);
	sf += sf_coeff[3] * sin(4 * MY_PI * s2);
	sf *= 2 * qI * qI;
	f[i][1] += qfactor * (eky * qI - sf);
	
	if (slabflag != 2) {
	  sf = sf_coeff[4] * sin(2 * MY_PI * s3);
	  sf += sf_coeff[5] * sin(4 * MY_PI * s3);
	  sf *= 2 * qI * qI;
	  f[i][2] += qfactor * (ekz * qI - sf);
	}
      } else {
	s1 = xRing[iRing][0] * hx_inv;
	s2 = xRing[iRing][1] * hy_inv;
	s3 = xRing[iRing][2] * hz_inv;
	sf = sf_coeff[0] * sin(2 * MY_PI * s1);
	sf += sf_coeff[1] * sin(4 * MY_PI * s1);
	sf *= 2 * qI * qI;
	fRing[iRing][0] += qfactor * (ekx * qI - sf);
	
	sf = sf_coeff[2] * sin(2 * MY_PI * s2);
	sf += sf_coeff[3] * sin(4 * MY_PI * s2);
	sf *= 2 * qI * qI;
	fRing[iRing][1] += qfactor * (eky * qI - sf);
	
	if (slabflag != 2) {
	  sf = sf_coeff[4] * sin(2 * MY_PI * s3);
	  sf += sf_coeff[5] * sin(4 * MY_PI * s3);
	  sf *= 2 * qI * qI;
	  fRing[iRing][2] += qfactor * (ekz * qI - sf);
	}
      }

    } // Loop iRing
    if(RingI_q) map_force_Ring(i); // Map ring forces back onto hydroxide atoms

  } // Loop i
}

/* ----------------------------------------------------------------------
  find 1 H atom bonded to O atom iOxy
  compute position xRing of fictitious charge sites for O atom
------------------------------------------------------------------------- */

void EVB_PPPMCRHYDROXIDE::find_Ring(int iOxy)
{
  double xO[3], xH[3];
  double l1[3], l2p[3], l2[3], l3[3];
  double u1[3],         u2[3], u3[3];
  double norm,proj;
  double dTheta,theta,cost,sint;

  double PI = 4*atan(1.0);

  double **x = atom->x;

  // test that O is correctly bonded to 1 succesive H atoms
  int iHyd = atom->map(atom->tag[iOxy] + 1);

  // Search for bond of this hydroxide ion: MS-EVB
  int found_bond = 0;
  if(atom->type[iHyd] != typeH) {
    int **bondlist = neighbor->bondlist;
    int nbondlist = neighbor->nbondlist;
    for (int i=0; i<nbondlist; i++) {
      if(bondlist[i][2] == typeB) {
	found_bond = 1;
	int i1 = bondlist[i][0];
	int i2 = bondlist[i][1];
	if(atom->tag[i1] == atom->tag[iOxy]) {
	  iHyd = i2;
	  break;
	} 
	if (atom->tag[i2] == atom->tag[iOxy]) {
	  iHyd = i1;
	  break;
	}
      }
    }
  }

  xO[0] = x[iOxy][0]; // Hydroxide oxygen coordinate
  xO[1] = x[iOxy][1];
  xO[2] = x[iOxy][2];

  // If bond is not on local processor, then loop over atoms until hydroxide hydrogen within distance tolerance of oxygen is found
  if(!found_bond) {
    double tol = 10.0; // Distance in Angstroms
    for(int i=0; i<(atom->nlocal+atom->nghost); i++) if(atom->type[i] == typeH) {
	double bond[3];
	bond[0] = x[i][0] - xO[0];
	bond[1] = x[i][1] - xO[1];
	bond[2] = x[i][2] - xO[2];
	domain->minimum_image(bond[0],bond[1],bond[2]);
	if(vector_norm(bond) < tol) {
	  tol = vector_norm(bond);
	  iHyd = i;
	  if(debug_q) fprintf(stdout,"   Candidate for hydroxide found: norm = %f tol = %f  i = %i  type = %i\n",vector_norm(bond),tol,i,atom->type[i]);
	}
      }
  }

  if (atom->type[iHyd] != typeH) {
    fprintf(stdout,"\nUnable to find hydroxide atoms on proc %i.\n",comm->me);
    for(int i=0; i<atom->nlocal; i++) if(atom->type[i] == typeH) fprintf(stdout,"   local   i = %i %i  type = %i\n",i,atom->tag[i],atom->type[i]);
    for(int i=atom->nlocal; i<atom->nghost; i++) if(atom->type[i] == typeH) fprintf(stdout,"   ghost   i = %i %i  type = %i\n",i,atom->tag[i],atom->type[i]);
    if(!found_bond) fprintf(stdout,"Hydroxide bond was not found\n");
    fprintf(stdout,"iOxy = %i %i type = %i   iHyd = %i %i  type = %i\n",iOxy,atom->tag[iOxy],atom->type[iOxy],iHyd,atom->tag[iHyd],atom->type[iHyd]);
    fprintf(stdout,"typeO = %i  typeH = %i\n",typeO,typeH);
    error->one(FLERR,"PPPM(find): CR-HYDROXIDE hydrogen has incorrect atom type\n");
  }

  l1[0] = x[iHyd][0] - xO[0];
  l1[1] = x[iHyd][1] - xO[1];
  l1[2] = x[iHyd][2] - xO[2];
  domain->minimum_image(l1[0],l1[1],l1[2]);

  xH[0] = xO[0] + l1[0]; // Hydroxide hydrogen coordinate
  xH[1] = xO[1] + l1[1];
  xH[2] = xO[2] + l1[2];

  // Coordinate system for hydroxide
  vector_unit(l1, u1); // Vector along OH bond
  
  l2p[0] = u1[0] + COEF_L2P;
  l2p[1] = u1[1] + COEF_L2P;
  l2p[2] = u1[2] + COEF_L2P;

  proj = vector_dot(u1, l2p);
  l2[0] = l2p[0] - proj * u1[0];
  l2[1] = l2p[1] - proj * u1[1];
  l2[2] = l2p[2] - proj * u1[2];
  vector_unit(l2, u2);  // Vector perpendicular to OH bond

  vector_cross(l1, l2, l3);
  vector_unit(l3, u3); // 2nd vector perpendicular to OH bond

  if(debug_q) {
    fprintf(stdout,"\nOxygen   = %f %f %f\n",xO[0],xO[1],xO[2]);
    fprintf(stdout,"Hydrogen = %f %f %f\n",xH[0],xH[1],xH[2]);
    fprintf(stdout,"Basis Vector #1 = %f %f %f\n",u1[0],u1[1],u1[2]);
    fprintf(stdout,"Basis Vector #2 = %f %f %f\n",u2[0],u2[1],u2[2]);
    fprintf(stdout,"Basis Vector #3 = %f %f %f\n",u3[0],u3[1],u3[2]);
    fprintf(stdout,"r1.r2 = %f\n",vector_dot(u1, u2));
    fprintf(stdout,"r1.r3 = %f\n",vector_dot(u1, u3));
    fprintf(stdout,"r2.r3 = %f\n",vector_dot(u2, u3));
    fprintf(stdout,"\n");
  }

  // Angle between adjacent ring particles
  dTheta = 2.0 * PI / double(cr_N);
  
  // Translate and scale basis vectors by ring dimensions
  double b1[3], b2[3], b3[3];
  b1[0] = xO[0] - u1[0] * cr_height;
  b1[1] = xO[1] - u1[1] * cr_height;
  b1[2] = xO[2] - u1[2] * cr_height;

  double cr_radius = cr_diameter / 2.0;
  b2[0] = u2[0] * cr_radius;
  b2[1] = u2[1] * cr_radius;
  b2[2] = u2[2] * cr_radius;

  b3[0] = u3[0] * cr_radius;
  b3[1] = u3[1] * cr_radius;
  b3[2] = u3[2] * cr_radius;

  // Generate ring particle positions
  for(int i=0; i<cr_N; i++) {
    theta = double(i) * dTheta;
    cost = cos(theta);
    sint = sin(theta);
    xRing[i][0] = b1[0] + cost * b2[0] + sint * b3[0];
    xRing[i][1] = b1[1] + cost * b2[1] + sint * b3[1];
    xRing[i][2] = b1[2] + cost * b2[2] + sint * b3[2];
    fRing[i][0] = 0.0;
    fRing[i][1] = 0.0;
    fRing[i][2] = 0.0;
  }
  
  if(debug_q) {
    fprintf(stdout,"Scaled Basis Vectors\n");
    fprintf(stdout,"Basis Vector #1 = %f %f %f\n",b1[0],b1[1],b1[2]);
    fprintf(stdout,"Basis Vector #2 = %f %f %f\n",b2[0],b2[1],b2[2]);
    fprintf(stdout,"Basis Vector #3 = %f %f %f\n",b3[0],b3[1],b3[2]);
    for(int i=0; i<cr_N; i++) fprintf(stdout,"Ring %i = %f %f %f  dist O = %f  dist H = %f\n",i,xRing[i][0],xRing[i][1],xRing[i][2],vector_dist(xO,xRing[i]),vector_dist(xH,xRing[i]));
  }
}

/* ----------------------------------------------------------------------
  Map force on ring particles to O and H atoms of hydroxide
------------------------------------------------------------------------- */

void EVB_PPPMCRHYDROXIDE::map_force_Ring(int iOxy)
{
  if(debug_q) fprintf(stdout,"Entered map_force_Ring()\n");

  double **x = atom->x;
  double xO[3], xH[3], x_com[3], bond[3], fO[3], fH[3];

  // Index of hydrogen atom
  int iHyd = atom->map(atom->tag[iOxy] + 1);

  // Search for bond of this hydroxide ion: MS-EVB
  int found_bond = 0;
  if(atom->type[iHyd] != typeH) {
    int **bondlist = neighbor->bondlist;
    int nbondlist = neighbor->nbondlist;
    for (int i=0; i<nbondlist; i++) {
      if(bondlist[i][2] == typeB) {
	found_bond = 1;
	int i1 = bondlist[i][0];
	int i2 = bondlist[i][1];
	if(atom->tag[i1] == atom->tag[iOxy]) {
	  iHyd = i2;
	  break;
	} 
	if (atom->tag[i2] == atom->tag[iOxy]) {
	  iHyd = i1;
	  break;
	}
      }
    }
  }

  xO[0] = x[iOxy][0]; // Hydroxide oxygen coordinate
  xO[1] = x[iOxy][1];
  xO[2] = x[iOxy][2];

  // If bond is not on local processor, then loop over atoms until hydroxide hydrogen within distance tolerance of oxygen is found
  if(!found_bond) {
    double tol = 10.0; // Distance in Angstroms
    for(int i=0; i<(atom->nlocal+atom->nghost); i++) if(atom->type[i] == typeH) {
	double bond[3];
	bond[0] = x[i][0] - xO[0];
	bond[1] = x[i][1] - xO[1];
	bond[2] = x[i][2] - xO[2];
	domain->minimum_image(bond[0],bond[1],bond[2]);
	if(vector_norm(bond) < tol) {
	  tol = vector_norm(bond);
	  iHyd = i;
	  if(debug_q) fprintf(stdout,"   Candidate for hydroxide found: norm = %f tol = %f  i = %i  type = %i\n",vector_norm(bond),tol,i,atom->type[i]);
	}
      }
  }

  if (atom->type[iHyd] != typeH) {
    fprintf(stdout,"\nUnable to find hydroxide atoms on proc %i.\n",comm->me);
    for(int i=0; i<atom->nlocal; i++) if(atom->type[i] == typeH) fprintf(stdout,"   local   i = %i %i  type = %i\n",i,atom->tag[i],atom->type[i]);
    for(int i=atom->nlocal; i<atom->nghost; i++) if(atom->type[i] == typeH) fprintf(stdout,"   ghost   i = %i %i  type = %i\n",i,atom->tag[i],atom->type[i]);
    if(!found_bond) fprintf(stdout,"Hydroxide bond was not found\n");
    fprintf(stdout,"iOxy = %i %i type = %i   iHyd = %i %i  type = %i\n",iOxy,atom->tag[iOxy],atom->type[iOxy],iHyd,atom->tag[iHyd],atom->type[iHyd]);
    error->one(FLERR,"PPPM(map): CR-HYDROXIDE hydrogen has incorrect atom type\n");
  }

  if(debug_q) {
    fprintf(stdout,"iOxy = %i  iHyd = %i  type = %i\n",iOxy,iHyd,atom->type[iOxy]);
    fprintf(stdout,"Forces on Ring Particles\n");
    for(int i=0; i<cr_N; i++) fprintf(stdout,"i = %i:  f = %f %f %f\n",i,fRing[i][0],fRing[i][1],fRing[i][2]);
  }

  xO[0] = x[iOxy][0]; // Hydroxide oxygen coordinate
  xO[1] = x[iOxy][1];
  xO[2] = x[iOxy][2];

  bond[0] = x[iHyd][0] - xO[0];
  bond[1] = x[iHyd][1] - xO[1];
  bond[2] = x[iHyd][2] - xO[2];
  domain->minimum_image(bond[0],bond[1],bond[2]);

  xH[0] = xO[0] + bond[0]; // Hydroxide hydrogen coordinate
  xH[1] = xO[1] + bond[1];
  xH[2] = xO[2] + bond[2];

  // Calculate force on atoms
  map_force_analytic(xO, xH, fO, fH);

  // Add forces due to ring to particle forces
  double **f = atom->f;
  f[iOxy][0]+= fO[0];
  f[iOxy][1]+= fO[1];
  f[iOxy][2]+= fO[2];
  
  f[iHyd][0]+= fH[0];
  f[iHyd][1]+= fH[1];
  f[iHyd][2]+= fH[2];

  double fTot[3];
  fTot[0] = 0.0;  fTot[1] = 0.0;  fTot[2] = 0.0;
  for(int i=0; i<cr_N; i++) {
    fTot[0]+= fRing[i][0];
    fTot[1]+= fRing[i][1];
    fTot[2]+= fRing[i][2];
  }

  double fSum[3];
  fSum[0] = fO[0] + fH[0];
  fSum[1] = fO[1] + fH[1];
  fSum[2] = fO[2] + fH[2];

  if(debug_q) {
    fprintf(stdout,"\n Force on Oxygen(Tot)   = %f %f %f\n",fO[0],fO[1],fO[2]);
    fprintf(stdout," Force on Hydrogen(Tot)   = %f %f %f\n",fH[0],fH[1],fH[2]);
    fprintf(stdout," Total                    = %f %f %f\n",fSum[0],fSum[1],fSum[2]);
    fprintf(stdout," Sum of forces            = %f %f %f\n",fTot[0],fTot[1],fTot[2]);
  }

  double diff[3];
  diff[0] = fTot[0] - fSum[0];
  diff[1] = fTot[1] - fSum[1];
  diff[2] = fTot[2] - fSum[2];

  if(vector_norm(diff) > 0.1) {
    fprintf(stdout,"\n Total                    = %f %f %f\n",fSum[0],fSum[1],fSum[2]);
    fprintf(stdout," Sum of forces            = %f %f %f\n",fTot[0],fTot[1],fTot[2]);
    fprintf(stdout," Difference larger than tolerance: tol = 0.1  diff = %f\n",vector_norm(diff));
    error->one(FLERR,"PPPMCRHYDROXIDE::map_force_Ring()\n");
  }

  // Spread error in forces over both atoms so total force is conserved
  f[iOxy][0]+= diff[0] * 0.5;
  f[iOxy][1]+= diff[1] * 0.5;
  f[iOxy][2]+= diff[2] * 0.5;
  
  f[iHyd][0]+= diff[0] * 0.5;
  f[iHyd][1]+= diff[1] * 0.5;
  f[iHyd][2]+= diff[2] * 0.5;
    
  if(debug_q) fprintf(stdout,"Leaving map_force_Ring()\n");
}

/* ---------------------------------------------------------------------- */

double EVB_PPPMCRHYDROXIDE::vector_norm(double r1[3])
{
  double norm = sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2]);
  return norm;
}

double EVB_PPPMCRHYDROXIDE::vector_dot(double r1[3], double r2[3])
{
  double dot = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2];
  return dot;
}

double EVB_PPPMCRHYDROXIDE::vector_dist(double r1[3], double r2[3])
{
  double r3[3];
  r3[0] = r2[0] - r1[0];
  r3[1] = r2[1] - r1[1];
  r3[2] = r2[2] - r1[2];
  double dist = sqrt(r3[0]*r3[0] + r3[1]*r3[1] + r3[2]*r3[2]);
  return dist;
}

void EVB_PPPMCRHYDROXIDE::vector_unit(double r1[3])
{
  double norm = sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2]);
  r1[0] /= norm;
  r1[1] /= norm;
  r1[2] /= norm;
}

void EVB_PPPMCRHYDROXIDE::vector_unit(double r1[3], double r2[3])
{
  double norm = sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2]);
  if(norm < 0.00001) norm = 0.0;
  else norm = 1.0 / norm;
  r2[0] = r1[0] * norm;
  r2[1] = r1[1] * norm;
  r2[2] = r1[2] * norm;
}

void EVB_PPPMCRHYDROXIDE::vector_cross(double r1[3], double r2[3], double r3[3])
{
  r3[0] =  r1[1]*r2[2] - r1[2]*r2[1];
  r3[1] = -r1[0]*r2[2] + r1[2]*r2[0];
  r3[2] =  r1[0]*r2[1] - r1[1]*r2[0];
}

void EVB_PPPMCRHYDROXIDE::map_force_analytic(double xO[3], double xH[3], double fO[3], double fH[3])
{
  if(debug_q) {
    fprintf(stdout,"\nEntered PPPM::map_force_analytic()\n");
    fprintf(stdout,"Force on Oxygen =   %f %f %f\n",fO[0],fO[1],fO[2]);
    fprintf(stdout,"Force on Hydrogen = %f %f %f\n",fH[0],fH[1],fH[2]);
  }

  double xRingDel[3];
  double l1[3], l2p[3], l2[3], l3[3];
  double u1[3],         u2[3], u3[3];
  double proj, scale;

  double sign = -1.0;  // -1.0 for oxygen forces; 1.0 for hydrogen forces

  // Calculate displaced basis vectors
  l1[0] = xH[0] - xO[0];
  l1[1] = xH[1] - xO[1];
  l1[2] = xH[2] - xO[2];
  vector_unit(l1, u1);
  
  l2p[0] = u1[0] + COEF_L2P;
  l2p[1] = u1[1] + COEF_L2P;
  l2p[2] = u1[2] + COEF_L2P;

  proj = vector_dot(u1, l2p);
  l2[0] = l2p[0] - proj * u1[0];
  l2[1] = l2p[1] - proj * u1[1];
  l2[2] = l2p[2] - proj * u1[2];
  vector_unit(l2, u2);  // Vector perpendicular to OH bond
  
  vector_cross(l1, l2, l3);
  vector_unit(l3, u3); // 2nd vector perpendicular to OH bond
  
  // Derivative of 1st basis vector
  double dr1rk[3][3];
  dr1rk[0][0] = sign;   dr1rk[0][1] = 0.0;    dr1rk[0][2] = 0.0;
  dr1rk[1][0] = 0.0;    dr1rk[1][1] = sign;   dr1rk[1][2] = 0.0;
  dr1rk[2][0] = 0.0;    dr1rk[2][1] = 0.0;    dr1rk[2][2] = sign;

  // Derivative of 1st basis unitvector
  double dru1rk[3][3];
  scale = sign / vector_norm(l1); // +1 for hydrogen, -1 for oxygen
  dru1rk[0][0] = (1.0 -u1[0]*u1[0])*scale;   dru1rk[0][1] = (    -u1[0]*u1[1])*scale;   dru1rk[0][2] = (    -u1[0]*u1[2])*scale;
  dru1rk[1][0] = dru1rk[0][1];               dru1rk[1][1] = (1.0 -u1[1]*u1[1])*scale;   dru1rk[1][2] = (    -u1[1]*u1[2])*scale;
  dru1rk[2][0] = dru1rk[0][2];               dru1rk[2][1] = dru1rk[1][2];               dru1rk[2][2] = (1.0 -u1[2]*u1[2])*scale;

  // Derivative of 2nd basis vector
  double dr2rk[3][3];
  scale = 1.0 - vector_dot(u1, l2p);
  dr2rk[0][0] = dru1rk[0][0] * scale;  dr2rk[0][1] = dru1rk[0][1] * scale;  dr2rk[0][2] = dru1rk[0][2] * scale;
  dr2rk[1][0] = dru1rk[1][0] * scale;  dr2rk[1][1] = dru1rk[1][1] * scale;  dr2rk[1][2] = dru1rk[1][2] * scale;
  dr2rk[2][0] = dru1rk[2][0] * scale;  dr2rk[2][1] = dru1rk[2][1] * scale;  dr2rk[2][2] = dru1rk[2][2] * scale;
  
  double tmp[3];
  double rnorm1 = 1.0 / vector_norm(l1);
  scale = vector_dot(u1, l2p) * rnorm1 * rnorm1 * sign;
  tmp[0] = scale * l1[0] - rnorm1 * sign * (rnorm1 * l1[0] + COEF_L2P);
  tmp[1] = scale * l1[1] - rnorm1 * sign * (rnorm1 * l1[1] + COEF_L2P);
  tmp[2] = scale * l1[2] - rnorm1 * sign * (rnorm1 * l1[2] + COEF_L2P);

  dr2rk[0][0]+= tmp[0] * u1[0];  dr2rk[0][1]+= tmp[0] * u1[1];  dr2rk[0][2]+= tmp[0] * u1[2];
  dr2rk[1][0]+= tmp[1] * u1[0];  dr2rk[1][1]+= tmp[1] * u1[1];  dr2rk[1][2]+= tmp[1] * u1[2];
  dr2rk[2][0]+= tmp[2] * u1[0];  dr2rk[2][1]+= tmp[2] * u1[1];  dr2rk[2][2]+= tmp[2] * u1[2];

  // Derivative of 2nd basis unitvector
  double dru2rk[3][3];
  scale = -rnorm1 * rnorm1 * rnorm1 * (l1[0] + l1[1] + l1[2]) * sign;
  tmp[0] = scale * l1[0] + sign * rnorm1;
  tmp[1] = scale * l1[1] + sign * rnorm1;
  tmp[2] = scale * l1[2] + sign * rnorm1;

  double rnorm2 = 1.0 / vector_norm(l2);
  scale = -rnorm2 * rnorm2 * rnorm2 * COEF_L2P * (1.0 - vector_dot(u1, l2p));
  tmp[0]*= scale;
  tmp[1]*= scale;
  tmp[2]*= scale;

  dru2rk[0][0] = tmp[0] * l2[0] + rnorm2 * dr2rk[0][0];  dru2rk[0][1] = tmp[0] * l2[1] + rnorm2 * dr2rk[0][1];  dru2rk[0][2] = tmp[0] * l2[2] + rnorm2 * dr2rk[0][2];
  dru2rk[1][0] = tmp[1] * l2[0] + rnorm2 * dr2rk[1][0];  dru2rk[1][1] = tmp[1] * l2[1] + rnorm2 * dr2rk[1][1];  dru2rk[1][2] = tmp[1] * l2[2] + rnorm2 * dr2rk[1][2];
  dru2rk[2][0] = tmp[2] * l2[0] + rnorm2 * dr2rk[2][0];  dru2rk[2][1] = tmp[2] * l2[1] + rnorm2 * dr2rk[2][1];  dru2rk[2][2] = tmp[2] * l2[2] + rnorm2 * dr2rk[2][2];

  // Derivative of 3rd basis vector
  double dr3rk[3][3];
  double tmp2[3];
  vector_cross(dr1rk[0], l2, tmp);  vector_cross(l1, dr2rk[0], tmp2);
  dr3rk[0][0] = tmp[0] + tmp2[0];  dr3rk[0][1] = tmp[1] + tmp2[1];  dr3rk[0][2] = tmp[2] + tmp2[2];

  vector_cross(dr1rk[1], l2, tmp);  vector_cross(l1, dr2rk[1], tmp2);
  dr3rk[1][0] = tmp[0] + tmp2[0];  dr3rk[1][1] = tmp[1] + tmp2[1];  dr3rk[1][2] = tmp[2] + tmp2[2];

  vector_cross(dr1rk[2], l2, tmp);  vector_cross(l1, dr2rk[2], tmp2);
  dr3rk[2][0] = tmp[0] + tmp2[0];  dr3rk[2][1] = tmp[1] + tmp2[1];  dr3rk[2][2] = tmp[2] + tmp2[2];

  // Derivative of 3rd basis unitvector
  double dru3rk[3][3];
  double rnorm3 = 1.0 / vector_norm(l3);
  scale = -COEF_L2P * COEF_L2P * sign * rnorm3 * rnorm3 * rnorm3;
  tmp[0] = scale * ((l1[0] - l1[1]) - (l1[2] - l1[0]));
  tmp[1] = scale * ((l1[1] - l1[2]) - (l1[0] - l1[1]));
  tmp[2] = scale * ((l1[2] - l1[0]) - (l1[1] - l1[2]));

  dru3rk[0][0] = tmp[0] * l3[0] + rnorm3 * dr3rk[0][0];  dru3rk[0][1] = tmp[0] * l3[1] + rnorm3 * dr3rk[0][1];  dru3rk[0][2] = tmp[0] * l3[2] + rnorm3 * dr3rk[0][2];
  dru3rk[1][0] = tmp[1] * l3[0] + rnorm3 * dr3rk[1][0];  dru3rk[1][1] = tmp[1] * l3[1] + rnorm3 * dr3rk[1][1];  dru3rk[1][2] = tmp[1] * l3[2] + rnorm3 * dr3rk[1][2];
  dru3rk[2][0] = tmp[2] * l3[0] + rnorm3 * dr3rk[2][0];  dru3rk[2][1] = tmp[2] * l3[1] + rnorm3 * dr3rk[2][1];  dru3rk[2][2] = tmp[2] * l3[2] + rnorm3 * dr3rk[2][2];

  if(debug_q) {
    fprintf(stdout,"l1 = %f %f %f   u1 = %f %f %f\n",l1[0],l1[1],l1[2],u1[0],u1[1],u1[2]);
    fprintf(stdout,"l2p= %f %f %f\n",l2p[0],l2p[1],l2p[2]);
    fprintf(stdout,"l2 = %f %f %f   u2 = %f %f %f\n",l2[0],l2[1],l2[2],u2[0],u2[1],u2[2]);
    fprintf(stdout,"l3 = %f %f %f   u3 = %f %f %f\n",l3[0],l3[1],l3[2],u3[0],u3[1],u3[2]);
    fprintf(stdout,"dru1rk = \n");
    fprintf(stdout,"  %f %f %f\n",dru1rk[0][0],dru1rk[0][1],dru1rk[0][2]);
    fprintf(stdout,"  %f %f %f\n",dru1rk[1][0],dru1rk[1][1],dru1rk[1][2]);
    fprintf(stdout,"  %f %f %f\n",dru1rk[2][0],dru1rk[2][1],dru1rk[2][2]);
    fprintf(stdout,"dr2rk = \n");
    fprintf(stdout,"  %f %f %f\n",dr2rk[0][0],dr2rk[0][1],dr2rk[0][2]);
    fprintf(stdout,"  %f %f %f\n",dr2rk[1][0],dr2rk[1][1],dr2rk[1][2]);
    fprintf(stdout,"  %f %f %f\n",dr2rk[2][0],dr2rk[2][1],dr2rk[2][2]);
    fprintf(stdout,"dru2rk = \n");
    fprintf(stdout,"  %f %f %f\n",dru2rk[0][0],dru2rk[0][1],dru2rk[0][2]);
    fprintf(stdout,"  %f %f %f\n",dru2rk[1][0],dru2rk[1][1],dru2rk[1][2]);
    fprintf(stdout,"  %f %f %f\n",dru2rk[2][0],dru2rk[2][1],dru2rk[2][2]);
    fprintf(stdout,"dr3rk = \n");
    fprintf(stdout,"  %f %f %f\n",dr3rk[0][0],dr3rk[0][1],dr3rk[0][2]);
    fprintf(stdout,"  %f %f %f\n",dr3rk[1][0],dr3rk[1][1],dr3rk[1][2]);
    fprintf(stdout,"  %f %f %f\n",dr3rk[2][0],dr3rk[2][1],dr3rk[2][2]);
    fprintf(stdout,"dru3rk = \n");
    fprintf(stdout,"  %f %f %f\n",dru3rk[0][0],dru3rk[0][1],dru3rk[0][2]);
    fprintf(stdout,"  %f %f %f\n",dru3rk[1][0],dru3rk[1][1],dru3rk[1][2]);
    fprintf(stdout,"  %f %f %f\n",dru3rk[2][0],dru3rk[2][1],dru3rk[2][2]);
  }

  // Calculate derivatives of ring particle positions w/r to hydroxide atom coordinates
  double s = 1.0;
  if(sign > 0.0) s = 0.0;
  double PI = 4 * atan(1.0);
  double dTheta = 2.0 * PI / double(cr_N);
  double xRingDer[3][3];
  double cost, sint, theta;
  fO[0] = 0.0;  fO[1] = 0.0;  fO[2] = 0.0;
  double fRingTot[3];
  fRingTot[0] = 0.0;  fRingTot[1] = 0.0;  fRingTot[2] = 0.0;
  for(int i=0; i<cr_N; i++) {
    theta = double(i) * dTheta;
    cost = cos(theta);
    sint = sin(theta);
    xRingDer[0][0] = s -cr_height * dru1rk[0][0];  xRingDer[0][1] =   -cr_height * dru1rk[0][1];  xRingDer[0][2] =   -cr_height * dru1rk[0][2];
    xRingDer[1][0] =   -cr_height * dru1rk[1][0];  xRingDer[1][1] = s -cr_height * dru1rk[1][1];  xRingDer[1][2] =   -cr_height * dru1rk[1][2];
    xRingDer[2][0] =   -cr_height * dru1rk[2][0];  xRingDer[2][1] =   -cr_height * dru1rk[2][1];  xRingDer[2][2] = s -cr_height * dru1rk[2][2];

    scale = cr_diameter / 2.0 * cost;
    xRingDer[0][0]+= scale * dru2rk[0][0];  xRingDer[0][1]+= scale * dru2rk[0][1];  xRingDer[0][2]+= scale * dru2rk[0][2];
    xRingDer[1][0]+= scale * dru2rk[1][0];  xRingDer[1][1]+= scale * dru2rk[1][1];  xRingDer[1][2]+= scale * dru2rk[1][2];
    xRingDer[2][0]+= scale * dru2rk[2][0];  xRingDer[2][1]+= scale * dru2rk[2][1];  xRingDer[2][2]+= scale * dru2rk[2][2];

    scale = cr_diameter / 2.0 * sint;
    xRingDer[0][0]+= scale * dru3rk[0][0];  xRingDer[0][1]+= scale * dru3rk[0][1];  xRingDer[0][2]+= scale * dru3rk[0][2];
    xRingDer[1][0]+= scale * dru3rk[1][0];  xRingDer[1][1]+= scale * dru3rk[1][1];  xRingDer[1][2]+= scale * dru3rk[1][2];
    xRingDer[2][0]+= scale * dru3rk[2][0];  xRingDer[2][1]+= scale * dru3rk[2][1];  xRingDer[2][2]+= scale * dru3rk[2][2];

    if(debug_q) {
      fprintf(stdout,"\ni = %i\n",i);
      fprintf(stdout,"  %f %f %f\n",xRingDer[0][0],xRingDer[0][1],xRingDer[0][2]);
      fprintf(stdout,"  %f %f %f\n",xRingDer[1][0],xRingDer[1][1],xRingDer[1][2]);
      fprintf(stdout,"  %f %f %f\n",xRingDer[2][0],xRingDer[2][1],xRingDer[2][2]);
    }

    fO[0]+= vector_dot(xRingDer[0], fRing[i]);
    fO[1]+= vector_dot(xRingDer[1], fRing[i]);
    fO[2]+= vector_dot(xRingDer[2], fRing[i]);

    fRingTot[0]+= fRing[i][0];
    fRingTot[1]+= fRing[i][1];
    fRingTot[2]+= fRing[i][2];
  }

  fH[0] = fRingTot[0] - fO[0];
  fH[1] = fRingTot[1] - fO[1];
  fH[2] = fRingTot[2] - fO[2];

  if(debug_q) {
    fprintf(stdout,"Force on Oxygen =   %f %f %f\n",fO[0],fO[1],fO[2]);
    fprintf(stdout,"Force on Hydrogen = %f %f %f\n",fH[0],fH[1],fH[2]);
    fprintf(stdout,"Leaving PPPM::map_force_analytic()\n");
  }
 
}
