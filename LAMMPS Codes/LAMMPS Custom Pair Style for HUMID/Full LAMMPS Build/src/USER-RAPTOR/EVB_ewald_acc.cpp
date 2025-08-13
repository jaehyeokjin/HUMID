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
   Contributing authors: Roy Pollock (LLNL), Paul Crozier (SNL)

   Splitted for MS-EVB by: Tianying, Chris and Yuxing
------------------------------------------------------------------------- */

#include "mpi.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "math.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "domain.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

#include "EVB_ewald.h"
#include "EVB_ewald_acc.h"
#include "EVB_offdiag.h"
#include "EVB_complex.h"
#include "EVB_matrix.h"
#include "EVB_engine.h"
#include "EVB_effpair.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.00001

/* ---------------------------------------------------------------------- */

EVB_EwaldACC::EVB_EwaldACC(LAMMPS *lmp, int narg, char **arg) : 
  EVB_Ewald(lmp, narg, arg)
{
  kmax_acc = 0;
  kcount_acc = 0;
  
  kxvecs_acc = kyvecs_acc = kzvecs_acc = NULL;
  ug_acc = NULL;
  eg_acc = vg_acc = NULL;

  sfacrl_env_acc = NULL;
  sfacim_env_acc = NULL;

  eikrrl_acc = eikrim_acc = NULL;
}

/* ----------------------------------------------------------------------
   free all memory 
------------------------------------------------------------------------- */

EVB_EwaldACC::~EVB_EwaldACC()
{

}

/* ----------------------------------------------------------------------
   adjust EVB_Ewald coeffs, called initially and whenever volume has changed 
------------------------------------------------------------------------- */

void EVB_EwaldACC::setup()
{
  evb_engine->flag_ACC = 1;

  // volume-dependent factors

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  
  // adjustment of z dimension for 2d slab EVB_Ewald
  // 3d EVB_Ewald just uses zprd since slab_volfactor = 1.0

  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  unitk[0] = 2.0*MY_PI/xprd;
  unitk[1] = 2.0*MY_PI/yprd;
  unitk[2] = 2.0*MY_PI/zprd_slab;

  // determine kmax
  // function of current box size, precision, G_ewald (short-range cutoff)

  bigint natoms = atom->natoms;
  double err;
  kxmax = 1;
  kymax = 1;
  kzmax = 1;
  
  err = rms(kxmax,xprd,natoms,q2);
  while (err > accuracy) {
    kxmax++;
    err = rms(kxmax,xprd,natoms,q2);
  }

  err = rms(kymax,yprd,natoms,q2);
  while (err > accuracy) {
    kymax++;
    err = rms(kymax,yprd,natoms,q2);
  }

  err = rms(kzmax,zprd_slab,natoms,q2);
  while (err > accuracy) {
    kzmax++;
    err = rms(kzmax,zprd_slab,natoms,q2);
  } 

  int kmax_old = kmax;
  kmax = MAX(kxmax,kymax);
  kmax = MAX(kmax,kzmax);
  kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

  double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
  double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
  double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
  gsqmx = MAX(gsqxmx,gsqymx);
  gsqmx = MAX(gsqmx,gsqzmx);
  gsqmx *= 1.00001;

  kxmax_acc = kxmax / 2; // Factor of 2 or kmax_acc could be input parameter
  kymax_acc = kymax / 2;
  kzmax_acc = kzmax / 2;

  kmax_acc = MAX(kxmax_acc,kymax_acc);
  kmax_acc = MAX(kmax_acc, kzmax_acc);
  kmax3d_acc = 4*kmax_acc*kmax_acc*kmax_acc + 6*kmax_acc*kmax_acc + 3*kmax_acc;

  // if size has grown, reallocate k-dependent and nlocal-dependent arrays

  if (kmax > kmax_old) {
    deallocate();
    deallocate_acc();
    allocate();
    allocate_acc();
    
    memory->destroy(ek);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    
    nmax = atom->nmax;
    memory->create(ek,nmax,3,"ewald:ek");
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald:sn");

    memory->destroy(eikrrl);
    memory->destroy(eikrim);
    memory->create(eikrrl,kmax3d,nmax_cplx,"ewald:eikrrl");
    memory->create(eikrim,kmax3d,nmax_cplx,"ewald:eikrim");
    
    kmax_created = kmax;
  }

  // pre-compute EVB_Ewald coefficients

  coeffs();
  coeffs_acc();
}

/* ---------------------------------------------------------------------- */

void EVB_EwaldACC::evb_setup()
{  
  // extend size of per-atom arrays if necessary
  
  int nlocal = atom->nlocal;
  
  if (atom->nlocal > nmax) {
    
    memory->destroy(ek);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    
    nmax = atom->nmax;
    memory->create(ek,nmax,3,"ewald:ek");
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald:sn");
    kmax_created = kmax;
  }
  
  int _nmax_cplx = 0;
  
  for(int i=0; i<evb_engine->ncomplex; i++)
    if(evb_engine->all_complex[i]->nlocal_cplx >_nmax_cplx)
      _nmax_cplx = evb_engine->all_complex[i]->nlocal_cplx;
  
  if (_nmax_cplx > nmax_cplx) 
    {
      nmax_cplx = _nmax_cplx;
      
      memory->destroy(eikrrl);
      memory->destroy(eikrim);
      memory->create(eikrrl,kmax3d,nmax_cplx,"ewald:eikrrl");
      memory->create(eikrim,kmax3d,nmax_cplx,"ewald:eikrim");
      
      memory->destroy(eikrrl_acc);
      memory->destroy(eikrim_acc);
      memory->create(eikrrl_acc,kmax3d_acc,nmax_cplx,"ewald:eikrrl_acc");
      memory->create(eikrim_acc,kmax3d_acc,nmax_cplx,"ewald:eikrim_acc");
    }
}
/* ---------------------------------------------------------------------- */

void EVB_EwaldACC::compute_env(int vflag)
{
  int i,n,k;
  int * is_cplx_atom = evb_engine->complex_atom;
  
  energy = 0.0;
  if (vflag) for (n = 0; n < 6; n++) virial[n] = 0.0;
  
  // partial structure factors on each processor
  // total structure factor by summing over procs

  eik_dot_r_env();
  eik_dot_r_env_acc();
   
  double uk;
  
  for (k = 0; k < kcount; k++) {
      uk = ug[k] * (sfacrl_env[k]*sfacrl_env[k] + 
                    sfacim_env[k]*sfacim_env[k]);
      energy += uk;
      if (vflag)
          for (n = 0; n < 6; n++) virial[n] += uk*vg[k][n];
  }

  qsqsum = evb_engine->qsqsum_env = evb_engine->qsqsum_sys - evb_engine->evb_complex->qsqsum;

  energy -= g_ewald*qsqsum/1.772453851 + 
      0.5*MY_PI*qsum*qsum / (g_ewald*g_ewald*volume);
  energy *= qqrd2e;

  if (vflag)
      for (n = 0; n < 6; n++) virial[n] *= qqrd2e;
  
  energy /= comm->nprocs;

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
}

/* --------------------------------------------------------------------- */

void EVB_EwaldACC::compute_cplx(int vflag)
{
  int i, k, n, iatm;

  energy = 0.0; 
  if (vflag) for (n = 0; n < 6; n++) virial[n] = 0.0; 
  
  eik_dot_r_cplx_acc();

  for (k = 0; k < kcount_acc; k++) {
    sfacrl[k] = sfacrl_cplx[k] + sfacrl_env_acc[k];
    sfacim[k] = sfacim_cplx[k] + sfacim_env_acc[k];
  }
  
  double **f = atom->f;
  double *q = atom->q;
  double partial;


  // electric field on cplx from cplx
  
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  
  // energy, virial if requested

  double uk;
  
  for (k = 0; k < kcount_acc; k++) {
    uk = 2.0 * ug_acc[k] * (sfacrl_cplx[k]*sfacrl_env_acc[k] +
			    sfacim_cplx[k]*sfacim_env_acc[k]);
    uk += ug_acc[k] * (sfacrl_cplx[k]*sfacrl_cplx[k] +
		       sfacim_cplx[k]*sfacim_cplx[k]);
    energy += uk;
    if (vflag)
      for (n = 0; n < 6; n++) virial[n] += uk*vg_acc[k][n];
  }

  qsqsum = evb_engine->evb_complex->qsqsum;
  
  energy -= g_ewald*qsqsum/1.772453851;

  energy *= qqrd2e;
  if (vflag)
    for (n = 0; n < 6; n++) virial[n] *= qqrd2e;
 
  energy /= comm->nprocs;

  // Add cplx contribution to dipole for slab correction
  // Don't calculate slab correction here in SCI simulations
  if(slabflag && evb_engine->ncomplex == 1) slabcorr_cplx();
}

/* ---------------------------------------------------------------------- */

void EVB_EwaldACC::compute_exch(int vflag)
{
  off_diag_energy = 0.0;
  if (vflag) for (int n = 0; n < 6; n++) off_diag_virial[n] = 0.0;
}

/* ---------------------------------------------------------------------- */

void EVB_EwaldACC::compute_eff(int vflag)
{
  int i,k,n;
  int iatm;
  
  if(evb_engine->ncomplex==1) eik_dot_r_cplx();
  else
  {
    for(k=0; k<kcount; k++) sfacrl_inter[k] = sfacim_inter[k] = 0.0;

    for(int i=0; i<evb_engine->ncomplex; i++)
    {
      evb_engine->evb_complex = evb_engine->all_complex[i];
      sci_setup_init();
      eik_dot_r_cplx();
      for(k=0; k<kcount; k++) 
      {
	sfacrl_inter[k] += sfacrl_cplx[k];
	sfacim_inter[k] += sfacim_cplx[k];
      }
    }

    for(k=0; k<kcount; k++)
    {
      sfacrl_cplx[k] = sfacrl_inter[k];
      sfacim_cplx[k] = sfacim_inter[k];
    }
  }
 
  for (k = 0; k < kcount; k++) {
    sfacrl[k] = sfacrl_cplx[k] + sfacrl_env[k];
    sfacim[k] = sfacim_cplx[k] + sfacim_env[k];
  }

  // K-space portion of electric field
  // double loop over K-vectors and local atoms

  double **f = atom->f;
  double  *q = atom->q;
  
  // electric field on env from cplx

  int nlocal = atom->nlocal;
  int* complex_atom = evb_engine->complex_atom;
  
  for (iatm = 0; iatm < nlocal; iatm++) {
    ek[iatm][0] = 0.0;
    ek[iatm][1] = 0.0;
    ek[iatm][2] = 0.0;
  }
  
  int kx, ky, kz;
  double cypz,sypz,exprl,expim,partial;
  
  for (k = 0; k < kcount; k++) {
	
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];
    
    for (iatm = 0; iatm < nlocal; iatm++) {
      cypz = cs[ky][1][iatm]*cs[kz][2][iatm] - sn[ky][1][iatm]*sn[kz][2][iatm];
      sypz = sn[ky][1][iatm]*cs[kz][2][iatm] + cs[ky][1][iatm]*sn[kz][2][iatm];
      exprl = cs[kx][0][iatm]*cypz - sn[kx][0][iatm]*sypz;
      expim = sn[kx][0][iatm]*cypz + cs[kx][0][iatm]*sypz;
      partial = expim*sfacrl[k] - exprl*sfacim[k];
      ek[iatm][0] += partial*eg[k][0];
      ek[iatm][1] += partial*eg[k][1];
      ek[iatm][2] += partial*eg[k][2];
    }
  }
  
  // convert E-field to force on env from cplx
  
  for (iatm = 0; iatm < nlocal ; iatm++) {
    double scale = qqrd2e * q[iatm];
    f[iatm][0] += scale * ek[iatm][0];
    f[iatm][1] += scale * ek[iatm][1];
    f[iatm][2] += scale * ek[iatm][2];
  }
  
  // Slab correction to forces
  if(slabflag) {
    double **x = atom->x;
    int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
    int *cplx_list = evb_engine->evb_complex->cplx_list;
    
    double dipole = 0.0;
    for(int i=0; i<nlocal_cplx; i++) {
      iatm = cplx_list[i];
      dipole += q[iatm] * x[iatm][2];
    }

    double dipole_all;
    MPI_Allreduce(&dipole,&dipole_all,1,MPI_DOUBLE,MPI_SUM,world);
    dipole_all += dipole_env; // Total System Dipole

    double ffact = -4.0 * MY_PI * qqrd2e * dipole_all / volume;
    for(int i=0; i<nlocal; i++) f[i][2] += q[i] * ffact;
  }
}

/* ---------------------------------------------------------------------- */

void EVB_EwaldACC::eik_dot_r_env_acc()
{
  int i,k,l,m,n,ic;
  int iatm;
  int i_cplx;
  double cstr1,sstr1,cstr2,sstr2,cstr3,sstr3,cstr4,sstr4;
  double sqk,clpm,slpm;

  double **x = atom->x;
  double *q = atom->q;
  int nlocal = atom->nlocal;
  int cplx_id = evb_engine->evb_complex->id;
  int *complex_atom = evb_engine->complex_atom;

  n = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      cstr1 = 0.0;
      sstr1 = 0.0;
      i_cplx = 0;
      for (i = 0; i < nlocal; i++) {
        cs[0][ic][i] = 1.0;
        sn[0][ic][i] = 0.0;
        cs[1][ic][i] = cos(unitk[ic]*x[i][ic]);
        sn[1][ic][i] = sin(unitk[ic]*x[i][ic]);
        cs[-1][ic][i] = cs[1][ic][i];
        sn[-1][ic][i] = -sn[1][ic][i];
        if (complex_atom[i]==cplx_id) {
          eikrrl_acc[n][i_cplx] = cs[1][ic][i];
          eikrim_acc[n][i_cplx++] = sn[1][ic][i];
        }
        if (complex_atom[i]==0) {
          cstr1 += q[i]*cs[1][ic][i];
          sstr1 += q[i]*sn[1][ic][i];
        }
      }
      sfacrl[n] = cstr1;
      sfacim[n++] = sstr1;
    }
  }

  for (m = 2; m <= kmax_acc; m++) {
    for (ic = 0; ic < 3; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
          cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
            sn[m-1][ic][i]*sn[1][ic][i];
          sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
            cs[m-1][ic][i]*sn[1][ic][i];
          cs[-m][ic][i] = cs[m][ic][i];
          sn[-m][ic][i] = -sn[m][ic][i];
          if (complex_atom[i]==cplx_id) {
            eikrrl_acc[n][i_cplx] = cs[m][ic][i];
            eikrim_acc[n][i_cplx++] = sn[m][ic][i];
          }
          if (complex_atom[i]==0) {
            cstr1 += q[i]*cs[m][ic][i];
            sstr1 += q[i]*sn[m][ic][i];
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax_acc; k++) {
    for (l = 1; l <= kymax_acc; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
          if (complex_atom[i]==cplx_id) {
            eikrrl_acc[n][i_cplx] = cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i];
            eikrim_acc[n][i_cplx] = sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i];
            eikrrl_acc[n+1][i_cplx] = cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i];
            eikrim_acc[n+1][i_cplx++] = sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i];
          }
          if(complex_atom[i]==0) {
            cstr1 += q[i]*(cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i]);
            sstr1 += q[i]*(sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i]);
            cstr2 += q[i]*(cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i]);
            sstr2 += q[i]*(sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i]);
          }
        } 
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax_acc; l++) {
    for (m = 1; m <= kzmax_acc; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
          if (complex_atom[i]==cplx_id) {
            eikrrl_acc[n][i_cplx] = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            eikrim_acc[n][i_cplx] = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            eikrrl_acc[n+1][i_cplx] = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            eikrim_acc[n+1][i_cplx++] = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
          }
          if (complex_atom[i]==0) {
            cstr1 += q[i]*(cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i]);
            sstr1 += q[i]*(sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i]);
            cstr2 += q[i]*(cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i]);
            sstr2 += q[i]*(sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i]);
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax_acc; k++) {
    for (m = 1; m <= kzmax_acc; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
          if (complex_atom[i]==cplx_id) {
            eikrrl_acc[n][i_cplx] = cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i];
            eikrim_acc[n][i_cplx] = sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i];
            eikrrl_acc[n+1][i_cplx] = cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i];
            eikrim_acc[n+1][i_cplx++] = sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i];
          }
          if( complex_atom[i]==0) {
            cstr1 += q[i]*(cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i]);
            sstr1 += q[i]*(sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i]);
            cstr2 += q[i]*(cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i]);
            sstr2 += q[i]*(sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i]);
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax_acc; k++) {
    for (l = 1; l <= kymax_acc; l++) { 
      for (m = 1; m <= kzmax_acc; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          cstr3 = 0.0;
          sstr3 = 0.0;
          cstr4 = 0.0;
          sstr4 = 0.0;
          i_cplx = 0;
          for (i = 0; i < nlocal; i++) {
            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            if (complex_atom[i]==cplx_id) {
              eikrrl_acc[n][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim_acc[n][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
            }
            if (complex_atom[i]==0) {
              cstr1 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr1 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            if (complex_atom[i]==cplx_id) {
              eikrrl_acc[n+1][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim_acc[n+1][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
            }
            if (complex_atom[i]==0) {
              cstr2 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr2 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            if (complex_atom[i]==cplx_id) {
              eikrrl_acc[n+2][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim_acc[n+2][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
            }
            if(complex_atom[i]==0)  {
              cstr3 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr3 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }

            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            if (complex_atom[i]) {
              eikrrl_acc[n+3][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim_acc[n+3][i_cplx++] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
            }
            if(complex_atom[i]==0) {
              cstr4 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr4 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }
          }
          sfacrl[n] = cstr1;
          sfacim[n++] = sstr1;
          sfacrl[n] = cstr2;
          sfacim[n++] = sstr2;
          sfacrl[n] = cstr3;
          sfacim[n++] = sstr3;
          sfacrl[n] = cstr4;
          sfacim[n++] = sstr4;
        }
      }
    }
  }

   
  MPI_Allreduce(sfacrl,sfacrl_env_acc,kcount_acc,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_env_acc,kcount_acc,MPI_DOUBLE,MPI_SUM,world);

}

/* ---------------------------------------------------------------------- */

void EVB_EwaldACC::eik_dot_r_cplx_acc()
{
  int i,k;
  int iatm;
  double *q = atom->q;

  int ncplx_local = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  
  for (k = 0; k < kcount_acc; k++) {
      sfacrl[k] = 0.0;
      sfacim[k] = 0.0;
      for (i = 0; i < ncplx_local; i++) {
          iatm = cplx_list[i];
          
          sfacrl[k] += q[iatm] * eikrrl_acc[k][i];
          sfacim[k] += q[iatm] * eikrim_acc[k][i];
      }
  }
  
  MPI_Allreduce(sfacrl,sfacrl_cplx,kcount_acc,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_cplx,kcount_acc,MPI_DOUBLE,MPI_SUM,world);
}

/* ---------------------------------------------------------------------- */

void EVB_EwaldACC::eik_dot_r_exch()
{

}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each EVB_Ewald K-vector 
------------------------------------------------------------------------- */

void EVB_EwaldACC::coeffs_acc()
{
  int k,l,m;
  double sqk,vterm;

  double unitkx = unitk[0];
  double unitky = unitk[1];
  double unitkz = unitk[2];
  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MY_PI/volume;

  kcount_acc = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (m = 1; m <= kmax_acc; m++) {
    sqk = (m*unitkx) * (m*unitkx);
    if (sqk <= gsqmx) {
      kxvecs_acc[kcount_acc] = m;
      kyvecs_acc[kcount_acc] = 0;
      kzvecs_acc[kcount_acc] = 0;
      ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg_acc[kcount_acc][0] = 2.0*unitkx*m*ug_acc[kcount_acc];
      eg_acc[kcount_acc][1] = 0.0;
      eg_acc[kcount_acc][2] = 0.0;
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg_acc[kcount_acc][0] = 1.0 + vterm*(unitkx*m)*(unitkx*m);
      vg_acc[kcount_acc][1] = 1.0;
      vg_acc[kcount_acc][2] = 1.0;
      vg_acc[kcount_acc][3] = 0.0;
      vg_acc[kcount_acc][4] = 0.0;
      vg_acc[kcount_acc][5] = 0.0;
      kcount_acc++;
    }
    sqk = (m*unitky) * (m*unitky);
    if (sqk <= gsqmx) {
      kxvecs_acc[kcount_acc] = 0;
      kyvecs_acc[kcount_acc] = m;
      kzvecs_acc[kcount_acc] = 0;
      ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg_acc[kcount_acc][0] = 0.0;
      eg_acc[kcount_acc][1] = 2.0*unitky*m*ug_acc[kcount_acc];
      eg_acc[kcount_acc][2] = 0.0;
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg_acc[kcount_acc][0] = 1.0;
      vg_acc[kcount_acc][1] = 1.0 + vterm*(unitky*m)*(unitky*m);
      vg_acc[kcount_acc][2] = 1.0;
      vg_acc[kcount_acc][3] = 0.0;
      vg_acc[kcount_acc][4] = 0.0;
      vg_acc[kcount_acc][5] = 0.0;
      kcount_acc++;
    }
    sqk = (m*unitkz) * (m*unitkz);
    if (sqk <= gsqmx) {
      kxvecs_acc[kcount_acc] = 0;
      kyvecs_acc[kcount_acc] = 0;
      kzvecs_acc[kcount_acc] = m;
      ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg_acc[kcount_acc][0] = 0.0;
      eg_acc[kcount_acc][1] = 0.0;
      eg_acc[kcount_acc][2] = 2.0*unitkz*m*ug_acc[kcount_acc];
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg_acc[kcount_acc][0] = 1.0;
      vg_acc[kcount_acc][1] = 1.0;
      vg_acc[kcount_acc][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
      vg_acc[kcount_acc][3] = 0.0;
      vg_acc[kcount_acc][4] = 0.0;
      vg_acc[kcount_acc][5] = 0.0;
      kcount_acc++;
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax_acc; k++) {
    for (l = 1; l <= kymax_acc; l++) {
      sqk = (unitkx*k) * (unitkx*k) + (unitky*l) * (unitky*l);
      if (sqk <= gsqmx) {
	kxvecs_acc[kcount_acc] = k;
	kyvecs_acc[kcount_acc] = l;
	kzvecs_acc[kcount_acc] = 0;
	ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg_acc[kcount_acc][0] = 2.0*unitkx*k*ug_acc[kcount_acc];
	eg_acc[kcount_acc][1] = 2.0*unitky*l*ug_acc[kcount_acc];
	eg_acc[kcount_acc][2] = 0.0;
	vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
	vg_acc[kcount_acc][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	vg_acc[kcount_acc][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	vg_acc[kcount_acc][2] = 1.0;
	vg_acc[kcount_acc][3] = vterm*unitkx*k*unitky*l;
	vg_acc[kcount_acc][4] = 0.0;
	vg_acc[kcount_acc][5] = 0.0;
	kcount_acc++;

	kxvecs_acc[kcount_acc] = k;
	kyvecs_acc[kcount_acc] = -l;
	kzvecs_acc[kcount_acc] = 0;
	ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg_acc[kcount_acc][0] = 2.0*unitkx*k*ug_acc[kcount_acc];
	eg_acc[kcount_acc][1] = -2.0*unitky*l*ug_acc[kcount_acc];
	eg_acc[kcount_acc][2] = 0.0;
	vg_acc[kcount_acc][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	vg_acc[kcount_acc][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	vg_acc[kcount_acc][2] = 1.0;
	vg_acc[kcount_acc][3] = -vterm*unitkx*k*unitky*l;
	vg_acc[kcount_acc][4] = 0.0;
	vg_acc[kcount_acc][5] = 0.0;
	kcount_acc++;;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax_acc; l++) {
    for (m = 1; m <= kzmax_acc; m++) {
      sqk = (unitky*l) * (unitky*l) + (unitkz*m) * (unitkz*m);
      if (sqk <= gsqmx) {
	kxvecs_acc[kcount_acc] = 0;
	kyvecs_acc[kcount_acc] = l;
	kzvecs_acc[kcount_acc] = m;
	ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg_acc[kcount_acc][0] =  0.0;
	eg_acc[kcount_acc][1] =  2.0*unitky*l*ug_acc[kcount_acc];
	eg_acc[kcount_acc][2] =  2.0*unitkz*m*ug_acc[kcount_acc];
	vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
	vg_acc[kcount_acc][0] = 1.0;
	vg_acc[kcount_acc][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	vg_acc[kcount_acc][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	vg_acc[kcount_acc][3] = 0.0;
	vg_acc[kcount_acc][4] = 0.0;
	vg_acc[kcount_acc][5] = vterm*unitky*l*unitkz*m;
	kcount_acc++;

	kxvecs_acc[kcount_acc] = 0;
	kyvecs_acc[kcount_acc] = l;
	kzvecs_acc[kcount_acc] = -m;
	ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg_acc[kcount_acc][0] =  0.0;
	eg_acc[kcount_acc][1] =  2.0*unitky*l*ug_acc[kcount_acc];
	eg_acc[kcount_acc][2] = -2.0*unitkz*m*ug_acc[kcount_acc];
	vg_acc[kcount_acc][0] = 1.0;
	vg_acc[kcount_acc][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	vg_acc[kcount_acc][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	vg_acc[kcount_acc][3] = 0.0;
	vg_acc[kcount_acc][4] = 0.0;
	vg_acc[kcount_acc][5] = -vterm*unitky*l*unitkz*m;
	kcount_acc++;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax_acc; k++) {
    for (m = 1; m <= kzmax_acc; m++) {
      sqk = (unitkx*k) * (unitkx*k) + (unitkz*m) * (unitkz*m);
      if (sqk <= gsqmx) {
	kxvecs_acc[kcount_acc] = k;
	kyvecs_acc[kcount_acc] = 0;
	kzvecs_acc[kcount_acc] = m;
	ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg_acc[kcount_acc][0] =  2.0*unitkx*k*ug_acc[kcount_acc];
	eg_acc[kcount_acc][1] =  0.0;
	eg_acc[kcount_acc][2] =  2.0*unitkz*m*ug_acc[kcount_acc];
	vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
	vg_acc[kcount_acc][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	vg_acc[kcount_acc][1] = 1.0;
	vg_acc[kcount_acc][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	vg_acc[kcount_acc][3] = 0.0;
	vg_acc[kcount_acc][4] = vterm*unitkx*k*unitkz*m;
	vg_acc[kcount_acc][5] = 0.0;
	kcount_acc++;

	kxvecs_acc[kcount_acc] = k;
	kyvecs_acc[kcount_acc] = 0;
	kzvecs_acc[kcount_acc] = -m;
	ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg_acc[kcount_acc][0] =  2.0*unitkx*k*ug_acc[kcount_acc];
	eg_acc[kcount_acc][1] =  0.0;
	eg_acc[kcount_acc][2] = -2.0*unitkz*m*ug_acc[kcount_acc];
	vg_acc[kcount_acc][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	vg_acc[kcount_acc][1] = 1.0;
	vg_acc[kcount_acc][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	vg_acc[kcount_acc][3] = 0.0;
	vg_acc[kcount_acc][4] = -vterm*unitkx*k*unitkz*m;
	vg_acc[kcount_acc][5] = 0.0;
	kcount_acc++;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax_acc; k++) {
    for (l = 1; l <= kymax_acc; l++) {
      for (m = 1; m <= kzmax_acc; m++) {
	sqk = (unitkx*k) * (unitkx*k) + (unitky*l) * (unitky*l) + 
	  (unitkz*m) * (unitkz*m);
	if (sqk <= gsqmx) {
	  kxvecs_acc[kcount_acc] = k;
	  kyvecs_acc[kcount_acc] = l;
	  kzvecs_acc[kcount_acc] = m;
	  ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	  eg_acc[kcount_acc][0] = 2.0*unitkx*k*ug_acc[kcount_acc];
	  eg_acc[kcount_acc][1] = 2.0*unitky*l*ug_acc[kcount_acc];
	  eg_acc[kcount_acc][2] = 2.0*unitkz*m*ug_acc[kcount_acc];
	  vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
	  vg_acc[kcount_acc][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	  vg_acc[kcount_acc][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	  vg_acc[kcount_acc][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	  vg_acc[kcount_acc][3] = vterm*unitkx*k*unitky*l;
	  vg_acc[kcount_acc][4] = vterm*unitkx*k*unitkz*m;
	  vg_acc[kcount_acc][5] = vterm*unitky*l*unitkz*m;
	  kcount_acc++;

	  kxvecs_acc[kcount_acc] = k;
	  kyvecs_acc[kcount_acc] = -l;
	  kzvecs_acc[kcount_acc] = m;
	  ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	  eg_acc[kcount_acc][0] = 2.0*unitkx*k*ug_acc[kcount_acc];
	  eg_acc[kcount_acc][1] = -2.0*unitky*l*ug_acc[kcount_acc];
	  eg_acc[kcount_acc][2] = 2.0*unitkz*m*ug_acc[kcount_acc];
	  vg_acc[kcount_acc][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	  vg_acc[kcount_acc][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	  vg_acc[kcount_acc][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	  vg_acc[kcount_acc][3] = -vterm*unitkx*k*unitky*l;
	  vg_acc[kcount_acc][4] = vterm*unitkx*k*unitkz*m;
	  vg_acc[kcount_acc][5] = -vterm*unitky*l*unitkz*m;
	  kcount_acc++;

	  kxvecs_acc[kcount_acc] = k;
	  kyvecs_acc[kcount_acc] = l;
	  kzvecs_acc[kcount_acc] = -m;
	  ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	  eg_acc[kcount_acc][0] = 2.0*unitkx*k*ug_acc[kcount_acc];
	  eg_acc[kcount_acc][1] = 2.0*unitky*l*ug_acc[kcount_acc];
	  eg_acc[kcount_acc][2] = -2.0*unitkz*m*ug_acc[kcount_acc];
	  vg_acc[kcount_acc][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	  vg_acc[kcount_acc][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	  vg_acc[kcount_acc][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	  vg_acc[kcount_acc][3] = vterm*unitkx*k*unitky*l;
	  vg_acc[kcount_acc][4] = -vterm*unitkx*k*unitkz*m;
	  vg_acc[kcount_acc][5] = -vterm*unitky*l*unitkz*m;
	  kcount_acc++;

	  kxvecs_acc[kcount_acc] = k;
	  kyvecs_acc[kcount_acc] = -l;
	  kzvecs_acc[kcount_acc] = -m;
	  ug_acc[kcount_acc] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	  eg_acc[kcount_acc][0] = 2.0*unitkx*k*ug_acc[kcount_acc];
	  eg_acc[kcount_acc][1] = -2.0*unitky*l*ug_acc[kcount_acc];
	  eg_acc[kcount_acc][2] = -2.0*unitkz*m*ug_acc[kcount_acc];
	  vg_acc[kcount_acc][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	  vg_acc[kcount_acc][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	  vg_acc[kcount_acc][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	  vg_acc[kcount_acc][3] = -vterm*unitkx*k*unitky*l;
	  vg_acc[kcount_acc][4] = -vterm*unitkx*k*unitkz*m;
	  vg_acc[kcount_acc][5] = vterm*unitky*l*unitkz*m;
	  kcount_acc++;;
	}
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors 
------------------------------------------------------------------------- */

void EVB_EwaldACC::allocate_acc()
{
  kxvecs_acc = new int[kmax3d_acc];
  kyvecs_acc = new int[kmax3d_acc];
  kzvecs_acc = new int[kmax3d_acc];
  
  ug_acc = new double[kmax3d_acc];
  memory->create(eg_acc,kmax3d_acc,3,"ewald:eg_acc");
  memory->create(vg_acc,kmax3d_acc,6,"ewald:vg_acc");

  // /***** EVB part *****/
  
  sfacrl_env_acc = new double[kmax3d_acc];
  sfacim_env_acc = new double[kmax3d_acc];
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors 
------------------------------------------------------------------------- */

void EVB_EwaldACC::deallocate_acc()
{
  delete [] kxvecs_acc;
  delete [] kyvecs_acc;
  delete [] kzvecs_acc;
	
  delete [] ug_acc;
  memory->destroy(eg_acc);
  memory->destroy(vg_acc);

  // /***** EVB part *****/  
  
  delete [] sfacrl_env_acc;
  delete [] sfacim_env_acc;
}

/* ----------------------------------------------------------------------
   memory usage of local arrays 
------------------------------------------------------------------------- */

double EVB_EwaldACC::memory_usage()
{
  double bytes = 3 * kmax3d * sizeof(int);
  bytes += (1 + 3 + 6) * kmax3d * sizeof(double);
  bytes += 4 * kmax3d * sizeof(double);
  bytes += nmax*3 * sizeof(double);
  bytes += 2 * (2*kmax+1)*3*nmax * sizeof(double);
  return bytes;
}


