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
#include "EVB_offdiag.h"
#include "EVB_complex.h"
#include "EVB_matrix.h"
#include "EVB_engine.h"
#include "EVB_effpair.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.00001

/* ---------------------------------------------------------------------- */

EVB_Ewald::EVB_Ewald(LAMMPS *lmp, int narg, char **arg) : 
  EVB_KSpace(lmp, narg, arg)
{
  if (narg != 1) error->all(FLERR,"Illegal kspace_style ewald command");

  accuracy_relative = atof(arg[0]);

  kmax = 0;
  kxvecs = kyvecs = kzvecs = NULL;
  ug = NULL;
  eg = vg = NULL;
  sfacrl = sfacim = NULL;

  nmax = 0;
  ek = NULL; 
  cs = sn = NULL;

  kcount = 0;

  /***** EVB *****/
  bEff = true;

  sfacrl_env = sfacim_env = NULL;
  sfacrl_cplx = sfacim_cplx = NULL;
  sfacrl_exch = sfacim_exch = NULL;
  sfacrl_inter = sfacim_inter = NULL;
  
  nmax_cplx = 0;
  eikrrl = eikrim = NULL;
}

/* ----------------------------------------------------------------------
   free all memory 
------------------------------------------------------------------------- */

EVB_Ewald::~EVB_Ewald()
{
  deallocate();
  memory->destroy(ek);
  memory->destroy3d_offset(cs,-kmax_created);
  memory->destroy3d_offset(sn,-kmax_created);

  memory->destroy(eikrrl);
  memory->destroy(eikrim);
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::init()
{
  EVB_KSpace::init();

  if (comm->me == 0) {
    if (screen) fprintf(screen,"EVB_Ewald initialization ...\n");
    if (logfile) fprintf(logfile,"EVB_Ewald initialization ...\n");
  }

  // error check

  if (domain->triclinic) error->all(FLERR,"Cannot use EVB_Ewald with triclinic box");
  if (domain->dimension == 2) 
    error->all(FLERR,"Cannot use EVB_Ewald with 2d simulation");

  if (!atom->q_flag) error->all(FLERR,"Kspace style requires atom attribute q");

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use nonperiodic boundaries with EVB_Ewald");
  if (slabflag == 1) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 || 
	domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab EVB_Ewald");
  }

  if(comm->me == 0 && slabflag) {
    if(screen) fprintf(screen,"  Slab correction activated: volfactor = %f\n",slab_volfactor);
    if(logfile) fprintf(logfile,"  Slab correction activated: volfactor = %f\n",slab_volfactor);
  }

  // extract short-range Coulombic cutoff from pair style

  qqrd2e = force->qqrd2e;

  if (force->pair == NULL)
    error->all(FLERR,"KSpace style is incompatible with Pair style");
  int itmp;
  double *p_cutoff = (double *) force->pair->extract((char*)("cut_coul"),itmp);
  if (p_cutoff == NULL)
    error->all(FLERR,"KSpace style is incompatible with Pair style");
  double cutoff = *p_cutoff;

  qsum = qsqsum = 0.0;
  for (int i = 0; i < atom->nlocal; i++) {
    qsum += atom->q[i];
    qsqsum += atom->q[i]*atom->q[i];
  }

  double tmp;
  MPI_Allreduce(&qsum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_all = qsum = tmp;
  MPI_Allreduce(&qsqsum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsqsum_all = qsqsum = tmp;

  if (qsqsum == 0.0)
    error->all(FLERR,"Cannot use kspace solver on system with no charge");
  if (fabs(qsum) > SMALL && comm->me == 0) {
    char str[128];
    sprintf(str,"System is not charge neutral, net charge = %g",qsum);
    error->warning(FLERR,str);
  }

  // set accuracy (force units) from accuracy_relative or accuracy_absolute
  
  if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  else accuracy = accuracy_relative * two_charge_force;

  // setup K-space resolution

  q2 = qsqsum * force->qqrd2e / force->dielectric;
  bigint natoms = atom->natoms;

  // use xprd,yprd,zprd even if triclinic so grid size is the same
  // adjust z dimension for 2d slab Ewald
  // 3d Ewald just uses zprd since slab_volfactor = 1.0

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  
  // make initial g_ewald estimate
  // based on desired accuracy and real space cutoff
  // fluid-occupied volume used to estimate real-space error
  // zprd used rather than zprd_slab

  if (!gewaldflag) {
    g_ewald = accuracy*sqrt(natoms*cutoff*xprd*yprd*zprd) / (2.0*q2);
    if (g_ewald >= 1.0)
      error->all(FLERR,"KSpace accuracy too large to estimate G vector");
    g_ewald = sqrt(-log(g_ewald)) / cutoff;
  }

  // setup EVB_Ewald coefficients so can print stats

  setup();

  // final RMS accuracy

  double lprx = rms(kxmax,xprd,natoms,q2);
  double lpry = rms(kymax,yprd,natoms,q2);
  double lprz = rms(kzmax,zprd_slab,natoms,q2);
  double lpr = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
  double spr = 2.0*q2 * exp(-g_ewald*g_ewald*cutoff*cutoff) / 
    sqrt(natoms*cutoff*xprd*yprd*zprd_slab);

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen,"  G vector (1/distance) = %g\n",g_ewald);
      fprintf(screen,"  estimated absolute RMS force accuracy = %g\n",
	      MAX(lpr,spr));
      fprintf(screen,"  estimated relative force accuracy = %g\n",
	      MAX(lpr,spr)/two_charge_force);
      fprintf(screen,"  KSpace vectors: actual max1d max3d = %d %d %d\n",
	      kcount,kmax,kmax3d);
    }
    if (logfile) {
      fprintf(logfile,"  G vector (1/distnace) = %g\n",g_ewald);
      fprintf(logfile,"  estimated absolute RMS force accuracy = %g\n",
	      MAX(lpr,spr));
      fprintf(logfile,"  estimated relative force accuracy = %g\n",
	      MAX(lpr,spr)/two_charge_force);
      fprintf(logfile,"  KSpace vectors: actual max1d max3d = %d %d %d\n",
	      kcount,kmax,kmax3d);
    }
  }
}

/* ----------------------------------------------------------------------
   adjust EVB_Ewald coeffs, called initially and whenever volume has changed 
------------------------------------------------------------------------- */

void EVB_Ewald::setup()
{
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
  // function of current box size, accuracy, G_ewald (short-range cutoff)

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

  // if size has grown, reallocate k-dependent and nlocal-dependent arrays

  if (kmax > kmax_old) {
    deallocate();
    allocate();
    
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

}

/* ----------------------------------------------------------------------
   compute RMS accuracy for a dimension
------------------------------------------------------------------------- */

double EVB_Ewald::rms(int km, double prd, bigint natoms, double q2)
{
  double value = 2.0*q2*g_ewald/prd * 
    sqrt(1.0/(MY_PI*km*natoms)) * 
    exp(-MY_PI*MY_PI*km*km/(g_ewald*g_ewald*prd*prd));

  return value;
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::evb_setup()
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
    }
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::compute_env(int vflag)
{
  int i,n,k;
  int * is_cplx_atom = evb_engine->complex_atom;
  
  energy = 0.0;
  if (vflag) for (n = 0; n < 6; n++) virial[n] = 0.0;
  
  // partial structure factors on each processor
  // total structure factor by summing over procs

  eik_dot_r_env();
   
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
/* ---------------------------------------------------------------------- */

void EVB_Ewald::compute_env_density(int vflag)
{
  int i,n,k;
  
  // energy = 0.0;
  // if (vflag) for (n = 0; n < 6; n++) virial[n] = 0.0;
  
  // partial structure factors on each processor
  // total structure factor by summing over procs

  eik_dot_r_env();
   
  // double uk;
  
  // for (k = 0; k < kcount; k++) {
  //     uk = ug[k] * (sfacrl_env[k]*sfacrl_env[k] + 
  //                   sfacim_env[k]*sfacim_env[k]);
  //     energy += uk;
  //     if (vflag)
  //         for (n = 0; n < 6; n++) virial[n] += uk*vg[k][n];
  // }

  // qsqsum = evb_engine->qsqsum_env = evb_engine->qsqsum_sys - evb_engine->evb_complex->qsqsum;

  // energy -= g_ewald*qsqsum/1.772453851 + 
  //     0.5*MY_PI*qsum*qsum / (g_ewald*g_ewald*volume);
  // energy *= qqrd2e;

  // if (vflag)
  //     for (n = 0; n < 6; n++) virial[n] *= qqrd2e;
  
  // energy /= comm->nprocs;

  // // Environment contribution to dipole for slab correction
  // if(slabflag) {
  //   double *q = atom->q;
  //   double **x = atom->x;
    
  //   double dipole = 0.0;
  //   for(int i=0; i<atom->nlocal; i++) dipole += q[i] * x[i][2];
    
  //   MPI_Allreduce(&dipole,&dipole_env,1,MPI_DOUBLE,MPI_SUM,world);
  // }
}

/* --------------------------------------------------------------------- */

void EVB_Ewald::compute_cplx(int vflag)
{
  int i, k, n, iatm;

  energy = 0.0; 
  if (vflag) for (n = 0; n < 6; n++) virial[n] = 0.0; 
  
  eik_dot_r_cplx();

  for (k = 0; k < kcount; k++) {
    sfacrl[k] = sfacrl_cplx[k] + sfacrl_env[k];
    sfacim[k] = sfacim_cplx[k] + sfacim_env[k];
  }
  
  double **f = atom->f;
  double *q = atom->q;
  double partial;

  // electric field on cplx from cplx
  
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  
  for (i = 0; i < nlocal_cplx; i++) {
      iatm = cplx_list[i];
      ek[iatm][0] = 0.0;
      ek[iatm][1] = 0.0;
      ek[iatm][2] = 0.0;
  }
  
  for (k = 0; k < kcount; k++) {
      for (i = 0; i < nlocal_cplx; i++) {
          iatm = cplx_list[i];
          partial = eikrim[k][i]*sfacrl[k] - eikrrl[k][i]*sfacim[k];
          
          ek[iatm][0] += partial*eg[k][0];
          ek[iatm][1] += partial*eg[k][1];
          ek[iatm][2] += partial*eg[k][2];
      }
  }

  // convert E-field to force on cplx
  
  for (i = 0; i < nlocal_cplx; i++) {
      iatm = cplx_list[i];
      f[iatm][0] += qqrd2e*q[iatm]*ek[iatm][0];
      f[iatm][1] += qqrd2e*q[iatm]*ek[iatm][1];
      f[iatm][2] += qqrd2e*q[iatm]*ek[iatm][2];
  }
  
  // energy, virial if requested

  double uk;
  
  for (k = 0; k < kcount; k++) {
    uk = 2.0 * ug[k] * (sfacrl_cplx[k]*sfacrl_env[k] + sfacim_cplx[k]*sfacim_env[k]);
    uk += ug[k] * (sfacrl_cplx[k]*sfacrl_cplx[k] +
                   sfacim_cplx[k]*sfacim_cplx[k]);
    energy += uk;
    if (vflag)
      for (n = 0; n < 6; n++) virial[n] += uk*vg[k][n];
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

void EVB_Ewald::compute_exch(int vflag)
{
  int i,k,n;
  int iatm;

  off_diag_energy = 0.0;
  if (vflag) for (n = 0; n < 6; n++) off_diag_virial[n] = 0.0;
  
  eik_dot_r_exch();
	
  for (k = 0; k < kcount; k++) {
    sfacrl[k] += sfacrl_env[k];
    sfacim[k] += sfacim_env[k];
  }
  
  // K-space portion of electric field

  double **f = atom->f;
  double *q = atom->q;

  double partial;

  // electric field on exch 
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  
  for (i = 0; i < nlocal_cplx; i++) {
      iatm = cplx_list[i];
      ek[iatm][0] = 0.0;
      ek[iatm][1] = 0.0;
      ek[iatm][2] = 0.0;
  }

  for (k = 0; k < kcount; k++) {
      for (i = 0; i < nlocal_cplx; i++) {
          iatm = cplx_list[i];
          if (is_exch_chg[iatm]) partial = eikrim[k][i]*sfacrl[k] - eikrrl[k][i]*sfacim[k];
	  else partial = eikrim[k][i]*sfacrl_exch[k] - eikrrl[k][i]*sfacim_exch[k];
      
          ek[iatm][0] += partial*eg[k][0];
          ek[iatm][1] += partial*eg[k][1];
          ek[iatm][2] += partial*eg[k][2];
    }
  }

  // convert E-field to force on
  double new_qqrd2e = qqrd2e * A_Rq;
  
  for (i = 0; i < nlocal_cplx; i++) {
      iatm = cplx_list[i];
      f[iatm][0] += new_qqrd2e*q[iatm]*ek[iatm][0];
      f[iatm][1] += new_qqrd2e*q[iatm]*ek[iatm][1];
      f[iatm][2] += new_qqrd2e*q[iatm]*ek[iatm][2];
  }
     
  // energy, no self interactions for exchange charge interactions
  // virial if requested

  double uk;
  for (k = 0; k < kcount; k++) {
    uk = 2.0 * ug[k] * (sfacrl_exch[k]*sfacrl[k] +
                        sfacim_exch[k]*sfacim[k]);
    off_diag_energy += uk;
    if (vflag)
      for (n = 0; n < 6; n++) off_diag_virial[n] += uk*vg[k][n];
  }

  off_diag_energy *= qqrd2e;

  if (vflag)
    for (n = 0; n < 6; n++) off_diag_virial[n] *= new_qqrd2e;

  // Add cplx contribution to dipole for slab correction
  // Don't calculate slab correction here in SCI simulations
  if(slabflag && evb_engine->ncomplex == 1) slabcorr_exch();
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::compute_eff(int vflag)
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
  
  for (iatm = 0; iatm < nlocal; iatm++)
      if(!complex_atom[iatm])
      {
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
	
      for (iatm = 0; iatm < nlocal; iatm++)
          if(!complex_atom[iatm])
          {
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
  
  for (iatm = 0; iatm < nlocal ; iatm++)
    if(!complex_atom[iatm])
      {
	double scale = qqrd2e * q[iatm];
	f[iatm][0] += scale * ek[iatm][0];
	f[iatm][1] += scale * ek[iatm][1];
	f[iatm][2] += scale * ek[iatm][2];      
      }
  
  // Slab correction to forces
  if(slabflag) slabcorr_eff();
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if 
   adequate empty space is left between repeating slabs (J. Chem. Phys. 
   111, 3155).  Slabs defined here to be parallel to the xy plane. 
------------------------------------------------------------------------- */

void EVB_Ewald::slabcorr_cplx()
{
  // compute local cplx contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;

  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int * cplx_list = evb_engine->evb_complex->cplx_list;
  int * is_cplx_atom = evb_engine->complex_atom;

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

  // compute corrections

  const double e_slabcorr = MY_2PI * (dipole_all * dipole_all - qsum * dipole_r2_all - 
				      qsum * qsum * zprd * zprd / 12.0) / volume;
  
  energy += qqrd2e * e_slabcorr / comm->nprocs;

  // add on force corrections

  double ffact = -4.0 * MY_PI * qqrd2e / volume;
  double **f = atom->f;

  for(int i=0; i<nlocal; i++) f[i][2] += ffact * q[i] * (dipole_all - qsum*x[i][2]);
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void EVB_Ewald::slabcorr_exch()
{
  // compute local cplx contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;

  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;

  double dipole_cplx    = 0.0;
  double dipole_r2_cplx = 0.0;
  double dipole_exch    = 0.0;
  double dipole_r2_exch = 0.0;
  double qsum_exch      = 0.0;
  for(int i=0; i<nlocal_cplx; i++) {
    int iatm = cplx_list[i];
    double qx = q[iatm] * x[iatm][2];

    dipole_cplx    += qx;
    dipole_r2_cplx += qx * x[iatm][2];

    if(is_exch_chg[iatm]) {
      dipole_exch    += qx;
      dipole_r2_exch += qx * x[iatm][2];
      qsum_exch      += q[iatm];
    }
  }
  
  // sum local contributions to get global dipole moments
  
  double tmp = 0.0;
  MPI_Allreduce(&dipole_cplx, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  double dipole_all = dipole_env + tmp;

  tmp = 0.0;
  MPI_Allreduce(&dipole_r2_cplx, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  double dipole_r2_all = dipole_r2_env + tmp;
  
  tmp = dipole_exch;
  dipole_exch = 0.0;
  MPI_Allreduce(&tmp, &dipole_exch, 1, MPI_DOUBLE, MPI_SUM, world);

  tmp = dipole_r2_exch;
  dipole_r2_exch = 0.0;
  MPI_Allreduce(&tmp, &dipole_r2_exch, 1, MPI_DOUBLE, MPI_SUM, world);

  tmp = qsum_exch;
  qsum_exch = 0.0;
  MPI_Allreduce(&tmp, &qsum_exch, 1, MPI_DOUBLE, MPI_SUM, world);

  // compute corrections
  double dipole_mexch    = dipole_all - dipole_exch;
  double dipole_r2_mexch = dipole_r2_all - dipole_r2_exch;
  double qsum_mexch      = qsum - qsum_exch;

  double dip_int    = dipole_all * dipole_all - dipole_mexch * dipole_mexch - dipole_exch * dipole_exch;
  double dip_r2_int = qsum * dipole_r2_all - (qsum_mexch * dipole_r2_mexch) - (qsum_exch * dipole_r2_exch);
  double qsum_int   = qsum * qsum - qsum_mexch * qsum_mexch - qsum_exch * qsum_exch;

  const double e_slabcorr = MY_2PI * (dip_int - dip_r2_int - qsum_int * zprd * zprd / 12.0) / volume;

  off_diag_energy += qqrd2e * e_slabcorr / comm->nprocs;

  // add on force corrections

  // dip_int  = dipole_all - dipole_mexch - dipole_exch; // dip_int is identically zero
  // qsum_int = qsum - qsum_mexch - qsum_exch; // qsum_int is identically zero

  // double ffact = -4.0 * MY_PI * qqrd2e / volume;
  // double **f = atom->f;

  // for(int i=0; i<nlocal; i++) f[i][2] += ffact * q[i] * (dip_int - qsum_int * x[i][2]);
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::slabcorr_eff()
{
  // compute local cplx contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  int nlocal = atom->nlocal;

  int cplx_id = evb_engine->evb_complex->id;
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int * cplx_list = evb_engine->evb_complex->cplx_list;
  int * is_cplx_atom = evb_engine->complex_atom;

  double dipole_cplx    = 0.0;
  for(int i=0; i<nlocal; i++) if(is_cplx_atom[i]) dipole_cplx    += q[i] * x[i][2];
  
  // sum local contributions to get global dipole moment
  
  double dipole_all    = 0.0;
  MPI_Allreduce(&dipole_cplx,    &dipole_all,    1, MPI_DOUBLE, MPI_SUM, world);

  dipole_all    += dipole_env; // Total System Dipole

  // add on force corrections

  double ffact = -4.0 * MY_PI * qqrd2e / volume;
  double **f = atom->f;

  //  for(int i=0; i<nlocal; i++) if(is_cplx_atom[i] == 0) f[i][2] += ffact * q[i] * (dipole_all - qsum*x[i][2]);
  //for(int i=0; i<nlocal; i++) f[i][2] += ffact * q[i] * (dipole_all - qsum*x[i][2]);
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::eik_dot_r_env()
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
          eikrrl[n][i_cplx] = cs[1][ic][i];
          eikrim[n][i_cplx++] = sn[1][ic][i];
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

  for (m = 2; m <= kmax; m++) {
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
            eikrrl[n][i_cplx] = cs[m][ic][i];
            eikrim[n][i_cplx++] = sn[m][ic][i];
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

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
          if (complex_atom[i]==cplx_id) {
            eikrrl[n][i_cplx] = cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i];
            eikrim[n][i_cplx] = sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i];
            eikrrl[n+1][i_cplx] = cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i];
            eikrim[n+1][i_cplx++] = sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i];
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

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
          if (complex_atom[i]==cplx_id) {
            eikrrl[n][i_cplx] = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            eikrim[n][i_cplx] = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            eikrrl[n+1][i_cplx] = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            eikrim[n+1][i_cplx++] = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
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

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
          if (complex_atom[i]==cplx_id) {
            eikrrl[n][i_cplx] = cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i];
            eikrim[n][i_cplx] = sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i];
            eikrrl[n+1][i_cplx] = cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i];
            eikrim[n+1][i_cplx++] = sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i];
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

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) { 
      for (m = 1; m <= kzmax; m++) {
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
              eikrrl[n][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim[n][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
            }
            if (complex_atom[i]==0) {
              cstr1 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr1 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            if (complex_atom[i]==cplx_id) {
              eikrrl[n+1][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim[n+1][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
            }
            if (complex_atom[i]==0) {
              cstr2 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr2 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            if (complex_atom[i]==cplx_id) {
              eikrrl[n+2][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim[n+2][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
            }
            if(complex_atom[i]==0)  {
              cstr3 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr3 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }

            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            if (complex_atom[i]) {
              eikrrl[n+3][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim[n+3][i_cplx++] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
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

   
  MPI_Allreduce(sfacrl,sfacrl_env,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_env,kcount,MPI_DOUBLE,MPI_SUM,world);

}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::eik_dot_r_cplx()
{
  int i,k;
  int iatm;
  double *q = atom->q;

  int ncplx_local = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  
  for (k = 0; k < kcount; k++) {
      sfacrl[k] = 0.0;
      sfacim[k] = 0.0;
      for (i = 0; i < ncplx_local; i++) {
          iatm = cplx_list[i];
          
          sfacrl[k] += q[iatm] * eikrrl[k][i];
          sfacim[k] += q[iatm] * eikrim[k][i];
      }
  }
  
  MPI_Allreduce(sfacrl,sfacrl_cplx,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_cplx,kcount,MPI_DOUBLE,MPI_SUM,world);
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::eik_dot_r_exch()
{
    int i,k;
    double *q = atom->q;
    int cplx_id = evb_engine->evb_complex->id;
    int *cplx_list = evb_engine->evb_complex->cplx_list;
    int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
    
    for (k = 0; k < kcount; k++) {
        sfacrl[k] = 0.0;
        sfacim[k] = 0.0;
        sfacrl_cplx[k] = 0.0;
        sfacim_cplx[k] = 0.0;
        for (i = 0; i < nlocal_cplx; i++) {
            int iatm = cplx_list[i];
            if (is_exch_chg[iatm]) {
                sfacrl[k] += q[iatm] * eikrrl[k][i];
                sfacim[k] += q[iatm] * eikrim[k][i];
            }
            else 
            {
                sfacrl_cplx[k] += q[iatm] * eikrrl[k][i];
                sfacim_cplx[k] += q[iatm] * eikrim[k][i];
            }
        }
    }
  
    MPI_Allreduce(sfacrl,sfacrl_exch,kcount,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(sfacim,sfacim_exch,kcount,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(sfacrl_cplx,sfacrl,kcount,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(sfacim_cplx,sfacim,kcount,MPI_DOUBLE,MPI_SUM,world);
}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each EVB_Ewald K-vector 
------------------------------------------------------------------------- */

void EVB_Ewald::coeffs()
{
  int k,l,m;
  double sqk,vterm;

  double unitkx = unitk[0];
  double unitky = unitk[1];
  double unitkz = unitk[2];
  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MY_PI/volume;

  kcount = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (m = 1; m <= kmax; m++) {
    sqk = (m*unitkx) * (m*unitkx);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = m;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = 0;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 2.0*unitkx*m*ug[kcount];
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 0.0;
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0 + vterm*(unitkx*m)*(unitkx*m);
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0;
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
    sqk = (m*unitky) * (m*unitky);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = m;
      kzvecs[kcount] = 0;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 2.0*unitky*m*ug[kcount];
      eg[kcount][2] = 0.0;
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0 + vterm*(unitky*m)*(unitky*m);
      vg[kcount][2] = 1.0;
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
    sqk = (m*unitkz) * (m*unitkz);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 2.0*unitkz*m*ug[kcount];
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (unitkx*k) * (unitkx*k) + (unitky*l) * (unitky*l);
      if (sqk <= gsqmx) {
	kxvecs[kcount] = k;
	kyvecs[kcount] = l;
	kzvecs[kcount] = 0;
	ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg[kcount][0] = 2.0*unitkx*k*ug[kcount];
	eg[kcount][1] = 2.0*unitky*l*ug[kcount];
	eg[kcount][2] = 0.0;
	vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
	vg[kcount][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	vg[kcount][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	vg[kcount][2] = 1.0;
	vg[kcount][3] = vterm*unitkx*k*unitky*l;
	vg[kcount][4] = 0.0;
	vg[kcount][5] = 0.0;
	kcount++;

	kxvecs[kcount] = k;
	kyvecs[kcount] = -l;
	kzvecs[kcount] = 0;
	ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg[kcount][0] = 2.0*unitkx*k*ug[kcount];
	eg[kcount][1] = -2.0*unitky*l*ug[kcount];
	eg[kcount][2] = 0.0;
	vg[kcount][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	vg[kcount][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	vg[kcount][2] = 1.0;
	vg[kcount][3] = -vterm*unitkx*k*unitky*l;
	vg[kcount][4] = 0.0;
	vg[kcount][5] = 0.0;
	kcount++;;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitky*l) * (unitky*l) + (unitkz*m) * (unitkz*m);
      if (sqk <= gsqmx) {
	kxvecs[kcount] = 0;
	kyvecs[kcount] = l;
	kzvecs[kcount] = m;
	ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg[kcount][0] =  0.0;
	eg[kcount][1] =  2.0*unitky*l*ug[kcount];
	eg[kcount][2] =  2.0*unitkz*m*ug[kcount];
	vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
	vg[kcount][0] = 1.0;
	vg[kcount][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	vg[kcount][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	vg[kcount][3] = 0.0;
	vg[kcount][4] = 0.0;
	vg[kcount][5] = vterm*unitky*l*unitkz*m;
	kcount++;

	kxvecs[kcount] = 0;
	kyvecs[kcount] = l;
	kzvecs[kcount] = -m;
	ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg[kcount][0] =  0.0;
	eg[kcount][1] =  2.0*unitky*l*ug[kcount];
	eg[kcount][2] = -2.0*unitkz*m*ug[kcount];
	vg[kcount][0] = 1.0;
	vg[kcount][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	vg[kcount][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	vg[kcount][3] = 0.0;
	vg[kcount][4] = 0.0;
	vg[kcount][5] = -vterm*unitky*l*unitkz*m;
	kcount++;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitkx*k) * (unitkx*k) + (unitkz*m) * (unitkz*m);
      if (sqk <= gsqmx) {
	kxvecs[kcount] = k;
	kyvecs[kcount] = 0;
	kzvecs[kcount] = m;
	ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg[kcount][0] =  2.0*unitkx*k*ug[kcount];
	eg[kcount][1] =  0.0;
	eg[kcount][2] =  2.0*unitkz*m*ug[kcount];
	vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
	vg[kcount][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	vg[kcount][1] = 1.0;
	vg[kcount][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	vg[kcount][3] = 0.0;
	vg[kcount][4] = vterm*unitkx*k*unitkz*m;
	vg[kcount][5] = 0.0;
	kcount++;

	kxvecs[kcount] = k;
	kyvecs[kcount] = 0;
	kzvecs[kcount] = -m;
	ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	eg[kcount][0] =  2.0*unitkx*k*ug[kcount];
	eg[kcount][1] =  0.0;
	eg[kcount][2] = -2.0*unitkz*m*ug[kcount];
	vg[kcount][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	vg[kcount][1] = 1.0;
	vg[kcount][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	vg[kcount][3] = 0.0;
	vg[kcount][4] = -vterm*unitkx*k*unitkz*m;
	vg[kcount][5] = 0.0;
	kcount++;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
	sqk = (unitkx*k) * (unitkx*k) + (unitky*l) * (unitky*l) + 
	  (unitkz*m) * (unitkz*m);
	if (sqk <= gsqmx) {
	  kxvecs[kcount] = k;
	  kyvecs[kcount] = l;
	  kzvecs[kcount] = m;
	  ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	  eg[kcount][0] = 2.0*unitkx*k*ug[kcount];
	  eg[kcount][1] = 2.0*unitky*l*ug[kcount];
	  eg[kcount][2] = 2.0*unitkz*m*ug[kcount];
	  vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
	  vg[kcount][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	  vg[kcount][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	  vg[kcount][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	  vg[kcount][3] = vterm*unitkx*k*unitky*l;
	  vg[kcount][4] = vterm*unitkx*k*unitkz*m;
	  vg[kcount][5] = vterm*unitky*l*unitkz*m;
	  kcount++;

	  kxvecs[kcount] = k;
	  kyvecs[kcount] = -l;
	  kzvecs[kcount] = m;
	  ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	  eg[kcount][0] = 2.0*unitkx*k*ug[kcount];
	  eg[kcount][1] = -2.0*unitky*l*ug[kcount];
	  eg[kcount][2] = 2.0*unitkz*m*ug[kcount];
	  vg[kcount][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	  vg[kcount][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	  vg[kcount][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	  vg[kcount][3] = -vterm*unitkx*k*unitky*l;
	  vg[kcount][4] = vterm*unitkx*k*unitkz*m;
	  vg[kcount][5] = -vterm*unitky*l*unitkz*m;
	  kcount++;

	  kxvecs[kcount] = k;
	  kyvecs[kcount] = l;
	  kzvecs[kcount] = -m;
	  ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	  eg[kcount][0] = 2.0*unitkx*k*ug[kcount];
	  eg[kcount][1] = 2.0*unitky*l*ug[kcount];
	  eg[kcount][2] = -2.0*unitkz*m*ug[kcount];
	  vg[kcount][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	  vg[kcount][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	  vg[kcount][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	  vg[kcount][3] = vterm*unitkx*k*unitky*l;
	  vg[kcount][4] = -vterm*unitkx*k*unitkz*m;
	  vg[kcount][5] = -vterm*unitky*l*unitkz*m;
	  kcount++;

	  kxvecs[kcount] = k;
	  kyvecs[kcount] = -l;
	  kzvecs[kcount] = -m;
	  ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
	  eg[kcount][0] = 2.0*unitkx*k*ug[kcount];
	  eg[kcount][1] = -2.0*unitky*l*ug[kcount];
	  eg[kcount][2] = -2.0*unitkz*m*ug[kcount];
	  vg[kcount][0] = 1.0 + vterm*(unitkx*k)*(unitkx*k);
	  vg[kcount][1] = 1.0 + vterm*(unitky*l)*(unitky*l);
	  vg[kcount][2] = 1.0 + vterm*(unitkz*m)*(unitkz*m);
	  vg[kcount][3] = -vterm*unitkx*k*unitky*l;
	  vg[kcount][4] = -vterm*unitkx*k*unitkz*m;
	  vg[kcount][5] = vterm*unitky*l*unitkz*m;
	  kcount++;;
	}
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors 
------------------------------------------------------------------------- */

void EVB_Ewald::allocate()
{
  kxvecs = new int[kmax3d];
  kyvecs = new int[kmax3d];
  kzvecs = new int[kmax3d];
  
  ug = new double[kmax3d];
  memory->create(eg,kmax3d,3,"ewald:eg");
  memory->create(vg,kmax3d,6,"ewald:vg");

  sfacrl = new double[kmax3d];
  sfacim = new double[kmax3d];

  /***** EVB part *****/
  
  sfacrl_env = new double[kmax3d];
  sfacim_env = new double[kmax3d];
  sfacrl_cplx = new double[kmax3d];
  sfacim_cplx = new double[kmax3d];
  sfacrl_exch = new double[kmax3d];
  sfacim_exch = new double[kmax3d];
  sfacrl_inter = new double[kmax3d];
  sfacim_inter = new double[kmax3d];
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors 
------------------------------------------------------------------------- */

void EVB_Ewald::deallocate()
{
  delete [] kxvecs;
  delete [] kyvecs;
  delete [] kzvecs;
	
  delete [] ug;
  memory->destroy(eg);
  memory->destroy(vg);

  delete [] sfacrl;
  delete [] sfacim;

  /***** EVB part *****/  
  
  delete [] sfacrl_env;
  delete [] sfacim_env;
  delete [] sfacrl_cplx;
  delete [] sfacim_cplx;
  delete [] sfacrl_exch;
  delete [] sfacim_exch;
  delete [] sfacrl_inter;
  delete [] sfacim_inter;
}

/* ----------------------------------------------------------------------
   memory usage of local arrays 
------------------------------------------------------------------------- */

double EVB_Ewald::memory_usage()
{
  double bytes = 3 * kmax3d * sizeof(int);
  bytes += (1 + 3 + 6) * kmax3d * sizeof(double);
  bytes += 4 * kmax3d * sizeof(double);
  bytes += nmax*3 * sizeof(double);
  bytes += 2 * (2*kmax+1)*3*nmax * sizeof(double);
  return bytes;
}


