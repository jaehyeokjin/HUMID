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
     per-atom energy/virial & group/group energy/force added by Stan Moore (BYU)
     analytic diff (2 FFT) option added by Rolf Isele-Holder (Aachen University)
     triclinic added by Stan Moore (SNL)
------------------------------------------------------------------------- */

#if defined (_OPENMP)

#include "lmptype.h"
#include "mpi.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "pppm_electrode_omp.h"
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
#include "pair_electrode_omp.h"
#define ZPRD_CORR() (pair_et_omp->D * 3.0 * slab_volfactor)
/* End */

/* ---------------------------------------------------------------------- */

PPPM_ElectrodeOMP::PPPM_ElectrodeOMP(LAMMPS *lmp, int narg, char **arg) : 
  PPPM_Electrode(lmp, narg, arg), ThrOMP(lmp, THR_KSPACE)
{
  suffix_flag |= Suffix::OMP;
 }

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

PPPM_ElectrodeOMP::~PPPM_ElectrodeOMP()
{

}

/* ----------------------------------------------------------------------
   called once before run
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::init()
{
  /* Electrode Model */
  
  // call a function out of a class is not a good way, however, the pair->init
  // is called after the kspace->init, and kspace need something from pair->init
  // and pair->init dose more than that. Therefore, the effecient way is to call
  // pair->electrode_init from the kspace->init.

  if(comm->me==0 && screen) fprintf(screen,"Electrode | Create pppm/electrode.\n");

  if(strcmp(force->pair_style,"electrode/omp"))
    error->all(FLERR, "Incompatible pair-style is used.(force->pair_style != electrode/omp)");

  pair_et_omp = (PairElectrodeOMP*)(force->pair);
  
  /* End */
  
  if (me == 0) {
    if (screen) fprintf(screen,"PPPM_Electrode initialization ...\n");
    if (logfile) fprintf(logfile,"PPPM_Electrode initialization ...\n");
  }

  // error check

  triclinic_check();
  if (domain->triclinic && differentiation_flag == 1)
    error->all(FLERR,"Cannot (yet) use PPPM_Electrode with triclinic box "
               "and kspace_modify diff ad");
  if (domain->triclinic && slabflag)
    error->all(FLERR,"Cannot (yet) use PPPM_Electrode with triclinic box and "
               "slab correction");
  if (domain->dimension == 2) error->all(FLERR,
                                         "Cannot use PPPM_Electrode with 2d simulation");

  if (!atom->q_flag) error->all(FLERR,"Kspace style requires atom attribute q");

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use nonperiodic boundaries with PPPM_Electrode");
  if (slabflag) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab PPPM_Electrode");
  }

  if (order < 2 || order > MAXORDER) {
    char str[128];
    sprintf(str,"PPPM_Electrode order cannot be < 2 or > than %d",MAXORDER);
    error->all(FLERR,str);
  }

  // extract short-range Coulombic cutoff from pair style

  triclinic = domain->triclinic;
  scale = 1.0;

  pair_check();

  int itmp = 0;
  double *p_cutoff = (double *) force->pair->extract("cut_coul",itmp);
  if (p_cutoff == NULL)
    error->all(FLERR,"KSpace style is incompatible with Pair style");
  cutoff = *p_cutoff;

  // if kspace is TIP4P, extract TIP4P params from pair style
  // bond/angle are not yet init(), so insure equilibrium request is valid

  qdist = 0.0;

  if (tip4pflag) {
    double *p_qdist = (double *) force->pair->extract("qdist",itmp);
    int *p_typeO = (int *) force->pair->extract("typeO",itmp);
    int *p_typeH = (int *) force->pair->extract("typeH",itmp);
    int *p_typeA = (int *) force->pair->extract("typeA",itmp);
    int *p_typeB = (int *) force->pair->extract("typeB",itmp);
    if (!p_qdist || !p_typeO || !p_typeH || !p_typeA || !p_typeB)
      error->all(FLERR,"KSpace style is incompatible with Pair style");
    qdist = *p_qdist;
    typeO = *p_typeO;
    typeH = *p_typeH;
    int typeA = *p_typeA;
    int typeB = *p_typeB;

    if (force->angle == NULL || force->bond == NULL ||
        force->angle->setflag == NULL || force->bond->setflag == NULL)
      error->all(FLERR,"Bond and angle potentials must be defined for TIP4P");
    if (typeA < 1 || typeA > atom->nangletypes ||
        force->angle->setflag[typeA] == 0)
      error->all(FLERR,"Bad TIP4P angle type for PPPM_Electrode/TIP4P");
    if (typeB < 1 || typeB > atom->nbondtypes ||
        force->bond->setflag[typeB] == 0)
      error->all(FLERR,"Bad TIP4P bond type for PPPM_Electrode/TIP4P");
    double theta = force->angle->equilibrium_angle(typeA);
    double blen = force->bond->equilibrium_distance(typeB);
    alpha = qdist / (cos(0.5*theta) * blen);
    if (domain->triclinic)
      error->all(FLERR,"Cannot (yet) use PPPM_Electrode with triclinic box and TIP4P");
  }

  // compute qsum & qsqsum and warn if not charge-neutral

  qsum = qsqsum = 0.0;
  for (int i = 0; i < atom->nlocal; i++) {
    qsum += atom->q[i];
    qsqsum += atom->q[i]*atom->q[i];
  }

  double tmp;
  MPI_Allreduce(&qsum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum = tmp;
  MPI_Allreduce(&qsqsum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsqsum = tmp;
  
  q2 = qsqsum * force->qqrd2e / force->dielectric;
  
  /* Electrode Model */
  qsqsum *= 3.0;
  q2 *= 3.0;
  /* End */
  
  if (qsqsum == 0.0)
    error->all(FLERR,"Cannot use kspace solver on system with no charge");
  if (fabs(qsum) > SMALL && me == 0) {
    char str[128];
    sprintf(str,"System is not charge neutral, net charge = %g",qsum);
    error->warning(FLERR,str);
  }

  // set accuracy (force units) from accuracy_relative or accuracy_absolute

  if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  else accuracy = accuracy_relative * two_charge_force;

  // free all arrays previously allocated

  deallocate();
  if (peratom_allocate_flag) deallocate_peratom();
  if (group_allocate_flag) deallocate_groups();

  // setup FFT grid resolution and g_ewald
  // normally one iteration thru while loop is all that is required
  // if grid stencil does not extend beyond neighbor proc
  //   or overlap is allowed, then done
  // else reduce order and try again

  int (*procneigh)[2] = comm->procneigh;

  GridComm *cgtmp = NULL;
  int iteration = 0;

  while (order >= minorder) {
    if (iteration && me == 0)
      error->warning(FLERR,"Reducing PPPM_Electrode order b/c stencil extends "
                     "beyond nearest neighbor processor");

    if (stagger_flag && !differentiation_flag) compute_gf_denom();
    set_grid_global();
    set_grid_local();
    if (overlap_allowed) break;

    cgtmp = new GridComm(lmp_pointer,world,1,1,
                         nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                         nxlo_out,nxhi_out,nylo_out,nyhi_out,nzlo_out,nzhi_out,
                         procneigh[0][0],procneigh[0][1],procneigh[1][0],
                         procneigh[1][1],procneigh[2][0],procneigh[2][1]);
    cgtmp->ghost_notify();
    if (!cgtmp->ghost_overlap()) break;
    delete cgtmp;

    order--;
    iteration++;
  }
  
  if (order < minorder) error->all(FLERR,"PPPM_Electrode order < minimum allowed order");
  if (!overlap_allowed && cgtmp->ghost_overlap())
    error->all(FLERR,"PPPM_Electrode grid stencil extends "
               "beyond nearest neighbor processor");
  if (cgtmp) delete cgtmp;

  // adjust g_ewald

  if (!gewaldflag) adjust_gewald();

  // calculate the final accuracy

  double estimated_accuracy = final_accuracy();

  // print stats

  int ngrid_max,nfft_both_max;
  MPI_Allreduce(&ngrid,&ngrid_max,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(&nfft_both,&nfft_both_max,1,MPI_INT,MPI_MAX,world);

  if (me == 0) {

#ifdef FFT_SINGLE
    const char fft_prec[] = "single";
#else
    const char fft_prec[] = "double";
#endif

    if (screen) {
      fprintf(screen,"  G vector (1/distance) = %g\n",g_ewald);
      fprintf(screen,"  grid = %d %d %d\n",nx_pppm,ny_pppm,nz_pppm);
      fprintf(screen,"  stencil order = %d\n",order);
      fprintf(screen,"  estimated absolute RMS force accuracy = %g\n",
              estimated_accuracy);
      fprintf(screen,"  estimated relative force accuracy = %g\n",
              estimated_accuracy/two_charge_force);
      fprintf(screen,"  using %s precision FFTs\n",fft_prec);
      fprintf(screen,"  3d grid and FFT values/proc = %d %d\n",
              ngrid_max,nfft_both_max);
    }
    if (logfile) {
      fprintf(logfile,"  G vector (1/distance) = %g\n",g_ewald);
      fprintf(logfile,"  grid = %d %d %d\n",nx_pppm,ny_pppm,nz_pppm);
      fprintf(logfile,"  stencil order = %d\n",order);
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
  // don't invoke allocate peratom() or group(), will be allocated when needed

  allocate();
  cg->ghost_notify();
  cg->setup();

  // pre-compute Green's function denomiator expansion
  // pre-compute 1d charge distribution coefficients

  compute_gf_denom();
  if (differentiation_flag == 1) compute_sf_precoeff();
  compute_rho_coeff();
}

/* ----------------------------------------------------------------------
   adjust PPPM_Electrode coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::setup()
{
  if (triclinic) {
    setup_triclinic();
    return;
  }

  int i,j,k,n;
  double *prd;

  // volume-dependent factors
  // adjust z dimension for 2d slab PPPM_Electrode
  // z dimension for 3d PPPM_Electrode is zprd since slab_volfactor = 1.0

  if (triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = ZPRD_CORR();
  volume = xprd * yprd * zprd_slab;

  delxinv = nx_pppm/xprd;
  delyinv = ny_pppm/yprd;
  delzinv = nz_pppm/zprd_slab;

  delvolinv = delxinv*delyinv*delzinv;

  double unitkx = (MY_2PI/xprd);
  double unitky = (MY_2PI/yprd);
  double unitkz = (MY_2PI/zprd_slab);

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
   adjust PPPM_Electrode coeffs, called initially and whenever volume has changed
   for a triclinic system
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::setup_triclinic()
{
  int i,j,k,n;
  double *prd;

  // volume-dependent factors
  // adjust z dimension for 2d slab PPPM_Electrode
  // z dimension for 3d PPPM_Electrode is zprd since slab_volfactor = 1.0

  prd = domain->prd;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = ZPRD_CORR();
  volume = xprd * yprd * zprd_slab;

  // use lamda (0-1) coordinates

  delxinv = nx_pppm;
  delyinv = ny_pppm;
  delzinv = nz_pppm;
  delvolinv = delxinv*delyinv*delzinv/volume;

  // fkx,fky,fkz for my FFT grid pts

  double per_i,per_j,per_k;

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++) {
    per_k = k - nz_pppm*(2*k/nz_pppm);
    for (j = nylo_fft; j <= nyhi_fft; j++) {
      per_j = j - ny_pppm*(2*j/ny_pppm);
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        per_i = i - nx_pppm*(2*i/nx_pppm);

        double unitk_lamda[3];
        unitk_lamda[0] = 2.0*MY_PI*per_i;
        unitk_lamda[1] = 2.0*MY_PI*per_j;
        unitk_lamda[2] = 2.0*MY_PI*per_k;
        x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
        fkx[n] = unitk_lamda[0];
        fky[n] = unitk_lamda[1];
        fkz[n] = unitk_lamda[2];
        n++;
      }
    }
  }

  // virial coefficients

  double sqk,vterm;

  for (n = 0; n < nfft; n++) {
    sqk = fkx[n]*fkx[n] + fky[n]*fky[n] + fkz[n]*fkz[n];
    if (sqk == 0.0) {
      vg[n][0] = 0.0;
      vg[n][1] = 0.0;
      vg[n][2] = 0.0;
      vg[n][3] = 0.0;
      vg[n][4] = 0.0;
      vg[n][5] = 0.0;
    } else {
      vterm = -2.0 * (1.0/sqk + 0.25/(g_ewald*g_ewald));
      vg[n][0] = 1.0 + vterm*fkx[n]*fkx[n];
      vg[n][1] = 1.0 + vterm*fky[n]*fky[n];
      vg[n][2] = 1.0 + vterm*fkz[n]*fkz[n];
      vg[n][3] = vterm*fkx[n]*fky[n];
      vg[n][4] = vterm*fkx[n]*fkz[n];
      vg[n][5] = vterm*fky[n]*fkz[n];
    }
  }

  compute_gf_ik_triclinic();
}

/* ----------------------------------------------------------------------
   compute the PPPM_Electrode long-range force, energy, virial
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::compute(int eflag, int vflag)
{ 
  int i,j;

  // set energy/virial flags
  // invoke allocate_peratom() if needed for first time

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = evflag_atom = eflag_global = vflag_global =
         eflag_atom = vflag_atom = 0;

  if (evflag_atom && !peratom_allocate_flag) {
    allocate_peratom();
    cg_peratom->ghost_notify();
    cg_peratom->setup();
  }
// double fff=atom->f[0][2];
  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }

  // extend size of per-atom arrays if necessary

  if (atom->nlocal > nmax) {
    memory->destroy(part2grid);
    nmax = atom->nmax;
    memory->create(part2grid,nmax,3,"pppm:part2grid");
  }

  // find grid points for all my particles
  // map my particle charge onto my local 3d density grid

  particle_map();
  make_rho();

  // all procs communicate density values from their ghost cells
  //   to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition

  cg->reverse_comm(this,REVERSE_RHO);

  brick2fft();

  // compute potential gradient on my FFT grid and
  //   portion of e_long on this proc's FFT grid
  // return gradients (electric fields) in 3d brick decomposition
  // also performs per-atom calculations via poisson_peratom()

  poisson();

  // all procs communicate E-field values
  // to fill ghost cells surrounding their 3d bricks

  if (differentiation_flag == 1) cg->forward_comm(this,FORWARD_AD);
  else cg->forward_comm(this,FORWARD_IK);

  // extra per-atom energy/virial communication

  if (evflag_atom) {
    if (differentiation_flag == 1 && vflag_atom) 
      cg_peratom->forward_comm(this,FORWARD_AD_PERATOM);
    else if (differentiation_flag == 0)
      cg_peratom->forward_comm(this,FORWARD_IK_PERATOM);
  }

  // calculate the force on my particles

  fieldforce();

  // extra per-atom energy/virial communication

  if (evflag_atom) fieldforce_peratom();

  // sum global energy across procs and add in volume-dependent term

  const double qscale = force->qqrd2e * scale;

  if (eflag_global) {
    double energy_all;
    MPI_Allreduce(&energy,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
    energy = energy_all;
  
    energy *= 0.5*volume;
    energy -= g_ewald*qsqsum/3.0/MY_PIS +
      MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
    energy *= qscale;
  }

  // sum global virial across procs

  if (vflag_global) {
    double virial_all[6];
    MPI_Allreduce(virial,virial_all,6,MPI_DOUBLE,MPI_SUM,world);
    for (i = 0; i < 6; i++) virial[i] = 0.5*qscale*volume*virial_all[i];
  }

  // per-atom energy/virial
  // energy includes self-energy correction
  // notal accounts for TIP4P tallying eatom/vatom for ghost atoms

  if (evflag_atom) {
    double *q = atom->q;
    int nlocal = atom->nlocal;
    int ntotal = nlocal;
    if (tip4pflag) ntotal += atom->nghost;

    if (eflag_atom) {
      for (i = 0; i < nlocal; i++) {
        eatom[i] *= 0.5;
        eatom[i] -= g_ewald*q[i]*q[i]/MY_PIS + MY_PI2*q[i]*qsum /
          (g_ewald*g_ewald*volume);
        eatom[i] *= qscale;
      }
      for (i = nlocal; i < ntotal; i++) eatom[i] *= 0.5*qscale;
    }

    if (vflag_atom) {
      for (i = 0; i < ntotal; i++)
        for (j = 0; j < 6; j++) vatom[i][j] *= 0.5*qscale;
    }
  }

  // 2d slab correction

  if (slabflag == 1) slabcorr();
// printf("ABC %lf %lf %lf\n", atom->x[0][2], atom->f[0][2]-fff, energy);
  // convert atoms back from lamda to box coords

  if (triclinic) domain->lamda2x(atom->nlocal);

  
#if defined(_OPENMP)
#pragma omp parallel default(none) shared(eflag,vflag)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    ThrData *thr = fix->get_thr(tid);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors and order
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::allocate()
{
  PPPM_Electrode::allocate();

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
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors and order
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::deallocate()
{
  PPPM_Electrode::deallocate();

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
}

/* ----------------------------------------------------------------------
   set global size of PPPM_Electrode grid = nx,ny,nz_pppm
   used for charge accumulation, FFTs, and electric field interpolation
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::set_grid_global()
{
  // use xprd,yprd,zprd (even if triclinic, and then scale later)
  // adjust z dimension for 2d slab PPPM_Electrode
  // 3d PPPM_Electrode just uses zprd since slab_volfactor = 1.0

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  
  // make initial g_ewald estimate
  // based on desired accuracy and real space cutoff
  // fluid-occupied volume used to estimate real-space error
  // zprd used rather than zprd_slab

  double h;
  bigint natoms = atom->natoms;

  /* Electrode Model */
  double zprd_slab = ZPRD_CORR();
  natoms *= 3;
  /* End */
  
  if (!gewaldflag) {
    if (accuracy <= 0.0)
      error->all(FLERR,"KSpace accuracy must be > 0");
    g_ewald = accuracy*sqrt(natoms*cutoff*xprd*yprd*zprd) / (2.0*q2);
    if (g_ewald >= 1.0) g_ewald = (1.35 - 0.15*log(accuracy))/cutoff;
    else g_ewald = sqrt(-log(g_ewald)) / cutoff;
  }

  // set optimal nx_pppm,ny_pppm,nz_pppm based on order and accuracy
  // nz_pppm uses extended zprd_slab instead of zprd
  // reduce it until accuracy target is met

  if (!gridflag) {

    if (differentiation_flag == 1 || stagger_flag) {

      h = h_x = h_y = h_z = 4.0/g_ewald;
      int count = 0;
      while (1) {

        // set grid dimension
        nx_pppm = static_cast<int> (xprd/h_x);
        ny_pppm = static_cast<int> (yprd/h_y);
        nz_pppm = static_cast<int> (zprd_slab/h_z);

        if (nx_pppm <= 1) nx_pppm = 2;
        if (ny_pppm <= 1) ny_pppm = 2;
        if (nz_pppm <= 1) nz_pppm = 2;

        //set local grid dimension
        int npey_fft,npez_fft;
        if (nz_pppm >= nprocs) {
          npey_fft = 1;
          npez_fft = nprocs;
        } else procs2grid2d(nprocs,ny_pppm,nz_pppm,&npey_fft,&npez_fft);

        int me_y = me % npey_fft;
        int me_z = me / npey_fft;

        nxlo_fft = 0;
        nxhi_fft = nx_pppm - 1;
        nylo_fft = me_y*ny_pppm/npey_fft;
        nyhi_fft = (me_y+1)*ny_pppm/npey_fft - 1;
        nzlo_fft = me_z*nz_pppm/npez_fft;
        nzhi_fft = (me_z+1)*nz_pppm/npez_fft - 1;

        double df_kspace = compute_df_kspace();

        count++;

        // break loop if the accuracy has been reached or
        // too many loops have been performed

        if (df_kspace <= accuracy) break;
        if (count > 500) error->all(FLERR, "Could not compute grid size");
        h *= 0.95;
        h_x = h_y = h_z = h;
      }

    } else {

      double err;
      h_x = h_y = h_z = 1.0/g_ewald;

      nx_pppm = static_cast<int> (xprd/h_x) + 1;
      ny_pppm = static_cast<int> (yprd/h_y) + 1;
      nz_pppm = static_cast<int> (zprd_slab/h_z) + 1;

      err = estimate_ik_error(h_x,xprd,natoms);
      while (err > accuracy) {
        err = estimate_ik_error(h_x,xprd,natoms);
        nx_pppm++;
        h_x = xprd/nx_pppm;
      }

      err = estimate_ik_error(h_y,yprd,natoms);
      while (err > accuracy) {
        err = estimate_ik_error(h_y,yprd,natoms);
        ny_pppm++;
        h_y = yprd/ny_pppm;
      }

      err = estimate_ik_error(h_z,zprd_slab,natoms);
      while (err > accuracy) {
        err = estimate_ik_error(h_z,zprd_slab,natoms); 
        nz_pppm++;
        h_z = zprd_slab/nz_pppm;
      }

    }

    // scale grid for triclinic skew
    
    if (triclinic) {
      double tmp[3];
      tmp[0] = nx_pppm/xprd;
      tmp[1] = ny_pppm/yprd;
      tmp[2] = nz_pppm/zprd;
      lamda2xT(&tmp[0],&tmp[0]);
      nx_pppm = static_cast<int>(tmp[0]) + 1;
      ny_pppm = static_cast<int>(tmp[1]) + 1;
      nz_pppm = static_cast<int>(tmp[2]) + 1;
    }
  }

  // boost grid size until it is factorable

  while (!factorable(nx_pppm)) nx_pppm++;
  while (!factorable(ny_pppm)) ny_pppm++;
  while (!factorable(nz_pppm)) nz_pppm++;

  if (triclinic == 0) {
    h_x = xprd/nx_pppm;
    h_y = yprd/ny_pppm;
    h_z = zprd_slab/nz_pppm;
  } else {
    double tmp[3];
    tmp[0] = nx_pppm;
    tmp[1] = ny_pppm;
    tmp[2] = nz_pppm;
    x2lamdaT(&tmp[0],&tmp[0]);
    h_x = 1.0/tmp[0];
    h_y = 1.0/tmp[1];
    h_z = 1.0/tmp[2];
  }

  if (nx_pppm >= OFFSET || ny_pppm >= OFFSET || nz_pppm >= OFFSET)
    error->all(FLERR,"PPPM_Electrode grid is too large");
}

/* ----------------------------------------------------------------------
   adjust the g_ewald parameter to near its optimal value
   using a Newton-Raphson solver
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::adjust_gewald()
{
  double dx;

  for (int i = 0; i < LARGE; i++) {
    dx = newton_raphson_f() / derivf();
    g_ewald -= dx;
    if (fabs(newton_raphson_f()) < SMALL) return;
  }

  char str[128];
  sprintf(str, "Could not compute g_ewald");
  error->all(FLERR, str);
}

/* ----------------------------------------------------------------------
 Calculate f(x) using Newton-Raphson solver
 ------------------------------------------------------------------------- */

double PPPM_ElectrodeOMP::newton_raphson_f()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  bigint natoms = atom->natoms;

  double df_rspace = 2.0*q2*exp(-g_ewald*g_ewald*cutoff*cutoff) /
       sqrt(natoms*cutoff*xprd*yprd*zprd);

  double df_kspace = compute_df_kspace();

  return df_rspace - df_kspace;
}

/* ----------------------------------------------------------------------
 Calculate numerical derivative f'(x) using forward difference
 [f(x + h) - f(x)] / h
 ------------------------------------------------------------------------- */

double PPPM_ElectrodeOMP::derivf()
{
  double h = 0.000001;  //Derivative step-size
  double df,f1,f2,g_ewald_old;

  f1 = newton_raphson_f();
  g_ewald_old = g_ewald;
  g_ewald += h;
  f2 = newton_raphson_f();
  g_ewald = g_ewald_old;
  df = (f2 - f1)/h;

  return df;
}

/* ----------------------------------------------------------------------
   compute estimated kspace force error
------------------------------------------------------------------------- */

double PPPM_ElectrodeOMP::compute_df_kspace()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  bigint natoms = atom->natoms;
  
  /* Electrode Model */
  zprd_slab = ZPRD_CORR();
  natoms *= 3;
  /* End */
  
  double df_kspace = 0.0;
  if (differentiation_flag == 1 || stagger_flag) {
    double qopt = compute_qopt();
    df_kspace = sqrt(qopt/natoms)*q2/(xprd*yprd*zprd_slab);
  } else {
    double lprx = estimate_ik_error(h_x,xprd,natoms);
    double lpry = estimate_ik_error(h_y,yprd,natoms);
    double lprz = estimate_ik_error(h_z,zprd_slab,natoms);
    df_kspace = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
  }
  return df_kspace;
}

/* ----------------------------------------------------------------------
   compute qopt
------------------------------------------------------------------------- */

double PPPM_ElectrodeOMP::compute_qopt()
{
  double qopt = 0.0;
  double *prd = domain->prd;
  
  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = ZPRD_CORR();
  volume = xprd * yprd * zprd_slab;

  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double u1, u2, sqk;
  double sum1,sum2,sum3,sum4,dot2;

  int k,l,m,nx,ny,nz;
  const int twoorder = 2*order;

  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    const int mper = m - nz_pppm*(2*m/nz_pppm);

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      const int lper = l - ny_pppm*(2*l/ny_pppm);

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        const int kper = k - nx_pppm*(2*k/nx_pppm);

        sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);

        if (sqk != 0.0) {

          sum1 = 0.0;
          sum2 = 0.0;
          sum3 = 0.0;
          sum4 = 0.0;
          for (nx = -2; nx <= 2; nx++) {
            qx = unitkx*(kper+nx_pppm*nx);
            sx = exp(-0.25*square(qx/g_ewald));
            argx = 0.5*qx*xprd/nx_pppm;
            wx = powsinxx(argx,twoorder);
            qx *= qx;

            for (ny = -2; ny <= 2; ny++) {
              qy = unitky*(lper+ny_pppm*ny);
              sy = exp(-0.25*square(qy/g_ewald));
              argy = 0.5*qy*yprd/ny_pppm;
              wy = powsinxx(argy,twoorder);
              qy *= qy;

              for (nz = -2; nz <= 2; nz++) {
                qz = unitkz*(mper+nz_pppm*nz);
                sz = exp(-0.25*square(qz/g_ewald));
                argz = 0.5*qz*zprd_slab/nz_pppm;
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
   Calculate the final estimate of the accuracy
------------------------------------------------------------------------- */

double PPPM_ElectrodeOMP::final_accuracy()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  bigint natoms = atom->natoms;

  /* Electrode Model */
  zprd *= 3.0;
  natoms *= 3;
  /* End */
  
  double df_kspace = compute_df_kspace();
  double q2_over_sqrt = q2 / sqrt(natoms*cutoff*xprd*yprd*zprd);
  double df_rspace = 2.0 * q2_over_sqrt * exp(-g_ewald*g_ewald*cutoff*cutoff);
  double df_table = estimate_table_accuracy(q2_over_sqrt,df_rspace);
  double estimated_accuracy = sqrt(df_kspace*df_kspace + df_rspace*df_rspace +
   df_table*df_table);

  return estimated_accuracy;
}

/* ----------------------------------------------------------------------
   set local subset of PPPM_Electrode/FFT grid that I own
   n xyz lo/hi in = 3d brick that I own (inclusive)
   n xyz lo/hi out = 3d brick + ghost cells in 6 directions (inclusive)
   n xyz lo/hi fft = FFT columns that I own (all of x dim, 2d decomp in yz)
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::set_grid_local()
{
  // global indices of PPPM_Electrode grid range from 0 to N-1
  // nlo_in,nhi_in = lower/upper limits of the 3d sub-brick of
  //   global PPPM_Electrode grid that I own without ghost cells
  // for slab PPPM_Electrode, assign z grid as if it were not extended

  nxlo_in = static_cast<int> (comm->xsplit[comm->myloc[0]] * nx_pppm);
  nxhi_in = static_cast<int> (comm->xsplit[comm->myloc[0]+1] * nx_pppm) - 1;

  nylo_in = static_cast<int> (comm->ysplit[comm->myloc[1]] * ny_pppm);
  nyhi_in = static_cast<int> (comm->ysplit[comm->myloc[1]+1] * ny_pppm) - 1;

/* Electrode Model */
  nzlo_in = static_cast<int>
      (comm->zsplit[comm->myloc[2]] * nz_pppm/(slab_volfactor*pair_et_omp->D * 3/domain->prd[2]));
  nzhi_in = static_cast<int>
      (comm->zsplit[comm->myloc[2]+1] * nz_pppm/(slab_volfactor*pair_et_omp->D * 3/domain->prd[2])) - 1;
/* End */

/* replaced
  nzlo_in = static_cast<int>
      (comm->zsplit[comm->myloc[2]] * nz_pppm/slab_volfactor);
  nzhi_in = static_cast<int>
      (comm->zsplit[comm->myloc[2]+1] * nz_pppm/slab_volfactor) - 1;
*/ 

  // nlower,nupper = stencil size for mapping particles to PPPM_Electrode grid

  nlower = -(order-1)/2;
  nupper = order/2;

  // shift values for particle <-> grid mapping
  // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

  if (order % 2) shift = OFFSET + 0.5;
  else shift = OFFSET;
  if (order % 2) shiftone = 0.0;
  else shiftone = 0.5;

  // nlo_out,nhi_out = lower/upper limits of the 3d sub-brick of
  //   global PPPM_Electrode grid that my particles can contribute charge to
  // effectively nlo_in,nhi_in + ghost cells
  // nlo,nhi = global coords of grid pt to "lower left" of smallest/largest
  //           position a particle in my box can be at
  // dist[3] = particle position bound = subbox + skin/2.0 + qdist
  //   qdist = offset due to TIP4P fictitious charge
  //   convert to triclinic if necessary
  // nlo_out,nhi_out = nlo,nhi + stencil size for particle mapping
  // for slab PPPM_Electrode, assign z grid as if it were not extended

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
  double zprd_slab = ZPRD_CORR();

  double dist[3];
  double cuthalf = 0.5*neighbor->skin + qdist;
  if (triclinic == 0) dist[0] = dist[1] = dist[2] = cuthalf;
  else kspacebbox(cuthalf,&dist[0]);

  int nlo,nhi;

  nlo = static_cast<int> ((sublo[0]-dist[0]-boxlo[0]) *
                            nx_pppm/xprd + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[0]+dist[0]-boxlo[0]) *
                            nx_pppm/xprd + shift) - OFFSET;
  nxlo_out = nlo + nlower;
  nxhi_out = nhi + nupper;

  nlo = static_cast<int> ((sublo[1]-dist[1]-boxlo[1]) *
                            ny_pppm/yprd + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[1]+dist[1]-boxlo[1]) *
                            ny_pppm/yprd + shift) - OFFSET;
  nylo_out = nlo + nlower;
  nyhi_out = nhi + nupper;

  nlo = static_cast<int> ((sublo[2]-dist[2]-boxlo[2]) *
                            nz_pppm/zprd_slab + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[2]+dist[2]-boxlo[2]) *
                            nz_pppm/zprd_slab + shift) - OFFSET;
  nzlo_out = nlo + nlower;
  nzhi_out = nhi + nupper;

  if (stagger_flag) {
    nxhi_out++;
    nyhi_out++;
    nzhi_out++;
  }

  // for slab PPPM_Electrode, change the grid boundary for processors at +z end
  //   to include the empty volume between periodically repeating slabs
  // for slab PPPM_Electrode, want charge data communicated from -z proc to +z proc,
  //   but not vice versa, also want field data communicated from +z proc to
  //   -z proc, but not vice versa
  // this is accomplished by nzhi_in = nzhi_out on +z end (no ghost cells)
  // also insure no other procs use ghost cells beyond +z limit

  if (slabflag == 1) {
    if (comm->myloc[2] == comm->procgrid[2]-1)
      nzhi_in = nzhi_out = nz_pppm - 1;
    nzhi_out = MIN(nzhi_out,nz_pppm-1);
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

/* replaced 
  int npey_fft,npez_fft;
  if (nz_pppm >= nprocs) {
    npey_fft = 1;
    npez_fft = nprocs;
  } else procs2grid2d(nprocs,ny_pppm,nz_pppm,&npey_fft,&npez_fft);

  int me_y = me % npey_fft;
  int me_z = me / npey_fft;
  
  nxlo_fft = 0;
  nxhi_fft = nx_pppm - 1;
  nylo_fft = me_y*ny_pppm/npey_fft;
  nyhi_fft = (me_y+1)*ny_pppm/npey_fft - 1;
  nzlo_fft = me_z*nz_pppm/npez_fft;
  nzhi_fft = (me_z+1)*nz_pppm/npez_fft - 1;
*/
  
  /* Eletrode Model */
  
  int npex_fft,npey_fft;
  if (nx_pppm >= nprocs) {
    npey_fft = 1;
    npex_fft = nprocs;
  } else procs2grid2d(nprocs,nx_pppm,ny_pppm,&npex_fft,&npey_fft);

  int me_x = me % npex_fft;
  int me_y = me / npex_fft;
  
  nzlo_fft = 0;
  nzhi_fft = nz_pppm - 1;
  
  nylo_fft = me_y*ny_pppm/npey_fft;
  nyhi_fft = (me_y+1)*ny_pppm/npey_fft - 1;
  
  nxlo_fft = me_x*nx_pppm/npex_fft;
  nxhi_fft = (me_x+1)*nx_pppm/npex_fft - 1;

  /* End */

  // PPPM_Electrode grid pts owned by this proc, including ghosts

  ngrid = (nxhi_out-nxlo_out+1) * (nyhi_out-nylo_out+1) *
    (nzhi_out-nzlo_out+1);

  // FFT grids owned by this proc, without ghosts
  // nfft = FFT points in FFT decomposition on this proc
  // nfft_brick = FFT points in 3d brick-decomposition on this proc
  // nfft_both = greater of 2 values

  nfft = (nxhi_fft-nxlo_fft+1) * (nyhi_fft-nylo_fft+1) *
    (nzhi_fft-nzlo_fft+1);
  int nfft_brick = (nxhi_in-nxlo_in+1) * (nyhi_in-nylo_in+1) *
    (nzhi_in-nzlo_in+1);
  nfft_both = MAX(nfft,nfft_brick);
}

/* ----------------------------------------------------------------------
   pre-compute modified (Hockney-Eastwood) Coulomb Green's function
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::compute_gf_ik()
{
  const double * const prd = domain->prd;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = ZPRD_CORR();
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double snx,sny,snz;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double sum1,dot1,dot2;
  double numerator,denominator;
  double sqk;

  int k,l,m,n,nx,ny,nz,kper,lper,mper;

  const int nbx = static_cast<int> ((g_ewald*xprd/(MY_PI*nx_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nby = static_cast<int> ((g_ewald*yprd/(MY_PI*ny_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nbz = static_cast<int> ((g_ewald*zprd_slab/(MY_PI*nz_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int twoorder = 2*order;

  /* Eletrode Model */
  double dz1 = - 2.0 * (pair_et_omp->pos_lo - domain->boxlo[2]);
  double dz2 = dz1 - 2.0 * pair_et_omp->D;
  /* End */
  
  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);
    snz = square(sin(0.5*unitkz*mper*zprd_slab/nz_pppm));

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);
      sny = square(sin(0.5*unitky*lper*yprd/ny_pppm));

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm*(2*k/nx_pppm);
        snx = square(sin(0.5*unitkx*kper*xprd/nx_pppm));

        sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);
	
	/* Electrode Model */
	
	img1_rl[n] = -cos(unitkz*mper*dz1);
	img1_im[n] =  sin(unitkz*mper*dz1);
	img2_rl[n] = -cos(unitkz*mper*dz2);
	img2_im[n] =  sin(unitkz*mper*dz2);
	
	/* End */

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
          greensfn[n++] = numerator*sum1/denominator;
        } else greensfn[n++] = 0.0;
      }
    }
  }
  
  /* Electrode */
  
  int block = (nxhi_fft - nxlo_fft + 1) * (nyhi_fft - nylo_fft + 1);
  int end  = block * (nzhi_fft - nzlo_fft + 1);

  for(int i=0; i<nz_pppm/2; i++)
  {
    int a = (end-block*(i+1));
    int b = block*i;
    
    for(int j=0; j<block; j++)
    {
      greensfn_rev[b+j] = (a + j)*2;
      greensfn_rev[a+j] = (b + j)*2;
    }
  }
  
  /* End */
}

/* ----------------------------------------------------------------------
   compute optimized Green's function for energy calculation
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::compute_gf_ad()
{
  const double * const prd = domain->prd;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab =ZPRD_CORR();
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double snx,sny,snz,sqk;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double numerator,denominator;
  int k,l,m,n,kper,lper,mper;

  const int twoorder = 2*order;

  for (int i = 0; i < 6; i++) sf_coeff[i] = 0.0;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);
    qz = unitkz*mper;
    snz = square(sin(0.5*qz*zprd_slab/nz_pppm));
    sz = exp(-0.25*square(qz/g_ewald));
    argz = 0.5*qz*zprd_slab/nz_pppm;
    wz = powsinxx(argz,twoorder);

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);
      qy = unitky*lper;
      sny = square(sin(0.5*qy*yprd/ny_pppm));
      sy = exp(-0.25*square(qy/g_ewald));
      argy = 0.5*qy*yprd/ny_pppm;
      wy = powsinxx(argy,twoorder);

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
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
          sf_coeff[0] += sf_precoeff1[n]*greensfn[n];
          sf_coeff[1] += sf_precoeff2[n]*greensfn[n];
          sf_coeff[2] += sf_precoeff3[n]*greensfn[n];
          sf_coeff[3] += sf_precoeff4[n]*greensfn[n];
          sf_coeff[4] += sf_precoeff5[n]*greensfn[n];
          sf_coeff[5] += sf_precoeff6[n]*greensfn[n];
          n++;
        } else {
          greensfn[n] = 0.0;
          sf_coeff[0] += sf_precoeff1[n]*greensfn[n];
          sf_coeff[1] += sf_precoeff2[n]*greensfn[n];
          sf_coeff[2] += sf_precoeff3[n]*greensfn[n];
          sf_coeff[3] += sf_precoeff4[n]*greensfn[n];
          sf_coeff[4] += sf_precoeff5[n]*greensfn[n];
          sf_coeff[5] += sf_precoeff6[n]*greensfn[n];
          n++;
        }
      }
    }
  }

  // compute the coefficients for the self-force correction

  double prex, prey, prez;
  prex = prey = prez = MY_PI/volume;
  prex *= nx_pppm/xprd;
  prey *= ny_pppm/yprd;
  prez *= nz_pppm/zprd_slab;
  sf_coeff[0] *= prex;
  sf_coeff[1] *= prex*2;
  sf_coeff[2] *= prey;
  sf_coeff[3] *= prey*2;
  sf_coeff[4] *= prez;
  sf_coeff[5] *= prez*2;

  // communicate values with other procs

  double tmp[6];
  MPI_Allreduce(sf_coeff,tmp,6,MPI_DOUBLE,MPI_SUM,world);
  for (n = 0; n < 6; n++) sf_coeff[n] = tmp[n];
}

/* ----------------------------------------------------------------------
   find center grid pt for each of my particles
   check that full stencil for the particle will fit in my 3d brick
   store central grid pt indices in part2grid array
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::particle_map()
{
  int nx,ny,nz;

  double **x = atom->x;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (int i = 0; i < nlocal; i++) {

    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // current particle coord can be outside global and local box
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

    nx = static_cast<int> ((x[i][0]-boxlo[0])*delxinv+shift) - OFFSET;
    ny = static_cast<int> ((x[i][1]-boxlo[1])*delyinv+shift) - OFFSET;
    nz = static_cast<int> ((x[i][2]-boxlo[2])*delzinv+shift) - OFFSET;

    part2grid[i][0] = nx;
    part2grid[i][1] = ny;
    part2grid[i][2] = nz;

    // check that entire stencil around nx,ny,nz will fit in my 3d brick

    if (nx+nlower < nxlo_out || nx+nupper > nxhi_out ||
        ny+nlower < nylo_out || ny+nupper > nyhi_out ||
        nz+nlower < nzlo_out || nz+nupper > nzhi_out)
      flag = 1;
  }

  if (flag) error->one(FLERR,"Out of range atoms - cannot compute PPPM_Electrode");
}

/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::make_rho()
{

  // clear 3d density array

  FFT_SCALAR * _noalias const d = &(density_brick[nzlo_out][nylo_out][nxlo_out]); 
  memset(d, 0, ngrid*sizeof(FFT_SCALAR));

  // no local atoms => nothing else to do

  const int nlocal = atom->nlocal;
  if(nlocal == 0) return;

  const int ix = nxhi_out - nxlo_out + 1;
  const int iy = nyhi_out - nylo_out + 1;

#if defined(_OPENMP)
#pragma omp parallel default(none)
#endif
  {
    const double * _noalias const q = atom->q;
    const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
    const int3_t * _noalias const p2g = (int3_t *) part2grid[0];

    const double boxlox = boxlo[0];
    const double boxloy = boxlo[1];
    const double boxloz = boxlo[2];

    // determine range of grid points handled by this thread
    int i, jfrom, jto, tid;
    loop_setup_thr(jfrom,jto,tid,ngrid,comm->nthreads);

    // get per thread data
    ThrData * thr = fix->get_thr(tid);
    FFT_SCALAR * const * const r1d = static_cast<FFT_SCALAR **>(thr->get_rho1d());

    // loop over my charges, add their contribution to nearby grid points
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // (dx,dy,dz) = distance to "lower left" grid pt
    
    for (int i = 0; i < nlocal; i++) {

      const int nx = p2g[i].a;
      const int ny = p2g[i].b;
      const int nz = p2g[i].t;

      // pre-screen whether this atom will ever come within
      // reach of the data segment this thread is updating
      
      if ( ((nz+nlower-nzlo_out)*ix*iy >= jto)
           || ((nz+nupper-nzlo_out+1)*ix*iy < jfrom) ) continue;
      
      const FFT_SCALAR dx = nx+shiftone - (x[i].x - boxlox)*delxinv;
      const FFT_SCALAR dy = ny+shiftone - (x[i].y - boxloy)*delyinv;
      const FFT_SCALAR dz = nz+shiftone - (x[i].z - boxloz)*delzinv;
      
      compute_rho1d_thr(r1d,dx,dy,dz);

      const FFT_SCALAR z0 = delvolinv * q[i];

      for (int n = nlower; n <= nupper; n++) {
	const int jn = (nz+n-nzlo_out)*ix*iy;
	const FFT_SCALAR y0 = z0*r1d[2][n];
	
	for (int m = nlower; m <= nupper; m++) {
	  const int jm = jn+(ny+m-nylo_out)*ix;
	  const FFT_SCALAR x0 = y0*r1d[1][m];

	  for (int l = nlower; l <= nupper; l++) {
	    const int jl = jm+nx+l-nxlo_out;
	    // make sure each thread only updates
	    // "his" elements of the density grid
	    if(jl >= jto) break;
	    if(jl < jfrom) continue;

	    d[jl] += x0*r1d[0][l];
	  }
	}
      }
    }
  }
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::poisson_ik()
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

  /* Eletrode Model */
  
  n = 0;
  for(i = 0; i < nfft; i++) 
  {
    int off = greensfn_rev[i];
    work1_img[n]    = 0.0;
    work1_img[n+1]  = 0.0;
 
    work1_img[n]    = img1_rl[i]*work1[off]   - img1_im[i]*work1[off+1];
    work1_img[n+1]  = img1_rl[i]*work1[off+1] + img1_im[i]*work1[off];

    work1_img[n]   += img2_rl[i]*work1[off]   - img2_im[i]*work1[off+1];
    work1_img[n+1] += img2_rl[i]*work1[off+1] + img2_im[i]*work1[off];
   
    n += 2;
  }
  
  /* End */
  
  // global energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  if (eflag_global || vflag_global) {
    if (vflag_global) {
      n = 0;
      for (i = 0; i < nfft; i++) {
      
        double sqr = 0.0;
        sqr += work1[n]*work1[n] + work1[n+1]*work1[n+1];
        sqr += work1[n]*work1_img[n] + work1[n+1]*work1_img[n+1];
	
        eng = s2 * greensfn[i] * (sqr);
        for (j = 0; j < 6; j++) virial[j] += eng*vg[i][j];
        if (eflag_global) energy += eng;
        n += 2;
      }
    } else {
      n = 0;
      for (i = 0; i < nfft; i++) {
        
	double sqr = 0.0;
        sqr += work1[n]*work1[n]     + work1[n+1]*work1[n+1];
        sqr += work1[n]*work1_img[n] + work1[n+1]*work1_img[n+1];
	
        energy += greensfn[i] * (sqr);
        n += 2;
      }
      energy *= s2;
    }
  }

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

  // extra FFTs for per-atom energy/virial

  if (evflag_atom) poisson_peratom();

  // triclinic system

  if (triclinic) {
    poisson_ik_triclinic();
    return;
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
   FFT-based Poisson solver for ad
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::poisson_ad()
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

  if (eflag_global || vflag_global) {
    if (vflag_global) {
      n = 0;
      for (i = 0; i < nfft; i++) {
        eng = s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        for (j = 0; j < 6; j++) virial[j] += eng*vg[i][j];
        if (eflag_global) energy += eng;
        n += 2;
      }
    } else {
      n = 0;
      for (i = 0; i < nfft; i++) {
        energy +=
          s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
        n += 2;
      }
    }
  }

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] *= scaleinv * greensfn[i];
    work1[n++] *= scaleinv * greensfn[i];
  }

  // extra FFTs for per-atom energy/virial

  if (vflag_atom) poisson_peratom();

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = work1[n];
    work2[n+1] = work1[n+1];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        u_brick[k][j][i] = work2[n];
        n += 2;
      }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ik
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::fieldforce_ik()
{
  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  const int nthreads = comm->nthreads;
  const int nlocal = atom->nlocal;

  // no local atoms => nothing to do
  if(nlocal == 0) return;

  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  const double * _noalias const q = atom->q;
  const int3_t * _noalias const p2g = (int3_t *) part2grid[0];
  
  const double qqrd2e = force->qqrd2e;
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

#if defined(_OPENMP)
#pragma omp parallel default(none)
#endif
  {
    FFT_SCALAR x0, y0, z0, ekx, eky, ekz;
    int i, ifrom, ito, tid, l, m, n, mx, my, mz;

    loop_setup_thr(ifrom,ito,tid,nlocal,nthreads);

    // get per thread data
    ThrData * thr = fix->get_thr(tid);
    dbl3_t * _noalias const f = (dbl3_t *) thr->get_f()[0];
    FFT_SCALAR * const * const r1d = static_cast<FFT_SCALAR **>(thr->get_rho1d());

    for (i = ifrom; i < ito; i++) {
      const int nx = p2g[i].a;
      const int ny = p2g[i].b;
      const int nz = p2g[i].t;
      const FFT_SCALAR dx = nx+shiftone - (x[i].x - boxlox)*delxinv;
      const FFT_SCALAR dy = ny+shiftone - (x[i].y - boxloy)*delyinv;
      const FFT_SCALAR dz = nz+shiftone - (x[i].z - boxloz)*delzinv;

      compute_rho1d_thr(r1d,dx,dy,dz);

      ekx = eky = ekz = ZEROF;
      for (n = nlower; n <= nupper; n++) {
	mz = n+nz;
	z0 = r1d[2][n];
	for (m = nlower; m <= nupper; m++) {
	  my = m+ny;
	  y0 = z0*r1d[1][m];
	  for (l = nlower; l <= nupper; l++) {
	    mx = l+nx;
	    x0 = y0*r1d[0][l];
	    ekx -= x0*vdx_brick[mz][my][mx];
	    eky -= x0*vdy_brick[mz][my][mx];
	    ekz -= x0*vdz_brick[mz][my][mx];
	  }
	}
      }

      // convert E-field to force

      const double qfactor = qqrd2e * scale * q[i];
      f[i].x += qfactor * ekx;
      f[i].y += qfactor * eky;
      if (slabflag != 2) f[i].z += qfactor * ekz;
    }
  } // end of parallel region
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ad
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::fieldforce_ad()
{
  const int nthreads = comm->nthreads;
  const int nlocal = atom->nlocal;

  // no local atoms => nothing to do
  if(nlocal == 0) return;

  const double * prd = (triclinic == 0) ? domain->prd : domain->prd_lamda;
  const double hx_inv = nx_pppm / prd[0];
  const double hy_inv = ny_pppm / prd[1];
  const double hz_inv = nz_pppm / prd[2];

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  const dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  const double * _noalias const q = atom->q;
  const int3_t * _noalias const p2g = (int3_t *) part2grid[0];
  const double qqrd2e = force->qqrd2e;
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

#if defined(_OPENMP)
#pragma omp parallel default(none)
#endif
  {
    double s1, s2, s3, sf;
    FFT_SCALAR ekx, eky, ekz;
    int i, ifrom, ito, tid, l, m, n, mx, my, mz;

    loop_setup_thr(ifrom,ito,tid,nlocal,nthreads);

    // get per thread data
    ThrData * thr = fix->get_thr(tid);
    dbl3_t * _noalias const f = (dbl3_t *) thr->get_f()[0];
    FFT_SCALAR * const * const r1d = static_cast<FFT_SCALAR **>(thr->get_rho1d());
    FFT_SCALAR * const * const d1d = static_cast<FFT_SCALAR **>(thr->get_drho1d());
    
    for (i = ifrom; i < ito; i++) {
      const int nx = p2g[i].a;
      const int ny = p2g[i].b;
      const int nz = p2g[i].t;
      const FFT_SCALAR dx = nx+shiftone - (x[i].x - boxlox)*delxinv;
      const FFT_SCALAR dy = ny+shiftone - (x[i].y - boxloy)*delyinv;
      const FFT_SCALAR dz = nz+shiftone - (x[i].z - boxloz)*delzinv;

      compute_rho1d_thr(r1d,dx,dy,dz);
      compute_drho1d_thr(d1d,dx,dy,dz);

      ekx = eky = ekz = ZEROF;
      for (n = nlower; n <= nupper; n++) {
	mz = n+nz;
	for (m = nlower; m <= nupper; m++) {
	  my = m+ny;
	  for (l = nlower; l <= nupper; l++) {
	    mx = l+nx;
	    ekx += d1d[0][l] * r1d[1][m] * r1d[2][n] * u_brick[mz][my][mx];
	    eky += r1d[0][l] * d1d[1][m] * r1d[2][n] * u_brick[mz][my][mx];
	    ekz += r1d[0][l] * r1d[1][m] * d1d[2][n] * u_brick[mz][my][mx];
        }
      }
    }
    ekx *= hx_inv;
    eky *= hy_inv;
    ekz *= hz_inv;

    // convert E-field to force and substract self forces

    const double qi = q[i];
    const double qfactor = force->qqrd2e * scale * qi;

    s1 = x[i].x*hx_inv;
    sf = sf_coeff[0]*sin(MY_2PI*s1);
    sf += sf_coeff[1]*sin(MY_4PI*s1);
    sf *= 2.0 * qi;
    f[i].x += qfactor * (ekx - sf);

    s2 = x[i].y*hy_inv;
    sf = sf_coeff[2]*sin(MY_2PI*s2);
    sf += sf_coeff[3]*sin(MY_4PI*s2);
    sf *= 2.0 * qi;
    f[i].y += qfactor*(eky - sf);

    s3 = x[i].z*hz_inv;
    sf = sf_coeff[4]*sin(MY_2PI*s3);
    sf += sf_coeff[5]*sin(MY_4PI*s3);
    sf *= 2.0 * qi;
    if (slabflag != 2) f[i].z += qfactor*(ekz - sf);
    }
  } // end of parallel region
}

/* ----------------------------------------------------------------------
   charge assignment into rho1d
   dx,dy,dz = distance of particle from "lower left" grid point
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::compute_rho1d_thr(FFT_SCALAR * const * const r1d, const FFT_SCALAR &dx, 
					  const FFT_SCALAR &dy, const FFT_SCALAR &dz)
{
  int k,l;
  FFT_SCALAR r1,r2,r3;

  for (k = (1-order)/2; k <= order/2; k++) {
    r1 = r2 = r3 = ZEROF;

    for (l = order-1; l >= 0; l--) {
      r1 = rho_coeff[l][k] + r1*dx;
      r2 = rho_coeff[l][k] + r2*dy;
      r3 = rho_coeff[l][k] + r3*dz;
    }
    r1d[0][k] = r1;
    r1d[1][k] = r2;
    r1d[2][k] = r3;
  }
}

/* ----------------------------------------------------------------------
   charge assignment into drho1d
   dx,dy,dz = distance of particle from "lower left" grid point
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::compute_drho1d_thr(FFT_SCALAR * const * const d1d, const FFT_SCALAR &dx,
					   const FFT_SCALAR &dy, const FFT_SCALAR &dz)
{
  int k,l;
  FFT_SCALAR r1,r2,r3;

  for (k = (1-order)/2; k <= order/2; k++) {
    r1 = r2 = r3 = ZEROF;

    for (l = order-2; l >= 0; l--) {
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
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::slabcorr()
{
  // compute local contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;

  double et_factor = 3.0;
  if(pair_et_omp->et_table) et_factor = 2.0;
  
  double dipole = 0.0;
  for (int i = 0; i < nlocal; i++) dipole += q[i] * x[i][2];
  dipole *= et_factor;
  
  // sum local contributions to get global dipole moment

  double dipole_all;
  MPI_Allreduce(&dipole,&dipole_all,1,MPI_DOUBLE,MPI_SUM,world);

  // need to make non-neutral systems and/or
  //  per-atom energy translationally invariant

  double dipole_r2 = 0.0;
  if (eflag_atom || fabs(qsum) > SMALL) {
    for (int i = 0; i < nlocal; i++)
      dipole_r2 += q[i]*x[i][2]*x[i][2];

    // sum local contributions

    double tmp;
    MPI_Allreduce(&dipole_r2,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
    dipole_r2 = tmp;
  }

  // compute corrections

  const double e_slabcorr = MY_2PI*(dipole_all*dipole_all -
    qsum*dipole_r2 - qsum*qsum*zprd*zprd/12.0)/volume;
  const double qscale = force->qqrd2e * scale;

  if (eflag_global) energy += qscale * e_slabcorr / et_factor;

  // per-atom energy

  if (eflag_atom) {
    double efact = qscale * MY_2PI/volume/et_factor;
    for (int i = 0; i < nlocal; i++)
      eatom[i] += efact * q[i]*(x[i][2]*dipole_all - 0.5*(dipole_r2 +
        qsum*x[i][2]*x[i][2]) - qsum*zprd*zprd/12.0);
  }

  // add on force corrections

  double ffact = qscale * (-4.0*MY_PI/volume);
  double **f = atom->f;

  for (int i = 0; i < nlocal; i++) f[i][2] += ffact * q[i]*(dipole_all - qsum*x[i][2]);
}

/* ----------------------------------------------------------------------
   Raptor Support
------------------------------------------------------------------------- */

void PPPM_ElectrodeOMP::compute_cplx(int vflag) { compute(1,vflag); energy /= comm->nprocs; }

void PPPM_ElectrodeOMP::compute_exch(int vflag)
{
  off_diag_energy = 0.0;
  if (vflag) for (int n = 0; n < 6; n++) off_diag_virial[n] = 0.0;
  
  pair_et_omp->A_Rq = A_Rq;
  pair_et_omp->is_exch_chg = is_exch_chg;

  double epair = pair_et_omp->compute_exch(vflag);
  MPI_Allreduce(&epair,&off_diag_energy,1,MPI_DOUBLE,MPI_SUM,world);
  off_diag_energy /= A_Rq;
}

#endif
