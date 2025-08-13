/* ----------------------------------------------------------------------
  This is variation from the LAMMPS code pair_lj_cut_coul_long
  Coded for the electrode image-plain model
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "force.h"
#include "kspace.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "modify.h"
#include "fix.h"

#include "pair_electrode.h"

#include "fix_evb.h"
#include "EVB_engine.h"
#include "EVB_list.h"
#include "EVB_complex.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

#define SMALL 0.000001
#define VECTOR_SUB(c,a,b) c[0]=a[0]-b[0];c[1]=a[1]-b[1];c[2]=a[2]-b[2]
#define VECTOR_PBC(a) domain->minimum_image(a[0],a[1],a[2])
#define VECTOR_R2(b,a) b=a[0]*a[0]+a[1]*a[1]+a[2]*a[2]

/* ---------------------------------------------------------------------- */

PairElectrode::PairElectrode(LAMMPS *lmp) : Pair(lmp)
{
  ewaldflag = pppmflag = 1;
  respa_enable = 1;
  ftable = NULL;
  qdist = 0.0;

  et_create();
}

/* ---------------------------------------------------------------------- */

PairElectrode::~PairElectrode()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
  if (ftable) free_tables();
  
  et_destroy();
}

/* ---------------------------------------------------------------------- */

void PairElectrode::compute(int eflag, int vflag)
{ 
  nlocal = atom->nlocal;
  nall = nlocal + atom->nghost;
  
  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  /* -------------------------------------------------------------------- */
  /* -----  This is the part that for the electrode model           ----- */
  
    
  if(fixevb)
  {  
    if(fixevb->Engine->evb_list->indicator==-1) return;
    else complex_atom = fixevb->Engine->complex_atom;
  }
  
  eimage = 0.0;
  et_setup();
  et_compute();

  /* -------------------------------------------------------------------- */

  int i,ii,j,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

  evdwl = ecoul = 0.0;

  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // electrode model
    
    double dr_i2l  = x[i][2] - pos_lo;
    double dr_i2h  = pos_hi - x[i][2];
    double ddr_i2l = cut_coul - dr_i2l;
    double ddr_i2h = cut_coul - dr_i2h;
    
    // end
      
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      
      /* --- Traditional compute for the pair (i,j) --- */

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;

        if (rsq < cut_coulsq) {
          if (!ncoultablebits || rsq <= tabinnersq) {
            r = sqrt(rsq);
            grij = g_ewald * r;
            expm2 = exp(-grij*grij);
            t = 1.0 / (1.0 + EWALD_P*grij);
            erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
            prefactor = qqrd2e * qtmp*q[j]/r;
            forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
          } else {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            itable = rsq_lookup.i & ncoulmask;
            itable >>= ncoulshiftbits;
            fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
            table = ftable[itable] + fraction*dftable[itable];
            forcecoul = qtmp*q[j] * table;
            if (factor_coul < 1.0) {
              table = ctable[itable] + fraction*dctable[itable];
              prefactor = qtmp*q[j] * table;
              forcecoul -= (1.0-factor_coul)*prefactor;
            }
          }
        } else forcecoul = 0.0;

        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        } else forcelj = 0.0;

        fpair = (forcecoul + factor_lj*forcelj) * r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
	
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          if (rsq < cut_coulsq) {
            if (!ncoultablebits || rsq <= tabinnersq)
              ecoul = prefactor*erfc;
            else {
              table = etable[itable] + fraction*detable[itable];
              ecoul = qtmp*q[j] * table;
            }
            if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
          } else ecoul = 0.0;

          if (rsq < cut_ljsq[itype][jtype]) {
            evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
              offset[itype][jtype];
            evdwl *= factor_lj;
          } else evdwl = 0.0;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,ecoul,fpair,delx,dely,delz);
      }
      
      /* --- Compute atoms with non-self image copies --- */
            
      double dr_j2l = x[j][2] - pos_lo;
      double dr_j2h = pos_hi - x[j][2];
      double ddr_j2l = cut_coul - dr_j2l;
      double ddr_j2h = cut_coul - dr_j2h;

      if(ddr_i2l > 0.0 && dr_j2l < ddr_i2l ) eimage += single_coul(x[i], x_image [j], q[i], -q[j], f[i]);
      if(ddr_i2h > 0.0 && dr_j2h < ddr_i2h ) eimage += single_coul(x[i], x_image2[j], q[i], -q[j], f[i]);
      
      if(ddr_j2l > 0.0 && dr_i2l < ddr_j2l ) eimage += single_coul(x[j], x_image [i], q[j], -q[i], f[j]);
      if(ddr_j2h > 0.0 && dr_i2h < ddr_j2h ) eimage += single_coul(x[j], x_image2[i], q[j], -q[i], f[j]);
      
      /* --- End of the loop of pair-list --- */
    }
  }
  
  eng_coul += eimage * 0.5;
  
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairElectrode::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cut_lj,n+1,n+1,"pair:cut_lj");
  memory->create(cut_ljsq,n+1,n+1,"pair:cut_ljsq");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairElectrode::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 2) error->all(FLERR,"Illegal pair_style command: wrong number of paramters.");

  cut_lj_global = force->numeric(FLERR,arg[0]);
  if (narg == 1) cut_coul = cut_lj_global;
  else cut_coul = force->numeric(FLERR,arg[1]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairElectrode::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(arg[0],atom->ntypes,ilo,ihi);
  force->bounds(arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_lj_one = cut_lj_global;
  if (narg == 5) cut_lj_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut_lj[i][j] = cut_lj_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairElectrode::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair style lj/cut/coul/long requires atom attribute q");

  // request regular or rRESPA neighbor lists

  int irequest;

  if (update->whichflag == 1 && strstr(update->integrate_style,"respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;

    if (respa == 0) irequest = neighbor->request(this,instance_me);
    else if (respa == 1) {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    } else {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 2;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respamiddle = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    }

  } else irequest = neighbor->request(this,instance_me);

  cut_coulsq = cut_coul * cut_coul;

  // set rRESPA cutoffs

  if (strstr(update->integrate_style,"respa") &&
      ((Respa *) update->integrate)->level_inner >= 0)
    cut_respa = ((Respa *) update->integrate)->cutoff;
  else cut_respa = NULL;

  // insure use of KSpace long-range solver, set g_ewald

  if (force->kspace == NULL)
    error->all(FLERR,"Pair style requires a KSpace style");
  g_ewald = force->kspace->g_ewald;

  // setup force tables

  if (ncoultablebits) init_tables(cut_coul,cut_respa);
  
  et_init();
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairElectrode::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listinner = ptr;
  else if (id == 2) listmiddle = ptr;
  else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairElectrode::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut_lj[i][j] = mix_distance(cut_lj[i][i],cut_lj[j][j]);
  }

  // include TIP4P qdist in full cutoff, qdist = 0.0 if not TIP4P

  double cut = MAX(cut_lj[i][j],cut_coul+2.0*qdist);
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag) {
    double ratio = sigma[i][j] / cut_lj[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  cut_ljsq[j][i] = cut_ljsq[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

  if (cut_respa && MIN(cut_lj[i][j],cut_coul) < cut_respa[3])
    error->all(FLERR,"Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double sig2 = sigma[i][j]*sigma[i][j];
    double sig6 = sig2*sig2*sig2;
    double rc3 = cut_lj[i][j]*cut_lj[i][j]*cut_lj[i][j];
    double rc6 = rc3*rc3;
    double rc9 = rc3*rc6;
    etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
    ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
  }

  return cut;
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairElectrode::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut_lj[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairElectrode::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&cut_lj[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairElectrode::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
  fwrite(&ncoultablebits,sizeof(int),1,fp);
  fwrite(&tabinner,sizeof(double),1,fp);  
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairElectrode::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_lj_global,sizeof(double),1,fp);
    fread(&cut_coul,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
    fread(&ncoultablebits,sizeof(int),1,fp);
    fread(&tabinner,sizeof(double),1,fp);    
  }
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
  MPI_Bcast(&ncoultablebits,1,MPI_INT,0,world);
  MPI_Bcast(&tabinner,1,MPI_DOUBLE,0,world);
}

/* ---------------------------------------------------------------------- */

double PairElectrode::single(int i, int j, int itype, int jtype,
                                 double rsq,
                                 double factor_coul, double factor_lj,
                                 double &fforce)
{
  error->warning(FLERR, "No single() function is defined in pair/electrode module.");
}

/* ---------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */

void *PairElectrode::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"cut_coul") == 0) return (void *) &cut_coul;
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  return NULL;
}

double PairElectrode::single_coul(double *xi, double* xj, double qi, double qj, double* fi)
{
  double r,grij,expm2,t,erfc,prefactor;
  double fraction,table,forcecoul,phicoul;
  int itable;

  double dx = xi[0] - xj[0];
  double dy = xi[1] - xj[1];
  double dz = xi[2] - xj[2];
  double rsq = dx*dx + dy*dy + dz*dz;
  double r2inv = 1.0/rsq;

  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq) {
      r = sqrt(rsq);
      grij = g_ewald * r;
      expm2 = exp(-grij*grij);
      t = 1.0 / (1.0 + EWALD_P*grij);
      erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
      prefactor = force->qqrd2e * qi * qj / r;
      forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
    } else {
      union_int_float_t rsq_lookup_single;
      rsq_lookup_single.f = rsq;
      itable = rsq_lookup_single.i & ncoulmask;
      itable >>= ncoulshiftbits;
      fraction = (rsq_lookup_single.f - rtable[itable]) * drtable[itable];
      table = ftable[itable] + fraction*dftable[itable];
      forcecoul = qi * qj * table;
    }
  } else forcecoul = 0.0;

  double fforce = forcecoul * r2inv;

  fi[0] += dx * fforce;
  fi[1] += dy * fforce;
  fi[2] += dz * fforce;

  double eng = 0.0;

  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq)
      phicoul = prefactor*erfc;
    else {
      table = etable[itable] + fraction*detable[itable];
      phicoul = qi * qj * table;
    }

    eng = phicoul;
  }

  return eng;
}

/* -------------------------------------------------------------------------
 *  CGIS based pair-wise interaction for MSEVB exchange-charge part
 * ------------------------------------------------------------------------- */
 
void PairElectrode::_cgis_init(double cut)
{
  _cgis_cut = 0.81 * cut;
  _cgis_cut_sq = _cgis_cut * _cgis_cut;
  _cgis_cut_inv = 1.0 / cut;
  
  _cgis_A = -1.0 / (cut * cut * _cgis_cut * (cut - _cgis_cut));
  _cgis_B = -1.0 / (cut * cut * _cgis_cut);
  _cgis_C =  1.0 / (cut * cut);
  
  double dr = _cgis_cut - cut;
  _cgis_eng_core = _cgis_A/3.0*dr*dr*dr + _cgis_B/2.0*dr*dr + _cgis_C*dr - _cgis_cut_inv;
}

/* ------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------- */

double PairElectrode::_cgis_single(double *rij, double qiqj, double *fi, double *fj)
{
  if(fabs(qiqj)<SMALL) return 0.0;
  double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
  double qqrd2e = force->qqrd2e;
  double ecoul = 0.0;
  
  if (r2 < cut_coulsq) 
  {  
    double r = sqrt(r2);
    double r2inv = 1.0 / r2;
    double rinv = 1.0/ r;
	    
    ecoul = rinv;
    double fpair = r2inv;
	    
    if(r < _cgis_cut) 
    {
      ecoul += _cgis_eng_core - _cgis_B/2.0 * (r2 - _cgis_cut_sq);
      fpair += _cgis_B * r;
    } 
    else 
    {
      double dr = r - cut_coul;
      double dr2 = dr * dr;
      ecoul += _cgis_A/3.0*dr2*dr + _cgis_B/2.0*dr2 + _cgis_C*dr - _cgis_cut_inv;
      fpair += - _cgis_A*dr2 - _cgis_B*dr - _cgis_C;
    }
    
    ecoul*= qqrd2e * qiqj;
    fpair*= qqrd2e * qiqj * rinv;

    double ftmpx = fpair * rij[0];
    double ftmpy = fpair * rij[1];
    double ftmpz = fpair * rij[2];
            
    fi[0] += ftmpx;
    fi[1] += ftmpy;
    fi[2] += ftmpz;
            
    if(fj) 
    {
      fj[0] -= ftmpx;
      fj[1] -= ftmpy;
      fj[2] -= ftmpz;
    }          
  }

  return ecoul;
}

/* ------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------- */

double PairElectrode::_cgis_single_eng(double *rij, double qiqj)
{
  if(fabs(qiqj)<SMALL) return 0.0;
  double r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
  double qqrd2e = force->qqrd2e;
  double ecoul = 0.0;
  
  if (r2 < cut_coulsq) {  
    double r = sqrt(r2);
    double r2inv = 1.0 / r2;
    double rinv = 1.0/ r;
	    
    ecoul = rinv;
	    
    if(r < _cgis_cut) ecoul += _cgis_eng_core - _cgis_B/2.0 * (r2 - _cgis_cut_sq);
    else {
      double dr = r - cut_coul;
      double dr2 = dr * dr;
      ecoul += _cgis_A/3.0*dr2*dr + _cgis_B/2.0*dr2 + _cgis_C*dr - _cgis_cut_inv;
    }
    
    ecoul*= qqrd2e * qiqj;
  }

  return ecoul;
}

/* ------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------- */

double PairElectrode::compute_exch(int vflag)
{
  double energy_offdiag = 0.0;
  double eimg = 0.0;

  int     nlocal = atom->nlocal;
  double      *q = atom->q; 
  double     **f = atom->f;
  double     **x = atom->x;

  //et_setup(); // This does nothing when EVB neighbor list is loaded.

  _cgis_init(cut_coul);
  
  for(int i=0; i<nlocal; i++) if(is_exch_chg[i])
  {
    /* --- Compute atoms with self image copies --- */

    double dr_i2l = x[i][2] - pos_lo;
    double dr_i2h = pos_hi - x[i][2];
    double ddr_i2l = cut_coul - dr_i2l * 2.0;
    double ddr_i2h = cut_coul - dr_i2h * 2.0;
    
    double rij[3];
    double qiqj = - q[i] * q[i] * A_Rq;
    
    VECTOR_SUB(rij, x[i], x_image[i]);
    if(ddr_i2l > 0.0) eimg += _cgis_single(rij, qiqj, f[i], NULL);
    
    VECTOR_SUB(rij, x[i], x_image2[i]);
    if(ddr_i2h > 0.0) eimg += _cgis_single(rij, qiqj, f[i], NULL);
    
    /* --- Plane-atom interaction ---     
       Above is the original part, however, I deleted this because it may cause 
       unconserved energy for Raptor. The reason is that we don't have an energy decomposition
       method yet to get the cross-term energy between the exchange-charge and plane charges
    */
    
    //plane->compute_one(i, x[i], f[i], q[i]*A_Rq, 0);
    //energy_offdiag += plane->eng_coul_one;
  }
    
  double qqrd2e = force->qqrd2e;
  
  int         inum = list->inum;
  int       *ilist = list->ilist;
  int    *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
    
  for(int i=0; i<inum; i++) 
  {
    bool iflag = false;
     
    int atomi = ilist[i];
    if(is_exch_chg[atomi]) iflag = true;
      
    int jnum = numneigh[atomi];
    int *jlist = firstneigh[atomi];
      
    for(int j=0; j<jnum; j++)
    {
      int atomj = jlist[j];
      atomj &=NEIGHMASK;	  
	
      bool jflag = false;
      if(is_exch_chg[atomj]) jflag = true;
        
      if( (iflag && (!jflag)) || (jflag && (!iflag)) ) 
      {
	/*************************************************************/
	/*************************************************************/
	  
	double qiqj = q[atomi] * q[atomj] * A_Rq;
        double rij[3];
	
	VECTOR_SUB(rij,x[atomi],x[atomj]);
	energy_offdiag += _cgis_single(rij, qiqj, f[atomi], f[atomj]);
	  
	double dr_i2l  = x[atomi][2] - pos_lo;
        double dr_i2h  = pos_hi - x[atomi][2];
        double ddr_i2l = cut_coul - dr_i2l;
        double ddr_i2h = cut_coul - dr_i2h;
	double dr_j2l  = x[atomj][2] - pos_lo;
        double dr_j2h  = pos_hi - x[atomj][2];
        double ddr_j2l = cut_coul - dr_j2l;
        double ddr_j2h = cut_coul - dr_j2h;
          
        qiqj = - qiqj;
    
	if(iflag)
	{
          if(ddr_i2l > 0.0 && dr_j2l < ddr_i2l ) 
	  {
	    VECTOR_SUB(rij, x[atomi], x_image[atomj]);
	    eimg += _cgis_single(rij, qiqj, f[atomi], NULL);
	  }
	 
          if(ddr_i2h > 0.0 && dr_j2h < ddr_i2h ) 
	  {
	    VECTOR_SUB(rij, x[atomi], x_image2[atomj]);
	    eimg += _cgis_single(rij, qiqj, f[atomi], NULL);
	  }
        }
	else
	{
          if(ddr_j2l > 0.0 && dr_i2l < ddr_j2l ) 
	  {
	    VECTOR_SUB(rij, x[atomj], x_image[atomi]);
	    eimg += _cgis_single(rij, qiqj, f[atomj], NULL);
          }
	 
	  if(ddr_j2h > 0.0 && dr_i2h < ddr_j2h ) 
	  {
	    VECTOR_SUB(rij, x[atomj], x_image2[atomi]);
	    eimg += _cgis_single(rij, qiqj, f[atomj], NULL);
	  }
	}
	  
	/*************************************************************/
	/*************************************************************/	  
      } // End of calculation
    } // End of loop atom j
  } // End of loop atom i

  energy_offdiag += 0.5 * eimg;
  return energy_offdiag;
}

/* -------------------------------------------------------------------------
   Functions used by the SCI code.
   ------------------------------------------------------------------------- */

double PairElectrode::single_ener_noljcoul(int i, int j, int itype, int jtype, double rsq, double factor_lj)
{
  double eimage = 0.0;

  double ** x = atom->x;
  double * q = atom->q;

  double dr_i2l  = x[i][2] - pos_lo;
  double dr_i2h  = pos_hi - x[i][2];
  double ddr_i2l = cut_coul - dr_i2l;
  double ddr_i2h = cut_coul - dr_i2h;
  
  double dr_j2l = x[j][2] - pos_lo;
  double dr_j2h = pos_hi - x[j][2];
  double ddr_j2l = cut_coul - dr_j2l;
  double ddr_j2h = cut_coul - dr_j2h;
  
  if(ddr_i2l > 0.0 && dr_j2l < ddr_i2l ) eimage += single_coul_eng(x[i], x_image [j], q[i], -q[j]);
  if(ddr_i2h > 0.0 && dr_j2h < ddr_i2h ) eimage += single_coul_eng(x[i], x_image2[j], q[i], -q[j]);
  
  if(ddr_j2l > 0.0 && dr_i2l < ddr_j2l ) eimage += single_coul_eng(x[j], x_image [i], q[j], -q[i]);
  if(ddr_j2h > 0.0 && dr_i2h < ddr_j2h ) eimage += single_coul_eng(x[j], x_image2[i], q[j], -q[i]);
  
  return eimage;
}

/* ---------------------------------------------------------------------- */

double PairElectrode::single_coul_eng(double *xi, double* xj, double qi, double qj)
{
  double r,grij,expm2,t,erfc,prefactor;
  double fraction,table,forcecoul,phicoul;
  int itable;

  double dx = xi[0] - xj[0];
  double dy = xi[1] - xj[1];
  double dz = xi[2] - xj[2];
  double rsq = dx*dx + dy*dy + dz*dz;
  double r2inv = 1.0 / rsq;

  double eng = 0.0;

  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq) {
      r = sqrt(rsq);
      grij = g_ewald * r;
      expm2 = exp(-grij*grij);
      t = 1.0 / (1.0 + EWALD_P*grij);
      erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
      prefactor = force->qqrd2e * qi * qj / r;
      
      eng = prefactor*erfc;
    } else {
      union_int_float_t rsq_lookup_single;
      rsq_lookup_single.f = rsq;
      itable = rsq_lookup_single.i & ncoulmask;
      itable >>= ncoulshiftbits;
      fraction = (rsq_lookup_single.f - rtable[itable]) * drtable[itable];
      
      table = etable[itable] + fraction*detable[itable];
      eng = qi * qj * table;
    }
  }
  
  return eng;
}

/* ---------------------------------------------------------------------- */

void PairElectrode::single_fpair_noljcoul(int i, int j, double * fi, double * fj)
{
  double ** x = atom->x;
  double * q = atom->q;

  double dr_i2l  = x[i][2] - pos_lo;
  double dr_i2h  = pos_hi - x[i][2];
  double ddr_i2l = cut_coul - dr_i2l;
  double ddr_i2h = cut_coul - dr_i2h;
  
  double dr_j2l = x[j][2] - pos_lo;
  double dr_j2h = pos_hi - x[j][2];
  double ddr_j2l = cut_coul - dr_j2l;
  double ddr_j2h = cut_coul - dr_j2h;

  fi[0] = fi[1] = fi[2] = 0.0;
  fj[0] = fj[1] = fj[2] = 0.0;
  
  if(ddr_i2l > 0.0 && dr_j2l < ddr_i2l ) single_coul_frc(x[i], x_image [j], q[i], -q[j], fi);
  if(ddr_i2h > 0.0 && dr_j2h < ddr_i2h ) single_coul_frc(x[i], x_image2[j], q[i], -q[j], fi);
  
  if(ddr_j2l > 0.0 && dr_i2l < ddr_j2l ) single_coul_frc(x[j], x_image [i], q[j], -q[i], fj);
  if(ddr_j2h > 0.0 && dr_i2h < ddr_j2h ) single_coul_frc(x[j], x_image2[i], q[j], -q[i], fj);
}

/* ---------------------------------------------------------------------- */

void PairElectrode::single_coul_frc(double *xi, double* xj, double qi, double qj, double * ff)
{
  double r,grij,expm2,t,erfc,prefactor;
  double fraction,table,forcecoul,phicoul;
  int itable;

  double dx = xi[0] - xj[0];
  double dy = xi[1] - xj[1];
  double dz = xi[2] - xj[2];
  double rsq = dx*dx + dy*dy + dz*dz;
  double r2inv = 1.0/rsq;

  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq) {
      r = sqrt(rsq);
      grij = g_ewald * r;
      expm2 = exp(-grij*grij);
      t = 1.0 / (1.0 + EWALD_P*grij);
      erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
      prefactor = force->qqrd2e * qi * qj / r;
      forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
    } else {
      union_int_float_t rsq_lookup_single;
      rsq_lookup_single.f = rsq;
      itable = rsq_lookup_single.i & ncoulmask;
      itable >>= ncoulshiftbits;
      fraction = (rsq_lookup_single.f - rtable[itable]) * drtable[itable];
      table = ftable[itable] + fraction*dftable[itable];
      forcecoul = qi * qj * table;
    }
  } else forcecoul = 0.0;

  double fforce = forcecoul * r2inv;

  ff[0] += dx * fforce;
  ff[1] += dy * fforce;
  ff[2] += dz * fforce;
}
