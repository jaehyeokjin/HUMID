/* ----------------------------------------------------------------------
  This is variation from the LAMMPS code pair_lj_cut_coul_long
  Coded for the electrode image-plain model
------------------------------------------------------------------------- */

#if defined (_OPENMP)

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

#include "pair_electrode_omp.h"

#include "fix_evb.h"
#include "EVB_engine.h"
#include "EVB_list.h"
#include "EVB_complex.h"

#include "suffix.h"

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

PairElectrodeOMP::PairElectrodeOMP(LAMMPS *lmp) : PairElectrode(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;

  single_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairElectrodeOMP::~PairElectrodeOMP()
{

}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairElectrodeOMP::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 2) error->all(FLERR,"Illegal pair_style command: wrong number of paramters.");

  cut_lj_global = force->numeric(FLERR,arg[0]);
  if (narg == 1) cut_coul = cut_lj_global;
  else cut_coul = force->numeric(FLERR,arg[1]);

  _cgis_init(cut_coul);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
  }
}

/* ---------------------------------------------------------------------- */

void PairElectrodeOMP::compute(int eflag, int vflag)
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

/* ------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------- */

double PairElectrodeOMP::compute_exch(int vflag)
{
  double energy_offdiag = 0.0;
  double eimg = 0.0;

  int     nlocal = atom->nlocal;
  double      *q = atom->q; 
  double     **f = atom->f;
  double     **x = atom->x;

  et_setup();
  
  for(int i=0; i<nlocal; i++) if(is_exch_chg[i]) {
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
  
  for(int i=0; i<inum; i++) {
    bool iflag = false;
    
    int atomi = ilist[i];
    if(is_exch_chg[atomi]) iflag = true;
    
    int jnum = numneigh[atomi];
    int *jlist = firstneigh[atomi];
    
    for(int j=0; j<jnum; j++) {
      int atomj = jlist[j];
      atomj &=NEIGHMASK;	  
      
      bool jflag = false;
      if(is_exch_chg[atomj]) jflag = true;
      
      if( (iflag && (!jflag)) || (jflag && (!iflag)) ) {
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
    
	if(iflag) {
          if(ddr_i2l > 0.0 && dr_j2l < ddr_i2l ) {
	    VECTOR_SUB(rij, x[atomi], x_image[atomj]);
	    eimg += _cgis_single(rij, qiqj, f[atomi], NULL);
	  }
	  
          if(ddr_i2h > 0.0 && dr_j2h < ddr_i2h ) {
	    VECTOR_SUB(rij, x[atomi], x_image2[atomj]);
	    eimg += _cgis_single(rij, qiqj, f[atomi], NULL);
	  }
        } else {
          if(ddr_j2l > 0.0 && dr_i2l < ddr_j2l ) {
	    VECTOR_SUB(rij, x[atomj], x_image[atomi]);
	    eimg += _cgis_single(rij, qiqj, f[atomj], NULL);
          }
	  
	  if(ddr_j2h > 0.0 && dr_i2h < ddr_j2h ) {
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

#endif
