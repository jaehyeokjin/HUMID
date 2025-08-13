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
     Based on lj/cut/coul/long and table by Paul Crozier (SNL)
              lj/gulp, lennard/gulp, and buck/gulp edited by Paolo Raiteri (Curtin)
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_gulp_coul_long.h"
#include "atom.h"
#include "comm.h"
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

using namespace LAMMPS_NS;
using namespace MathConst;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

enum{NONE,RLINEAR,RSQ,BMP};

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

PairGulpCoulLong::PairGulpCoulLong(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  ewaldflag = pppmflag = 1;
  respa_enable = 0;
  ftable = NULL;
  qdist = 0.0;

  ntables = 0;
  tables = NULL;
}

/* ---------------------------------------------------------------------- */

PairGulpCoulLong::~PairGulpCoulLong()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    // lj/cut
    memory->destroy(cut_ljcut);
    memory->destroy(cutsq_ljcut);
    memory->destroy(epsilon_ljcut);
    memory->destroy(sigma_ljcut);
    memory->destroy(setflag_ljcut);
    memory->destroy(lj1_ljcut);
    memory->destroy(lj2_ljcut);
    memory->destroy(lj3_ljcut);
    memory->destroy(lj4_ljcut);
    memory->destroy(offset_ljcut);

    // lj/gulp
    memory->destroy(cut_ljgulp);
    memory->destroy(cutsq_ljgulp);
    memory->destroy(cut_inner_ljgulp);
    memory->destroy(cutsq_inner_ljgulp);
    memory->destroy(epsilon_ljgulp);
    memory->destroy(sigma_ljgulp);
    memory->destroy(setflag_ljgulp);
    memory->destroy(lj1_ljgulp);
    memory->destroy(lj2_ljgulp);
    memory->destroy(lj3_ljgulp);
    memory->destroy(lj4_ljgulp);

    // lennard/gulp
    memory->destroy(cut_lengulp);
    memory->destroy(cutsq_lengulp);
    memory->destroy(cut_inner_lengulp);
    memory->destroy(cutsq_inner_lengulp);
    memory->destroy(a_lengulp);
    memory->destroy(b_lengulp);
    memory->destroy(setflag_lengulp);
    memory->destroy(lj1_lengulp);
    memory->destroy(lj2_lengulp);
    memory->destroy(lj3_lengulp);
    memory->destroy(lj4_lengulp);

    // buck/gulp
    memory->destroy(cut_buckgulp);
    memory->destroy(cutsq_buckgulp);
    memory->destroy(cut_inner_buckgulp);
    memory->destroy(cutsq_inner_buckgulp);
    memory->destroy(a_buckgulp);
    memory->destroy(rho_buckgulp);
    memory->destroy(c_buckgulp);
    memory->destroy(setflag_buckgulp);
    memory->destroy(rhoinv_buckgulp);
    memory->destroy(buck1_buckgulp);
    memory->destroy(buck2_buckgulp);
    memory->destroy(offset_buckgulp);

    //table
    memory->destroy(setflag_table);
    memory->destroy(cut_table);
    memory->destroy(cutsq_table);
    memory->destroy(tabindex);
    memory->destroy(tab_scale);
  }

  if (ftable) free_tables();

  if(ntables > 0) {
    for (int m = 0; m < ntables; m++) free_table(&tables[m]);
    memory->sfree(tables);
  }
}

/* ---------------------------------------------------------------------- */

void PairGulpCoulLong::compute(int eflag, int vflag)
{
  int i,ii,j,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

  Table *tb;
  double value, a, b;
  union_int_float_t rsq_lookup;
  int tlm1 = tablength - 1;

  evdwl = ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
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

      if(rsq < cutsq[itype][jtype]) {
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
	}
	
	double fvdwl;
	forcelj = 0.0;
	evdwl = 0.0;

	// lj/cut
	if(setflag_ljcut[itype][jtype] && rsq < cutsq_ljcut[itype][jtype]) {
	  r6inv = r2inv*r2inv*r2inv;
	  forcelj += r6inv * (lj1_ljcut[itype][jtype]*r6inv - lj2_ljcut[itype][jtype]);
	  
	  if(eflag) {
	    double eng = r6inv*(lj3_ljcut[itype][jtype]*r6inv-lj4_ljcut[itype][jtype]) - offset_ljcut[itype][jtype];
	    evdwl += eng;
	  }
	}
	
	// lj/gulp
	if(setflag_ljgulp[itype][jtype] && rsq < cutsq_ljgulp[itype][jtype]) {
	  double philj, rr, dp, d, dd, tt, dt;

	  r6inv = r2inv*r2inv*r2inv;
	  fvdwl = r6inv * (lj1_ljgulp[itype][jtype]*r6inv - lj2_ljgulp[itype][jtype]);

	  if (rsq > cutsq_inner_ljgulp[itype][jtype]) {
	    philj = r6inv*(lj3_ljgulp[itype][jtype]*r6inv-lj4_ljgulp[itype][jtype]);
	    
	    rr = sqrt(rsq);
	    dp = (cut_ljgulp[itype][jtype] - cut_inner_ljgulp[itype][jtype]);
	    d = (rr-cut_inner_ljgulp[itype][jtype]) / dp;
	    dd = 1.-d;
	    // taperig function - gulp style
	    tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;
	    // minus the derivative of the tapering function
	    dt = 30.* d*d * dd*dd * rr / dp;
	    
	    fvdwl = fvdwl*tt + philj*dt;
	  } else tt = 1;
	  
	  forcelj += fvdwl;

	  if(eflag) {
	    double eng = r6inv * (lj3_ljgulp[itype][jtype]*r6inv - lj4_ljgulp[itype][jtype]);
	    evdwl += eng * tt;
	  }
	}

	// lennard/gulp
	if(setflag_lengulp[itype][jtype] && rsq < cutsq_lengulp[itype][jtype]) {
	  double philj, rr, dp, d, dd, tt, dt;

	  r6inv = r2inv*r2inv*r2inv;
	  fvdwl = r6inv * (lj1_lengulp[itype][jtype]*r6inv - lj2_lengulp[itype][jtype]);

	  if (rsq > cutsq_inner_lengulp[itype][jtype]) {
	    philj = r6inv*(lj3_lengulp[itype][jtype]*r6inv-lj4_lengulp[itype][jtype]);  
	    
	    rr = sqrt(rsq);
	    dp = (cut_lengulp[itype][jtype] - cut_inner_lengulp[itype][jtype]);
	    d = (rr-cut_inner_lengulp[itype][jtype]) / dp;
	    dd = 1.-d;
	    // taperig function - gulp style
	    tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;
	    // minus the derivative of the tapering function
	    dt = 30.* d*d * dd*dd * rr / dp;
	    
	    fvdwl = fvdwl*tt + philj*dt;
	  } else tt = 1;
	  forcelj += fvdwl;

	  if(eflag) {
	    double eng = r6inv * (lj3_lengulp[itype][jtype]*r6inv - lj4_lengulp[itype][jtype]);
	    evdwl += eng * tt;
	  }
	}

	// buck/gulp
	if(setflag_buckgulp[itype][jtype] && rsq < cutsq_buckgulp[itype][jtype]) {
	  double rexp, phibuck, dp, d, dd, tt, dt;

	  r6inv = r2inv*r2inv*r2inv;
	  r = sqrt(rsq);
	  rexp = exp(-r*rhoinv_buckgulp[itype][jtype]);
	  fvdwl = buck1_buckgulp[itype][jtype]*r*rexp - buck2_buckgulp[itype][jtype]*r6inv;
	  if (rsq > cutsq_inner_buckgulp[itype][jtype]) {
	    phibuck = a_buckgulp[itype][jtype]*rexp - c_buckgulp[itype][jtype]*r6inv;
	    dp = (cut_buckgulp[itype][jtype] - cut_inner_buckgulp[itype][jtype]);
	    d = (r-cut_inner_buckgulp[itype][jtype]) / dp;
	    dd = 1.-d;
	    // taperig function - gulp style
	    tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;
	    // minus the derivative of the tapering function
	    dt = 30.* d*d * dd*dd * r / dp;
	    
	    fvdwl = fvdwl*tt + phibuck*dt;
	  } else tt = 1;
	  forcelj += fvdwl;

	  if(eflag) {
	    double eng = a_buckgulp[itype][jtype]*rexp - c_buckgulp[itype][jtype]*r6inv; 
	    evdwl += eng * tt;
	  }
	}

	// table
	if(setflag_table[itype][jtype] && rsq < cutsq_table[itype][jtype]) {
	  tb = &tables[tabindex[itype][jtype]];
	  double scale = tab_scale[itype][jtype];

  	  if (rsq < tb->innersq) {
	    fprintf(stdout,"\nTable %i   r = %f\n",tabindex[itype][jtype],sqrt(rsq));
  	    error->one(FLERR,"Pair distance < table inner cutoff");
	  }
 	  
  	  if (tabstyle == LOOKUP) {
  	    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
  	    if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
  	    forcelj += tb->f[itable] * scale / r2inv;
  	  } else if (tabstyle == LINEAR) {
  	    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
  	    if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
  	    fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
  	    value = tb->f[itable] + fraction*tb->df[itable];
  	    forcelj += value * scale / r2inv;
  	  } else if (tabstyle == SPLINE) {
  	    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
  	    if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
  	    b = (rsq - tb->rsq[itable]) * tb->invdelta;
  	    a = 1.0 - b;
  	    value = a * tb->f[itable] + b * tb->f[itable+1] + 
  	      ((a*a*a-a)*tb->f2[itable] + (b*b*b-b)*tb->f2[itable+1]) * 
  	      tb->deltasq6;
  	    forcelj += value * scale / r2inv;
  	  } else {
  	    rsq_lookup.f = rsq;
  	    itable = rsq_lookup.i & tb->nmask;
  	    itable >>= tb->nshiftbits;
  	    fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
  	    value = tb->f[itable] + fraction*tb->df[itable];
  	    forcelj += value * scale / r2inv;
  	  }

  	  if (eflag) {
	    double etable = 0.0;
  	    if (tabstyle == LOOKUP)
  	      etable = tb->e[itable];
  	    else if (tabstyle == LINEAR || tabstyle == BITMAP)
  	      etable = tb->e[itable] + fraction*tb->de[itable];
  	    else
  	      etable = a * tb->e[itable] + b * tb->e[itable+1] + 
  		((a*a*a-a)*tb->e2[itable] + (b*b*b-b)*tb->e2[itable+1]) * 
  		tb->deltasq6;
	    evdwl += etable * scale;
  	  }
	}
	
	fpair = (forcecoul + factor_lj*forcelj) * r2inv;

	f[i][0] += delx*fpair;
	f[i][1] += dely*fpair;
	f[i][2] += delz*fpair;
	if (newton_pair || j < nlocal) {
	  f[j][0] -= delx*fpair;
	  f[j][1] -= dely*fpair;
	  f[j][2] -= delz*fpair;
	}

	evdwl *= factor_lj;
         if (evflag) ev_tally(i,j,nlocal,newton_pair,
                              evdwl,ecoul,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairGulpCoulLong::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,  n+1,n+1,"pair:cutsq");

  // lj/cut
  memory->create(cut_ljcut,    n+1,n+1,"pair:cut_ljcut");
  memory->create(cutsq_ljcut,  n+1,n+1,"pair:cutsq_ljcut");
  memory->create(epsilon_ljcut,n+1,n+1,"pair:epsilon_ljcut");
  memory->create(sigma_ljcut,  n+1,n+1,"pair:sigma_ljcut");
  memory->create(setflag_ljcut,n+1,n+1,"pair:setflag_ljcut");
  memory->create(lj1_ljcut,    n+1,n+1,"pair:lj1_ljcut");
  memory->create(lj2_ljcut,    n+1,n+1,"pair:lj2_ljcut");
  memory->create(lj3_ljcut,    n+1,n+1,"pair:lj3_ljcut");
  memory->create(lj4_ljcut,    n+1,n+1,"pair:lj4_ljcut");
  memory->create(offset_ljcut, n+1,n+1,"pair:offset_ljcut");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++) {
      setflag_ljcut[i][j] = 0;
      cutsq_ljcut[i][j] = 0.0;
    }

  // lj/gulp
  memory->create(cut_ljgulp,         n+1,n+1,"pair:cut_ljgulp");
  memory->create(cutsq_ljgulp,       n+1,n+1,"pair:cutsq_ljgulp");
  memory->create(cut_inner_ljgulp,   n+1,n+1,"pair:cut_inner_ljgulp");
  memory->create(cutsq_inner_ljgulp, n+1,n+1,"pair:cutsq_inner_ljgulp");
  memory->create(epsilon_ljgulp,     n+1,n+1,"pair:epsilon_ljgulp");
  memory->create(sigma_ljgulp,       n+1,n+1,"pair:sigma_ljgulp");
  memory->create(setflag_ljgulp,     n+1,n+1,"pair:setflag_ljgulp");
  memory->create(lj1_ljgulp,         n+1,n+1,"pair:lj1_ljgulp");
  memory->create(lj2_ljgulp,         n+1,n+1,"pair:lj2_ljgulp");
  memory->create(lj3_ljgulp,         n+1,n+1,"pair:lj3_ljgulp");
  memory->create(lj4_ljgulp,         n+1,n+1,"pair:lj4_ljgulp");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++) {
      setflag_ljgulp[i][j] = 0;
      cutsq_ljgulp[i][j] = 0.0;
    }

  // lennard/gulp
  memory->create(cut_lengulp,         n+1,n+1,"pair:cut_lengulp");
  memory->create(cutsq_lengulp,       n+1,n+1,"pair:cutsq_lengulp");
  memory->create(cut_inner_lengulp,   n+1,n+1,"pair:cut_inner_lengulp");
  memory->create(cutsq_inner_lengulp, n+1,n+1,"pair:cutsq_inner_lengulp");
  memory->create(a_lengulp,           n+1,n+1,"pair:epsilon_lengulp");
  memory->create(b_lengulp,           n+1,n+1,"pair:sigma_lengulp");
  memory->create(setflag_lengulp,     n+1,n+1,"pair:setflag_lengulp");
  memory->create(lj1_lengulp,         n+1,n+1,"pair:lj1_lengulp");
  memory->create(lj2_lengulp,         n+1,n+1,"pair:lj2_lengulp");
  memory->create(lj3_lengulp,         n+1,n+1,"pair:lj3_lengulp");
  memory->create(lj4_lengulp,         n+1,n+1,"pair:lj4_lengulp");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++) {
      setflag_lengulp[i][j] = 0;
      cutsq_lengulp[i][j] = 0.0;
    }

  // lennard/gulp
  memory->create(cut_buckgulp,         n+1,n+1,"pair:cut_buckgulp");
  memory->create(cutsq_buckgulp,       n+1,n+1,"pair:cutsq_buckgulp");
  memory->create(cut_inner_buckgulp,   n+1,n+1,"pair:cut_inner_buckgulp");
  memory->create(cutsq_inner_buckgulp, n+1,n+1,"pair:cutsq_inner_buckgulp");
  memory->create(a_buckgulp,           n+1,n+1,"pair:a_buckgulp");
  memory->create(rho_buckgulp,         n+1,n+1,"pair:rho_buckgulp");
  memory->create(c_buckgulp,           n+1,n+1,"pair:c_buckgulp");
  memory->create(setflag_buckgulp,     n+1,n+1,"pair:setflag_buckgulp");
  memory->create(rhoinv_buckgulp,      n+1,n+1,"pair:rhoinv_buckgulp");
  memory->create(buck1_buckgulp,       n+1,n+1,"pair:buck1_buckgulp");
  memory->create(buck2_buckgulp,       n+1,n+1,"pair:buck2_buckgulp");
  memory->create(offset_buckgulp,      n+1,n+1,"pair:offset_buckgulp");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++) {
      setflag_buckgulp[i][j] = 0;
      cutsq_buckgulp[i][j] = 0.0;
    }

  // table
  memory->create(cut_table,     n+1,n+1,"pair:cut_table");
  memory->create(cutsq_table,   n+1,n+1,"pair:cutsq_table");
  memory->create(setflag_table, n+1,n+1,"pair:setflag_table");
  memory->create(tabindex,      n+1,n+1,"pair:tabindex");
  memory->create(tab_scale,     n+1,n+1,"pair:tab_scale");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++) {
      setflag_table[i][j] = 0;
      tabindex[i][j] = 0;
      cut_table[i][j] = 0.0;
      cutsq_table[i][j] = 0.0;
      tab_scale[i][j] = 1.0;
    }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairGulpCoulLong::settings(int narg, char **arg)
{ 
  cut_inner_global = force->numeric(FLERR,arg[0]);
  cut_global = force->numeric(FLERR,arg[1]);
  
  int indx = 0;
  do_tables = 0;
  if (narg == 2) cut_coul = cut_global;
  else if (narg == 3) cut_coul = force->numeric(FLERR,arg[2]);
  else if (narg == 4) {
    cut_coul = cut_global;
    do_tables = 1;
    indx = 2;
  } else if (narg == 5) {
    cut_coul = force->numeric(FLERR,arg[2]);
    do_tables = 1;
    indx = 3;
  } else error->all(FLERR,"Illegal pair_style command: incorrect number of arguments; expecting 2, 3, 4, or 5");
  
  if (cut_inner_global <= 0.0 || cut_inner_global > cut_global)
    error->all(FLERR,"Illegal pair_style command: cut_inner_global is negative or larger than cut_global");

  // Activate tabulated potentials
  if(do_tables) {
    if (strcmp(arg[indx],"lookup") == 0) tabstyle = LOOKUP;
    else if (strcmp(arg[indx],"linear") == 0) tabstyle = LINEAR; 
    else if (strcmp(arg[indx],"spline") == 0) tabstyle = SPLINE;
    else if (strcmp(arg[indx],"bitmap") == 0) tabstyle = BITMAP;
    else error->one(FLERR,"Unknown table style in pair_style command");
    
    if(tabstyle != LINEAR) error->all(FLERR,"Table style not yet supported.  Only linear.");
    
    tablength = force->inumeric(FLERR,arg[indx+1]);
    if(tablength < 1) error->all(FLERR,"tablength < 1");

    // delete old tables, since cannot just change settings
    
    for (int m = 0; m < ntables; m++) free_table(&tables[m]);
    memory->sfree(tables);
    
    ntables = 0;
    tables = NULL;
  }
  
  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++) 
      for (j = i+1; j <= atom->ntypes; j++) {
        if(setflag_ljcut[i][j]) cut_ljcut[i][j] = cut_global;

        if(setflag_ljgulp[i][j]) {
	  cut_inner_ljgulp[i][j] = cut_inner_global;
	  cut_ljgulp[i][j] = cut_global;
	}

	if(setflag_lengulp[i][j]) {
	  cut_inner_lengulp[i][j] = cut_inner_global;
	  cut_lengulp[i][j] = cut_global;
	}

	if(setflag_buckgulp[i][j]) {
	  cut_inner_buckgulp[i][j] = cut_inner_global;
	  cut_buckgulp[i][j] = cut_global;
	}

	if(setflag_table[i][j]) cut_table[i][j] = cut_global;
      }
  }

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairGulpCoulLong::coeff(int narg, char **arg)
{
  if (narg < 3 || narg > 8)
    error->all(FLERR,"Incorrect number of args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR, arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR, arg[1],atom->ntypes,jlo,jhi);

  int count = 0;
  if(strcmp(arg[2],"coul/long") == 0) {
    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo,i); j <= jhi; j++) {
	setflag[i][j] = 1;
	count++;
      }
    }

  } else if(strcmp(arg[2],"lj/cut") == 0) {

    double epsilon_one = force->numeric(FLERR,arg[3]);
    double sigma_one = force->numeric(FLERR,arg[4]);

    double cut_one = cut_global;
    if (narg == 6) cut_one = force->numeric(FLERR,arg[5]);

    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo,i); j <= jhi; j++) {
	epsilon_ljcut[i][j] = epsilon_one;
	sigma_ljcut[i][j]   = sigma_one;
	cut_ljcut[i][j]     = cut_one;
	setflag_ljcut[i][j] = 1;
	count++;
      }
    }

  } else if(strcmp(arg[2],"lj/gulp") == 0) {
    double epsilon_one = force->numeric(FLERR,arg[3]);
    double sigma_one = force->numeric(FLERR,arg[4]);
    
    double cut_inner_one = cut_inner_global;
    double cut_one = cut_global;
    if (narg == 7) {
      cut_inner_one = force->numeric(FLERR,arg[5]);
      cut_one = force->numeric(FLERR,arg[6]);

      if(cut_inner_one <= 0.0 || cut_inner_one > cut_one) error->all(FLERR,"Illegal pair_style command");
    }

    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo,i); j <= jhi; j++) {
	epsilon_ljgulp[i][j]   = epsilon_one;
	sigma_ljgulp[i][j]     = sigma_one;
	cut_inner_ljgulp[i][j] = cut_inner_one;
	cut_ljgulp[i][j]       = cut_one;
	setflag_ljgulp[i][j]   = 1;
	count++;
      }
    }

  } else if(strcmp(arg[2],"lennard/gulp") == 0) {
    double aparm_one = force->numeric(FLERR,arg[3]);
    double bparm_one = force->numeric(FLERR,arg[4]);
    
    double cut_inner_one = cut_inner_global;
    double cut_one = cut_global;
    if (narg == 7) {
      cut_inner_one = force->numeric(FLERR,arg[5]);
      cut_one = force->numeric(FLERR,arg[6]);

      if(cut_inner_one <= 0.0 || cut_inner_one > cut_one) error->all(FLERR,"Illegal pair_style command");
    }

    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo,i); j <= jhi; j++) {
	a_lengulp[i][j]         = aparm_one;
	b_lengulp[i][j]         = bparm_one;
	cut_inner_lengulp[i][j] = cut_inner_one;
	cut_lengulp[i][j]       = cut_one;
	setflag_lengulp[i][j]   = 1;
	count++;
      }
    }

  } else if(strcmp(arg[2],"buck/gulp") == 0) {
    double a_one = force->numeric(FLERR,arg[3]);
    double rho_one = force->numeric(FLERR,arg[4]);
    if(rho_one <= 0.0) error->all(FLERR,"Incorrect args for pair coefficients: rho_one <= 0.0");
    double c_one = force->numeric(FLERR,arg[5]);
    
    double cut_inner_one = cut_inner_global;
    double cut_one = cut_global;
    if (narg == 8) {
      cut_inner_one = force->numeric(FLERR,arg[6]);
      cut_one = force->numeric(FLERR,arg[7]);

      if(cut_inner_one <= 0.0 || cut_inner_one > cut_one) error->all(FLERR,"Illegal pair_style command");
    }

    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo,i); j <= jhi; j++) {
	a_buckgulp[i][j]         = a_one;
	rho_buckgulp[i][j]       = rho_one;
	c_buckgulp[i][j]         = c_one;
	cut_inner_buckgulp[i][j] = cut_inner_one;
	cut_buckgulp[i][j]       = cut_one;
	setflag_buckgulp[i][j]   = 1;
	count++;
      }
    }
  } else if(strcmp(arg[2],"table") == 0) {
    int me;
    MPI_Comm_rank(world,&me);

    tables = (Table *) memory->srealloc(tables,(ntables+1)*sizeof(Table),"pair:tables");
    Table *tb = &tables[ntables];
    null_table(tb);
    if (me == 0) read_table(tb,arg[3],arg[4]);
    bcast_table(tb);
    
    // set table cutoff
    
    if (narg == 6) tb->cut = force->numeric(FLERR,arg[5]);
    else if (tb->rflag) tb->cut = tb->rhi;
    else tb->cut = tb->rfile[tb->ninput-1];
    
    // error check on table parameters
    // insure cutoff is within table
    // for BITMAP tables, file values can be in non-ascending order
    
    if (tb->ninput <= 1) error->one(FLERR,"Invalid pair table length");
    double rlo,rhi;
    if (tb->rflag == 0) {
      rlo = tb->rfile[0];
      rhi = tb->rfile[tb->ninput-1];
    } else {
      rlo = tb->rlo;
      rhi = tb->rhi;
    }
    if (tb->cut <= rlo || tb->cut > rhi) error->one(FLERR,"Invalid pair table cutoff");
    if (rlo <= 0.0) error->one(FLERR,"Invalid pair table cutoff");
    
    // match = 1 if don't need to spline read-in tables
    // this is only the case if r values needed by final tables
    //   exactly match r values read from file
    
    tb->match = 0;
    if (tabstyle == LINEAR && tb->ninput == tablength && 
     	tb->rflag == RSQ && tb->rhi == tb->cut) tb->match = 1;
    if (tabstyle == SPLINE && tb->ninput == tablength && 
     	tb->rflag == RSQ && tb->rhi == tb->cut) tb->match = 1;
    if (tabstyle == BITMAP && tb->ninput == 1 << tablength && 
     	tb->rflag == BMP && tb->rhi == tb->cut) tb->match = 1;
    
    if (tb->rflag == BMP && tb->match == 0)
      error->one(FLERR,"Bitmapped table in file does not match requested table");
    
    // spline read-in values and compute r,e,f vectors within table
    
    if (tb->match == 0) spline_table(tb);  //breaks RAPTOR (parent_id array)
    compute_table(tb);
    
    // store ptr to table in tabindex
    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo,i); j <= jhi; j++) {
	setflag_table[i][j] = 1;
	cut_table[i][j] = tb->cut;
      	tabindex[i][j] = ntables;
      	count++;
      }
    }

    ntables++;
    
  } else error->all(FLERR,"Illegal pair_coeff command: unsupported style");

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients: count == 0");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairGulpCoulLong::init_style()
{
  if (!atom->q_flag) error->all(FLERR,"Pair style coul/long requires atom attribute q");

  // request regular or rRESPA neighbor lists

  int irequest;

  if (update->whichflag == 1 && strstr(update->integrate_style,"respa")) {
    error->all(FLERR,"respa not yet supported with pair_style");

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

  if (force->kspace == NULL) error->all(FLERR,"Pair style requires a KSpace style");
  g_ewald = force->kspace->g_ewald;

  // setup force tables

  if (ncoultablebits) init_tables(cut_coul,cut_respa);
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairGulpCoulLong::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listinner = ptr;
  else if (id == 2) listmiddle = ptr;
  else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGulpCoulLong::init_one(int i, int j)
{
  double cut = cut_coul;

  if(setflag_ljcut[i][j]) {
    cut = MAX(cut, cut_ljcut[i][j]);
    cutsq_ljcut[i][j] = cut_ljcut[i][j] * cut_ljcut[i][j];
    
    lj1_ljcut[i][j] = 48.0 * epsilon_ljcut[i][j] * pow(sigma_ljcut[i][j],12.0);
    lj2_ljcut[i][j] = 24.0 * epsilon_ljcut[i][j] * pow(sigma_ljcut[i][j], 6.0);
    lj3_ljcut[i][j] =  4.0 * epsilon_ljcut[i][j] * pow(sigma_ljcut[i][j],12.0);
    lj4_ljcut[i][j] =  4.0 * epsilon_ljcut[i][j] * pow(sigma_ljcut[i][j], 6.0);
    
    if (offset_flag) {
      double ratio = sigma_ljcut[i][j] / cut_ljcut[i][j];
      offset_ljcut[i][j] = 4.0 * epsilon_ljcut[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
    } else offset_ljcut[i][j] = 0.0;
    
    cutsq_ljcut[j][i]  = cutsq_ljcut[i][j];
    lj1_ljcut[j][i]    = lj1_ljcut[i][j];
    lj2_ljcut[j][i]    = lj2_ljcut[i][j];
    lj3_ljcut[j][i]    = lj3_ljcut[i][j];
    lj4_ljcut[j][i]    = lj4_ljcut[i][j];
    offset_ljcut[j][i] = offset_ljcut[i][j];
    setflag_ljcut[j][i] = setflag_ljcut[i][j];
    
    // check interior rRESPA cutoff
    
    if (cut_respa && MIN(cut_ljcut[i][j], cut_coul) < cut_respa[3])
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
      
      double sig2 = sigma_ljcut[i][j]*sigma_ljcut[i][j];
      double sig6 = sig2*sig2*sig2;
      double rc3 = cut_ljcut[i][j]*cut_ljcut[i][j]*cut_ljcut[i][j];
      double rc6 = rc3*rc3;
      double rc9 = rc3*rc6;
      etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon_ljcut[i][j] *
	sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
      ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon_ljcut[i][j] *
	sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
    }
  }
  
  if(setflag_ljgulp[i][j]) {
    cut = MAX(cut, cut_ljgulp[i][j]);

    cutsq_ljgulp[i][j] = cut_ljgulp[i][j]*cut_ljgulp[i][j];
    cutsq_inner_ljgulp[i][j] = cut_inner_ljgulp[i][j]*cut_inner_ljgulp[i][j];
    lj1_ljgulp[i][j] = 48.0 * epsilon_ljgulp[i][j] * pow(sigma_ljgulp[i][j],12.0);
    lj2_ljgulp[i][j] = 24.0 * epsilon_ljgulp[i][j] * pow(sigma_ljgulp[i][j], 6.0);
    lj3_ljgulp[i][j] =  4.0 * epsilon_ljgulp[i][j] * pow(sigma_ljgulp[i][j],12.0);
    lj4_ljgulp[i][j] =  4.0 * epsilon_ljgulp[i][j] * pow(sigma_ljgulp[i][j], 6.0);
    
    cut_ljgulp[j][i] = cut_ljgulp[i][j];
    cut_inner_ljgulp[j][i] = cut_inner_ljgulp[i][j];
    cutsq_ljgulp[j][i] = cutsq_ljgulp[i][j];
    cutsq_inner_ljgulp[j][i] = cutsq_inner_ljgulp[i][j];
    lj1_ljgulp[j][i] = lj1_ljgulp[i][j];
    lj2_ljgulp[j][i] = lj2_ljgulp[i][j];
    lj3_ljgulp[j][i] = lj3_ljgulp[i][j];
    lj4_ljgulp[j][i] = lj4_ljgulp[i][j];
    setflag_ljgulp[j][i] = setflag_ljgulp[i][j];
  }

  if(setflag_lengulp[i][j]) {
    cut = MAX(cut, cut_lengulp[i][j]);
    
    cutsq_lengulp[i][j] = cut_lengulp[i][j]*cut_lengulp[i][j];
    cutsq_inner_lengulp[i][j] = cut_inner_lengulp[i][j]*cut_inner_lengulp[i][j];
    
    lj1_lengulp[i][j] = 12.0 * a_lengulp[i][j];
    lj2_lengulp[i][j] =  6.0 * b_lengulp[i][j];
    lj3_lengulp[i][j] = a_lengulp[i][j];
    lj4_lengulp[i][j] = b_lengulp[i][j];
    
    cut_lengulp[j][i] = cut_lengulp[i][j];
    cut_inner_lengulp[j][i]   = cut_inner_lengulp[i][j];
    cutsq_lengulp[j][i] = cutsq_lengulp[i][j];
    cutsq_inner_lengulp[j][i] = cutsq_inner_lengulp[i][j];
    lj1_lengulp[j][i] = lj1_lengulp[i][j];
    lj2_lengulp[j][i] = lj2_lengulp[i][j];
    lj3_lengulp[j][i] = lj3_lengulp[i][j];
    lj4_lengulp[j][i] = lj4_lengulp[i][j];
    setflag_lengulp[j][i] = setflag_lengulp[i][j];
  }

  if(setflag_buckgulp[i][j]) {
    cut = MAX(cut, cut_buckgulp[i][j]);

    rhoinv_buckgulp[i][j] = 1.0 / rho_buckgulp[i][j];
    buck1_buckgulp[i][j]  = a_buckgulp[i][j] / rho_buckgulp[i][j];
    buck2_buckgulp[i][j]  = 6.0 * c_buckgulp[i][j];
    
    if (offset_flag) {
      double rexp = exp(-cut_buckgulp[i][j] / rho_buckgulp[i][j]);
      offset_buckgulp[i][j] = a_buckgulp[i][j]*rexp - c_buckgulp[i][j] / pow(cut_buckgulp[i][j],6.0);
    } else offset_buckgulp[i][j] = 0.0;
    
    cutsq_buckgulp[i][j] = cut_buckgulp[i][j]*cut_buckgulp[i][j];
    cutsq_inner_buckgulp[i][j] = cut_inner_buckgulp[i][j]*cut_inner_buckgulp[i][j];

    cut_buckgulp[j][i] = cut_buckgulp[i][j];
    cutsq_buckgulp[j][i] = cutsq_buckgulp[i][j];
    cut_inner_buckgulp[j][i] = cut_inner_buckgulp[i][j];
    cutsq_inner_buckgulp[j][i] = cutsq_inner_buckgulp[i][j];    
    a_buckgulp[j][i]      = a_buckgulp[i][j];
    c_buckgulp[j][i]      = c_buckgulp[i][j];
    rhoinv_buckgulp[j][i] = rhoinv_buckgulp[i][j];
    buck1_buckgulp[j][i]  = buck1_buckgulp[i][j];
    buck2_buckgulp[j][i]  = buck2_buckgulp[i][j];
    offset_buckgulp[j][i] = offset_buckgulp[i][j];
    setflag_buckgulp[j][i] = setflag_buckgulp[i][j];
    
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
      
      double rho1 = rho_buckgulp[i][j];
      double rho2 = rho1*rho1;
      double rho3 = rho2*rho1;
      double rc = cut_buckgulp[i][j];
      double rc2 = rc*rc;
      double rc3 = rc2*rc;
      etail_ij = 2.0*MY_PI*all[0]*all[1]*
	(a_buckgulp[i][j]*exp(-rc/rho1)*rho1*(rc2 + 2.0*rho1*rc + 2.0*rho2) -
	 c_buckgulp[i][j]/(3.0*rc3));
      ptail_ij = (-1/3.0)*2.0*MY_PI*all[0]*all[1]*
	(-a_buckgulp[i][j]*exp(-rc/rho1)*
	 (rc3 + 3.0*rho1*rc2 + 6.0*rho2*rc + 6.0*rho3) + 2.0*c_buckgulp[i][j]/rc3);
    }
  }

  if(setflag_table[i][j]) {
    cut = MAX(cut, cut_table[i][j]);
    cutsq_table[i][j] = cut_table[i][j] * cut_table[i][j];
    
    setflag_table[j][i] = setflag_table[i][j];
    cut_table[j][i] = cut_table[i][j];
    cutsq_table[j][i] = cutsq_table[i][j];
    tabindex[j][i] = tabindex[i][j];
    tab_scale[j][i] = tab_scale[i][j];
  }

  return cut;
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGulpCoulLong::write_restart(FILE *fp)
{

}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGulpCoulLong::read_restart(FILE *fp)
{

}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairGulpCoulLong::write_restart_settings(FILE *fp)
{

}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairGulpCoulLong::read_restart_settings(FILE *fp)
{

}

/* ---------------------------------------------------------------------- */

double PairGulpCoulLong::single(int i, int j, int itype, int jtype,
                                 double rsq,
                                 double factor_coul, double factor_lj,
                                 double &fforce)
{  
  int inum,jnum,itable;
  double evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcelj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;

  Table *tb;
  double value, a, b;
  union_int_float_t rsq_lookup;
  int tlm1 = tablength - 1;

  fforce = 0.0;
  evdwl = 0.0;
  ecoul = 0.0;
  if(rsq < cutsq[itype][jtype]) {
    r2inv = 1.0/rsq;

    if (rsq < cut_coulsq) {
      if (!ncoultablebits || rsq <= tabinnersq) {
	r = sqrt(rsq);
	grij = g_ewald * r;
	expm2 = exp(-grij*grij);
	t = 1.0 / (1.0 + EWALD_P*grij);
	erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
	prefactor = force->qqrd2e * atom->q[i] * atom->q[j] / r;
	forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
	if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
      } else {
	union_int_float_t rsq_lookup;
	rsq_lookup.f = rsq;
	itable = rsq_lookup.i & ncoulmask;
	itable >>= ncoulshiftbits;
	fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
	table = ftable[itable] + fraction*dftable[itable];
	forcecoul = atom->q[i] * atom->q[j] * table;
	if (factor_coul < 1.0) {
	  table = ctable[itable] + fraction*dctable[itable];
	  prefactor = atom->q[i] * atom->q[j] * table;
	  forcecoul -= (1.0-factor_coul)*prefactor;
	}
      }
    } else forcecoul = 0.0;

    if (rsq < cut_coulsq) {
      if (!ncoultablebits || rsq <= tabinnersq)
	ecoul = prefactor*erfc;
      else {
	table = etable[itable] + fraction*detable[itable];
	ecoul = atom->q[i]*atom->q[j] * table;
      }
      if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
    } else ecoul = 0.0;
    
    double fvdwl;
    forcelj = 0.0;
    evdwl = 0.0;

    // lj/cut
    if(setflag_ljcut[itype][jtype] && rsq < cutsq_ljcut[itype][jtype]) {
      r6inv = r2inv*r2inv*r2inv;
      forcelj += r6inv * (lj1_ljcut[itype][jtype]*r6inv - lj2_ljcut[itype][jtype]);

      double etmp = r6inv*(lj3_ljcut[itype][jtype]*r6inv-lj4_ljcut[itype][jtype]) - offset_ljcut[itype][jtype];
      evdwl += etmp;
    }
	
    // lj/gulp
    if(setflag_ljgulp[itype][jtype] && rsq < cutsq_ljgulp[itype][jtype]) {
      double philj, rr, dp, d, dd, tt, dt;
      
      r6inv = r2inv*r2inv*r2inv;
      fvdwl = r6inv * (lj1_ljgulp[itype][jtype]*r6inv - lj2_ljgulp[itype][jtype]);
      
      if (rsq > cutsq_inner_ljgulp[itype][jtype]) {
	philj = r6inv*(lj3_ljgulp[itype][jtype]*r6inv-lj4_ljgulp[itype][jtype]);
	
	rr = sqrt(rsq);
	dp = (cut_ljgulp[itype][jtype] - cut_inner_ljgulp[itype][jtype]);
	d = (rr-cut_inner_ljgulp[itype][jtype]) / dp;
	dd = 1.-d;
	// taperig function - gulp style
	tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;
	// minus the derivative of the tapering function
	dt = 30.* d*d * dd*dd * rr / dp;
	
	fvdwl = fvdwl*tt + philj*dt;
      } else tt = 1;
      
      forcelj += fvdwl;
      
      double etmp = r6inv * (lj3_ljgulp[itype][jtype]*r6inv - lj4_ljgulp[itype][jtype]);
      evdwl += etmp * tt;
    }
    
    // lennard/gulp
    if(setflag_lengulp[itype][jtype] && rsq < cutsq_lengulp[itype][jtype]) {
      double philj, rr, dp, d, dd, tt, dt;
      
      r6inv = r2inv*r2inv*r2inv;
      fvdwl = r6inv * (lj1_lengulp[itype][jtype]*r6inv - lj2_lengulp[itype][jtype]);
      
      if (rsq > cutsq_inner_lengulp[itype][jtype]) {
	philj = r6inv*(lj3_lengulp[itype][jtype]*r6inv-lj4_lengulp[itype][jtype]);  
	
	rr = sqrt(rsq);
	dp = (cut_lengulp[itype][jtype] - cut_inner_lengulp[itype][jtype]);
	d = (rr-cut_inner_lengulp[itype][jtype]) / dp;
	dd = 1.-d;
	// taperig function - gulp style
	tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;
	// minus the derivative of the tapering function
	dt = 30.* d*d * dd*dd * rr / dp;
	
	fvdwl = fvdwl*tt + philj*dt;
      } else tt = 1;
      forcelj += fvdwl;
      
      double etmp = r6inv * (lj3_lengulp[itype][jtype]*r6inv - lj4_lengulp[itype][jtype]);
      evdwl += etmp * tt;
    }

    // buck/gulp
    if(setflag_buckgulp[itype][jtype] && rsq < cutsq_buckgulp[itype][jtype]) {
      double rexp, phibuck, dp, d, dd, tt, dt;
      
      r6inv = r2inv*r2inv*r2inv;
      r = sqrt(rsq);
      rexp = exp(-r*rhoinv_buckgulp[itype][jtype]);
      fvdwl = buck1_buckgulp[itype][jtype]*r*rexp - buck2_buckgulp[itype][jtype]*r6inv;
      if (rsq > cutsq_inner_buckgulp[itype][jtype]) {
	phibuck = a_buckgulp[itype][jtype]*rexp - c_buckgulp[itype][jtype]*r6inv;
	dp = (cut_buckgulp[itype][jtype] - cut_inner_buckgulp[itype][jtype]);
	d = (r-cut_inner_buckgulp[itype][jtype]) / dp;
	dd = 1.-d;
	// taperig function - gulp style
	tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;
	// minus the derivative of the tapering function
	dt = 30.* d*d * dd*dd * r / dp;
	
	fvdwl = fvdwl*tt + phibuck*dt;
      } else tt = 1;
      forcelj += fvdwl;
      
      double etmp = a_buckgulp[itype][jtype]*rexp - c_buckgulp[itype][jtype]*r6inv; 
      evdwl += etmp * tt;
    }
    
    // table
    if(setflag_table[itype][jtype] && rsq < cutsq_table[itype][jtype]) {
      tb = &tables[tabindex[itype][jtype]];
      double scale = tab_scale[itype][jtype];
      
      if (rsq < tb->innersq) {
	fprintf(stdout,"\nTable %i   r = %f\n",tabindex[itype][jtype],sqrt(rsq));
	error->one(FLERR,"Pair distance < table inner cutoff");
      }
      
      if (tabstyle == LOOKUP) {
	itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
	if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
	forcelj += tb->f[itable] * scale / r2inv;
      } else if (tabstyle == LINEAR) {
	itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
	if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
	fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
	value = tb->f[itable] + fraction*tb->df[itable];
	forcelj += value * scale / r2inv;
      } else if (tabstyle == SPLINE) {
	itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
	if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
	b = (rsq - tb->rsq[itable]) * tb->invdelta;
	a = 1.0 - b;
	value = a * tb->f[itable] + b * tb->f[itable+1] + 
	  ((a*a*a-a)*tb->f2[itable] + (b*b*b-b)*tb->f2[itable+1]) * 
	  tb->deltasq6;
	forcelj += value * scale / r2inv;
      } else {
	rsq_lookup.f = rsq;
	itable = rsq_lookup.i & tb->nmask;
	itable >>= tb->nshiftbits;
	fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
	value = tb->f[itable] + fraction*tb->df[itable];
	forcelj += value * scale / r2inv;
      }
      
      double etable = 0.0;
      if (tabstyle == LOOKUP)
	etable = tb->e[itable];
      else if (tabstyle == LINEAR || tabstyle == BITMAP)
	etable = tb->e[itable] + fraction*tb->de[itable];
      else
	etable = a * tb->e[itable] + b * tb->e[itable+1] + 
	  ((a*a*a-a)*tb->e2[itable] + (b*b*b-b)*tb->e2[itable+1]) * 
	  tb->deltasq6;
      evdwl += etable * scale;
    }

    fforce = (forcecoul + factor_lj*forcelj) * r2inv;
  }
  
  double eng = ecoul + evdwl * factor_lj;
  return eng;
}

double PairGulpCoulLong::single_ener_noljcoul(int i, int j, int itype, int jtype,
                                 double rsq,
                                 double factor_lj)
{  
  int inum,jnum,itable;
  double evdwl,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcelj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;

  Table *tb;
  double value, a, b;
  union_int_float_t rsq_lookup;
  int tlm1 = tablength - 1;

  evdwl = 0.0;
  if(rsq < cutsq[itype][jtype]) {
    r2inv = 1.0/rsq;
	
    // lj/gulp
    if(setflag_ljgulp[itype][jtype] && rsq < cutsq_ljgulp[itype][jtype]) {
      double philj, rr, dp, d, dd, tt, dt;
      
      r6inv = r2inv*r2inv*r2inv;
      
      if (rsq > cutsq_inner_ljgulp[itype][jtype]) {
	philj = r6inv*(lj3_ljgulp[itype][jtype]*r6inv-lj4_ljgulp[itype][jtype]);
	
	rr = sqrt(rsq);
	dp = (cut_ljgulp[itype][jtype] - cut_inner_ljgulp[itype][jtype]);
	d = (rr-cut_inner_ljgulp[itype][jtype]) / dp;
	dd = 1.-d;
	// tapering function - gulp style
	tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;
      } else tt = 1;
      
      double etmp = r6inv * (lj3_ljgulp[itype][jtype]*r6inv - lj4_ljgulp[itype][jtype]);
      evdwl += etmp * tt;
    }
    
    // lennard/gulp
    if(setflag_lengulp[itype][jtype] && rsq < cutsq_lengulp[itype][jtype]) {
      double philj, rr, dp, d, dd, tt, dt;
      
      r6inv = r2inv*r2inv*r2inv;
      
      if (rsq > cutsq_inner_lengulp[itype][jtype]) {
	philj = r6inv*(lj3_lengulp[itype][jtype]*r6inv-lj4_lengulp[itype][jtype]);  
	
	rr = sqrt(rsq);
	dp = (cut_lengulp[itype][jtype] - cut_inner_lengulp[itype][jtype]);
	d = (rr-cut_inner_lengulp[itype][jtype]) / dp;
	dd = 1.-d;
	// tapering function - gulp style
	tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;

      } else tt = 1;
      
      double etmp = r6inv * (lj3_lengulp[itype][jtype]*r6inv - lj4_lengulp[itype][jtype]);
      evdwl += etmp * tt;
    }

    // buck/gulp
    if(setflag_buckgulp[itype][jtype] && rsq < cutsq_buckgulp[itype][jtype]) {
      double rexp, phibuck, dp, d, dd, tt, dt;
      
      r6inv = r2inv*r2inv*r2inv;
      r = sqrt(rsq);
      rexp = exp(-r*rhoinv_buckgulp[itype][jtype]);
      if (rsq > cutsq_inner_buckgulp[itype][jtype]) {
	phibuck = a_buckgulp[itype][jtype]*rexp - c_buckgulp[itype][jtype]*r6inv;
	dp = (cut_buckgulp[itype][jtype] - cut_inner_buckgulp[itype][jtype]);
	d = (r-cut_inner_buckgulp[itype][jtype]) / dp;
	dd = 1.-d;
	// tapering function - gulp style
	tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;

      } else tt = 1;
      
      double etmp = a_buckgulp[itype][jtype]*rexp - c_buckgulp[itype][jtype]*r6inv; 
      evdwl += etmp * tt;
    }

    // table
    if(setflag_table[itype][jtype] && rsq < cutsq_table[itype][jtype]) {
      tb = &tables[tabindex[itype][jtype]];
      double scale = tab_scale[itype][jtype];
      
      if (rsq < tb->innersq) {
	fprintf(stdout,"\nTable %i   r = %f\n",tabindex[itype][jtype],sqrt(rsq));
	error->one(FLERR,"Pair distance < table inner cutoff");
      }
      
      if (tabstyle == LOOKUP) {
	itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
	if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
      } else if (tabstyle == LINEAR) {
	itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
	if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
	fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
      } else if (tabstyle == SPLINE) {
	itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
	if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
	b = (rsq - tb->rsq[itable]) * tb->invdelta;
	a = 1.0 - b;
      } else {
	rsq_lookup.f = rsq;
	itable = rsq_lookup.i & tb->nmask;
	itable >>= tb->nshiftbits;
	fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
      }
      
      double etable = 0.0;
      if (tabstyle == LOOKUP)
	etable = tb->e[itable];
      else if (tabstyle == LINEAR || tabstyle == BITMAP)
	etable = tb->e[itable] + fraction*tb->de[itable];
      else
	etable = a * tb->e[itable] + b * tb->e[itable+1] + 
	  ((a*a*a-a)*tb->e2[itable] + (b*b*b-b)*tb->e2[itable+1]) * 
	  tb->deltasq6;
      evdwl += etable * scale;
    }
  }

  double eng = evdwl * factor_lj;
  return eng;
}

/* ---------------------------------------------------------------------- */

double PairGulpCoulLong::single_fpair_noljcoul(int i, int j, int itype, int jtype,
                                 double rsq,
                                 double factor_lj)
{  
  int inum,jnum,itable;
  double evdwl,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcelj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;

  Table *tb;
  double value, a, b;
  union_int_float_t rsq_lookup;
  int tlm1 = tablength - 1;

  double fforce = 0.0;
  if(rsq < cutsq[itype][jtype]) {
    r2inv = 1.0/rsq;
    
    double fvdwl;
    forcelj = 0.0;
	
    // lj/gulp
    if(setflag_ljgulp[itype][jtype] && rsq < cutsq_ljgulp[itype][jtype]) {
      double philj, rr, dp, d, dd, tt, dt;
      
      r6inv = r2inv*r2inv*r2inv;
      fvdwl = r6inv * (lj1_ljgulp[itype][jtype]*r6inv - lj2_ljgulp[itype][jtype]);
      
      if (rsq > cutsq_inner_ljgulp[itype][jtype]) {
	philj = r6inv*(lj3_ljgulp[itype][jtype]*r6inv-lj4_ljgulp[itype][jtype]);
	
	rr = sqrt(rsq);
	dp = (cut_ljgulp[itype][jtype] - cut_inner_ljgulp[itype][jtype]);
	d = (rr-cut_inner_ljgulp[itype][jtype]) / dp;
	dd = 1.-d;
	// taperig function - gulp style
	tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;
	// minus the derivative of the tapering function
	dt = 30.* d*d * dd*dd * rr / dp;
	
	fvdwl = fvdwl*tt + philj*dt;
      } else tt = 1;
      
      forcelj += fvdwl;
    }
    
    // lennard/gulp
    if(setflag_lengulp[itype][jtype] && rsq < cutsq_lengulp[itype][jtype]) {
      double philj, rr, dp, d, dd, tt, dt;
      
      r6inv = r2inv*r2inv*r2inv;
      fvdwl = r6inv * (lj1_lengulp[itype][jtype]*r6inv - lj2_lengulp[itype][jtype]);
      
      if (rsq > cutsq_inner_lengulp[itype][jtype]) {
	philj = r6inv*(lj3_lengulp[itype][jtype]*r6inv-lj4_lengulp[itype][jtype]);  
	
	rr = sqrt(rsq);
	dp = (cut_lengulp[itype][jtype] - cut_inner_lengulp[itype][jtype]);
	d = (rr-cut_inner_lengulp[itype][jtype]) / dp;
	dd = 1.-d;
	// taperig function - gulp style
	tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;
	// minus the derivative of the tapering function
	dt = 30.* d*d * dd*dd * rr / dp;
	
	fvdwl = fvdwl*tt + philj*dt;
      } else tt = 1;
      forcelj += fvdwl;
    }

    // buck/gulp
    if(setflag_buckgulp[itype][jtype] && rsq < cutsq_buckgulp[itype][jtype]) {
      double rexp, phibuck, dp, d, dd, tt, dt;
      
      r6inv = r2inv*r2inv*r2inv;
      r = sqrt(rsq);
      rexp = exp(-r*rhoinv_buckgulp[itype][jtype]);
      fvdwl = buck1_buckgulp[itype][jtype]*r*rexp - buck2_buckgulp[itype][jtype]*r6inv;
      if (rsq > cutsq_inner_buckgulp[itype][jtype]) {
	phibuck = a_buckgulp[itype][jtype]*rexp - c_buckgulp[itype][jtype]*r6inv;
	dp = (cut_buckgulp[itype][jtype] - cut_inner_buckgulp[itype][jtype]);
	d = (r-cut_inner_buckgulp[itype][jtype]) / dp;
	dd = 1.-d;
	// taperig function - gulp style
	tt = (1. + 3.*d + 6.*d*d)*dd*dd*dd;
	// minus the derivative of the tapering function
	dt = 30.* d*d * dd*dd * r / dp;
	
	fvdwl = fvdwl*tt + phibuck*dt;
      } else tt = 1;
      forcelj += fvdwl;
    }
    
    // table
    if(setflag_table[itype][jtype] && rsq < cutsq_table[itype][jtype]) {
      tb = &tables[tabindex[itype][jtype]];
      double scale = tab_scale[itype][jtype];
      
      if (rsq < tb->innersq) {
	fprintf(stdout,"\nTable %i   r = %f\n",tabindex[itype][jtype],sqrt(rsq));
	error->one(FLERR,"Pair distance < table inner cutoff");
      }
      
      if (tabstyle == LOOKUP) {
	itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
	if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
	forcelj += tb->f[itable] * scale / r2inv;
      } else if (tabstyle == LINEAR) {
	itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
	if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
	fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
	value = tb->f[itable] + fraction*tb->df[itable];
	forcelj += value * scale / r2inv;
      } else if (tabstyle == SPLINE) {
	itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
	if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
	b = (rsq - tb->rsq[itable]) * tb->invdelta;
	a = 1.0 - b;
	value = a * tb->f[itable] + b * tb->f[itable+1] + 
	  ((a*a*a-a)*tb->f2[itable] + (b*b*b-b)*tb->f2[itable+1]) * 
	  tb->deltasq6;
	forcelj += value * scale / r2inv;
      } else {
	rsq_lookup.f = rsq;
	itable = rsq_lookup.i & tb->nmask;
	itable >>= tb->nshiftbits;
	fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
	value = tb->f[itable] + fraction*tb->df[itable];
	forcelj += value * scale / r2inv;
      }
    }

    fforce = factor_lj*forcelj * r2inv;
  }

  return fforce;
}

/* ---------------------------------------------------------------------- */

void *PairGulpCoulLong::extract(const char *str, int &dim)
{
  // coul/long: for kspace to grab real-space cutoff
  dim = 0;
  if(strcmp(str,"cut_coul") == 0) return (void *) &cut_coul;

  // lj/cut
  dim = 2;
  if (strcmp(str,"epsilon_ljcut") == 0) return (void *) epsilon_ljcut;

  // lj/gulp
  dim = 2;
  if (strcmp(str,"epsilon_ljgulp") == 0) return (void *) epsilon_ljgulp;

  // lennard/gulp
  dim = 2;
  if (strcmp(str,"a_lengulp") == 0) return (void *) a_lengulp;
  if (strcmp(str,"b_lengulp") == 0) return (void *) b_lengulp;

  // buck/gulp
  dim = 2;
  if (strcmp(str,"a_buckgulp") == 0) return (void *) a_buckgulp;
  if (strcmp(str,"c_buckgulp") == 0) return (void *) c_buckgulp;

  // table
  dim = 2;
  if (strcmp(str,"tab_scale") == 0) return (void *) tab_scale;

  if(screen) {
    fprintf(screen,"\n\nSupported extract() arguments for pair_style gulp/coul/long\n");
    fprintf(screen,"cut_coul\n");
    fprintf(screen,"epsilon_ljcut\n");
    fprintf(screen,"epsilon_ljgulp\n");
    fprintf(screen,"a_lengulp, b_lengulp\n");
    fprintf(screen,"a_buckgulp, c_buckgulp\n");
    fprintf(screen,"tab_scale");
  }

  return NULL;
}


/* ----------------------------------------------------------------------
   All table functions below copied from pair_table.cpp
------------------------------------------------------------------------- */


/* ----------------------------------------------------------------------
   read a table section from a tabulated potential file
   only called by proc 0
   this function sets these values in Table:
     ninput,rfile,efile,ffile,rflag,rlo,rhi,fpflag,fplo,fphi,ntablebits
------------------------------------------------------------------------- */

void PairGulpCoulLong::read_table(Table *tb, char *file, char *keyword)
{
  char line[MAXLINE];

  // open file

  FILE *fp = fopen(file,"r");
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open file %s",file);
    error->one(FLERR,str);
  }

  // loop until section found with matching keyword

  while (1) {
    if (fgets(line,MAXLINE,fp) == NULL)
      error->one(FLERR,"Did not find keyword in table file");
    if (strspn(line," \t\n\r") == strlen(line)) continue;  // blank line
    if (line[0] == '#') continue;                          // comment
    char *word = strtok(line," \t\n\r");
    if (strcmp(word,keyword) == 0) break;           // matching keyword
    fgets(line,MAXLINE,fp);                         // no match, skip section
    param_extract(tb,line);
    fgets(line,MAXLINE,fp);
    for (int i = 0; i < tb->ninput; i++) fgets(line,MAXLINE,fp);
  }

  // read args on 2nd line of section
  // allocate table arrays for file values

  fgets(line,MAXLINE,fp);
  param_extract(tb,line);
  memory->create(tb->rfile,tb->ninput,"pair:rfile");
  memory->create(tb->efile,tb->ninput,"pair:efile");
  memory->create(tb->ffile,tb->ninput,"pair:ffile");

  // setup bitmap parameters for table to read in

  tb->ntablebits = 0;
  int masklo,maskhi,nmask,nshiftbits;
  if (tb->rflag == BMP) {
    while (1 << tb->ntablebits < tb->ninput) tb->ntablebits++;
    if (1 << tb->ntablebits != tb->ninput)
      error->one(FLERR,"Bitmapped table is incorrect length in table file");
    init_bitmap(tb->rlo,tb->rhi,tb->ntablebits,masklo,maskhi,nmask,nshiftbits);
  }

  // read r,e,f table values from file
  // if rflag set, compute r
  // if rflag not set, use r from file

  int itmp;
  double rtmp;
  union_int_float_t rsq_lookup;

  fgets(line,MAXLINE,fp);
  for (int i = 0; i < tb->ninput; i++) {
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d %lg %lg %lg",&itmp,&rtmp,&tb->efile[i],&tb->ffile[i]);

    if (tb->rflag == RLINEAR)
      rtmp = tb->rlo + (tb->rhi - tb->rlo)*i/(tb->ninput-1);
    else if (tb->rflag == RSQ) {
      rtmp = tb->rlo*tb->rlo +
        (tb->rhi*tb->rhi - tb->rlo*tb->rlo)*i/(tb->ninput-1);
      rtmp = sqrt(rtmp);
    } else if (tb->rflag == BMP) {
      rsq_lookup.i = i << nshiftbits;
      rsq_lookup.i |= masklo;
      if (rsq_lookup.f < tb->rlo*tb->rlo) {
        rsq_lookup.i = i << nshiftbits;
        rsq_lookup.i |= maskhi;
      }
      rtmp = sqrtf(rsq_lookup.f);
    }

    tb->rfile[i] = rtmp;
  }

  // close file

  fclose(fp);
}

/* ----------------------------------------------------------------------
   broadcast read-in table info from proc 0 to other procs
   this function communicates these values in Table:
     ninput,rfile,efile,ffile,rflag,rlo,rhi,fpflag,fplo,fphi
------------------------------------------------------------------------- */

void PairGulpCoulLong::bcast_table(Table *tb)
{
  MPI_Bcast(&tb->ninput,1,MPI_INT,0,world);

  int me;
  MPI_Comm_rank(world,&me);
  if (me > 0) {
    memory->create(tb->rfile,tb->ninput,"pair:rfile");
    memory->create(tb->efile,tb->ninput,"pair:efile");
    memory->create(tb->ffile,tb->ninput,"pair:ffile");
  }

  MPI_Bcast(tb->rfile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->efile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->ffile,tb->ninput,MPI_DOUBLE,0,world);

  MPI_Bcast(&tb->rflag,1,MPI_INT,0,world);
  if (tb->rflag) {
    MPI_Bcast(&tb->rlo,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&tb->rhi,1,MPI_DOUBLE,0,world);
  }
  MPI_Bcast(&tb->fpflag,1,MPI_INT,0,world);
  if (tb->fpflag) {
    MPI_Bcast(&tb->fplo,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&tb->fphi,1,MPI_DOUBLE,0,world);
  }
}

/* ----------------------------------------------------------------------
   build spline representation of e,f over entire range of read-in table
   this function sets these values in Table: e2file,f2file
------------------------------------------------------------------------- */

void PairGulpCoulLong::spline_table(Table *tb)
{
  memory->create(tb->e2file,tb->ninput,"pair:e2file");
  memory->create(tb->f2file,tb->ninput,"pair:f2file");

  double ep0 = - tb->ffile[0];
  double epn = - tb->ffile[tb->ninput-1];
  spline(tb->rfile,tb->efile,tb->ninput,ep0,epn,tb->e2file);

  if (tb->fpflag == 0) {
    tb->fplo = (tb->ffile[1] - tb->ffile[0]) / (tb->rfile[1] - tb->rfile[0]);
    tb->fphi = (tb->ffile[tb->ninput-1] - tb->ffile[tb->ninput-2]) /
      (tb->rfile[tb->ninput-1] - tb->rfile[tb->ninput-2]);
  }

  double fp0 = tb->fplo;
  double fpn = tb->fphi;
  spline(tb->rfile,tb->ffile,tb->ninput,fp0,fpn,tb->f2file);
}

/* ----------------------------------------------------------------------
   extract attributes from parameter line in table section
   format of line: N value R/RSQ/BITMAP lo hi FP fplo fphi
   N is required, other params are optional
------------------------------------------------------------------------- */

void PairGulpCoulLong::param_extract(Table *tb, char *line)
{
  tb->ninput = 0;
  tb->rflag = NONE;
  tb->fpflag = 0;

  char *word = strtok(line," \t\n\r\f");
  while (word) {
    if (strcmp(word,"N") == 0) {
      word = strtok(NULL," \t\n\r\f");
      tb->ninput = atoi(word);
    } else if (strcmp(word,"R") == 0 || strcmp(word,"RSQ") == 0 ||
               strcmp(word,"BITMAP") == 0) {
      if (strcmp(word,"R") == 0) tb->rflag = RLINEAR;
      else if (strcmp(word,"RSQ") == 0) tb->rflag = RSQ;
      else if (strcmp(word,"BITMAP") == 0) tb->rflag = BMP;
      word = strtok(NULL," \t\n\r\f");
      tb->rlo = atof(word);
      word = strtok(NULL," \t\n\r\f");
      tb->rhi = atof(word);
    } else if (strcmp(word,"FP") == 0) {
      tb->fpflag = 1;
      word = strtok(NULL," \t\n\r\f");
      tb->fplo = atof(word);
      word = strtok(NULL," \t\n\r\f");
      tb->fphi = atof(word);
    } else {
      printf("WORD: %s\n",word);
      error->one(FLERR,"Invalid keyword in pair table parameters");
    }
    word = strtok(NULL," \t\n\r\f");
  }

  if (tb->ninput == 0) error->one(FLERR,"Pair table parameters did not set N");
}

/* ----------------------------------------------------------------------
   compute r,e,f vectors from splined values
------------------------------------------------------------------------- */

void PairGulpCoulLong::compute_table(Table *tb)
{
  int tlm1 = tablength-1;

  // inner = inner table bound
  // cut = outer table bound
  // delta = table spacing in rsq for N-1 bins

  double inner;
  if (tb->rflag) inner = tb->rlo;
  else inner = tb->rfile[0];
  tb->innersq = inner*inner;
  tb->delta = (tb->cut*tb->cut - tb->innersq) / tlm1;
  tb->invdelta = 1.0/tb->delta;

  // direct lookup tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // e,f = value at midpt of bin
  // e,f are N-1 in length since store 1 value at bin midpt
  // f is converted to f/r when stored in f[i]
  // e,f are never a match to read-in values, always computed via spline interp

  if (tabstyle == LOOKUP) {
    memory->create(tb->e,tlm1,"pair:e");
    memory->create(tb->f,tlm1,"pair:f");

    double r,rsq;
    for (int i = 0; i < tlm1; i++) {
      rsq = tb->innersq + (i+0.5)*tb->delta;
      r = sqrt(rsq);
      tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
      tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
    }
  }

  // linear tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // de,df values = delta from lower edge to upper edge of bin
  // rsq,e,f are N in length so de,df arrays can compute difference
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == LINEAR) {
    memory->create(tb->rsq,tablength,"pair:rsq");
    memory->create(tb->e,tablength,"pair:e");
    memory->create(tb->f,tablength,"pair:f");
    memory->create(tb->de,tlm1,"pair:de");
    memory->create(tb->df,tlm1,"pair:df");

    double r,rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i*tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i]/r;
      } else {
        tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
      }
    }

    for (int i = 0; i < tlm1; i++) {
      tb->de[i] = tb->e[i+1] - tb->e[i];
      tb->df[i] = tb->f[i+1] - tb->f[i];
    }
  }

  // cubic spline tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // e2,f2 = spline coefficient for each bin
  // rsq,e,f,e2,f2 are N in length so have N-1 spline bins
  // f is converted to f/r after e is splined
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == SPLINE) {
    memory->create(tb->rsq,tablength,"pair:rsq");
    memory->create(tb->e,tablength,"pair:e");
    memory->create(tb->f,tablength,"pair:f");
    memory->create(tb->e2,tablength,"pair:e2");
    memory->create(tb->f2,tablength,"pair:f2");

    tb->deltasq6 = tb->delta*tb->delta / 6.0;

    double r,rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i*tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i]/r;
      } else {
        tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r);
      }
    }

    // ep0,epn = dh/dg at inner and at cut
    // h(r) = e(r) and g(r) = r^2
    // dh/dg = (de/dr) / 2r = -f/2r

    double ep0 = - tb->f[0] / (2.0 * sqrt(tb->innersq));
    double epn = - tb->f[tlm1] / (2.0 * tb->cut);
    spline(tb->rsq,tb->e,tablength,ep0,epn,tb->e2);

    // fp0,fpn = dh/dg at inner and at cut
    // h(r) = f(r)/r and g(r) = r^2
    // dh/dg = (1/r df/dr - f/r^2) / 2r
    // dh/dg in secant approx = (f(r2)/r2 - f(r1)/r1) / (g(r2) - g(r1))

    double fp0,fpn;
    double secant_factor = 0.1;
    if (tb->fpflag) fp0 = (tb->fplo/sqrt(tb->innersq) - tb->f[0]/tb->innersq) /
      (2.0 * sqrt(tb->innersq));
    else {
      double rsq1 = tb->innersq;
      double rsq2 = rsq1 + secant_factor*tb->delta;
      fp0 = (splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,sqrt(rsq2)) /
             sqrt(rsq2) - tb->f[0] / sqrt(rsq1)) / (secant_factor*tb->delta);
    }

    if (tb->fpflag && tb->cut == tb->rfile[tb->ninput-1]) fpn =
      (tb->fphi/tb->cut - tb->f[tlm1]/(tb->cut*tb->cut)) / (2.0 * tb->cut);
    else {
      double rsq2 = tb->cut * tb->cut;
      double rsq1 = rsq2 - secant_factor*tb->delta;
      fpn = (tb->f[tlm1] / sqrt(rsq2) -
             splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,sqrt(rsq1)) /
             sqrt(rsq1)) / (secant_factor*tb->delta);
    }

    for (int i = 0; i < tablength; i++) tb->f[i] /= sqrt(tb->rsq[i]);
    spline(tb->rsq,tb->f,tablength,fp0,fpn,tb->f2);
  }

  // bitmapped linear tables
  // 2^N bins from inner to cut, spaced in bitmapped manner
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == BITMAP) {
    double r;
    union_int_float_t rsq_lookup;
    int masklo,maskhi;

    // linear lookup tables of length ntable = 2^n
    // stored value = value at lower edge of bin

    init_bitmap(inner,tb->cut,tablength,masklo,maskhi,tb->nmask,tb->nshiftbits);
    int ntable = 1 << tablength;
    int ntablem1 = ntable - 1;

    memory->create(tb->rsq,ntable,"pair:rsq");
    memory->create(tb->e,ntable,"pair:e");
    memory->create(tb->f,ntable,"pair:f");
    memory->create(tb->de,ntable,"pair:de");
    memory->create(tb->df,ntable,"pair:df");
    memory->create(tb->drsq,ntable,"pair:drsq");

    union_int_float_t minrsq_lookup;
    minrsq_lookup.i = 0 << tb->nshiftbits;
    minrsq_lookup.i |= maskhi;

    for (int i = 0; i < ntable; i++) {
      rsq_lookup.i = i << tb->nshiftbits;
      rsq_lookup.i |= masklo;
      if (rsq_lookup.f < tb->innersq) {
        rsq_lookup.i = i << tb->nshiftbits;
        rsq_lookup.i |= maskhi;
      }
      r = sqrtf(rsq_lookup.f);
      tb->rsq[i] = rsq_lookup.f;
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i]/r;
      } else {
        tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
      }
      minrsq_lookup.f = MIN(minrsq_lookup.f,rsq_lookup.f);
    }

    tb->innersq = minrsq_lookup.f;

    for (int i = 0; i < ntablem1; i++) {
      tb->de[i] = tb->e[i+1] - tb->e[i];
      tb->df[i] = tb->f[i+1] - tb->f[i];
      tb->drsq[i] = 1.0/(tb->rsq[i+1] - tb->rsq[i]);
    }

    // get the delta values for the last table entries
    // tables are connected periodically between 0 and ntablem1

    tb->de[ntablem1] = tb->e[0] - tb->e[ntablem1];
    tb->df[ntablem1] = tb->f[0] - tb->f[ntablem1];
    tb->drsq[ntablem1] = 1.0/(tb->rsq[0] - tb->rsq[ntablem1]);

    // get the correct delta values at itablemax
    // smallest r is in bin itablemin
    // largest r is in bin itablemax, which is itablemin-1,
    //   or ntablem1 if itablemin=0

    // deltas at itablemax only needed if corresponding rsq < cut*cut
    // if so, compute deltas between rsq and cut*cut
    //   if tb->match, data at cut*cut is unavailable, so we'll take
    //   deltas at itablemax-1 as a good approximation

    double e_tmp,f_tmp;
    int itablemin = minrsq_lookup.i & tb->nmask;
    itablemin >>= tb->nshiftbits;
    int itablemax = itablemin - 1;
    if (itablemin == 0) itablemax = ntablem1;
    int itablemaxm1 = itablemax - 1;
    if (itablemax == 0) itablemaxm1 = ntablem1;
    rsq_lookup.i = itablemax << tb->nshiftbits;
    rsq_lookup.i |= maskhi;
    if (rsq_lookup.f < tb->cut*tb->cut) {
      if (tb->match) {
        tb->de[itablemax] = tb->de[itablemaxm1];
        tb->df[itablemax] = tb->df[itablemaxm1];
        tb->drsq[itablemax] = tb->drsq[itablemaxm1];
      } else {
            rsq_lookup.f = tb->cut*tb->cut;
        r = sqrtf(rsq_lookup.f);
        e_tmp = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        f_tmp = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
        tb->de[itablemax] = e_tmp - tb->e[itablemax];
        tb->df[itablemax] = f_tmp - tb->f[itablemax];
        tb->drsq[itablemax] = 1.0/(rsq_lookup.f - tb->rsq[itablemax]);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   set all ptrs in a table to NULL, so can be freed safely
------------------------------------------------------------------------- */

void PairGulpCoulLong::null_table(Table *tb)
{
  tb->rfile = tb->efile = tb->ffile = NULL;
  tb->e2file = tb->f2file = NULL;
  tb->rsq = tb->drsq = tb->e = tb->de = NULL;
  tb->f = tb->df = tb->e2 = tb->f2 = NULL;
}

/* ----------------------------------------------------------------------
   free all arrays in a table
------------------------------------------------------------------------- */

void PairGulpCoulLong::free_table(Table *tb)
{
  memory->destroy(tb->rfile);
  memory->destroy(tb->efile);
  memory->destroy(tb->ffile);
  memory->destroy(tb->e2file);
  memory->destroy(tb->f2file);

  memory->destroy(tb->rsq);
  memory->destroy(tb->drsq);
  memory->destroy(tb->e);
  memory->destroy(tb->de);
  memory->destroy(tb->f);
  memory->destroy(tb->df);
  memory->destroy(tb->e2);
  memory->destroy(tb->f2);
}

/* ----------------------------------------------------------------------
   spline and splint routines modified from Numerical Recipes
------------------------------------------------------------------------- */

void PairGulpCoulLong::spline(double *x, double *y, int n,
                       double yp1, double ypn, double *y2)
{
  int i,k;
  double p,qn,sig,un;
  double *u = new double[n];

  if (yp1 > 0.99e30) y2[0] = u[0] = 0.0;
  else {
    y2[0] = -0.5;
    u[0] = (3.0/(x[1]-x[0])) * ((y[1]-y[0]) / (x[1]-x[0]) - yp1);
  }
  for (i = 1; i < n-1; i++) {
    sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);
    p = sig*y2[i-1] + 2.0;
    y2[i] = (sig-1.0) / p;
    u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1]);
    u[i] = (6.0*u[i] / (x[i+1]-x[i-1]) - sig*u[i-1]) / p;
  }
  if (ypn > 0.99e30) qn = un = 0.0;
  else {
    qn = 0.5;
    un = (3.0/(x[n-1]-x[n-2])) * (ypn - (y[n-1]-y[n-2]) / (x[n-1]-x[n-2]));
  }
  y2[n-1] = (un-qn*u[n-2]) / (qn*y2[n-2] + 1.0);
  for (k = n-2; k >= 0; k--) y2[k] = y2[k]*y2[k+1] + u[k];

  delete [] u;
}

/* ---------------------------------------------------------------------- */

double PairGulpCoulLong::splint(double *xa, double *ya, double *y2a, int n, double x)
{
  int klo,khi,k;
  double h,b,a,y;

  klo = 0;
  khi = n-1;
  while (khi-klo > 1) {
    k = (khi+klo) >> 1;
    if (xa[k] > x) khi = k;
    else klo = k;
  }
  h = xa[khi]-xa[klo];
  a = (xa[khi]-x) / h;
  b = (x-xa[klo]) / h;
  y = a*ya[klo] + b*ya[khi] +
    ((a*a*a-a)*y2a[klo] + (b*b*b-b)*y2a[khi]) * (h*h)/6.0;
  return y;
}
