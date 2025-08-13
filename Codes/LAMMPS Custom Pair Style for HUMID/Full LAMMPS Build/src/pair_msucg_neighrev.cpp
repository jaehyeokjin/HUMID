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
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_msucg_neighrev.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairMSUCG_NEIGHREV::PairMSUCG_NEIGHREV(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 1;
  writedata = 1;

  countiter = 0;
  countneigh = 0;
  nmax = 0;
  W = NULL;
  dW = NULL;
  comm_reverse = 4;
  comm_forward = 4;
}

/* ---------------------------------------------------------------------- */

PairMSUCG_NEIGHREV::~PairMSUCG_NEIGHREV()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);

    memory->destroy(type_linked); /*---YP--- Destroy coeff link array */
  }
}

/* ---------------------------------------------------------------------- */

/*---YP--- Function Name: P
/*---YP--- Purpose: Calcutating state probability and its partial deriv
/*---YP--- Number of Parameters: 3
/*---YP--- Parameters:
/*---YP---    type - type of particle
/*---YP---    x    - coordinates of particle
/*---YP---    f    - partial derivative (force)
/*---YP--- Return value: probability
*/

double PairMSUCG_NEIGHREV::P(int type, int state, double *x, double *f, double w, double *uval)
{

  if(type==1)
  {
    double p = 0.5 * (-1.0*tanh((w - p_constant)/(0.1 * p_constant))+1);

    if(state==1)
    {
      f[0] = 1.0/(0.2 * p_constant) * (1.0 - pow(tanh((w - p_constant)/(0.1 * p_constant)), 2.0)) * uval[0]/(0.001 * sigma_cutoff);
      f[1] = 1.0/(0.2 * p_constant) * (1.0 - pow(tanh((w - p_constant)/(0.1 * p_constant)), 2.0)) * uval[1]/(0.001 * sigma_cutoff);
      f[2] = 1.0/(0.2 * p_constant) * (1.0 - pow(tanh((w - p_constant)/(0.1 * p_constant)), 2.0)) * uval[2]/(0.001 * sigma_cutoff);
      return p;
    }
    else if(state==2)
    {
      f[0] = -1.0/(0.2 * p_constant) * (1.0 - pow(tanh((w - p_constant)/(0.1 * p_constant)), 2.0)) * uval[0]/(0.001 * sigma_cutoff);
      f[1] = -1.0/(0.2 * p_constant) * (1.0 - pow(tanh((w - p_constant)/(0.1 * p_constant)), 2.0)) * uval[1]/(0.001 * sigma_cutoff);
      f[2] = -1.0/(0.2 * p_constant) * (1.0 - pow(tanh((w - p_constant)/(0.1 * p_constant)), 2.0)) * uval[2]/(0.001 * sigma_cutoff);
      return 1.0-p;
    }
    else error->one(FLERR, "Wrong type");
  }
  else error->one(FLERR, "Wrong type");
}

/* ---------------------------------------------------------------------- */

int PairMSUCG_NEIGHREV::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++)
  {
    buf[m++] = W[i];
    buf[m++] = dW[i][0];
    buf[m++] = dW[i][1];
    buf[m++] = dW[i][2];
  }

  return m;
}

void PairMSUCG_NEIGHREV::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  for (i = 0; i < n; i++) {

    j = list[i];

    W[j] += buf[m++];
    dW[j][0] += buf[m++];
    dW[j][1] += buf[m++];
    dW[j][2] += buf[m++];
  }
}

int PairMSUCG_NEIGHREV::pack_forward_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++){
    j = list[i];
    buf[m++] = W[j];
    buf[m++] = dW[j][0];
    buf[m++] = dW[j][1];
    buf[m++] = dW[j][2];
  }
  return m;
}

void PairMSUCG_NEIGHREV::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    W[i] += buf[m++];
    dW[i][0] += buf[m++];
    dW[i][1] += buf[m++];
    dW[i][2] += buf[m++];
  }
}

void PairMSUCG_NEIGHREV::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj;
  int *ilist,*jlist,*klist,*numneigh,**firstneigh;
  double w_value, u_coef;
  double *u_value = (double *)malloc(sizeof(double)*3);
  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;


// First Loop: Calculate W(i)

  int nall = nlocal + atom->nghost;

  if(nall > nmax)
  {
    nmax = nall;
    memory->grow(W, nall, "pair/msucg:W");
    memory->grow(dW, nall, 3, "pair/msucg:dW");
  }

  for(int i=0; i<nall; i++)
    W[i] = dW[i][0] = dW[i][1] = dW[i][2] = 0.0;

  for (ii = 0; ii < inum; ii++) {

    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
/* Calculation check */
/*    if(ii == 2){
	if(countiter ==0){
	fprintf(screen,"%lf,%lf,%lf \n", x[i][0], x[i][1], x[i][2]);
      for(jj = 0 ; jj<jnum ; jj++){
	j = jlist[jj];
  	delx = xtmp - x[j][0];
  	dely = ytmp - x[j][1];
  	delz = ztmp - x[j][2];
  	rsq = delx*delx + dely*dely + delz*delz;
	if(rsq < 81.0){
  	fprintf(screen,"%d %lf %lf %lf %lf \n", jj, x[j][0], x[j][1], x[j][2], rsq);
	}
	}
	countiter += 1;
}
} */
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
	countneigh += 1;
        double w_value = 1.0 - tanh((sqrt(rsq) - sigma_cutoff)/(0.001 * sigma_cutoff));

	W[i] += w_value;

	double u_coef = 1.0 - pow(tanh((sqrt(rsq) - sigma_cutoff)/(0.001 * sigma_cutoff)),2.0);
        double fx = u_coef * delx/sqrt(rsq);
        double fy = u_coef * dely/sqrt(rsq);
        double fz = u_coef * delz/sqrt(rsq);

	dW[i][0] += fx;
	dW[i][1] += fy;
	dW[i][2] += fz;

	W[j] += w_value;
	dW[j][0] -= fx;
	dW[j][1] -= fy;
	dW[j][2] -= fz;

/* Calculation check part from the tag position of the atom
	if( atom->tag[i] == 1 | atom->tag[j] == 1){

		if (atom->tag[i] == 1){
			fprintf(screen, "ith atom position : %lf, %lf, %lf \n", x[atom->tag[i]][0], x[atom->tag[i]][1], x[atom->tag[i]][2]);
		}
		if (atom->tag[j] == 1){
			fprintf(screen, "ith atom position : %lf, %lf, %lf \n", x[atom->tag[j]][0], x[atom->tag[j]][1], x[atom->tag[j]][2]);
		}
		fprintf(screen, "neighbor pair: %d, %d \n", i, j);
	}
  */
      }
    }
  }

  comm->reverse_comm_pair(this);
  comm->forward_comm_pair(this);

/* W value check
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
  if( atom->tag[i] == 1 ){
    fprintf(screen, "Particle position ###: 1st: (%lf,%lf,%lf), ith: (%lf, %lf,%lf) \n", x[i][0], x[i][1], x[i][2], x[0][0], x[0][1], x[0][2]);
    fprintf(screen, "1st particle's w value %lf,ith particle assigned by lammps : %lf\n", W[i], W[0]);
    fprintf(screen, "1st particle's v value %lf,%lf,%lf\n", dW[i][0], dW[i][1], dW[i][2]);
    }
}
*/
// Second Loop: Calculate pair-wise interaction

  for (ii = 0; ii < inum; ii++) {

    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;

        int alpha = itype;
	int beta  = jtype;
	double pi, pj;

	while(alpha !=0)
	{
          if(alpha == 2){
            beta = 1;
          }

	  while(beta != 0)
	  {
	    double ffi[3], ffj[3];
            double pi = P(itype, alpha, x[i], ffi, W[i], dW[i]);
            double pj = P(jtype, beta, x[j], ffj, W[j], dW[j]);

	    forcelj = r6inv * (lj1[alpha][beta]*r6inv - lj2[alpha][beta]);
            fpair = factor_lj*forcelj*r2inv * pi * pj;
	    evdwl = r6inv*(lj3[alpha][beta]*r6inv-lj4[alpha][beta]) - offset[alpha][beta];
	    evdwl *= factor_lj;

	    f[i][0] += delx * fpair - ffi[0] * pj * evdwl;
            f[i][1] += dely * fpair - ffi[1] * pj * evdwl;
            f[i][2] += delz * fpair - ffi[2] * pj * evdwl;
/* ##################################################################
      if (countiter < 100){
        fprintf(screen,"################\n Step: %d, Type:(%d,%d)\n atom(%d): (%lf,%lf,%lf), atom(%d): (%lf,%lf,%lf), delta: (%lf, %lf, %lf) Rsq:%lf\nProbability check-- Pi: %lf, Pj: %lf, sigmavalue: %lf, w_value: %lf, u_value: (%lf,%lf,%lf)", countiter, alpha, beta, ii, x[i][0],x[i][1],x[i][2],jj, x[j][0],x[j][1],x[j][2],delx, dely, delz, rsq,pi,pj,sigma_cutoff,w_value,u_value[0],u_value[1],u_value[2]);
        countiter += 1;
      }

 ################################################################## */
	    if (newton_pair || j < nlocal)
	    {
              f[j][0] -= delx * fpair - ffi[0] * pj * evdwl;
              f[j][1] -= dely * fpair - ffi[1] * pj * evdwl;
              f[j][2] -= delz * fpair - ffi[2] * pj * evdwl;
            }

	    if (evflag) ev_tally(i,j,nlocal,newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
	    beta = type_linked[beta];
	  }
	  alpha = type_linked[alpha];
	}


      }
    }
  }

/*---YP--- In-house UCG nodes compute U(i) start here ---*/
/*
  for(int i=0; i<nlocal; i++)
  {
    int itype = type[i];
    int alpha = itype;

    while(alpha != 0)
    {
      double ff[3];
      double p = P(itype, alpha, x[i], ff,);

      f[i][0] += kT * log(p) * ff[0];
      f[i][1] += kT * log(p) * ff[1];
      f[i][2] += kT * log(p) * ff[2];

      alpha = type_linked[alpha];
    }
  }
*/
/*---YP--- In-house UCG nodes compute U(i) end here ---*/

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::compute_inner()
{
  error->all(FLERR, "Work not done here!"); /*---YP--- Haven't done this part yet */

  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listinner->inum;
  ilist = listinner->ilist;
  numneigh = listinner->numneigh;
  firstneigh = listinner->firstneigh;

  double cut_out_on = cut_respa[0];
  double cut_out_off = cut_respa[1];

  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        fpair = factor_lj*forcelj*r2inv;
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
          fpair *= 1.0 - rsw*rsw*(3.0 - 2.0*rsw);
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::compute_middle()
{
  error->all(FLERR, "Work not done here!"); /*---YP--- Haven't done this part yet */

  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listmiddle->inum;
  ilist = listmiddle->ilist;
  numneigh = listmiddle->numneigh;
  firstneigh = listmiddle->firstneigh;

  double cut_in_off = cut_respa[0];
  double cut_in_on = cut_respa[1];
  double cut_out_on = cut_respa[2];
  double cut_out_off = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq && rsq > cut_in_off_sq) {
        r2inv = 1.0/rsq;
        r6inv = r2inv*r2inv*r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        fpair = factor_lj*forcelj*r2inv;
        if (rsq < cut_in_on_sq) {
          rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff;
          fpair *= rsw*rsw*(3.0 - 2.0*rsw);
        }
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff;
          fpair *= 1.0 + rsw*rsw*(2.0*rsw - 3.0);
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::compute_outer(int eflag, int vflag)
{
  error->all(FLERR, "Work not done here!"); /*---YP--- Haven't done this part yet */

  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj,rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = listouter->inum;
  ilist = listouter->ilist;
  numneigh = listouter->numneigh;
  firstneigh = listouter->firstneigh;

  double cut_in_off = cut_respa[2];
  double cut_in_on = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        if (rsq > cut_in_off_sq) {
          r2inv = 1.0/rsq;
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
          fpair = factor_lj*forcelj*r2inv;
          if (rsq < cut_in_on_sq) {
            rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff;
            fpair *= rsw*rsw*(3.0 - 2.0*rsw);
          }

          f[i][0] += delx*fpair;
          f[i][1] += dely*fpair;
          f[i][2] += delz*fpair;
          if (newton_pair || j < nlocal) {
            f[j][0] -= delx*fpair;
            f[j][1] -= dely*fpair;
            f[j][2] -= delz*fpair;
          }
        }

        if (eflag) {
          r2inv = 1.0/rsq;
          r6inv = r2inv*r2inv*r2inv;
          evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
            offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (vflag) {
          if (rsq <= cut_in_off_sq) {
            r2inv = 1.0/rsq;
            r6inv = r2inv*r2inv*r2inv;
            forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
            fpair = factor_lj*forcelj*r2inv;
          } else if (rsq < cut_in_on_sq)
            fpair = factor_lj*forcelj*r2inv;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");

  memory->create(type_linked, n+1, "pair:type_linked"); /*---YP--- Allocate coeff link array */
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::settings(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);
  sigma_cutoff =  atof(arg[1]);
  p_constant = atof(arg[2]);
  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(arg[0],atom->ntypes,ilo,ihi);
  force->bounds(arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_one = cut_global;

  /*---YP--- Change the original definition for coefficient 5
  /*--- if (narg == 5) cut_one = force->numeric(FLERR,arg[4]); */
  if(narg == 5 && strcmp(arg[0], arg[1])==0) type_linked[atoi(arg[0])] = atoi(arg[4]); /*--- Add Linking list informatin here */

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::init_style()
{
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

  // set rRESPA cutoffs

  if (strstr(update->integrate_style,"respa") &&
      ((Respa *) update->integrate)->level_inner >= 0)
    cut_respa = ((Respa *) update->integrate)->cutoff;
  else cut_respa = NULL;

  /*---YP--- Obtain thermostat temperature */

  double *pT = NULL;
  int pdim;

  for(int ifix = 0; ifix < modify->nfix; ifix++)
  {
    pT = (double*) modify->fix[ifix]->extract("t_target", pdim);
    if(pT) { T = (*pT); break; }
  }

  if(pT==NULL) error->all(FLERR, "Cannot locate temperature target from thermostat.");
  else fprintf(screen, "Pair/MSUCG Ensemble temperature is %lf.\n", T);

  kT = force->boltz * T;
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listinner = ptr;
  else if (id == 2) listmiddle = ptr;
  else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMSUCG_NEIGHREV::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag) {
    double ratio = sigma[i][j] / cut[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

  if (cut_respa && cut[i][j] < cut_respa[3])
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
    double rc3 = cut[i][j]*cut[i][j]*cut[i][j];
    double rc6 = rc3*rc3;
    double rc9 = rc3*rc6;
    etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
    ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::read_restart(FILE *fp)
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
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,epsilon[i][i],sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairMSUCG_NEIGHREV::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g\n",i,j,epsilon[i][j],sigma[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairMSUCG_NEIGHREV::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  error->all(FLERR, "Work not done here!"); /*---YP--- Haven't done this part yet */

  double r2inv,r6inv,forcelj,philj;

  r2inv = 1.0/rsq;
  r6inv = r2inv*r2inv*r2inv;
  forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
  fforce = factor_lj*forcelj*r2inv;

  philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
    offset[itype][jtype];
  return factor_lj*philj;
}

/* ---------------------------------------------------------------------- */

void *PairMSUCG_NEIGHREV::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  return NULL;
}
