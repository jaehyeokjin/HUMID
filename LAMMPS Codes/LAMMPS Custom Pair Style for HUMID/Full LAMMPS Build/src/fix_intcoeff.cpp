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
   Contributing author: Naveen Michaud-Agrawal (Johns Hopkins University)
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_intcoeff.h"
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "respa.h"
#include "memory.h"
#include "error.h"
#include "math.h"
#include "pair.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "group.h"
#include "force.h"

using namespace LAMMPS_NS;

#define DELTA 10000

using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixIntCoeff::FixIntCoeff(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if ((narg < 4) || (narg > 5))
    error->all(FLERR,"Illegal fix spring/self command");
  dist_cutoff = force->numeric(FLERR,arg[3]);

  peratom_flag = 1;
  size_peratom_cols = 5;
  peratom_freq = 1;

  // Initialization of the state coefficient array

  nmax = atom->nmax;
  memory->create(statecoeff,nmax,5,"intcoeff:statecoeff");
  array_atom = statecoeff;

  countneigh = (int *)calloc(nmax,sizeof(int));
  int countneigh = {0,};

  // perform initial allocation of atom-based array
  // register with Atom class

  // Neighbor consideration - Implemented 07/16/15
  nmax_pair=0;
  vector= NULL;
  array= NULL;


  // zero the array since dump may access it on timestep 0
  // zero the array since a variable may access it before first run

}

/* ---------------------------------------------------------------------- */

FixIntCoeff::~FixIntCoeff()
{
  memory->destroy(statecoeff);
  memory->destroy(vector);
  memory->destroy(array);
}

/* ---------------------------------------------------------------------- */

void FixIntCoeff::init()
{
  if (strstr(update->integrate_style,"respa"))
  nlevels_respa = ((Respa *) update->integrate)->nlevels;

  int irequest = neighbor->request((void *) this);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

void FixIntCoeff::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet"))
    post_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
}

/* ---------------------------------------------------------------------- */

int FixIntCoeff::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

void FixIntCoeff::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixIntCoeff::post_force(int vflag)
{
//  invoked_local = update->ntimestep;

  ncount = atom->nlocal;
  if (ncount > nmax_pair) reallocate(ncount);
  size_local_rows = ncount;
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////    Main neighbor matching begins <- actually from compute_pairs() subroutine. Moved by 07/24/15    ////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
  int i,j,m,n,ii,jj,inum,jnum,itype,jtype;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsqi2,rsqi6;
  double dx_temp, dy_temp, dz_temp;
  double rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double *ptr;

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  //  Initialziation for countneigh array and index for corresponding matrix (countindex) <- Implemented by July 24
  memset(countneigh,0x00,atom->nmax);
  int countindex = 0;
  e_step=0.0;
  double x_comp=0.0;
  double y_comp=0.0;
  double z_comp=0.0;

  // Initiallization for the state coefficient vector for the first step
  for (int i = 0; i < nmax; i++){
    statecoeff[i][0]=0.0;
  }

  // invoke half neighbor list (will copy or build if necessary)

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  // skip if I or J are not in group
  // for flag = 0, just count pair interactions within force cutoff
  // for flag = 1, calculate requested output fields

  Pair *pair = force->pair;
  double **cutsq = force->pair->cutsq;

  m = 0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itag = tag[i];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      if (!(mask[j] & groupbit)) continue;
      // Appending into the countneigh array
      
      // rsq calculation (by original file)
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if(dist_cutoff != 0.0) if (rsq < dist_cutoff*dist_cutoff) statecoeff[i][0]=statecoeff[i][0]+1.0;

      m++;
      // Force consideration only if dist > certain criteria (here fixed at 1.0 angstrom)
      if(rsq < 4.0){
        countneigh[countindex]=j;
        countindex++;
      }
    }
    if(countindex>=2){
      for(int k = 0; k < countindex; k++){
        dx_temp=xtmp-x[countneigh[k]][0];
        dy_temp=ytmp-x[countneigh[k]][1];
        dz_temp=ztmp-x[countneigh[k]][2];
        rsqi2=(dx_temp*dx_temp+dy_temp*dy_temp+dz_temp*dz_temp);
//        rsqi6=rsqi2*rsqi2*rsqi2 ;
        e_step += 0.005* rsqi2;
        // i component
        x_comp -= dx_temp*0.01;
        y_comp -= dy_temp*0.01;
        z_comp -= dz_temp*0.01;
        // j component
        f[countneigh[k]][0]+=dx_temp;
        f[countneigh[k]][1]+=dy_temp;
        f[countneigh[k]][2]+=dz_temp;

      }
      f[i][0]+=x_comp;
      f[i][1]+=y_comp;
      f[i][2]+=z_comp;
    }
    memset(countneigh,0x00,atom->nmax);
    countindex = 0;
  }

  for (int i = 0; i < nmax; i++){
    statecoeff[i][1] = ((double) rand() / (RAND_MAX));
    statecoeff[i][2] = ((double) rand() / (RAND_MAX));
    statecoeff[i][3] = ((double) rand() / (RAND_MAX));
    statecoeff[i][4] = ((double) rand() / (RAND_MAX));
  }
}

/* ---------------------------------------------------------------------- */

void FixIntCoeff::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixIntCoeff::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

double FixIntCoeff::compute_scalar()
{
  double all;
  MPI_Allreduce(&e_step, &all, 1, MPI_DOUBLE, MPI_SUM, world);  
  return all;
}

/* ---------------------------------------------------------------------- */

void FixIntCoeff::reallocate(int n)
{
  // grow vector or array and indices array

  while (nmax_pair < n) nmax_pair += DELTA;

    memory->destroy(vector);
    memory->create(vector,nmax_pair,"intcoeff:vector");
    vector_local = vector;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixIntCoeff::memory_usage()
{
  double bytes = atom->nmax*5 * sizeof(double);
  return bytes;
}
