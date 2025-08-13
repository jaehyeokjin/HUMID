/* -------------------------------------------------------------------------
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
   Authors: Chris Knight 
             Derived from USER-MULTIPRO package
------------------------------------------------------------------------- */


#include "lmptype.h"
#include "stdlib.h"
#include "string.h"
#include "multipro_sci.h"
#include "domain.h"
#include "update.h"
#include "integrate.h"
#include "modify.h"
#include "output.h"
#include "finish.h"
#include "input.h"
#include "timer.h"
#include "error.h"
#include "universe.h"
#include "comm.h"

#include "read_data.h"
#include "read_restart.h"

#include "style_integrate.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MultiProSCI::MultiProSCI(LAMMPS *lmp) : Pointers(lmp) {}

/* ---------------------------------------------------------------------- */

void MultiProSCI::command(int narg, char **arg)
{
  MPI_Status mpi_status;

  if(narg!=2 || (strcmp(arg[0],"read_data") && strcmp(arg[0],"read_restart")) )
    error->universe_all(FLERR,"[MULIPRO_SCI] must be put before a valid command \"read_data\" or \"read_restart\"."); 
  
  if(universe->me==0) printf("[MULTIPRO_SCI] Multi-Program accelerator is loaded.\n");
  
#ifdef STATE_DECOMP
  error->universe_all(FLERR,"[MULTIPRO_SCI] Does not currently support -DSTATE_DECOMP.");
#endif
  
  int nprocs_master = universe->procs_per_world[0];
  int nprocs_all = universe->nprocs;
  
  // Make sure all partitions are the same size
  for(int i=1; i<universe->nworlds; i++) 
    if(nprocs_master != universe->procs_per_world[i])
      error->universe_all(FLERR,"[MULTIPRO_SCI] Number of processors in all partitions must be the same.\n");

  /* -------------------------------------------------------------- */
  
  if(strcmp(arg[0],"read_data")==0) {
    ReadData read_data(lmp);
    read_data.command(1,arg+1);
  } else {
    ReadRestart read_restart(lmp);
    read_restart.command(1,arg+1);
  }

  if(universe->me==0 && universe->uscreen) fprintf(universe->uscreen,"[MULTIPRO_SCI] Processors for partition %d: %d x %d x %d\n", 
						   universe->iworld,comm->procgrid[0],comm->procgrid[1],comm->procgrid[2]);

  // Make sure all partitions have the same layout
  int nxyz[3];
  nxyz[0] = comm->procgrid[0];
  nxyz[1] = comm->procgrid[1];
  nxyz[2] = comm->procgrid[2];
  MPI_Bcast(&(nxyz), 3, MPI_INT, 0, universe->uworld);
  if(nxyz[0] != comm->procgrid[0] || nxyz[1] != comm->procgrid[1] || nxyz[2] != comm->procgrid[2]) 
    error->universe_all(FLERR,"[MULTIPRO_SCI] Processor grid layout for all partitions must be the same");

  /* -------------------------------------------------------------- */
  
  MPI_Barrier(universe->uworld);
 
  /* -------------------------------------------------------------- */     
  
  delete [] update->integrate_style;
  delete update->integrate;

  char *str = (char *) "mp_verlet_sci";
  int n = strlen(str) + 1;
  update->integrate_style = new char[n];
  strcpy(update->integrate_style,str);
  update->integrate = new MP_Verlet_SCI(lmp,0,NULL);
}
