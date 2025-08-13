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
   Package: MULTIPRO
   Purpose: Improve the parallel effeciency of PPPM in MD simulations
   Authors: Yuxing Peng and Chris Knight 
            Voth Group, Department of Chemistry, University of Chicago
------------------------------------------------------------------------- */


#include "lmptype.h"
#include "stdlib.h"
#include "string.h"
#include "multipro.h"
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

MultiPro::MultiPro(LAMMPS *lmp) : Pointers(lmp) {}

/* ---------------------------------------------------------------------- */

void MultiPro::command(int narg, char **arg)
{
  MPI_Status mpi_status;
  ReadData read_data(lmp);
  ReadRestart read_restart(lmp);

  if(narg!=2 || (strcmp(arg[0],"read_data") && strcmp(arg[0],"read_restart")) )
    error->universe_all(FLERR,"[MULIPRO] must be put before a valid command \"read_data\" or \"read_restart\"."); 
  
  if(universe->me==0) printf("[MULTIPRO] Multi-Program accelerator is loaded.\n");

  if(universe->existflag==0 || universe->nworlds!=2)
  {
    if(universe->me==0) 
      printf("[MULTIPRO] must be run on 2-partition jobs. Quit MP-APPROACH.\n");
    
    if(strcmp(arg[0],"read_data")==0) read_data.command(1,arg+1);
    else read_restart.command(1,arg+1);    
    
    return;
  }
  
  int nprocs_master = universe->procs_per_world[0];
  int nprocs_kspace = universe->procs_per_world[1];
  
  if(nprocs_master%nprocs_kspace)
    error->universe_all(FLERR,"[MULTIPRO] Processors in Partition 1 must be integer times as in Partition 2.\n");

  int ratio = nprocs_master / nprocs_kspace;
  
  /* -------------------------------------------------------------- */
  
  // set procs on real-space partition first
  
  if(universe->iworld==0)
  {
    if(strcmp(arg[0],"read_data")==0) read_data.command(1,arg+1);
    else read_restart.command(1,arg+1);
  }

  // tell k-scpace partition
  
  int address_local[2] = {0,0};
  int address[2];
  
  if(universe->iworld==0 & comm->me==0) address_local[0] = universe->me;
  else if(universe->iworld==1 & comm->me==0) address_local[1] = universe->me;
  
  MPI_Allreduce(address_local, address, 2 ,MPI_INT, MPI_SUM, universe->uworld);
  
  if(universe->iworld==0)
  {
    if(comm->me==0) MPI_Send(comm->procgrid, 3, MPI_INT, address[1], 0, universe->uworld);
  }
  else
  {
    if(universe->me==address[1]) MPI_Recv(comm->user_procgrid, 3, MPI_INT, address[0], 0, universe->uworld, &(mpi_status));    
    MPI_Bcast(comm->user_procgrid, 3, MPI_INT, 0, world);
    
    if(comm->user_procgrid[2]%ratio==0) comm->user_procgrid[2]/=ratio;
    else if(comm->user_procgrid[1]%ratio==0) comm->user_procgrid[1]/=ratio;
    else comm->user_procgrid[0]/=ratio;
  }
  
  // set procs on k-space partition

  if(universe->iworld==1)
  {
    if(strcmp(arg[0],"read_data")==0) read_data.command(1,arg+1);
    else read_restart.command(1,arg+1);  
  }
  
  /* -------------------------------------------------------------- */
  
  if(universe->iworld==0 && comm->me==0) 
    fprintf(universe->uscreen,"[MULTIPRO] Processors for r-space: %d x %d x %d\n", 
      comm->procgrid[0],comm->procgrid[1],comm->procgrid[2]);
  
  if(universe->iworld==1 && comm->me==0)   
    fprintf(universe->uscreen,"[MULTIPRO] Processors for k-space: %d x %d x %d\n", 
      comm->procgrid[0],comm->procgrid[1],comm->procgrid[2]);
      
  MPI_Barrier(universe->uworld);
    
  /* -------------------------------------------------------------- */     
  
  delete [] update->integrate_style;
  delete update->integrate;

  char *str = (char *) "mp_verlet";
  int n = strlen(str) + 1;
  update->integrate_style = new char[n];
  strcpy(update->integrate_style,str);
  update->integrate = new MP_Verlet(lmp,0,NULL);  
  
  if(comm->me==0) fprintf(universe->uscreen,"[MULTIPRO] Initiation finished on partition %d\n", universe->iworld);
  MPI_Barrier(universe->uworld);

  if(universe->iworld ==0) screen = stdout;
}
