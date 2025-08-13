/*
*******************************************************************************
*                                                                             *
*                                PLUMED                                       *
*   A Portable Plugin for Free Energy Calculations with Molecular Dynamics    *
*                              VERSION 1.2                                    *
*                                                                             *
*******************************************************************************
*
*  
*  Copyright (c) 2010 The PLUMED team.
*  See http://www.plumed-code.org for more information. 
*
*  This file is part of PLUMED.
*
*  PLUMED is free software: you can redistribute it and/or modify
*  it under the terms of the GNU Lesser General Public License as 
*  published by the Free Software Foundation, either version 3 of 
*  the License, or (at your option) any later version.
*
*  PLUMED is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU Lesser General Public License for more details.
*
*  You should have received a copy of the GNU Lesser General
*  Public License along with PLUMED.  
*  If not, see <http://www.gnu.org/licenses/>.
*
*  For more info, see:  http://www.plumed-code.org
*  or subscribe to plumed-users@googlegroups.com
*
*/


#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "respa.h"
#include "domain.h"
#include "error.h"
#include "group.h"
#include "comm.h"
#include "fix_plumed.h"

static const double PLUMED_UNSET=-1.23456789e12;

using namespace LAMMPS_NS;

/*UPDATE*/ using namespace FixConst; 
/*UPDATE*/ #include "EVB_api.h"
/*UPDATE*/ FixEVB *evb;
/*UPDATE*/ EVB_Complex *cplx;
/*UPDATE*/ 

/* ---------------------------------------------------------------------- */

FixPlumed::FixPlumed(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  double *mss,*chg;
  int *int_p,size;
  int i,j,k,i_c,nn,mm;
  int me;

  if (narg != 7) error->all(FLERR,"Illegal fix plumed command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&size);

  if (!atom->tag_enable)
    error->all(FLERR,"fix plumed requires atom tags");

  if (!atom->tag_consecutive())
    error->all(FLERR,"fix plumed requires consecutive atom tags");

  if (strcmp(arg[1],"all") != 0)
    error->all(FLERR,"fix plumed requires using group all");

//  printf("PROCESSOR %d SIZE %d\n",me,size);

  if (strcmp(arg[3],"plumedfile") == 0) {
     printf("PLUMED: FOUND THE INPUT FILE %s\n",arg[4]);
     if (strcmp(arg[5],"outfile") == 0) {
     printf("PLUMED: FOUND THE OUTPUT FILE %s\n",arg[6]);
     } else {
          error->all(FLERR,"Illegal fix plumed command");
     }
     // detect the number of atoms, mass and charge
     // number of atoms in this group (default must be all )

/*UPDATE*/ EVB_GetFixObj(modify,&evb);
/*UPDATE*/   
/*UPDATE*/ if(evb)
/*UPDATE*/ {
/*UPDATE*/   if(me==0 && screen) fprintf(screen,"[MESSAGE] Found FixEVB Object.\n");
/*UPDATE*/   atom->natoms += (evb->Engine->ncomplex *2);
/*UPDATE*/ }
/*UPDATE*/ else if(me==0 && screen) fprintf(screen,"[MESSAGE] No FixEVB Object.\n");

     nn=int(atom->natoms);

     atom->check_mass();
  printf("PROCESSOR %d NATOMS %d\n",me,nn);
     mss=(double *)malloc(nn*sizeof(double));          
     chg=(double *)malloc(nn*sizeof(double));          
  //   printf("FOUND NATOMS FOR GROUP %d : %d\n",0,nn);
  //   printf("FOUND MASS  %f DT %f\n",group->mass(0),update->dt);
     //for(i=0;i<nn;i++){chg[i]=atom->q[i];} 
     for(i=0;i<nn;i++){chg[i]=PLUMED_UNSET;} 
     for(i=0;i<nn;i++){mss[i]=PLUMED_UNSET;} 
     //use the groups for massses
     MPI_Request request;
     int groupbit = group->bitmask[igroup]; //get the groupbit of all atoms
     // retrieve masses 
     if (atom->rmass) {
      for (int i = 0; i < atom->nlocal; i++){
        if (atom->mask[i] & groupbit) { 
             mm=atom->tag[i]-1; 
             mss[mm] = atom->rmass[i];
//             printf("I AM PROC %d AND I OWN AT %d R-MASS %f TAG %d \n",me,mm,mss[mm],atom->tag[i]);
             if (me) MPI_Isend(&(mss[mm]),1,MPI_DOUBLE,0,mm,world,&request);
        }
      }
     } else {
        for (int i = 0; i < atom->nlocal; i++){
          if (atom->mask[i] & groupbit){ 
               mm=atom->tag[i]-1; 
               mss[mm] = atom->mass[atom->type[i]];
//               printf("I AM PROC %d AND I OWN AT %d MASS %f TAG %d \n",me,mm,mss[mm],atom->tag[i]);
               if (me) MPI_Isend(&(mss[mm]),1,MPI_DOUBLE,0,mm,world,&request);
          }
        }
     }

/*UPDATE*/ if(evb && me==0) 
/*UPDATE*/ {
/*UPDATE*/   int aaa=evb->Engine->ncomplex*2; int ddd = nn-aaa;
/*UPDATE*/   for(int iii=0; iii<aaa; iii++) mss[ddd+iii]=1.0;
/*UPDATE*/   MPI_Isend(&(mss[ddd]),aaa,MPI_DOUBLE,0,ddd,world,&request);
/*UPDATE*/ }
  
//     printf("ME: %dEVERYONE SENT \n",me); 

     if(!me){
        MPI_Status status;
        for (i=0;i<nn;i++){
          if(mss[i]==PLUMED_UNSET)MPI_Recv(&(mss[i]),1,MPI_DOUBLE,MPI_ANY_SOURCE,i,world,&status); 
        }
     }
  
//     printf("ME: %d DONE RECEIVE : Q_FLAG %d \n",me,atom->q_flag); 

     // retrieve charges

     for (i = 0; i < atom->nlocal; i++){
        if (atom->mask[i] & groupbit){ 
             mm=atom->tag[i]-1; 
            if(atom->q_flag){
               chg[mm] = atom->q[i];
             } else  {
               chg[mm] = 0.0 ;
             }
//             printf("I AM PROC %d AND I OWN AT %d CHG %f \n",me,mm,chg[mm]);
             if (me) MPI_Isend(&(chg[mm]),1,MPI_DOUBLE,0,mm,world,&request);
        }
     }

/*UPDATE*/ if(evb && me==0) 
/*UPDATE*/ {
/*UPDATE*/   int aaa=evb->Engine->ncomplex*2; int ddd = nn-aaa;
/*UPDATE*/   for(int iii=0; iii<aaa; iii++) chg[ddd+iii]=1.0;
/*UPDATE*/   MPI_Isend(&(chg[ddd]),aaa,MPI_DOUBLE,0,ddd,world,&request);
/*UPDATE*/ }

//     printf("ME: %dEVERYONE SENT \n",me); 

     if(!me){
        MPI_Status status;
        for (i=0;i<nn;i++){
          if(chg[i]==PLUMED_UNSET)MPI_Recv(&(chg[i]),1,MPI_DOUBLE,MPI_ANY_SOURCE,i,world,&status); 
        }
     }
   
//     printf("ME: %d DONE RECEIVE  \n",me); 

     MPI_Bcast(mss,nn,MPI_DOUBLE,0,world);
     MPI_Bcast(chg,nn,MPI_DOUBLE,0,world);
  
//     if(!me){for(i=0;i<nn;i++)printf("ME %d I %d MASS %f CHG %f\n",me,i,mss[i],chg[i]);}
      //  the plumed object 
     int_p=(int *)malloc(atom->natoms*sizeof(int));
     for(i=0;i<atom->natoms;i++)int_p[i]=0;
      
     if(!me){
         plumed_ptr= new Plumed(arg[4],arg[6],&nn,mss,chg,&(update->dt),force->boltz); 
         // delete the vectors
         // create a group with only the needed atoms
         int narg_p;                 
         // now create a  group for plumed purpose 
         for(i_c=0;i_c<plumed_ptr->colvar.nconst;i_c++){
              for(j=0;j<plumed_ptr->colvar.natoms[i_c];j++){
                  int_p[plumed_ptr->colvar.cvatoms[i_c][j]]=1; // int p has the "c" like definition (start from 0 )    
                  //printf("PROC %d ATOMS: CV %d PROG %d IND %d \n",me,i_c,j,plumed_ptr->colvar.cvatoms[i_c][j]);
              }
         }
     }
     MPI_Bcast(int_p,nn,MPI_INT,0,world);

     group->create_plumed((char*)"plumed",int_p);
     plumedgroup=group->find("plumed");
     if (plumedgroup == -1) 
       error->all(FLERR,"Fix plumed group ID does not exist"); 
      plumedgroup2bit = group->bitmask[plumedgroup];
     //for (int iarg = 0; iarg < narg_p; iarg++) { fprintf(screen,"ARG %s ",arg_p[iarg]);}fprintf(screen,"\n");
     //printf("ASSIGNMENT DONE TO INDEX %d\n",plumedgroup); 
  }
  else error->all(FLERR,"Illegal fix plumed command");
  // commit my special mpi type rvec
  MPI_Type_contiguous( 3 ,MPI_DOUBLE,&MPI_RVEC);
  MPI_Type_commit(&MPI_RVEC); 
  free(mss);
  free(chg);
  free(int_p);
//  fprintf(screen,"EXITING PLUMED PROC %d\n",me);   

//  MPI_Finalize();
//  abort();

/*UPDATE*/ if(evb) atom->natoms -= (evb->Engine->ncomplex *2);

  return;
}

/* ---------------------------------------------------------------------- */

FixPlumed::~FixPlumed()
{

}

/* ---------------------------------------------------------------------- */

int FixPlumed::setmask()
{
  // set with a bitmask how and when apply the force from plumed 
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= THERMO_ENERGY;
  mask |= PRE_FORCE_RESPA;
  mask |= MIN_PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixPlumed::init()
{

  if (strcmp(update->integrate_style,"respa") == 0)
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixPlumed::setup(int vflag)
{
/*UPDATE*/  int nlocal = atom->nlocal;
/*UPDATE*/  int nall = nlocal + atom->nghost;
/*UPDATE*/  for(int i=nlocal; i<nall; i++)
/*UPDATE*/  atom->f[i][0] = atom->f[i][1] = atom->f[i][2] = 0.0;
  
  if (strcmp(update->integrate_style,"verlet") == 0)
    pre_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    pre_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
  
/*UPDATE*/ if(evb) comm->reverse_comm();
}

/* ---------------------------------------------------------------------- */

void FixPlumed::min_setup(int vflag)
{
  pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixPlumed::pre_force(int vflag)
{
  plumed_interface();
}

/* ---------------------------------------------------------------------- */

void FixPlumed::plumed_interface()
{
/*UPDATE*/ if(evb) atom->natoms += (evb->Engine->ncomplex *2);

 //printf("ENTERING PLUMED INTERFACE\n");
  int me,nprocs;
  MPI_Comm_rank(world,&me) ;
  MPI_Comm_size(world,&nprocs ) ;
 // printf("MY PROC IS %d SIZE IS %d\n",me,nprocs  );

  // alias the atoms

  double **f = atom->f;
  int   *tag = atom->tag;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double **x = atom->x;

  // retrieve positions in a single vector
  int *my_atoms;
  int *my_idx,*my_backtable;
  rvec *my_pos,*my_force;
  int i,j;
  my_atoms=( int *)calloc(nprocs,sizeof(int));
  my_backtable=( int *)malloc(atom->natoms*sizeof(int));
  my_pos=( rvec *)malloc(atom->natoms*sizeof(rvec));
  my_force=( rvec *)malloc(atom->natoms*sizeof(rvec));
  my_idx=( int *)calloc(atom->natoms,sizeof(int));
  for (i = 0; i < nlocal; i++){ // loop over local atoms 
    if (mask[i] & plumedgroup2bit) {   // 
//      printf("LOCAL ATOM %d :I BELONG TO PLUMED IN PROC %d TAG: %d\n",i,me,tag[i]);
      // place it in the right vector:
      // tag to mypos :  index is needed
      j=my_atoms[me]; 
      my_pos[j][0]=x[i][0];  
      my_pos[j][1]=x[i][1];  
      my_pos[j][2]=x[i][2];  
      my_backtable[j]=i;
      my_idx[j]=tag[i]-1;  
 //     printf("PROC %d IDX %d LOCAL_IDX %d\n",me, my_idx[j],my_backtable[j]);
      my_atoms[me]++;
    }
  }

/*UPDATE*/ if(evb && me==0) for(int icplx=0; icplx<evb->Engine->ncomplex; icplx++)
/*UPDATE*/ {
/*UPDATE*/   j=my_atoms[me];
/*UPDATE*/   my_idx[j]=((int)(atom->natoms))-(evb->Engine->ncomplex *2)+icplx*2;
/*UPDATE*/   my_idx[j+1]=my_idx[j]+1;
/*UPDATE*/   EVB_GetCplx(evb,icplx,&cplx);
/*UPDATE*/   EVB_GetCEC(cplx,my_pos[j]); 
/*UPDATE*/   //printf("%d %lf %lf %lf\n", my_idx[j], my_pos[j][0], my_pos[j][1], my_pos[j][2]);
/*UPDATE*/   EVB_GetCECV2(cplx,my_pos[j+1]);
/*UPDATE*/   my_atoms[me]+=2;
/*UPDATE*/ }
  

  // gather all the infos    
  int sum_allatoms, meatoms;
  meatoms=my_atoms[me];
  MPI_Allreduce(&meatoms,&sum_allatoms,1,MPI_INT,MPI_SUM,world);
  MPI_Allgather(&meatoms,1,MPI_INT,my_atoms,1,MPI_INT,world);
  // test the allgather 

//   for (i=0;i<nprocs;i++){
//      printf("----PROC %d RIGHT %d %d\n",me,i,my_atoms[i]);
//   }

  // put all the atoms in the same bucket: allocate it
//  printf("total number of atoms %d\n",sum_allatoms); 
  // incr pos contains the position in the buffer  
  int  *incr_pos;
  incr_pos=(int *)calloc(nprocs,sizeof(int));
  j=0;
  // only the root has informations
  if(me==0)for(i=0;i<nprocs;i++){incr_pos[i]=j;j+=my_atoms[i];}
  // this is the final vector to be allocated for store

  rvec *allpos, *allforce;
  int *allidx; 
  allpos=(rvec *)calloc(sum_allatoms,sizeof(rvec));
  allforce=(rvec *)calloc(sum_allatoms,sizeof(rvec));
  allidx=(int *)calloc(sum_allatoms,sizeof(int));
  MPI_Gatherv(my_pos,my_atoms[me],MPI_RVEC,allpos,&my_atoms[me],&incr_pos[me],MPI_RVEC,0,world); 
  MPI_Gatherv(my_idx,my_atoms[me],MPI_INT,allidx,&my_atoms[me],&incr_pos[me],MPI_INT,0,world); 


//if(me==0){
//   printf("PROC %d RIGHT \n",me);
//   for (i=0;i<nprocs;i++){
//      printf("P %d : ",i);
//      for (j=0;j<my_atoms[i];j++){
//            printf(" %d ",allidx[incr_pos[i]+j]);
//      }
//      printf("\n");
//   }
//   printf("\n");
//  }else{
////   printf("PROC %d WRONG \n",me);
////   for (i=0;i<nprocs;i++){
////      printf("P %d :  ",i);
////      for (j=0;j<my_atoms[i];j++){
////            printf(" %d ",allidx[incr_pos[i]+j]);
////      }
////      printf("\n");
////   }
////   printf("\n");
//  }
 

  
  if(!me){
   // for(i=0;i<sum_allatoms;i++){
   //     printf("COORD IDX %d COORD %f %f %f\n",allidx[i],allpos[i][0],allpos[i][1],allpos[i][2]);
   // }
    // now do the plumed calculation 
    plumed_ptr->meta_force_calculation(allidx,allpos,allforce,sum_allatoms,domain);

  } 
  // add the forces on each node 
  // scatter on the vector
  MPI_Scatterv(allforce,&my_atoms[me],&incr_pos[me],MPI_RVEC,my_force,my_atoms[me],MPI_RVEC,0,world); 
  // now each vector should have its force: loop on them


/*UPDATE*/  #define MAX_CEC 200  
/*UPDATE*/  if(evb)
/*UPDATE*/  {
/*UPDATE*/    double buf[MAX_CEC*3]; double *ff=buf;
/*UPDATE*/    
/*UPDATE*/    if(me==0) 
/*UPDATE*/    { 
/*UPDATE*/      for(int icplx=0; icplx<evb->Engine->ncomplex; icplx++) 
/*UPDATE*/      {
/*UPDATE*/        int id = my_atoms[me]-evb->Engine->ncomplex*2+icplx;
/*UPDATE*/	
/*UPDATE*/        ff[icplx*6  ] = my_force[id][0];
/*UPDATE*/        ff[icplx*6+1] = my_force[id][1];
/*UPDATE*/        ff[icplx*6+2] = my_force[id][2];
/*UPDATE*/	  ff[icplx*6+3] = my_force[id+1][0];
/*UPDATE*/        ff[icplx*6+4] = my_force[id+1][1];
/*UPDATE*/        ff[icplx*6+5] = my_force[id+1][2];
/*UPDATE*/      }
/*UPDATE*/    
/*UPDATE*/      my_atoms[me] -= evb->Engine->ncomplex*2;
/*UPDATE*/    }
/*UPDATE*/    
/*UPDATE*/    MPI_Bcast(ff,MAX_CEC*3,MPI_DOUBLE,0,world);
/*UPDATE*/    
/*UPDATE*/    for(int icplx=0; icplx<evb->Engine->ncomplex; icplx++)
/*UPDATE*/    {
/*UPDATE*/      #define SMALL 0.0001
/*UPDATE*/      EVB_GetCplx(evb,icplx,&cplx);
/*UPDATE*/      EVB_PutCEC(cplx,ff);
/*UPDATE*/      ff+=3;
/*UPDATE*/      EVB_PutCECV2(cplx,ff);
/*UPDATE*/      ff+=3;
/*UPDATE*/    }
/*UPDATE*/  }

  for (i = 0; i < my_atoms[me] ; i++){ // loop over local atoms 
      // place it in the right vector:
      j=my_backtable[i];
      f[j][0]+=my_force[i][0];
      f[j][1]+=my_force[i][1];
      f[j][2]+=my_force[i][2];
  }

  // free all the vectors
  free(allpos); 
  free(allforce); 
  free(allidx); 
  free(my_atoms);
  free(my_pos);
  free(my_force);
  free(my_idx);
  free(my_backtable);
  free(incr_pos);
//  MPI_Finalize();
//  abort();

/*UPDATE*/ if(evb) atom->natoms -= (evb->Engine->ncomplex *2);
 
//  printf("EXITING  PLUMED INTERFACE\n");
};

/* ---------------------------------------------------------------------- */

void FixPlumed::pre_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixPlumed::min_pre_force(int vflag)
{
  pre_force(vflag);
}
