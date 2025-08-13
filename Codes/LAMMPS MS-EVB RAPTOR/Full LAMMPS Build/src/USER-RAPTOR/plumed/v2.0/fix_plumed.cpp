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
#include "fix_plumed.h"

using namespace LAMMPS_NS;
using namespace PLMD;
using namespace FixConst;

// RAPTOR - START
#include "memory.h"
#include "comm.h"
#include "EVB_api.h"
FixEVB *evb;
EVB_Complex *cplx;
// RAPTOR - STOP

FixPlumed::FixPlumed(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  p(NULL),
  nlocal(0),
  gatindex(NULL),
  masses(NULL),
  charges(NULL)
{
// Not sure this is really necessary:
  if (!atom->tag_enable) error->all(FLERR,"fix plumed requires atom tags");
// Initialize plumed:
  p=new PLMD::Plumed;
  p->cmd("setMPIComm",&world);

// Set up units
// LAMMPS units wrt kj/mol - nm - ps
// Set up units

  if (force->boltz == 1.0){
// LAMMPS units lj
    p->cmd("setNaturalUnits");
  } else {
    double energyUnits=1.0;
    double lengthUnits=1.0;
    double timeUnits=1.0;
    if (force->boltz == 0.0019872067){
// LAMMPS units real :: kcal/mol; angstrom; fs
      energyUnits=4.184;
      lengthUnits=0.1;
      timeUnits=0.001;
    } else if (force->boltz == 8.617343e-5){
// LAMMPS units metal :: eV; angstrom; ps
      energyUnits=96.48530749925792;
      lengthUnits=0.1;
      timeUnits=1.0;
    } else if (force->boltz == 1.3806504e-23){
// LAMMPS units si :: Joule, m; s
      energyUnits=0.001;
      lengthUnits=1.e-9;
      timeUnits=1.e-12;
    } else if (force->boltz == 1.3806504e-16){
// LAMMPS units cgs :: erg; cms;, s
      energyUnits=6.0221418e13;
      lengthUnits=1.e-7;
      timeUnits=1.e-12;
    } else if (force->boltz == 3.16681534e-6){
// LAMMPS units electron :: Hartree, bohr, fs
      energyUnits=2625.5257;
      lengthUnits=0.052917725;
      timeUnits=0.001;
    } else error->all(FLERR,"Odd LAMMPS units, plumed cannot work with that");
    p->cmd("setMDEnergyUnits",&energyUnits);
    p->cmd("setMDLengthUnits",&lengthUnits);
    p->cmd("setMDTimeUnits",&timeUnits);
  }

// Read fix parameters:
  int next=0;
  for(int i=3;i<narg;++i){
    if(!strcmp(arg[i],"outfile")) next=1;
    else if(next==1){
      p->cmd("setLogFile",arg[i]);
      next=0;
    }
    else if(!strcmp(arg[i],"plumedfile"))next=2;
    else if(next==2){
      p->cmd("setPlumedDat",arg[i]);
      next=0;
    }
    else error->all(FLERR,"syntax error in fix plumed - use 'fix name plumed plumedfile plumed.dat outfile plumed.out' ");
  }
  if(next==1) error->all(FLERR,"missing argument for outfile option");
  if(next==2) error->all(FLERR,"missing argument for plumedfile option");

  p->cmd("setMDEngine","LAMMPS");

  // RAPTOR - START
  MPI_Comm_rank(world, &me);

  lmax = 0;
  lx = NULL; // local copy of x
  lf = NULL; // local copy of f

  EVB_GetFixObj(modify,&evb);
  if(evb) if(me==0 && screen) fprintf(screen,"[MESSAGE] Found FixEVB Object.\n");
  else if(me==0 && screen) fprintf(screen,"[MESSAGE] No FixEVB Object.\n");
  // RAPTOR - STOP

  int natoms=int(atom->natoms);

  // RAPTOR: Update the total number of atoms with 2*ncomplex
  if(evb) natoms += evb->Engine->ncomplex * 2;

  p->cmd("setNatoms",&natoms);

  double dt=update->dt;
  p->cmd("setTimestep",&dt);

  virial_flag=1;

// This is the real initialization:
  p->cmd("init");

}

FixPlumed::~FixPlumed()
{
  delete p;

  memory->destroy(lx);
  memory->destroy(lf);
}

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

void FixPlumed::init()
{
  if (strcmp(update->integrate_style,"respa") == 0)
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

void FixPlumed::setup(int vflag)
{
  // RAPTOR - START
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;

  for(int i=nlocal; i<nall; i++) atom->f[i][0] = atom->f[i][1] = atom->f[i][2] = 0.0;
  // RAPTOR - STOP

  if (strcmp(update->integrate_style,"verlet") == 0)
    pre_force(vflag);
  else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    pre_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }

  // RAPTOR - START
  if(evb) comm->reverse_comm();
  // RAPTOR - STOP

}

void FixPlumed::min_setup(int vflag)
{
  pre_force(vflag);
}

void FixPlumed::pre_force(int vflag)
{
  int update_gatindex=0;

  // RAPTOR - START : Update atom->nlocal with local RAPTOR cecs
  nlocal_raptor = atom->nlocal;
  if(evb) {
    for(int icplx=0; icplx<evb->Engine->ncomplex; icplx++) 
      if(evb->Engine->rc_rank[icplx] == me) nlocal_raptor += 2;
  }
  // RAPTOR - STOP

// Try to find out if the domain decomposition has been updated:
  if(nlocal != nlocal_raptor) {
    if(charges)  delete [] charges;
    if(masses)   delete [] masses;
    if(gatindex) delete [] gatindex;
    nlocal = nlocal_raptor;
    gatindex = new int [nlocal];
    masses   = new double [nlocal];
    charges  = new double [nlocal];
    update_gatindex = 1;
  } else {
    for(int i=0; i<atom->nlocal; i++){
      if(gatindex[i] != atom->tag[i]-1){
        update_gatindex=1;
        break;
      }
    }
    
    // Check tags for cecs
    if(evb) {
      int indx = 0;
      int istart = atom->nlocal;
      for(int icplx=0; icplx<evb->Engine->ncomplex; icplx++) {
	if(evb->Engine->rc_rank[icplx] == me) {
	  int tg = atom->natoms + icplx*2;  // tag-1 of cec
	  if(gatindex[istart + indx*2] != tg) {
	    update_gatindex = 1;
	    break;
	  }
	  
	  tg++; // tag-1 of cec2
	  if(gatindex[istart + indx*2 + 1] != tg) {
	    update_gatindex = 1;
	    break;
	  }
	  
	  indx++; // Increment for next complex
	}
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,&update_gatindex,1,MPI_INT,MPI_SUM,world);

// In case it has been updated, rebuild the local mass/charges array
// and tell plumed about the change:
  if(update_gatindex){
    // Loop over local atoms
    for(int i=0; i<atom->nlocal; i++) {
      gatindex[i] = atom->tag[i]-1;
      masses[i]   = atom->mass[atom->type[i]];
      charges[i]  = atom->q[atom->type[i]];
    }
    
    // Loop over local cecs
    if(evb) {
      int indx = 0;
      int istart = atom->nlocal;
      for(int icplx=0; icplx<evb->Engine->ncomplex; icplx++) {
	if(evb->Engine->rc_rank[icplx] == me) {
	  int tg = atom->natoms + icplx*2; // tag-1 of cec
	  int ii = istart + indx*2;
	  gatindex[ii] = tg;
	  masses[ii]   = 1.0; // default value
	  charges[ii]  = 1.0; // default value

	  tg++;
	  ii++;
	  gatindex[ii] = tg;
	  masses[ii]   = 1.0; // default value
	  charges[ii]  = 1.0; // default value

	  indx++;
	}
      }
    }

    // Update plumed with info
    p->cmd("setAtomsNlocal",&nlocal_raptor);
    p->cmd("setAtomsGatindex",gatindex);
  }


// set up local virial/box. plumed uses full 3x3 matrices
  double virial[3][3];
  for(int i=0;i<3;i++) for(int j=0;j<3;j++) virial[i][j]=0.0;
  double box[3][3];
  for(int i=0;i<3;i++) for(int j=0;j<3;j++) box[i][j]=0.0;
  box[0][0]=domain->h[0];
  box[1][1]=domain->h[1];
  box[2][2]=domain->h[2];
  box[2][1]=domain->h[3];
  box[2][0]=domain->h[4];
  box[1][0]=domain->h[5];
  
// local variable with timestep:
  int step=update->ntimestep;

  // Grow local arrays if needed
  if(nlocal_raptor > lmax) {
    lmax = nlocal_raptor + 100;
    memory->grow(lx, lmax, 3, "FixPlumed:lx");
    memory->grow(lf, lmax, 3, "FixPlumed:lf");
  }

  // Pack local arrays with coordinates and forces to send to plumed
  memcpy(&lx[0][0], &atom->x[0][0], atom->nlocal*3*sizeof(double));
  memcpy(&lf[0][0], &atom->f[0][0], atom->nlocal*3*sizeof(double));

  // Pack cec coordinates and zero forces
  if(evb) {
    int indx = 0;
    int istart = atom->nlocal;
    for(int icplx=0; icplx<evb->Engine->ncomplex; icplx++) {
      if(evb->Engine->rc_rank[icplx] == me) {
	EVB_GetCplx(evb, icplx, &cplx);

	int ii = istart + indx*2;
	EVB_GetCEC(cplx, lx[ii]);
	lf[ii][0] = lf[ii][1] = lf[ii][2] = 0.0;
	
	ii++;
	EVB_GetCECV2(cplx, lx[ii]);
	lf[ii][0] = lf[ii][1] = lf[ii][2] = 0.0;
	
	indx++;
      }
    }
  }

  // pass all pointers to plumed:
  p->cmd("setStep",&step);
  p->cmd("setPositions",&lx[0][0]);
  p->cmd("setBox",&box[0][0]);
  p->cmd("setForces",&lf[0][0]);
  p->cmd("setMasses",&masses[0]);
  p->cmd("setCharges",&charges[0]);
  p->cmd("setVirial",&virial[0][0]);
  
// do the real calculation:
  p->cmd("calc");

// retransform virial to lammps representation:
  Fix::virial[0]=-virial[0][0];
  Fix::virial[1]=-virial[1][1];
  Fix::virial[2]=-virial[2][2];
  Fix::virial[3]=-virial[0][1];
  Fix::virial[4]=-virial[0][2];
  Fix::virial[5]=-virial[1][2];

  // Update forces on real particles
  memcpy(&atom->f[0][0], &lf[0][0], atom->nlocal*3*sizeof(double));
  
  // Grab cec forces
  if(evb) {
    int indx = 0;
    int istart = atom->nlocal;
    for(int icplx=0; icplx<evb->Engine->ncomplex; icplx++) {
      EVB_GetCplx(evb, icplx, &cplx);
      double ff[3];
      int ii;
      if(evb->Engine->rc_rank[icplx] == me) {
       	ii = istart + indx*2;
	ff[0] = lf[ii][0];
	ff[1] = lf[ii][1];
	ff[2] = lf[ii][2];
       }
      MPI_Bcast(&ff[0], 3, MPI_DOUBLE, evb->Engine->rc_rank[icplx], world);

      EVB_PutCEC(cplx, &ff[0]);
      
      if(evb->Engine->rc_rank[icplx] == me) {
       	ii++;
       	ff[0] = lf[ii][0];
	ff[1] = lf[ii][1];
	ff[2] = lf[ii][2];
      }
      MPI_Bcast(&ff[0], 3, MPI_DOUBLE, evb->Engine->rc_rank[icplx], world);
      EVB_PutCECV2(cplx, &ff[0]);
      
      if(evb->Engine->rc_rank[icplx] == me) indx++;
    }
  }
}

void FixPlumed::pre_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) pre_force(vflag);
}

void FixPlumed::min_pre_force(int vflag)
{
  pre_force(vflag);
}

