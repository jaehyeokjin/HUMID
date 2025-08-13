/* ------------------------------------------------------------------------- 
 * 
 *  This file contains the electrode-model algorithm related functions
 *  while the pair_electrode.cpp contains mostly the LAMMPS related content.
 *
 *  To use this package, RAPTOR is required.
 *
 * ------------------------------------------------------------------------- */

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
#include "universe.h"

#include "pair_electrode.h"
using namespace LAMMPS_NS;

#include "fix_evb.h"
#include "EVB_engine.h"
#include "EVB_list.h"
#include "EVB_complex.h"
#include "EVB_effpair.h"

/* -------------------------------------------------------------------------
 *  Construct/Destruct the electrode model
 * ------------------------------------------------------------------------- */

#define _MPI_FILE_IO() FILE* _mpi_fp=NULL; int _mpi_nchar=0; char _MPI_LINE[1001]=""

#define _MPI_FOPEN(a,b) if(master) { \
  _mpi_fp = fopen(#a,#b); \
  if(!_mpi_fp) error->one(FLERR, "_MPI_FILE_IO can't find the file" #a); \
}

#define NEXTWORD strtok(NULL," \t\n")

#define _MPI_FCLOSE() if(master) fclose(_mpi_fp)
#define _MPI_FGETS() { if(master) { char *re = fgets(_MPI_LINE, 1000, _mpi_fp); \
    if(re) _mpi_nchar=strlen(_MPI_LINE); else _mpi_nchar = 0;} \
  MPI_Bcast(&_mpi_nchar, 1, MPI_INT, 0, universe->uworld); \
  MPI_Bcast(_MPI_LINE, _mpi_nchar+1, MPI_CHAR, 0, universe->uworld); }

void PairElectrode::et_create()
{  

  // Rank 0 of partition 0 is the master process
  int master = 0;
  if(universe->iworld == 0 && comm->me == 0) master = 1;

  if(comm->me==0 && screen) fprintf(screen,"Electrode | Create pair/electrode.\n");
  
  last_step = -1;
  fixevb = NULL;
  
  nplanes = 2;
  planes[0][0] = planes[0][1] = planes[1][0] = planes[1][1] = 0.0;  
  
  max_image = 1000;
  x_image = memory->create(x_image, max_image * 2, 3, "pair_electrode:x_image");
  Qc_proc = memory->create(Qc_proc, comm->nprocs, "pair_electrode:Qc_proc");
  
  /* -------------------------------------------------------------------- */
  /* -----  Read the parameter file                                 ----- */
  /* -------------------------------------------------------------------- */
  
  _MPI_FILE_IO();
  _MPI_FOPEN(electrode.inp, r+);
  
  while(true)
  {
    _MPI_FGETS(); if(_mpi_nchar == 0) break;
    char *ch = strtok(_MPI_LINE, " \t\n");
    if(!ch) continue;
    else if(strcmp(ch,"VOLTAGE")==0) volt = atof(strtok(NULL," \t\n"));  
    
    else if(strcmp(ch,"POSITION")==0)
    {
      pos_lo = atof(NEXTWORD);
      pos_hi = atof(NEXTWORD);
      D = pos_hi - pos_lo;
      planes[0][2] = pos_lo;
      planes[1][2] = pos_hi;

      rD = 1.0 / D;
    }

    else if(strcmp(ch,"LJ")==0)
    {
      et_cut = atof(NEXTWORD);
      et_cutsq = et_cut * et_cut;
      
      et_epsnl = atof(NEXTWORD);
      et_sigma = atof(NEXTWORD);
    }
    
    else if(strcmp(ch,"CELL")==0)
    {
      cell_x1 = atof(NEXTWORD);
      cell_x2 = atof(NEXTWORD);
      cell_y2 = atof(NEXTWORD);
      skin = 2.5*sqrt((cell_x1+cell_x2)*(cell_x1+cell_x2) + cell_y2*cell_y2);

      rcell_x1 = 1.0 / cell_x1;
      rcell_y2 = 1.0 / cell_y2;
    }
    
    else if(strcmp(ch,"PLANE")==0)
    {
      if(nplanes == MAXPLN) error->all(FLERR, "Reach the maximum number of electrode planes.");
      planes[nplanes][0] = atof(NEXTWORD);
      planes[nplanes][1] = atof(NEXTWORD);
      planes[nplanes][2] = atof(NEXTWORD);
      nplanes ++;
    }
    
    else if(strcmp(ch,"TABLE")==0)
    {
      char *p = NEXTWORD;
      if(strcmp(p,"none")==0) et_table = false;
      else
      {
	et_table = true;
	tbl_nx = atoi(p);
	tbl_ny = atoi(NEXTWORD);
	tbl_nz = atoi(NEXTWORD);
	tbl_np = atoi(NEXTWORD);
	tbl_dz = atof(NEXTWORD);

	rtbl_dz = 1.0 / tbl_dz;
	rtbl_np = 1.0 / tbl_np;
      }
    }
    
    else if(strcmp(ch, "3BODY")==0)
    {
      npar_3body = atoi(NEXTWORD);
      if(npar_3body) par_3body = new parameters_3body [npar_3body];    
  
      for(int i=0; i<npar_3body; i++)
      {  
        _MPI_FGETS(); sscanf(_MPI_LINE, "%d", &(par_3body[i].type));
        _MPI_FGETS(); sscanf(_MPI_LINE, "%lf %lf %lf", &(par_3body[i].rcut), &(par_3body[i].alpha), &(par_3body[i].beta));
        _MPI_FGETS(); sscanf(_MPI_LINE, "%lf %lf %lf %lf %lf %lf", &(par_3body[i].A), &(par_3body[i].B), &(par_3body[i].C),
        &(par_3body[i].D), &(par_3body[i].E), &(par_3body[i].F)); 
      }   

      if(npar_3body) { _MPI_FGETS(); sscanf(_MPI_LINE, "%d", &nbond_3body);

      bonds_3body = NULL;
      bonds_3body = memory->grow(bonds_3body, nbond_3body, 5, "plane_siepmann::bonds");
      
      for(int i=0; i<nbond_3body; i++)
      {
        double *bond = bonds_3body[i]; _MPI_FGETS(); 
        sscanf(_MPI_LINE, "%lf %lf %lf", &(bond[0]), &(bond[1]), &(bond[2])); 
        bond[3] = bond[0]*bond[0] + bond[1]*bond[1] + bond[2]*bond[2];
        bond[4] = sqrt(bond[3]);
      }
      
      }
    }
    
    else
    {
      if(comm->me==0 && screen) 
        fprintf(screen, "Electrode | Warning of ignored key-word line: %s  %s\n", ch, strtok(NULL,"\n"));
    }    
  }
  
  _MPI_FCLOSE();
  
  /* -------------------------------------------------------------------- */
  /* -----  Read the LJ xyz file                                    ----- */
  /* -------------------------------------------------------------------- */
  
  FILE* fp;
  
  if(master)
  {
    fp = fopen("cryst.xyz","r");
    fscanf(fp, "%d", &cryst_size);
  }
  
  MPI_Bcast(&cryst_size, 1, MPI_INT, 0, universe->uworld);  
  cryst = new double [cryst_size*3];
    
  if(master)
  {
    char line[1001];
    fgets(line,1000,fp);
    for(int i=0; i<cryst_size; i++) { fscanf(fp, "%s %lf %lf %lf\n", line, &(cryst[i*3]), &(cryst[i*3+1]), &(cryst[i*3+2])); }
    fclose(fp);
  }
    
  MPI_Bcast(cryst, 3.0 * cryst_size, MPI_DOUBLE, 0, universe->uworld);

  /* -------------------------------------------------------------------- */
  /* -----  Read the tabulated force                                ----- */
  /* -------------------------------------------------------------------- */
  
  if(et_table)
  {
    memory->create(tbl_force_x,      tbl_nx, tbl_ny, tbl_nz, "PairElectrode:PlaneTable:tbl_force_x");
    memory->create(tbl_force_y,      tbl_nx, tbl_ny, tbl_nz, "PairElectrode:PlaneTable:tbl_force_y");
    memory->create(tbl_force_z,      tbl_nx, tbl_ny, tbl_nz, "PairElectrode:PlaneTable:tbl_force_z");
    memory->create(tbl_energy_const, tbl_nx, tbl_ny, tbl_nz, "PairElectrode:PlaneTable:tbl_energy_const");
    memory->create(tbl_energy_polar, tbl_nx, tbl_ny, tbl_nz, "PairElectrode:PlaneTable:tbl_energy_table");
  
    if(master)
    {
      char line[1001];
      FILE* fp = fopen("force.tbl","r");
      
      for(int i=0; i<tbl_nx; i++) for(int j=0; j<tbl_ny; j++)
      {
        fgets(line,1000,fp);
	for(int k=0; k<tbl_nz; k++)
	{
	  fgets(line,1000,fp);
	  sscanf(line, "%lE %lE %lE", &(tbl_force_x[i][j][k]), &(tbl_force_y[i][j][k]), &(tbl_force_z[i][j][k]));
	}
      }  
    }
  
    MPI_Bcast(&(tbl_force_x[0][0][0]), tbl_nx*tbl_ny*tbl_nz, MPI_DOUBLE, 0, universe->uworld);
    MPI_Bcast(&(tbl_force_y[0][0][0]), tbl_nx*tbl_ny*tbl_nz, MPI_DOUBLE, 0, universe->uworld);
    MPI_Bcast(&(tbl_force_z[0][0][0]), tbl_nx*tbl_ny*tbl_nz, MPI_DOUBLE, 0, universe->uworld);
    
    tbl_nx--; tbl_ny--; tbl_nz--;  
    tbl_di  = cell_x1 / tbl_nx;
    tbl_djx = cell_x2 / tbl_ny;
    tbl_djy = cell_y2 / tbl_ny;

    rtbl_di = 1.0 / tbl_di;
    rtbl_djy = 1.0 / tbl_djy;
    
    // Construct the energy table
  
    tbl_energy_const[0][0][0] = 0.0;
    tbl_energy_polar[0][0][0] = 0.0;
  
    double alpha = 1.0 * rD * rtbl_np, beta;
    double dz3by3 = 1.0/3.0*tbl_dz*tbl_dz*tbl_dz;
    double dz2by2 = 1.0/2.0*tbl_dz*tbl_dz;
    D0 = (pos_lo + pos_hi) * 0.5;
  
    for(int i=0; i<=tbl_nx; i++)
    {
      double fx, fy, fz;
      
      if(i!=0)
      {
        fx = (tbl_force_x[i-1][0][0] + tbl_force_x[i][0][0]) * 0.5;
        tbl_energy_const[i][0][0] = tbl_energy_const[i-1][0][0] - fx * tbl_di;
        tbl_energy_polar[i][0][0] = 0.0;
      }
    
      for(int j=0; j<=tbl_ny; j++)
      {
      
        if(j!=0)
        {
          fx = (tbl_force_x[i][j-1][0] + tbl_force_x[i][j][0]) * 0.5;
          fy = (tbl_force_y[i][j-1][0] + tbl_force_y[i][j][0]) * 0.5;
          tbl_energy_const[i][j][0] = tbl_energy_const[i][j-1][0] - (fx * tbl_djx) - (fy * tbl_djy);
	  tbl_energy_polar[i][j][0] = 0.0;
        }

        double q0 = 0.0;
      
        for(int k=1; k<=tbl_nz; k++)
        {
          fz = (tbl_force_z[i][j][k-1] + tbl_force_z[i][j][k]) * 0.5;
	  tbl_energy_const[i][j][k] = tbl_energy_const[i][j][k-1] - (fz * tbl_dz);
	
	
	  beta = (tbl_force_z[i][j][k] - tbl_force_z[i][j][k-1]) * rtbl_dz;
	  tbl_energy_polar[i][j][k] = tbl_energy_polar[i][j][k-1] 
	                          - alpha * beta * dz3by3
				  - (alpha * tbl_force_z[i][j][k-1] + beta * q0) * dz2by2
				  - tbl_force_z[i][j][k-1] * q0 * tbl_dz;

	  q0 += tbl_dz * rD * rtbl_np;
        }
      
      } // loop y
    } // loop x
    
    // end of et_table construction
  }
  
  /* -------------------------------------------------------------------- */
  /* -----  End of the construction                                 ----- */
  /* -------------------------------------------------------------------- */
  
  if(comm->me==0 && screen) 
    fprintf(screen,"Electrode | Position of the planes: Lo=%lf  Hi=%lf\n", pos_lo, pos_hi);
  
  fixevb = NULL;
}

void PairElectrode::et_destroy()
{
  memory->destroy(x_image);
  memory->destroy(Qc_proc);
}

void PairElectrode::et_setup()
{
  double **x = atom->x;

  nall = atom->nlocal + atom->nghost; // Needed because not all partitions in sci_mp code call compute().

  if(fixevb==NULL || fixevb->Engine->evb_list->indicator==ENV_LIST)
  {
    // create image charge

    if(nall > max_image) 
    {
      max_image = nall + 1000;
      memory->grow(x_image, max_image*2, 3, "pair_electrode:ximg_local");
    }
  
    x_image2 = x_image + nall;
  
    for(int i=0; i<nall; i++)
    {
      x_image[i][0] = x_image2[i][0] = x[i][0];
      x_image[i][1] = x_image2[i][1] = x[i][1];      
      x_image [i][2] =  pos_lo * 2.0 - x[i][2];
      x_image2[i][2] =  pos_hi * 2.0 - x[i][2];
    }
  }
}

void PairElectrode::et_compute()
{
  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  
  // Don't evaluate Q0-dependent terms in SCI simulation
  if(fixevb==NULL) et_compute_pln();
  else if(fixevb->Engine->evb_list->indicator==EVB_LIST && fixevb->Engine->ncomplex == 1) et_compute_pln();
  else if(fixevb->Engine->ncomplex > 1) et_compute_pln_noQ0();

  // self-image interaction
  
  for(int i=0; i<nlocal; i++)
  {
    if(fixevb)
    {
      if(fixevb->Engine->evb_list->indicator==ENV_LIST && complex_atom[i]!=0) continue;
      else if(fixevb->Engine->evb_list->indicator==EVB_LIST && complex_atom[i]==0) continue;
    }
  
    double dr_i2l = x[i][2] - pos_lo;
    double dr_i2h = pos_hi - x[i][2];
    double ddr_i2l = cut_coul - dr_i2l * 2.0;
    double ddr_i2h = cut_coul - dr_i2h * 2.0;
    
    if(ddr_i2l > 0.0) eimage += single_coul(x[i], x_image [i], q[i], -q[i], f[i]);
    if(ddr_i2h > 0.0) eimage += single_coul(x[i], x_image2[i], q[i], -q[i], f[i]);
  }
}

void PairElectrode::et_init()
{
  /* -------------------------------------------------------------------- */
  /* -----  Initialize Plane LJ parameters                          ----- */
  /* -------------------------------------------------------------------- */
  
  et_lj1 = new double [atom->ntypes+1];
  et_lj2 = new double [atom->ntypes+1];
  et_lj3 = new double [atom->ntypes+1];
  et_lj4 = new double [atom->ntypes+1];
    
  for(int i=1; i<=atom->ntypes; i++)
  {
    double _eps_mix = sqrt  (et_epsnl * epsilon[i][i]);
    double _sig_mix = 0.5 * (et_sigma + sigma[i][i]  );
      
    et_lj1[i] = 48.0 *  _eps_mix * pow(_sig_mix,12.0);
    et_lj2[i] = 24.0 *  _eps_mix * pow(_sig_mix, 6.0);
    et_lj3[i] =  4.0 *  _eps_mix * pow(_sig_mix,12.0);
    et_lj4[i] =  4.0 *  _eps_mix * pow(_sig_mix, 6.0);
  }
  
  fixevb = NULL;
  for(int i=0; i<modify->nfix; i++) if(strcmp(modify->fix[i]->style, "evb")==0) 
  {
    fixevb = (FixEVB*)(modify->fix[i]);
    break;
  }
  
  if(fixevb && comm->me==0 && screen) 
    fprintf(screen,"Electrode | RAPTOR is supported by PairElectrode potential.\n");    
    
  ipar_3body = new int [atom->ntypes+1];
  for(int i=0; i<atom->ntypes+1; i++) ipar_3body[i] = -1;
  for(int i=0; i<npar_3body; i++) ipar_3body[par_3body[i].type] = i; 
}

/* -------------------------------------------------------------------------
 *  Compute the Plane-Charge interaction
 * ------------------------------------------------------------------------- */

#define TABLE(i,j,k) (tbl[offset[0]+i][offset[1]+j][offset[2]+k])

double PairElectrode::interpolate3d(double ***tbl, int *offset, double *frac)
{
  double onemfrac0 = 1.0 - frac[0];
  double c00 = TABLE(0,0,0) * onemfrac0 + TABLE(1,0,0) * frac[0];
  double c10 = TABLE(0,1,0) * onemfrac0 + TABLE(1,1,0) * frac[0];
  double c01 = TABLE(0,0,1) * onemfrac0 + TABLE(1,0,1) * frac[0];
  double c11 = TABLE(0,1,1) * onemfrac0 + TABLE(1,1,1) * frac[0];
  double c0  = c00 * (1-frac[1]) + c10 * frac[1];
  double c1  = c01 * (1-frac[1]) + c11 * frac[1];
  return c0 * (1-frac[2]) + c1 * frac[2];
}

void PairElectrode::locate3d(double *x, int *offset, double *frac)
{
  double b = x[1] * rcell_y2;
  double a = (x[0] - b*cell_x2) * rcell_x1;
  
  b = b - floor(b);
  if(b<0) b += 1.0;
  
  offset[1] = (int)(floor(b * rtbl_djy));
  frac[1] = (b - tbl_djy * offset[1]) * rtbl_djy;
  
  a = a - floor(a);
  if(a<0) a += 1.0;
  
  offset[0] = (int)(floor(a * rtbl_di));
  frac[0] = (a - tbl_di * offset[0]) * rtbl_di;
  
  offset[1] = (int)(floor(b * rtbl_djy));
  frac[1] = (b - tbl_djy * offset[1]) * rtbl_djy;
  
  double result;
  double z = fabs(x[2]-D0);
  
  offset[2] = (int)(floor(z * rtbl_dz));
  frac[2] = (z - tbl_dz * offset[2]) * rtbl_dz;
  
  if(offset[0]>= tbl_nx || offset[1]>= tbl_ny || offset[2]>= tbl_nz)
    printf("ERROR: TABLE OVERFLOW! %d:%d:%d\n", offset[0], offset[1], offset[2]);
}

void PairElectrode::et_compute_one(int iatm, double *x, double *f, double q, int atp)
{
  for(int ipln=0; ipln<nplanes; ipln++)
  {
    double dz = x[2] - planes[ipln][2];
    if(fabs(dz)>et_cut) continue;
    
    double n2 = floor((x[1]-planes[ipln][1]) * rcell_y2);
    double n1 = floor((x[0]-planes[ipln][0] - cell_x2 * n2) * rcell_x1);
    double dx0 = x[0] - planes[ipln][0] - n1 * cell_x1 - n2 * cell_x2;
    double dy0 = x[1] - planes[ipln][1] - n2 * cell_y2;
  
    double cutLJ = skin;
    double ddz2 = et_cutsq - dz * dz;
    if(ddz2>0.0) cutLJ += sqrt(ddz2);  

    // loop the crystal list
    
    double *table = cryst;
    
    while(table[2]<cutLJ)
    {      
      double dx = dx0 - table[0];
      double dy = dy0 - table[1];
      double rsq = dx*dx + dy*dy + dz*dz;
      
      if (rsq<et_cutsq)
      {
	double r2inv = 1.0/rsq;
	double r6inv = r2inv*r2inv*r2inv;
	double forcelj = r6inv * (et_lj1[atp]*r6inv - et_lj2[atp]);
	double fpair = forcelj*r2inv;
	eng_vdwl += r6inv*(et_lj3[atp]*r6inv - et_lj4[atp]);
	
	f[0] += dx * fpair;
	f[1] += dy * fpair;
	f[2] += dz * fpair;
	
	if(npar_3body && ipar_3body[atp]!=-1) 
	  et_compute_3body(iatm, ipar_3body[atp], dx, dy, dz, rsq);
      }
      
      table += 3;
    }
  }
  
  // QQ

  if(et_table)
  {
    int offset[3];
    double frac[3];
    double prefac = q * rtbl_np;
    double pcharge = prefac * charge;
  
    locate3d(x, offset, frac);
    f[0] += pcharge * interpolate3d(tbl_force_x, offset, frac);
    f[1] += pcharge * interpolate3d(tbl_force_y, offset, frac);
    f[2] += pcharge * interpolate3d(tbl_force_z, offset, frac);

    double econst = prefac * Q0 * interpolate3d(tbl_energy_const, offset, frac);
    double epolar = prefac *  q * interpolate3d(tbl_energy_polar, offset, frac) * tbl_np;
    if(x[2] - D0 < 0.0) econst = - econst;
  
    eng_coul += econst + epolar;
  }
  else
  {
    f[2] += q * field;

    double dz = x[2] - pos_lo;
    eng_coul -= Q0 * q * dz * qE2f;
    eng_coul -= (0.5 * q * q * rD) * (x[2]*x[2] - pos_lo*pos_lo)  * qE2f;
  }
}

void PairElectrode::et_compute_pln()
{
  // compute the polarized charge on planes
  
  const double eps0 = 5.52678E-3;
  double A = domain->xprd * domain->yprd;
  double A_inv = 1.0 / A;
  qext_lo  = volt * eps0 * A * rD;
  
  double **x = atom->x;
  double *q = atom->q;
  double charge_node = 0.0;

  for(int i=0; i<nlocal; i++) charge_node += q[i] * (x[i][2]-pos_lo);
  
  charge_node *= rD;
  MPI_Allreduce(&charge_node, &charge, 1, MPI_DOUBLE, MPI_SUM, world);
  charge += qext_lo;

  if(!et_table) 
  {
    field = A_inv * charge * force->qqrd2e * 4.0 * M_PI;
    qE2f = A_inv * force->qqrd2e * 4.0 * M_PI;
  }
  
  // compute the interaction  

  double **f = atom->f;
  int  *type = atom->type;
  
  // first loop: calculated accumulated charges
  
  Q0 = qext_lo;
  // double Qc = Q0+charge_node;
  
  // for(int i=0; i<comm->nprocs-1; i++)
  // {
  //   if(comm->me==i) MPI_Send(&Qc, 1, MPI_DOUBLE, i+1, 0, world);
  //   else if(comm->me==i+1)
  //   {
  //     MPI_Status status;
  //     MPI_Recv(&Q0, 1, MPI_DOUBLE, i, 0, world, &status);
  //     Qc = Q0+charge_node;
  //   }
  // }

  memset(&(Qc_proc[0]), 0, sizeof(double)*comm->nprocs);
  Qc_proc[comm->me] = charge_node;
  MPI_Allreduce(MPI_IN_PLACE, &(Qc_proc[0]), comm->nprocs, MPI_DOUBLE, MPI_SUM, world);

  for(int i=0; i<comm->me; i++) Q0 += Qc_proc[i];
  double Qc = Q0 + charge_node;

  for(int i=0; i<nlocal; i++) {
    et_compute_one(i, x[i], f[i], q[i], type[i]);
    Q0 += q[i] * (x[i][2] - D0) * rD;
  }
}

void PairElectrode::et_compute_3body(int iatm, int ipar, double dx, double dy, double dz, double rsq)
{
  parameters_3body *par = par_3body+ipar;
  int *type = atom->type;
  double *q = atom->q;
  double *mass = atom->mass;
  double **x = atom->x;
  double **f = atom->f;
  
  double *xi = x[iatm];
  double *fi = f[iatm];
    
  // two body - distance
  
  double r2inv = 1.0 / rsq;
  double rij = sqrt(rsq);
  double rrij = 1.0 / rij;

  double e1 = par->A * pow (r2inv, 0.5 * par->alpha);
  double e2 = par->C * r2inv * r2inv * r2inv;
  eng_vdwl += e1 - e2;

  double fpair = (e1 * par->alpha - e2 * 6.0) * r2inv;
  fi[0] += dx * fpair;
  fi[1] += dy * fpair;
  fi[2] += dz * fpair;

  // metal-oxygen-dipole 3-body
  
  int nb = atom->num_bond[iatm];
  int* bondatom = atom->bond_atom[iatm];
  	  
  double dr = rij - par->rcut;
  double rdr = 1.0 / dr;
  double expr1 = exp(par->B * rdr);
    
  if(dr<0 && nb>0)
  {
    // compute dipole
	    
    double qall = q[iatm];
    double mall = mass[type[iatm]];
	    
    double dipole[3];
    dipole[0] = xi[0] * q[iatm];
    dipole[1] = xi[1] * q[iatm];
    dipole[2] = xi[2] * q[iatm];
	    
    double com[3];
    com[0] = xi[0] * mass[type[iatm]];
    com[1] = xi[1] * mass[type[iatm]];
    com[2] = xi[2] * mass[type[iatm]];	    
	    
    for(int j=0; j<nb; j++)
    {
      int id = atom->map(bondatom[j]);
      if(id==-1) error->one(FLERR, "Molecule is too big.");
      id = domain->closest_image(iatm, id);
            
      double qq = q[id];
      double mm = mass[type[id]];
      qall += qq;
      mall += mm;
	      
      dipole[0] += qq * x[id][0];
      dipole[1] += qq * x[id][1];
      dipole[2] += qq * x[id][2];
	      
      com[0] += mm * x[id][0];
      com[1] += mm * x[id][1];
      com[2] += mm * x[id][2];	      
    }
    
    double rmall = 1.0 / mall;

    com[0] *= rmall; 
    com[1] *= rmall;
    com[2] *= rmall;
	    
    if(fabs(qall)>1.0E-6)
    {
      dipole[0] -= qall * com[0];
      dipole[1] -= qall * com[1];
      dipole[2] -= qall * com[2];
    }
	    
    double rx = -dx, ry = -dy, rz = -dz;
	    
    double dd2 = dipole[0]*dipole[0] + dipole[1]*dipole[1] + dipole[2]*dipole[2];
    double rdd2 = 1.0 / dd2;
    double dot = rx * dipole[0] + ry * dipole[1] + rz * dipole[2];
    double dd = sqrt(dd2);
    double cs = dot * rrij / dd;  
    double expr2 = exp(-8.0 * pow( (cs - 1)*0.25, 4.0));
    double r3inv = r2inv * rrij;

    double ene =  par->D * expr1 * r3inv * expr2;
    eng_vdwl -= ene;
    
    double ff = - ene * (par->B * rdr * rdr * rrij + r2inv * 3);
    fi[0] += ff * dx;
    fi[1] += ff * dy;
    fi[2] += ff * dz; 

    ff = ene * 8 * pow((cs - 1)*0.25, 3.0) / dd * rrij;
    fi[0] += ff * (dipole[0] - rx * dot * r2inv);
    fi[1] += ff * (dipole[1] - ry * dot * r2inv);
    fi[2] += ff * (dipole[2] - rz * dot * r2inv);
	    
    double ffx = ff * (rx - dipole[0] * dot * rdd2);
    double ffy = ff * (ry - dipole[1] * dot * rdd2);
    double ffz = ff * (rz - dipole[2] * dot * rdd2);
	    
    fi[0] -= ffx * q[iatm];
    fi[1] -= ffy * q[iatm];
    fi[2] -= ffz * q[iatm];
	    
    if(fabs(qall)>1.0E-6) { 
      double scale = qall * rmall * mass[type[iatm]];
      fi[0] += ffx * scale; 
      fi[1] += ffy * scale; 
      fi[2] += ffz * scale; 
    }

    for(int j=0; j<nb; j++)
    {
      int id = atom->map(bondatom[j]);
      double *fj = f[id];
      double qi = q[id];
      double mi = mass[type[id]];
	      
      fj[0] -= ffx * qi; fj[1] -= ffy * qi; fj[2] -= ffz * qi;
 
      if(fabs(qall)>1.0E-6) { 
	double scale = qall * rmall * mi;
	fj[0] += ffx * scale; 
	fj[1] += ffy * scale; 
	fj[2] += ffz * scale; 
      }
    }	    	      
  }
	  
  // metal-metal-oxygen 3-body

  if(dr<0)
  {
    double rbetainv = pow(rij, -par->beta); 
    
    for(int j=0; j<nbond_3body; j++)
    {
      double z1z2 = dz * bonds_3body[j][2];
 
      double dot = bonds_3body[j][0] * dx + bonds_3body[j][1] * dy + z1z2;
      double cs = dot * rrij / bonds_3body[j][4];
      double expr2 = exp( par->F * 0.5 * (1.0+cs) );
      double ene = par->E * expr1 * rbetainv * expr2;
      eng_vdwl += ene;
      
      double ff =  ene * (par->B * rdr * rdr * rrij + r2inv * par->beta);
      fi[0] += ff * dx;
      fi[1] += ff * dy;
      fi[2] += ff * dz; 
      
      ff = ene * 0.5 * par->F * rrij / bonds_3body[j][4];
      fi[0] -= ff * ( bonds_3body[j][0] - dot * r2inv * dx );
      fi[1] -= ff * ( bonds_3body[j][1] - dot * r2inv * dy );
      fi[2] -= ff * ( bonds_3body[j][2] - dot * r2inv * dz );	            
    }	     
  }
}

void PairElectrode::et_compute_setup()
{

  if(fixevb) {  
    if(fixevb->Engine->evb_list->indicator==-1) return;
    else complex_atom = fixevb->Engine->complex_atom;
  }
  
  eimage = 0.0;
  et_setup();
  et_compute();
}

/* -------------------------------------------------------------------- */
/* -----  SCI functions that do not calculate Q0-dependent terms  ----- */
/* -------------------------------------------------------------------- */

void PairElectrode::et_compute_pln_noQ0()
{
  double ** x = atom->x;
  double * q  = atom->q;
  double ** f = atom->f;
  int * type  = atom->type;

  // compute the non-Q0 interactions  
  int id_list = fixevb->Engine->evb_list->indicator;
  int cplx_id = fixevb->Engine->evb_complex->id;

  if(id_list == ENV_LIST) {
    for(int i=0; i<nlocal; i++) if(complex_atom[i] == 0) et_compute_one_noQ0(i, x[i], f[i], q[i], type[i]);

  } else if(id_list == EVB_LIST) {
    for(int i=0; i<nlocal; i++) if(complex_atom[i] == cplx_id) et_compute_one_noQ0(i, x[i], f[i], q[i], type[i]);

  }
}

void PairElectrode::et_compute_one_noQ0(int iatm, double *x, double *f, double q, int atp)
{
  for(int ipln=0; ipln<nplanes; ipln++)
  {
    double dz = x[2] - planes[ipln][2];
    if(fabs(dz)>et_cut) continue;
    
    double n2 = floor((x[1]-planes[ipln][1]) * rcell_y2);
    double n1 = floor((x[0]-planes[ipln][0] - cell_x2 * n2) * rcell_x1);
    double dx0 = x[0] - planes[ipln][0] - n1 * cell_x1 - n2 * cell_x2;
    double dy0 = x[1] - planes[ipln][1] - n2 * cell_y2;
  
    double cutLJ = skin;
    double ddz2 = et_cutsq - dz * dz;
    if(ddz2>0.0) cutLJ += sqrt(ddz2);  

    // loop the crystal list
    
    double *table = cryst;
    
    while(table[2]<cutLJ)
    {      
      double dx = dx0 - table[0];
      double dy = dy0 - table[1];
      double rsq = dx*dx + dy*dy + dz*dz;
      
      if (rsq<et_cutsq)
      {
	double r2inv = 1.0/rsq;
	double r6inv = r2inv*r2inv*r2inv;
	double forcelj = r6inv * (et_lj1[atp]*r6inv - et_lj2[atp]);
	double fpair = forcelj*r2inv;
	eng_vdwl += r6inv*(et_lj3[atp]*r6inv - et_lj4[atp]);
	
	f[0] += dx * fpair;
	f[1] += dy * fpair;
	f[2] += dz * fpair;
  	
  	if(npar_3body && ipar_3body[atp]!=-1) 
  	  et_compute_3body(iatm, ipar_3body[atp], dx, dy, dz, rsq);
      }
      
      table += 3;
    }
  }

}

double PairElectrode::compute_exch_image_eng(int cplx_id)
{
  if(!fixevb) error->all(FLERR,"PairElectrode::compute_exch_image_eng() not called from RAPTOR");
  
  double energy_offdiag = 0.0;
  double eimg = 0.0;

  int    nlocal = atom->nlocal;
  double     *q = atom->q; 
  double    **f = atom->f;
  double    **x = atom->x;

  _cgis_init(cut_coul);
    
  double qqrd2e = force->qqrd2e;

  // RAPTOR pointers
  int * cplx_index = fixevb->Engine->complex_atom;
  int * is_exch    = fixevb->Engine->evb_effpair->is_exch;
  double * q_exch  = fixevb->Engine->evb_effpair->q_exch;
  bool * cut_coul_effpair  = fixevb->Engine->evb_effpair->cut_coul;

  int         inum = list->inum;
  int       *ilist = list->ilist;
  int    *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  int id = 0;
  double qq;
    
  for(int i=0; i<inum; i++) {
    int atomi = ilist[i];
    if(cplx_index[atomi] == 0) continue;
      
    int jnum = numneigh[atomi];
    int *jlist = firstneigh[atomi];
      
    for(int j=0; j<jnum; j++) {
      int atomj = jlist[j] & NEIGHMASK;

      if(!cut_coul_effpair[id]) {
	double rij[3];
	
	double dr_i2l  = x[atomi][2] - pos_lo;
	double dr_i2h  = pos_hi - x[atomi][2];
	double ddr_i2l = cut_coul - dr_i2l;
	double ddr_i2h = cut_coul - dr_i2h;
	double dr_j2l  = x[atomj][2] - pos_lo;
	double dr_j2h  = pos_hi - x[atomj][2];
	double ddr_j2l = cut_coul - dr_j2l;
	double ddr_j2h = cut_coul - dr_j2h;

	if(is_exch[atomi] && cplx_index[atomj] != cplx_id) {
	  qq = -q_exch[atomi] * q[atomj];
	  
	  if(ddr_i2l > 0.0 && dr_j2l < ddr_i2l )  {
	    VECTOR_SUB(rij, x[atomi], x_image[atomj]);
	    eimg += _cgis_single_eng(rij, qq);
	  }
	  
	  if(ddr_i2h > 0.0 && dr_j2h < ddr_i2h )  {
	    VECTOR_SUB(rij, x[atomi], x_image2[atomj]);
	    eimg += _cgis_single_eng(rij, qq);
	  }
	}
	
	else if(cplx_index[atomi] != cplx_id && is_exch[atomj]) {
	  qq = -q[atomi] * q_exch[atomj];

          if(ddr_j2l > 0.0 && dr_i2l < ddr_j2l ) {
	    VECTOR_SUB(rij, x[atomj], x_image[atomi]);
	    eimg += _cgis_single_eng(rij, qq);
          }
	  
	  if(ddr_j2h > 0.0 && dr_i2h < ddr_j2h ) {
	    VECTOR_SUB(rij, x[atomj], x_image2[atomi]);
	    eimg += _cgis_single_eng(rij, qq);
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

/* -------------------------------------------------------------------- */
/* -----  SCI functions to return Q0-dependent energies/forces    ----- */
/* -------------------------------------------------------------------- */

double PairElectrode::et_compute_pln_eng()
{
  // compute the polarized charge on planes

  double energy = 0.0;

  const double eps0 = 5.52678E-3;
  double A = domain->xprd * domain->yprd;
  double A_inv = 1.0 / A;
  qext_lo  = volt * eps0 * A / D;
  
  double **x = atom->x;
  double *q = atom->q;
  double charge_node = 0.0;

  for(int i=0; i<nlocal; i++) charge_node += q[i] * (x[i][2]-pos_lo);
  
  charge_node /= D;
  MPI_Allreduce(&charge_node, &charge, 1, MPI_DOUBLE, MPI_SUM, world);
  charge += qext_lo;

  if(!et_table) 
  {
    field = A_inv * charge * force->qqrd2e * 4.0 * M_PI;
    qE2f = A_inv * force->qqrd2e * 4.0 * M_PI;
  }
  
  // compute the interaction  

  int  *type = atom->type;
  
  // first loop: calculated accumulated charges
  
  Q0 = qext_lo;
  // double Qc = Q0+charge_node;
  
  // for(int i=0; i<comm->nprocs-1; i++)
  // {
  //   if(comm->me==i) MPI_Send(&Qc, 1, MPI_DOUBLE, i+1, 0, world);
  //   else if(comm->me==i+1)
  //   {
  //     MPI_Status status;
  //     MPI_Recv(&Q0, 1, MPI_DOUBLE, i, 0, world, &status);
  //     Qc = Q0+charge_node;
  //   }
  // }

  memset(&(Qc_proc[0]), 0, sizeof(double)*comm->nprocs);
  Qc_proc[comm->me] = charge_node;
  MPI_Allreduce(MPI_IN_PLACE, &(Qc_proc[0]), comm->nprocs, MPI_DOUBLE, MPI_SUM, world);

  for(int i=0; i<comm->me; i++) Q0 += Qc_proc[i];
  double Qc = Q0 + charge_node;

  for(int i=0; i<nlocal; i++) {
    energy += et_compute_one_eng(i, x[i], q[i], type[i]);
    Q0 += q[i] * (x[i][2] - D0) * rD;
  }

  return energy;
}

double PairElectrode::et_compute_one_eng(int iatm, double *x, double q, int atp)
{
  double ecoul = 0.0;

  // QQ

  if(et_table)
  {
    int offset[3];
    double frac[3];
    double prefac = q * rtbl_np;
  
    locate3d(x, offset, frac);

    double econst = prefac * Q0 * interpolate3d(tbl_energy_const, offset, frac);
    double epolar = prefac *  q * interpolate3d(tbl_energy_polar, offset, frac) * tbl_np;
    if(x[2] - D0 < 0.0) econst = - econst;
  
    ecoul += econst + epolar;
  }
  else
  {
    double dz = x[2] - pos_lo;
    ecoul -= Q0 * q * dz * qE2f;
    ecoul -= (0.5 * q * q * rD) * (x[2]*x[2] - pos_lo*pos_lo)  * qE2f;
  }

  const int ncomplex = fixevb->Engine->ncomplex;
  ecoul /= ncomplex;

  return ecoul;
}

void PairElectrode::et_compute_pln_frc()
{
  // compute the polarized charge on planes

  double energy = 0.0;

  const double eps0 = 5.52678E-3;
  double A = domain->xprd * domain->yprd;
  double A_inv = 1.0 / A;
  qext_lo  = volt * eps0 * A * rD;
  
  double **x = atom->x;
  double ** f = atom->f;
  double *q = atom->q;
  double charge_node = 0.0;

  for(int i=0; i<nlocal; i++) charge_node += q[i] * (x[i][2]-pos_lo);
  
  charge_node /= D;
  MPI_Allreduce(&charge_node, &charge, 1, MPI_DOUBLE, MPI_SUM, world);
  charge += qext_lo;

  if(!et_table) 
  {
    field = A_inv * charge * force->qqrd2e * 4.0 * M_PI;
    qE2f = A_inv * force->qqrd2e * 4.0 * M_PI;
  }
  
  // compute the interaction  

  int  *type = atom->type;
  
  // first loop: calculated accumulated charges
  
  Q0 = qext_lo;
  // double Qc = Q0+charge_node;
  
  // for(int i=0; i<comm->nprocs-1; i++)
  // {
  //   if(comm->me==i) MPI_Send(&Qc, 1, MPI_DOUBLE, i+1, 0, world);
  //   else if(comm->me==i+1)
  //   {
  //     MPI_Status status;
  //     MPI_Recv(&Q0, 1, MPI_DOUBLE, i, 0, world, &status);
  //     Qc = Q0+charge_node;
  //   }
  // }

  memset(&(Qc_proc[0]), 0, sizeof(double)*comm->nprocs);
  Qc_proc[comm->me] = charge_node;
  MPI_Allreduce(MPI_IN_PLACE, &(Qc_proc[0]), comm->nprocs, MPI_DOUBLE, MPI_SUM, world);

  for(int i=0; i<comm->me; i++) Q0 += Qc_proc[i];
  double Qc = Q0 + charge_node;

  for(int i=0; i<nlocal; i++) {
    et_compute_one_frc(i, x[i], f[i], q[i], type[i]);
    Q0 += q[i] * (x[i][2] - D0) * rD;
  }

}

void PairElectrode::et_compute_one_frc(int iatm, double *x, double * f, double q, int atp)
{
  if(et_table)
  {
    int offset[3];
    double frac[3];
    double prefac = q * rtbl_np;
    double pcharge = prefac * charge;
  
    locate3d(x, offset, frac);
    f[0] += pcharge * interpolate3d(tbl_force_x, offset, frac);
    f[1] += pcharge * interpolate3d(tbl_force_y, offset, frac);
    f[2] += pcharge * interpolate3d(tbl_force_z, offset, frac);

  } else {
    f[2] += q * field;
  }
}
