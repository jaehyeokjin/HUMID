/* ----------------------------------------------------------------------
  Copyright @ Voth Group
  Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "EVB_checkout.h"
#include "fix_evb.h"
#include "EVB_engine.h"
#include "EVB_complex.h"
#include "EVB_matrix.h"
#include "EVB_repul.h"
#include "EVB_offdiag_pt.h"

#include "verlet.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "atom.h"
#include "atom_vec_full.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "output.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "fix.h"
#include "timer.h"
#include "memory.h"
#include "error.h"
#include "write_restart.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

EVB_Checkout::EVB_Checkout(LAMMPS *lmp) : Pointers(lmp) 
{
  allatom = false;
  write_types = false;
  write_types_mol = false;
}

/* ---------------------------------------------------------------------- */

void __str2upper(char* src)
{
  if(!src) return;
  
  while(*src)
  {
    if(*src>='a' && *src<='z') *src -= 32;
    src++;
  }
}


void EVB_Checkout::init(char* input_name)
{
  // read input

  FILE *fp = fopen(input_name,"r");
  if(!fp) error->universe_all(FLERR,"EVB_CHECKOUT: Can't open input file.");
  
  char line[1025];
  int line_id=0;
  
  while(fgets(line,1024,fp))
  {
    // get one line
    
    line_id++;
    char *keyword = strtok(line," =\t\r\n");
    char *setting = strtok(NULL," =\t\r\n");
    __str2upper(keyword);
    __str2upper(setting);
    
    // blank line or comment line
    
    if(!keyword || keyword[0]=='#') continue;
    
    // lacking of settings
    
    if(!setting)
    {
      char errline[256];
      sprintf(errline,"EVB_CHECKOUT: Incompleted settings at line %d of input.",line_id);
      error->universe_all(FLERR,errline);
    }
    
    // deal with settings
    
    #define FOR_KEY(key) else if(strcmp(keyword,#key)==0)
    
    FOR_KEY(ALL_ATOM)
    {
      if(strcmp(setting,"TRUE")==0) allatom = true;
      else allatom = false;
    }

    FOR_KEY(WRITE_TYPES)
    {
      if(strcmp(setting,"TRUE")==0) write_types = true;
      else write_types = false;
    }
    
    FOR_KEY(WRITE_TYPES_MOL)
    {
      if(strcmp(setting,"TRUE")==0) write_types_mol = true;
      else write_types_mol = false;
    }
    
    else
    {
      char errline[256];
      sprintf(errline,"EVB_CHECKOUT: Unkown key [%s] at line %d of input.",keyword,line_id);
      error->universe_all(FLERR,errline);
    }
    
    printf("%s = %s\n",keyword,setting);
    
  } // while-fgets
  
  fclose(fp);
}

void EVB_Checkout::exit()
{

}

/* ---------------------------------------------------------------------- */

void EVB_Checkout::run()
{
  // run evb model

  lmp->init();
  update->whichflag = 1;
  update->setupflag = 1;
  update->ntimestep = 0;

  atom->setup();
  modify->setup_pre_exchange();
  domain->pbc();
  domain->reset_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
  comm->exchange();
  if (atom->sortfreq > 0) atom->sort();
  comm->borders();
  neighbor->build();
  neighbor->ncalls = 0;

  modify->setup_pre_force(0);

  if (force->kspace) force->kspace->setup();  
  if (force->newton) comm->reverse_comm();

  int ifix = modify->find_fix("evb");
  if(ifix==-1) error->universe_all(FLERR,"EVB_CHECKOUT: Cannot find [fix/evb] module.");
  FixEVB* fix = (FixEVB*)(modify->fix[ifix]);
  EVB_Engine*  evb_engine  = fix->Engine;

  modify->setup(0);
  while(evb_engine->evb_matrix->pivot_state) 
  { 
    comm->forward_comm();
    neighbor->build(); 
    modify->setup(0); 
  }
  
  WriteRestart write_restart(lmp);
  char *args[1];
  args[0] = (char*)"chk.restart";
  write_restart.command(1,args);
  
  write2txt();
}

void EVB_Checkout::write2txt()
{  
  // checkout data
  
  int ifix = modify->find_fix("evb");
  if(ifix==-1) error->universe_all(FLERR,"EVB_CHECKOUT: Cannot find [fix/evb] module.");
  FixEVB* fix = (FixEVB*)(modify->fix[ifix]);
  EVB_Engine*  evb_engine  = fix->Engine;
  EVB_Complex* evb_complex = evb_engine->evb_complex;
  EVB_Matrix*  evb_matrix  = evb_engine->evb_matrix;
  
  FILE *chk = fopen("evb.chk","w");
  
  // checkout box
  
  fprintf(chk,"BOX %18.12lf %18.12lf %18.12lf\n\n", domain->xprd, domain->yprd, domain->zprd);
  
  // checkout atoms
  
  int* list = evb_complex->cplx_list;  
  
  int natom;
  if(allatom) natom = atom->nlocal;
  else natom = evb_complex->nlocal_cplx;
  
  int* map = new int [atom->nlocal+1];
  for(int i=0; i<atom->nlocal+1; i++) map[i]=-2;
  
  char* element = new char[atom->ntypes+1];
  for(int i=1; i<=atom->ntypes; i++)
    if(abs(atom->mass[i]-1.0) < 0.001) element[i]='H';
    else if(abs(atom->mass[i]-12.0) < 0.01) element[i]='C';
    else if(abs(atom->mass[i]-14.0) < 0.01) element[i]='N';
    else if(abs(atom->mass[i]-16.0) < 0.01) element[i]='O';
    else element[i]='X';
    
  fprintf(chk,"ATOMS %d\n",natom);
  
  FILE *qmchk = fopen("qm.index","w");
  
  for(int i=0; i<natom; i++)
  {
    int id;
    if(allatom) id = i; else id = list[i];
    map[id]= i;
    
    fprintf(chk, "%8d %8d %18.12lf %18.12lf %18.12lf         %c\n",
      i+1, atom->tag[id],
      atom->x[id][0], atom->x[id][1], atom->x[id][2],
      element[atom->type[id]]
    );
    
    fprintf(qmchk,"&QM_KIND %cqm\n  MM_INDEX %6d\n&END QM_KIND\n", element[atom->type[id]],atom->tag[id]);
  }
  
  fclose(qmchk);
  
  // checkout matrix and energy
  
  fprintf(chk, "\nENV_ENERGY %18.12lf\n", evb_matrix->e_env[EDIAG_POT]);
  fprintf(chk, "\nNSTATE %3d   [I|V|T]\n", evb_complex->nstate);
  
  int atom_index[100];
    
  for(int i=0; i<evb_complex->nstate; i++)
  {
    evb_complex->load_avec(i);
    evb_complex->update_mol_map();
    
    int mol_id = evb_complex->molecule_B[i];
    int target_atom = evb_engine->molecule_map[mol_id][1];
    int target_etp = evb_engine->mol_type[target_atom];
    int irep = -1;
    
    for(int j=0; j<evb_engine->nrepulsive; j++) 
      if(evb_engine->all_repulsive[j]->etp_center == target_etp) { irep = j;  break; }
    
    evb_engine->all_repulsive[irep]->center_mol_id = mol_id;
    int n = evb_engine->all_repulsive[irep]->checkout(atom_index);
    
    fprintf(chk,"%3d %18.12lf   %-12s", i, evb_matrix->e_diagonal[i][EDIAG_POT], evb_engine->all_repulsive[irep]->name);

    int indx = 1;
    
    // Write some entries verbatim from checkout()
    if(atom_index[0] < 0) {
      for (int tt=indx; tt<=(-atom_index[0]); tt++) fprintf(chk, " %5d", atom_index[tt]);
      indx -= atom_index[0]; // subtract a negative number
    }

    // The remaining entries are atom indices; write as map[atom_index]
    for(int tt=indx; tt<n; tt++) 
      if(atom_index[tt]==-1) fprintf(chk," %5d",-1);
      else fprintf(chk, " %5d", map[atom_index[tt]]+1); // map of atom index
    
    fprintf(chk," %5d\n",-1); // Write '-1' because atom_index[0] is special and not written to checkpoint file.
  }
  
  fprintf(chk, "\nNCOUPLING %3d   [I|J|V|T|A|B|C]\n", evb_complex->nstate + evb_complex->nextra_coupling -1);

  int iextra = 0;
  
  for(int i=1; i<evb_complex->nstate; i++)
  {
    evb_complex->load_avec(i);
    evb_complex->update_mol_map();
    int n = evb_engine->all_offdiag[evb_complex->reaction[i]-1]->checkout(atom_index);

    fprintf(chk, "%3d %3d %18.12lf   %-12s",
      evb_complex->parent_id[i], i, evb_matrix->e_offdiag[i-1][EOFF_ENE],
        evb_engine->all_offdiag[evb_complex->reaction[i]-1]->name);
    for(int tt=0; tt<n; tt++) if(atom_index[tt]==-1) fprintf(chk," %5d",-1);
    else fprintf(chk, " %5d", map[atom_index[tt]]+1);
    fprintf(chk,"\n");
    
    int save_mol_A = evb_complex->molecule_A[i];
    for(int j=0; j<evb_complex->extra_coupling[i]; j++)
    {
      evb_complex->molecule_A[i] = evb_complex->molecule_B[i-j-1];
      int n = evb_engine->all_offdiag[evb_complex->reaction[i]-1]->checkout(atom_index);

      fprintf(chk, "%3d %3d %18.12lf %-12s",
        i-j-1, i, evb_matrix->e_extra[iextra++][EOFF_ENE],
          evb_engine->all_offdiag[evb_complex->reaction[i]-1]->name);
      for(int tt=0; tt<n; tt++) fprintf(chk, " %5d", map[atom_index[tt]]+1);
      fprintf(chk,"\n");
    }
    evb_complex->molecule_A[i] = save_mol_A;
  }
  
  evb_complex->load_avec(evb_matrix->pivot_state);
  evb_complex->update_mol_map();
  
  // checkout force
  
  fprintf(chk,"\nFORCE MATRIX\n\n");
  
  int nall = atom->nlocal + atom->nghost;
  
  #define CHECKOUT_FORCE() \
  for(int gatom=atom->nlocal; gatom<nall; gatom++) \
  { \
    int ddd = atom->map(atom->tag[gatom]); \
    atom->f[ddd][0]+=atom->f[gatom][0]; atom->f[ddd][1]+=atom->f[gatom][1]; atom->f[ddd][2]+=atom->f[gatom][2]; \
  } \
  for(int iatom=0; iatom<natom; iatom++) \
  { \
    int id; if(allatom) id=iatom; else id=list[iatom]; \
    if(write_types_mol) fprintf(chk,"%18.12lf %18.12lf %18.12lf  %d %d # %d\n", f[id][0], f[id][1], f[id][2], atom->type[id], atom->molecule[id], atom->tag[id]); \
    else if(write_types) fprintf(chk,"%18.12lf %18.12lf %18.12lf  %d # %d\n", f[id][0], f[id][1], f[id][2], atom->type[id], atom->tag[id]); \
    else fprintf(chk,"%18.12lf %18.12lf %18.12lf   # %d\n", f[id][0], f[id][1], f[id][2], atom->tag[id]); \
  } \
  fprintf(chk, "\n");
 
  double **f, **f_save=atom->f;
  atom->f = f = evb_matrix->f_env;  
  CHECKOUT_FORCE();
    
  for(int i=0; i<evb_complex->nstate; i++)
  {
    atom->f = f = evb_matrix->f_diagonal[i];

    evb_complex->load_avec(i); // To ensure correct types and charges get written to checkpoint file
    CHECKOUT_FORCE();
  }
  
  iextra=0;
  for(int i=1; i<evb_complex->nstate; i++)
  {
    atom->f = f = evb_matrix->f_off_diagonal[i-1];
    CHECKOUT_FORCE();
    
    for(int j=0; j<evb_complex->extra_coupling[i]; j++)
    {
      atom->f = f = evb_matrix->f_extra_coupling[iextra++];
      CHECKOUT_FORCE();
    }
  }
  
  evb_complex->load_avec(evb_matrix->pivot_state);
  atom->f = f_save;
  
  fclose(chk);
  delete [] map;
  delete [] element;
}

/* -----------

  This part of code is not used anymore

void EVB_Checkout::write2bin()
{
  int s_real = sizeof(double);
  int s_int  = sizeof(int);
  
  #define WRITEINT(A) fwrite(&(A), s_int, 1, chk)
  #define WRITEREAL(A) fwrite(&(A), s_real, 1, chk)
  #define WRITEINTARR(A,N) fwrite(A, s_int, N, chk)
  #define WRITEREALARR(A,N) fwrite(A, s_real, N, chk)
  
  // checkout data
  
  int ifix = modify->find_fix("evb");
  if(ifix==-1) error->universe_all(FLERR,"EVB_CHECKOUT: Cannot find [fix/evb] module.");
  FixEVB* fix = (FixEVB*)(modify->fix[ifix]);
  EVB_Engine*  evb_engine  = fix->Engine;
  EVB_Complex* evb_complex = evb_engine->evb_complex;
  EVB_Matrix*  evb_matrix  = evb_engine->evb_matrix;
  
  evb_engine->disable_reaction();
   
  FILE *chk = fopen("evb.chk","wb");
  
  // checkout box
  
  WRITEREAL(domain->xprd);
  WRITEREAL(domain->yprd);
  WRITEREAL(domain->zprd);
  
  // checkout atoms
  
  int* list = evb_complex->cplx_list;  
  int natom;
  if(allatom) natom = atom->nlocal;
  else natom = evb_complex->nlocal_cplx;
  int* map = new int [atom->nlocal+1];
  memset(map,0, sizeof(int)*(atom->nlocal));
  fwrite(&natom, sizeof(int), 1, chk);
  
  for(int i=0; i<natom; i++)
  {
    int id;
    if(allatom) id=i; else id=list[i];
    map[id]= i;    
    WRITEINT(atom->tag[id]);
    WRITEREALARR(atom->x[id],3);
  }
  
  // checkout matrix and energy
  
  WRITEINT(evb_complex->nstate);
  
  for(int i=0; i<evb_complex->nstate; i++)
  {
    evb_complex->load_avec(i);
    evb_complex->update_mol_map();
    
    int mol_id = evb_complex->molecule_B[i];
    int target_atom = evb_engine->molecule_map[mol_id][1];
    int target_etp = evb_engine->mol_type[target_atom];
    int irep = -1;
    
    for(int j=0; j<evb_engine->nrepulsive; j++) 
      if(evb_engine->all_repulsive[j]->etp_center == target_etp) { irep = j;  break; }
    
    WRITEREAL(evb_matrix->e_diagonal[i][EDIAG_POT]);
    irep++; WRITEINT(irep);
    irep=0; WRITEINT(irep);
  }
  
  int ncoupl = evb_complex->nstate + evb_complex->nextra_coupling -1;
  WRITEINT(ncoupl);
   
  int iextra=0;
  for(int i=1; i<evb_complex->nstate; i++)
  {
    evb_complex->load_avec(i);
    evb_complex->update_mol_map();
    
    EVB_OffDiag_PT* evb_offdiag = (EVB_OffDiag_PT*)(evb_engine->all_offdiag[evb_complex->reaction[i]-1]);
    int mol_A = evb_complex->molecule_A[i];
    int mol_B = evb_complex->molecule_B[i];
    
    int m1, m2, m3;
    if(evb_offdiag->mol_A_Rq[0]==1) m1 = mol_A; else m1 = mol_B;
    if(evb_offdiag->mol_A_Rq[1]==1) m2 = mol_A; else m2 = mol_B;
    if(evb_offdiag->mol_A_Rq[2]==1) m3 = mol_A; else m3 = mol_B;
    
    int A1 = map[evb_engine->molecule_map[m1][evb_offdiag->atom_A_Rq[0]]] +1;
    int A2 = map[evb_engine->molecule_map[m2][evb_offdiag->atom_A_Rq[1]]] +1;
    int A3 = map[evb_engine->molecule_map[m3][evb_offdiag->atom_A_Rq[2]]] +1;
    
    int tt=3;
    
    WRITEINT(evb_complex->parent_id[i]);
    WRITEINT(i);
    WRITEREAL(evb_matrix->e_offdiag[i-1][EOFF_ENE]);
    WRITEINT(evb_complex->reaction[i]);
    WRITEINT(tt);
    WRITEINT(A1);
    WRITEINT(A2);
    WRITEINT(A3);
 
    for(int j=0; j<evb_complex->extra_coupling[i]; j++)
    {
      mol_A = evb_complex->molecule_B[i-j-1];
      if(evb_offdiag->mol_A_Rq[0]==1) m1 = mol_A; else m1 = mol_B;
      A1 = map[evb_engine->molecule_map[m1][evb_offdiag->atom_A_Rq[0]]] +1;
      
      int ttt = i-j-1;
      WRITEINT(ttt);
      WRITEINT(i);
      WRITEREAL(evb_matrix->e_extra[iextra][EOFF_ENE]);
      WRITEINT(evb_complex->reaction[i]);      
      WRITEINT(tt);
      WRITEINT(A1);
      WRITEINT(A2);
      WRITEINT(A3);
      iextra++;
    }
  }

  // checkout force

  #define CHECKOUT_FORCE_BIN() \
  for(int iatom=0; iatom<natom; iatom++) \
  { \
    int id; if(allatom) id=iatom; else id=list[iatom]; \
    fwrite(f[id],s_real,3,chk); \
  } 
 
  double **f, **f_save=atom->f;
  atom->f = f = evb_matrix->f_env;
  CHECKOUT_FORCE_BIN();  
    
  for(int i=0; i<evb_complex->nstate; i++)
  {
    atom->f = f = evb_matrix->f_diagonal[i];
    CHECKOUT_FORCE_BIN();
  }
  
  iextra=0;
  for(int i=1; i<evb_complex->nstate; i++)
  {
    atom->f = f = evb_matrix->f_off_diagonal[i-1];
    CHECKOUT_FORCE_BIN();
    
    for(int j=0; j<evb_complex->extra_coupling[i]; j++)
    {
      atom->f = f = evb_matrix->f_extra_coupling[iextra++];
      CHECKOUT_FORCE_BIN();
    }
  }
  
  atom->f = f_save;
  
  fclose(chk);
  delete [] map;
}

--- */

/* ---------------------------------------------------------------------- */

void EVB_Checkout::command(int narg, char **arg)
{
  // Read Inputs of Fitting Settings
  
  if(comm->nprocs!=1) error->universe_all(FLERR,"EVB_CHECKOUT: Must be run in 1-processor mode.");
  if(narg!=1) error->universe_all(FLERR,"EVB_CHECKOUT: Wrong number of parameters.");
  
  init(arg[0]);
  run();
  exit();
  
}

/* ---------------------------------------------------------------------- */
