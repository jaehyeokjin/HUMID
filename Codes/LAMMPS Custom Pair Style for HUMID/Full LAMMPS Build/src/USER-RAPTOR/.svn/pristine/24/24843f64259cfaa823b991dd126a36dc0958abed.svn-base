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
   Contributing authors: Roy Pollock (LLNL), Paul Crozier (SNL)

   Splitted for MS-EVB by: Tianying, Chris and Yuxing
------------------------------------------------------------------------- */

#include "mpi.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "math.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "domain.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "universe.h"
#include "mp_verlet_sci.h"

#include "EVB_ewald.h"
#include "EVB_offdiag.h"
#include "EVB_complex.h"
#include "EVB_matrix.h"
#include "EVB_engine.h"
#include "EVB_effpair.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.00001

#define PI 3.1415926535897932
#define RSQRTPI 1.772453851

/* ----------------------------------------------------------------------
   SCI functions
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void EVB_Ewald::sci_compute_env(int vflag)
{
  int i,n,k;
  
  energy = 0.0; 
  if (vflag) for (n = 0; n < 6; n++) virial[n] = 0.0;
  
  // partial structure factors on each processor
  // total structure factor by summing over procs

  eik_dot_r_env();
   
  double uk;
  
  for (k = 0; k < kcount; k++) {
      uk = ug[k] * (sfacrl_env[k]*sfacrl_env[k] + 
                    sfacim_env[k]*sfacim_env[k]);
      energy += uk;
      if (vflag)
          for (n = 0; n < 6; n++) virial[n] += uk*vg[k][n];
  }

  qsqsum = evb_engine->qsqsum_sys;
  for(int i=0; i<evb_engine->ncomplex; i++) qsqsum -= evb_engine->all_complex[i]->qsqsum;
  evb_engine->qsqsum_env = qsqsum;

  energy -= g_ewald*qsqsum/1.772453851;
  energy *= qqrd2e;

  if (vflag)
      for (n = 0; n < 6; n++) virial[n] *= qqrd2e;

  energy /= comm->nprocs;
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::sci_setup_init()
{
  int i,k,l,m,n,ic;
  int iatm;
  int i_cplx;
  double sqk,clpm,slpm;

  double **x = atom->x;
  int nlocal = atom->nlocal;
  int *complex_atom = evb_engine->complex_atom;
  int cplx_id = evb_engine->evb_complex->id;
  i_cplx =0;
  n = 0;

    // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
     
      i_cplx = 0;
      for (i = 0; i < nlocal; i++) {
        if (complex_atom[i] == cplx_id) {
          
          cs[0][ic][i] = 1.0;
          sn[0][ic][i] = 0.0;
          cs[1][ic][i] = cos(unitk[ic]*x[i][ic]);
          sn[1][ic][i] = sin(unitk[ic]*x[i][ic]);
          cs[-1][ic][i] = cs[1][ic][i];
          sn[-1][ic][i] = -sn[1][ic][i];
    
        
          eikrrl[n][i_cplx] = cs[1][ic][i];
          eikrim[n][i_cplx++] = sn[1][ic][i];
        }
      }
      n++;
    }
  }
  
  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 3; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
     
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
           if (complex_atom[i]==cplx_id) {
            
               cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
                   sn[m-1][ic][i]*sn[1][ic][i];
               sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
                   cs[m-1][ic][i]*sn[1][ic][i];
               cs[-m][ic][i] = cs[m][ic][i];
               sn[-m][ic][i] = -sn[m][ic][i];
               
               eikrrl[n][i_cplx] = cs[m][ic][i];
               eikrim[n][i_cplx++] = sn[m][ic][i];
           } 
        }
        n++;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kmax; k++) {
    for (l = 1; l <= kmax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
  
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
            
          if (complex_atom[i]==cplx_id) {
            eikrrl[n][i_cplx] = cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i];
            eikrim[n][i_cplx] = sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i];
            eikrrl[n+1][i_cplx] = cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i];
            eikrim[n+1][i_cplx++] = sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i];
          } 
        } 
        n+=2;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kmax; l++) {
    for (m = 1; m <= kmax; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
      
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
            
          if (complex_atom[i]==cplx_id) {
            eikrrl[n][i_cplx] = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            eikrim[n][i_cplx] = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            eikrrl[n+1][i_cplx] = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            eikrim[n+1][i_cplx++] = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
          }
        }
        n+=2;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kmax; k++) {
    for (m = 1; m <= kmax; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
     
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
         
            
          if (complex_atom[i]==cplx_id) {
            eikrrl[n][i_cplx] = cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i];
            eikrim[n][i_cplx] = sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i];
            eikrrl[n+1][i_cplx] = cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i];
            eikrim[n+1][i_cplx++] = sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i];
          }
        }
        n+=2;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kmax; k++) {
    for (l = 1; l <= kmax; l++) { 
      for (m = 1; m <= kmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
       
          i_cplx = 0;
          for (i = 0; i < nlocal; i++) {
              
            if (complex_atom[i]==cplx_id) {
              
                clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
                slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
           
                eikrrl[n][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
                eikrim[n][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
           

                clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
                slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
          
                eikrrl[n+1][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
                eikrim[n+1][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
         

                clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
                slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
          
                eikrrl[n+2][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
                eikrim[n+2][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
                
                
                clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
                slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
                
                eikrrl[n+3][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
                eikrim[n+3][i_cplx++] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
                
            }
          }
          n+=4;
        }
      }
    }
  }
}

/***********************************************************************************/

void EVB_Ewald::sci_setup_iteration()
{
  int i,k,l,m,n,ic;
  int iatm;
  int i_cplx;
  double cstr1,sstr1,cstr2,sstr2,cstr3,sstr3,cstr4,sstr4;
  double sqk,clpm,slpm;

  double **x = atom->x;
  int nlocal = atom->nlocal;
  double *q = evb_engine->evb_effpair->q;
  int *complex_atom = evb_engine->complex_atom;
  int cplx_id = evb_engine->evb_complex->id;
  i_cplx =0;
  n = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      cstr1 = 0.0;
      sstr1 = 0.0;
      i_cplx = 0;
      for (i = 0; i < nlocal; i++) {
        if(!complex_atom[i]) continue;
          
        cs[0][ic][i] = 1.0;
        sn[0][ic][i] = 0.0;
        cs[1][ic][i] = cos(unitk[ic]*x[i][ic]);
        sn[1][ic][i] = sin(unitk[ic]*x[i][ic]);
        cs[-1][ic][i] = cs[1][ic][i];
        sn[-1][ic][i] = -sn[1][ic][i];
    
        if (complex_atom[i] == cplx_id) {
          eikrrl[n][i_cplx] = cs[1][ic][i];
          eikrim[n][i_cplx++] = sn[1][ic][i];
        } else {
          cstr1 += q[i]*cs[1][ic][i];
          sstr1 += q[i]*sn[1][ic][i];
        }
      }
      sfacrl[n] = cstr1;
      sfacim[n++] = sstr1;
    }
  }

  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 3; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
          if(!complex_atom[i]) continue;
            
          cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
            sn[m-1][ic][i]*sn[1][ic][i];
          sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
            cs[m-1][ic][i]*sn[1][ic][i];
          cs[-m][ic][i] = cs[m][ic][i];
          sn[-m][ic][i] = -sn[m][ic][i];

          if (complex_atom[i]==cplx_id) {
            eikrrl[n][i_cplx] = cs[m][ic][i];
            eikrim[n][i_cplx++] = sn[m][ic][i];
          } else {
            cstr1 += q[i]*cs[m][ic][i];
            sstr1 += q[i]*sn[m][ic][i];
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kmax; k++) {
    for (l = 1; l <= kmax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
          if(!complex_atom[i]) continue;
            
          if (complex_atom[i]==cplx_id) {
            eikrrl[n][i_cplx] = cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i];
            eikrim[n][i_cplx] = sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i];
            eikrrl[n+1][i_cplx] = cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i];
            eikrim[n+1][i_cplx++] = sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i];
          } else {
            cstr1 += q[i]*(cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i]);
            sstr1 += q[i]*(sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i]);
            cstr2 += q[i]*(cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i]);
            sstr2 += q[i]*(sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i]);
          }
        } 
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kmax; l++) {
    for (m = 1; m <= kmax; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
          if(!complex_atom[i]) continue;
            
          if (complex_atom[i]==cplx_id) {
            eikrrl[n][i_cplx] = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            eikrim[n][i_cplx] = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            eikrrl[n+1][i_cplx] = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            eikrim[n+1][i_cplx++] = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
          } else {
            cstr1 += q[i]*(cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i]);
            sstr1 += q[i]*(sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i]);
            cstr2 += q[i]*(cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i]);
            sstr2 += q[i]*(sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i]);
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kmax; k++) {
    for (m = 1; m <= kmax; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        i_cplx = 0;
        for (i = 0; i < nlocal; i++) {
          if(!complex_atom[i]) continue;
            
          if (complex_atom[i]==cplx_id) {
            eikrrl[n][i_cplx] = cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i];
            eikrim[n][i_cplx] = sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i];
            eikrrl[n+1][i_cplx] = cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i];
            eikrim[n+1][i_cplx++] = sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i];
          } else {
            cstr1 += q[i]*(cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i]);
            sstr1 += q[i]*(sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i]);
            cstr2 += q[i]*(cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i]);
            sstr2 += q[i]*(sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i]);
          }
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kmax; k++) {
    for (l = 1; l <= kmax; l++) { 
      for (m = 1; m <= kmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          cstr3 = 0.0;
          sstr3 = 0.0;
          cstr4 = 0.0;
          sstr4 = 0.0;
          i_cplx = 0;
          for (i = 0; i < nlocal; i++) {
            if(!complex_atom[i]) continue;
              
            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            if (complex_atom[i]==cplx_id) {
              eikrrl[n][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim[n][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
            } else {
              cstr1 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr1 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            if (complex_atom[i]==cplx_id) {
              eikrrl[n+1][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim[n+1][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
            } else {
              cstr2 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr2 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            if (complex_atom[i]==cplx_id) {
              eikrrl[n+2][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim[n+2][i_cplx] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
            } else {
              cstr3 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr3 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }

            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            if (complex_atom[i]==cplx_id) {
              eikrrl[n+3][i_cplx] = cs[k][0][i]*clpm - sn[k][0][i]*slpm;
              eikrim[n+3][i_cplx++] = sn[k][0][i]*clpm + cs[k][0][i]*slpm;
            } else {
              cstr4 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
              sstr4 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
            }
          }
          sfacrl[n] = cstr1;
          sfacim[n++] = sstr1;
          sfacrl[n] = cstr2;
          sfacim[n++] = sstr2;
          sfacrl[n] = cstr3;
          sfacim[n++] = sstr3;
          sfacrl[n] = cstr4;
          sfacim[n++] = sstr4;
        }
      }
    }
  }

  MPI_Allreduce(sfacrl,sfacrl_inter,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_inter,kcount,MPI_DOUBLE,MPI_SUM,world);
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::sci_compute_cplx(int vflag)
{
  int i, k, n;

  energy = 0.0; 
  if (vflag) for (n = 0; n < 6; n++) virial[n] = 0.0; 

  int nlocal = atom->nlocal;
  int cplx_id = evb_engine->evb_complex->id;
  int *complex_atom = evb_engine->complex_atom;
  double *q = atom->q;
  
  for (k = 0; k < kcount; k++) {
      int i_cplx = 0;
      sfacrl[k] = 0.0;
      sfacim[k] = 0.0;
      for (i = 0; i < nlocal; i++) {
          if(complex_atom[i] && complex_atom[i]==cplx_id)
          {
              sfacrl[k] += q[i] * eikrrl[k][i_cplx];
              sfacim[k] += q[i] * eikrim[k][i_cplx++];
          }
      }
  }
  
  MPI_Allreduce(sfacrl,sfacrl_cplx,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_cplx,kcount,MPI_DOUBLE,MPI_SUM,world);

  // energy, virial if requested

  double uk;
  
  for (k = 0; k < kcount; k++) {
    uk = 2.0 * ug[k] * (sfacrl_cplx[k]*sfacrl_inter[k] +
                        sfacim_cplx[k]*sfacim_inter[k]);

    energy += uk;
    if (vflag)
      for (n = 0; n < 6; n++) virial[n] += uk*vg[k][n];
  }
  
  energy *= qqrd2e;

  if (vflag)
    for (n = 0; n < 6; n++) virial[n] *= qqrd2e;
  
  energy /= comm->nprocs;

  if(slabflag) slabcorr_sci_cplx();
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::sci_compute_exch(int vflag)
{
  int i,k,n;
  int iatm;

  int* is_exch = evb_engine->evb_effpair->is_exch;
  double* q_exch = evb_engine->evb_effpair->q_exch;
  int nlocal = atom->nlocal;
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int *cplx_list = evb_engine->evb_complex->cplx_list;
  
  off_diag_energy = 0.0;

  if (vflag) for (n = 0; n < 6; n++) off_diag_virial[n] = 0.0;

  /*****************************************************************/

    for (k = 0; k < kcount; k++) {
        sfacrl[k] = 0.0;
        sfacim[k] = 0.0;
        sfacrl_cplx[k] = 0.0;
        sfacim_cplx[k] = 0.0;
        for (i = 0; i < nlocal_cplx; i++) {
            iatm = cplx_list[i];
            if (is_exch[iatm]) {
                sfacrl[k] += q_exch[iatm] * eikrrl[k][i];
                sfacim[k] += q_exch[iatm] * eikrim[k][i];
            } 
        }
    }
  
  MPI_Allreduce(sfacrl,sfacrl_exch,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_exch,kcount,MPI_DOUBLE,MPI_SUM,world);

  /*****************************************************************/

  double uk;
  for (k = 0; k < kcount; k++) {
    uk = 2.0 * ug[k] * (sfacrl_exch[k]*sfacrl_inter[k] +
                        sfacim_exch[k]*sfacim_inter[k]);
    off_diag_energy += uk;
    if (vflag)
      for (n = 0; n < 6; n++) off_diag_virial[n] += uk*vg[k][n];
  }
  
  off_diag_energy *= qqrd2e;
  off_diag_energy /= comm->nprocs;

  if(slabflag) slabcorr_exch();
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::sci_compute_eff(int vflag)
{
  int kx, ky, kz;
  double cypz,sypz,exprl,expim,partial;
  double *q = atom->q;
  double **f = atom->f;
  
  int ncomplex = evb_engine->ncomplex;
  EVB_Complex * cplx;
  
  // When a k-space method is not used in the off-diagonals, then only the diagonals contribute to the effective charges.
  if(evb_engine->flag_DIAG_QEFF) for(int i=0; i<ncomplex; i++) {
      evb_engine->evb_complex = evb_engine->all_complex[i];
      evb_engine->evb_effpair->compute_q_eff(false,false);
    }
  
  // Compute k-space forces on ENV atoms due to CPLX atoms
  compute_eff(vflag);
  
  // Accumulate \rho(k) for only the CPLX atoms
  // If using a k-space method in the off-diagonals, then sfacrl_inter and sfacim_inter 
  //  already have this from compute_eff().
  if(evb_engine->flag_DIAG_QEFF) {
    
    for(int k=0; k<kcount; k++) sfacrl_inter[k] = sfacim_inter[k] = 0.0;
    
    for(int i=0; i<ncomplex; i++) {
      evb_engine->evb_complex = evb_engine->all_complex[i];
      evb_engine->evb_effpair->compute_q_eff(true,false);
      
      sci_setup_init();
      eik_dot_r_cplx();
      for(int k=0; k<kcount; k++) {
	sfacrl_inter[k] += sfacrl_cplx[k];
	sfacim_inter[k] += sfacim_cplx[k];
      }
    }
  } else {
    for(int k=0; k<kcount; k++) {
      sfacrl_inter[k] = sfacrl_cplx[k];
      sfacim_inter[k] = sfacim_cplx[k];
    }
  }
  
  // Loop over complexes.
  //  Subtract \rho(k) for complex and then calculate force on CPLX atoms due to all other CPLXs.
  for(int i=0; i<ncomplex; i++) {
    cplx = evb_engine->evb_complex = evb_engine->all_complex[i];
    sci_setup_init();
    eik_dot_r_cplx();
    
    for(int k=0; k<kcount; k++) {
      sfacrl[k] = sfacrl_inter[k]-sfacrl_cplx[k];
      sfacim[k] = sfacim_inter[k]-sfacim_cplx[k];
    }
    
    int nlocal_cplx = cplx->nlocal_cplx;
    int* cplx_list = cplx->cplx_list;
    
    for(int i=0; i<nlocal_cplx; i++) {
      int iatm = cplx_list[i];
      ek[iatm][0] = 0.0;
      ek[iatm][1] = 0.0;
      ek[iatm][2] = 0.0;
    }
    
    for (int k=0; k<kcount; k++) {
      
      kx = kxvecs[k];
      ky = kyvecs[k];
      kz = kzvecs[k];
      
      for (int i=0; i<nlocal_cplx; i++) {
	int iatm = cplx_list[i];
	cypz = cs[ky][1][iatm]*cs[kz][2][iatm] - sn[ky][1][iatm]*sn[kz][2][iatm];
	sypz = sn[ky][1][iatm]*cs[kz][2][iatm] + cs[ky][1][iatm]*sn[kz][2][iatm];
	exprl = cs[kx][0][iatm]*cypz - sn[kx][0][iatm]*sypz;
	expim = sn[kx][0][iatm]*cypz + cs[kx][0][iatm]*sypz;
	partial = expim*sfacrl[k] - exprl*sfacim[k];
	ek[iatm][0] += partial*eg[k][0];
	ek[iatm][1] += partial*eg[k][1];
	ek[iatm][2] += partial*eg[k][2];
      }
    }
    
    // convert E-field to force on env from cplx
    
    for (int i=0; i<nlocal_cplx; i++) {
      int iatm = cplx_list[i];
      double scale = qqrd2e * q[iatm];
      f[iatm][0] += scale * ek[iatm][0];
      f[iatm][1] += scale * ek[iatm][1];
      f[iatm][2] += scale * ek[iatm][2];   
    }
  }

  if(slabflag) slabcorr_eff();
}
/* ---------------------------------------------------------------------- */

void EVB_Ewald::compute_eff_mp(int vflag)
{
  if(!evb_engine->mp_verlet_sci->is_master) return;

  int i,k,n;
  int iatm;

  memset(sfacrl_inter, 0, sizeof(double)*kcount);
  memset(sfacim_inter, 0, sizeof(double)*kcount);

  int istart = universe->iworld;
  int di = universe->nworlds;

  for(int i=istart; i<evb_engine->ncomplex; i+=di) {
    evb_engine->evb_complex = evb_engine->all_complex[i];
    sci_setup_init();
    eik_dot_r_cplx();
    for(k=0; k<kcount; k++) {
      sfacrl_inter[k] += sfacrl_cplx[k];
      sfacim_inter[k] += sfacim_cplx[k];
    }
  }

  memset(sfacrl_cplx, 0, sizeof(double)*kcount);
  memset(sfacim_cplx, 0, sizeof(double)*kcount);

  MPI_Allreduce(sfacrl_inter, sfacrl_cplx, kcount, MPI_DOUBLE, MPI_SUM, evb_engine->mp_verlet_sci->block);
  MPI_Allreduce(sfacim_inter, sfacim_cplx, kcount, MPI_DOUBLE, MPI_SUM, evb_engine->mp_verlet_sci->block);

  // Only Master partition calculates forces
  if(!evb_engine->mp_verlet_sci->is_master) return;
  
  for (k = 0; k < kcount; k++) {
    sfacrl[k] = sfacrl_cplx[k] + sfacrl_env[k];
    sfacim[k] = sfacim_cplx[k] + sfacim_env[k];
  }

  // K-space portion of electric field
  // double loop over K-vectors and local atoms

  double **f = atom->f;
  double  *q = atom->q;
  
  // electric field on env from cplx

  int nlocal = atom->nlocal;
  int* complex_atom = evb_engine->complex_atom;
  
  for (iatm = 0; iatm < nlocal; iatm++)
    if(!complex_atom[iatm]) {
      ek[iatm][0] = 0.0;
      ek[iatm][1] = 0.0;
      ek[iatm][2] = 0.0;
    }
  
  int kx, ky, kz;
  double cypz,sypz,exprl,expim,partial;
  
  for (k = 0; k < kcount; k++) {
    
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];
    
    for (iatm = 0; iatm < nlocal; iatm++)
      if(!complex_atom[iatm]) {
	cypz = cs[ky][1][iatm]*cs[kz][2][iatm] - sn[ky][1][iatm]*sn[kz][2][iatm];
	sypz = sn[ky][1][iatm]*cs[kz][2][iatm] + cs[ky][1][iatm]*sn[kz][2][iatm];
	exprl = cs[kx][0][iatm]*cypz - sn[kx][0][iatm]*sypz;
	expim = sn[kx][0][iatm]*cypz + cs[kx][0][iatm]*sypz;
	partial = expim*sfacrl[k] - exprl*sfacim[k];
	ek[iatm][0] += partial*eg[k][0];
	ek[iatm][1] += partial*eg[k][1];
	ek[iatm][2] += partial*eg[k][2];
      }
  }
  
  // convert E-field to force on env from cplx
  
  for (iatm = 0; iatm < nlocal ; iatm++)
    if(!complex_atom[iatm]) {
      double scale = qqrd2e * q[iatm];
      f[iatm][0] += scale * ek[iatm][0];
      f[iatm][1] += scale * ek[iatm][1];
      f[iatm][2] += scale * ek[iatm][2];      
    }
  

  if(slabflag) slabcorr_eff();
}

/* ---------------------------------------------------------------------- */

void EVB_Ewald::sci_compute_eff_mp(int vflag)
{
  int kx, ky, kz;
  double cypz,sypz,exprl,expim,partial;
  double *q = atom->q;
  double **f = atom->f;
  
  int ncomplex = evb_engine->ncomplex;
  EVB_Complex * cplx;
  
  // When a k-space method is not used in the off-diagonals, then only the diagonals contribute to the effective charges.
  if(evb_engine->flag_DIAG_QEFF) for(int i=0; i<ncomplex; i++) {
      evb_engine->evb_complex = evb_engine->all_complex[i];
      evb_engine->evb_effpair->compute_q_eff(false,false);
    }

  compute_eff(vflag);
  //compute_eff_mp(vflag);  // not currently supported with mp_verlet_sci->is_master3
  
  // Accumulate \rho(k) for only the CPLX atoms
  // If using a k-space method in the off-diagonals, then sfacrl_inter and sfacim_inter 
  //  already have this from compute_eff().
  if(evb_engine->flag_DIAG_QEFF) {
    
    for(int k=0; k<kcount; k++) sfacrl_inter[k] = sfacim_inter[k] = 0.0;
    
    for(int i=0; i<ncomplex; i++) {
      evb_engine->evb_complex = evb_engine->all_complex[i];
      evb_engine->evb_effpair->compute_q_eff(true,false);
      
      sci_setup_init();
      eik_dot_r_cplx();
      for(int k=0; k<kcount; k++) {
	sfacrl_inter[k] += sfacrl_cplx[k];
	sfacim_inter[k] += sfacim_cplx[k];
      }
    }
  } else {
    memcpy(sfacrl_inter, sfacrl_cplx, sizeof(double)*kcount);
    memcpy(sfacim_inter, sfacim_cplx, sizeof(double)*kcount);
  }

  // Loop over complexes.
  //  Subtract \rho(k) for complex and then calculate force on CPLX atoms due to all other CPLXs.
  int istart = universe->iworld;
  int di = universe->nworlds;
  
  for(int i=0; i<ncomplex; i++) {
    //for(int i=istart; i<ncomplex; i+=di) {  // not currently supported with mp_verlet_sci->is_master3
    cplx = evb_engine->evb_complex = evb_engine->all_complex[i];
    sci_setup_init();
    eik_dot_r_cplx();
    
    for(int k=0; k<kcount; k++) {
      sfacrl[k] = sfacrl_inter[k] - sfacrl_cplx[k];
      sfacim[k] = sfacim_inter[k] - sfacim_cplx[k];
    }
    
    int nlocal_cplx = cplx->nlocal_cplx;
    int* cplx_list = cplx->cplx_list;
    
    for(int i=0; i<nlocal_cplx; i++) {
      int iatm = cplx_list[i];
      ek[iatm][0] = 0.0;
      ek[iatm][1] = 0.0;
      ek[iatm][2] = 0.0;
    }
    
    for (int k=0; k<kcount; k++) {
      
      kx = kxvecs[k];
      ky = kyvecs[k];
      kz = kzvecs[k];
      
      for (int i=0; i<nlocal_cplx; i++) {
	int iatm = cplx_list[i];
	cypz = cs[ky][1][iatm]*cs[kz][2][iatm] - sn[ky][1][iatm]*sn[kz][2][iatm];
	sypz = sn[ky][1][iatm]*cs[kz][2][iatm] + cs[ky][1][iatm]*sn[kz][2][iatm];
	exprl = cs[kx][0][iatm]*cypz - sn[kx][0][iatm]*sypz;
	expim = sn[kx][0][iatm]*cypz + cs[kx][0][iatm]*sypz;
	partial = expim*sfacrl[k] - exprl*sfacim[k];
	ek[iatm][0] += partial*eg[k][0];
	ek[iatm][1] += partial*eg[k][1];
	ek[iatm][2] += partial*eg[k][2];
      }
    }
    
    // convert E-field to force on env from cplx
    
    for (int i=0; i<nlocal_cplx; i++) {
      int iatm = cplx_list[i];
      double scale = qqrd2e * q[iatm];
      f[iatm][0] += scale * ek[iatm][0];
      f[iatm][1] += scale * ek[iatm][1];
      f[iatm][2] += scale * ek[iatm][2];   
    }
  }

  if(slabflag) slabcorr_eff();
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if 
   adequate empty space is left between repeating slabs (J. Chem. Phys. 
   111, 3155).  Slabs defined here to be parallel to the xy plane. 
------------------------------------------------------------------------- */

void EVB_Ewald::slabcorr_sci_cplx()
{
  // compute local cplx contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;

  int cplx_id = evb_engine->evb_complex->id;
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int * cplx_list = evb_engine->evb_complex->cplx_list;
  int * is_cplx_atom = evb_engine->complex_atom;

  double dipole_cplx    = 0.0;
  double dipole_r2_cplx = 0.0;
  for(int i=0; i<nlocal; i++) if(is_cplx_atom[i]) {
    dipole_cplx    += q[i] * x[i][2];
    dipole_r2_cplx += q[i] * x[i][2] * x[i][2];
  }
  
  // sum local contributions to get global dipole moment
  
  double dipole_all    = 0.0;
  double dipole_r2_all = 0.0;
  MPI_Allreduce(&dipole_cplx,    &dipole_all,    1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&dipole_r2_cplx, &dipole_r2_all, 1, MPI_DOUBLE, MPI_SUM, world);

  dipole_all    += dipole_env; // Total System Dipole
  dipole_r2_all += dipole_r2_env;

  // compute corrections

  const double e_slabcorr = MY_2PI * (dipole_all * dipole_all - qsum * dipole_r2_all - 
				      qsum * qsum * zprd * zprd / 12.0) / volume;
  
  energy += qqrd2e * e_slabcorr / comm->nprocs;

  // add on force corrections

  double ffact = -4.0 * MY_PI * qqrd2e / volume;
  double **f = atom->f;

  //  for(int i=0; i<nlocal; i++) if(is_cplx_atom[i] == cplx_id) f[i][2] += ffact * q[i] * (dipole_all - qsum*x[i][2]);
}
